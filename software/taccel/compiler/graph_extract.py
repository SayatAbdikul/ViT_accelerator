"""Build the TACCEL IR graph for a ViT-family model.

Historically this file hard-coded DeiT-tiny's dimensions. It is now a
config-driven IR builder: :func:`extract_vit_graph` takes a
:class:`ModelConfig` and produces the canonical IRGraph for any ViT-family
encoder + classifier head.

The legacy module-level constants below default to DeiT-tiny values and are
retained for backward compatibility with downstream consumers that imported
them directly. They are *not* the source of truth — :class:`ModelConfig` is.
"""
from __future__ import annotations

from ..model_config import ModelConfig
from .ir import IRGraph, IRNode

# ── Legacy DeiT-tiny constants ────────────────────────────────────────────────
# Kept for backward compatibility with `from .graph_extract import EMBED_DIM`
# style imports in compiler.py / codegen.py / tools/. New code should derive
# these from a ModelConfig instance.
_DEFAULT = ModelConfig.deit_tiny()
EMBED_DIM = _DEFAULT.embed_dim          # 192
DEPTH = _DEFAULT.depth                  # 12
NUM_HEADS = _DEFAULT.num_heads          # 3
HEAD_DIM = _DEFAULT.head_dim            # 64
MLP_RATIO = _DEFAULT.mlp_ratio          # 4
MLP_DIM = _DEFAULT.mlp_dim              # 768
SEQ_LEN = _DEFAULT.seq_len              # 197
PATCH_SIZE = _DEFAULT.patch_size        # 16
IMAGE_SIZE = _DEFAULT.image_size        # 224
NUM_PATCHES = _DEFAULT.num_patches      # 196
PATCH_DIM = _DEFAULT.patch_dim          # 768
NUM_CLASSES = _DEFAULT.num_classes      # 1000
SEQ_LEN_PAD = _DEFAULT.seq_len_pad      # 208


def extract_vit_graph(cfg: ModelConfig) -> IRGraph:
    """Build the IR graph for a ViT-family model with the given dimensions.

    The graph structure (per-head Q/K/V loop, GELU inlined into FC1,
    strip-mined MLPs) is identical across ViT variants — only the dimensions
    differ. Parameter names use the canonical ``{cfg.module_prefix}.*``
    namespace that ``ViTForImageClassification`` exposes.
    """
    graph = IRGraph()
    prefix_root = cfg.module_prefix
    seq_len = cfg.seq_len
    embed_dim = cfg.embed_dim
    head_dim = cfg.head_dim
    num_heads = cfg.num_heads
    mlp_dim = cfg.mlp_dim
    num_classes = cfg.num_classes

    # Patch embedding is done by the host (CPU pre-processing) before invoking the accelerator.
    # The program starts with [num_patches, embed_dim] INT8 embedded patches already in ABUF.
    # CLS prepend: load cls_token from DRAM → ABUF[0], embedded patches at ABUF[row 1]
    graph.add_node(IRNode(
        op="cls_prepend", name="cls_prepend",
        inputs=["embedded_patches", f"{prefix_root}.embeddings.cls_token"],
        output_shape=(seq_len, embed_dim),
    ))

    # Add position embedding: [seq_len, embed_dim] + [seq_len, embed_dim]
    graph.add_node(IRNode(
        op="pos_embed_add", name="pos_embed_add",
        inputs=["cls_prepend", f"{prefix_root}.embeddings.position_embeddings"],
        output_shape=(seq_len, embed_dim),
    ))

    prev_output = "pos_embed_add"

    # --- Transformer Blocks ---
    for block_idx in range(cfg.depth):
        prefix = f"{prefix_root}.encoder.layer.{block_idx}"
        b = f"block{block_idx}"

        # LayerNorm 1
        ln1_name = f"{b}_ln1"
        graph.add_node(IRNode(
            op="layernorm", name=ln1_name,
            inputs=[prev_output,
                    f"{prefix}.layernorm_before.weight",
                    f"{prefix}.layernorm_before.bias"],
            output_shape=(seq_len, embed_dim),
        ))

        # Per-head Q, K, V projections interleaved with attention computation.
        # Compute Q/K/V for one head, run its attention, free, then next head.
        # This keeps only one head's Q/K/V live at a time in ABUF.
        for h in range(num_heads):
            # Q, K, V projections for head h
            for proj in ["query", "key", "value"]:
                graph.add_node(IRNode(
                    op="matmul", name=f"{b}_head{h}_{proj}",
                    inputs=[ln1_name,
                            f"{prefix}.attention.attention.{proj}.weight_h{h}"],
                    output_shape=(seq_len, head_dim),
                    attrs={"bias": f"{prefix}.attention.attention.{proj}.bias_h{h}"},
                ))

            # Q_h @ K_h^T → [seq_len, seq_len]
            graph.add_node(IRNode(
                op="matmul_qkt", name=f"{b}_head{h}_qkt",
                inputs=[f"{b}_head{h}_query", f"{b}_head{h}_key"],
                output_shape=(seq_len, seq_len),
                attrs={"head_idx": h, "transpose_b": True},
            ))

            # Scale by 1/sqrt(d_head)
            graph.add_node(IRNode(
                op="scale_mul", name=f"{b}_head{h}_scale",
                inputs=[f"{b}_head{h}_qkt"],
                output_shape=(seq_len, seq_len),
                attrs={"scale": head_dim ** -0.5},
            ))

            # Softmax
            graph.add_node(IRNode(
                op="softmax", name=f"{b}_head{h}_softmax",
                inputs=[f"{b}_head{h}_scale"],
                output_shape=(seq_len, seq_len),
            ))

            # Attn @ V_h → [seq_len, head_dim]
            graph.add_node(IRNode(
                op="matmul_attn_v", name=f"{b}_head{h}_attn_v",
                inputs=[f"{b}_head{h}_softmax", f"{b}_head{h}_value"],
                output_shape=(seq_len, head_dim),
                attrs={"head_idx": h},
            ))

        # Concat heads: [num_heads, seq_len, head_dim] → [seq_len, embed_dim]
        graph.add_node(IRNode(
            op="concat_heads", name=f"{b}_concat",
            inputs=[f"{b}_head{h}_attn_v" for h in range(num_heads)],
            output_shape=(seq_len, embed_dim),
        ))

        # Output projection
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_out_proj",
            inputs=[f"{b}_concat", f"{prefix}.attention.output.dense.weight"],
            output_shape=(seq_len, embed_dim),
            attrs={"bias": f"{prefix}.attention.output.dense.bias"},
        ))

        # Residual add 1
        graph.add_node(IRNode(
            op="vadd", name=f"{b}_residual1",
            inputs=[f"{b}_out_proj", prev_output],
            output_shape=(seq_len, embed_dim),
        ))

        # LayerNorm 2
        ln2_name = f"{b}_ln2"
        graph.add_node(IRNode(
            op="layernorm", name=ln2_name,
            inputs=[f"{b}_residual1",
                    f"{prefix}.layernorm_after.weight",
                    f"{prefix}.layernorm_after.bias"],
            output_shape=(seq_len, embed_dim),
        ))

        # MLP: FC1 (strip-mined; GELU is applied inline per strip)
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_fc1",
            inputs=[ln2_name, f"{prefix}.intermediate.dense.weight"],
            output_shape=(seq_len, mlp_dim),
            attrs={"bias": f"{prefix}.intermediate.dense.bias",
                   "strip_mine": True,
                   "inline_gelu": f"{b}_gelu"},
        ))

        # GELU — no-op at codegen time (handled inline in FC1 strip loop)
        graph.add_node(IRNode(
            op="gelu", name=f"{b}_gelu",
            inputs=[f"{b}_fc1"],
            output_shape=(seq_len, mlp_dim),
            attrs={"inline_with": f"{b}_fc1"},
        ))

        # MLP: FC2
        graph.add_node(IRNode(
            op="matmul", name=f"{b}_fc2",
            inputs=[f"{b}_gelu", f"{prefix}.output.dense.weight"],
            output_shape=(seq_len, embed_dim),
            attrs={"bias": f"{prefix}.output.dense.bias",
                   "strip_mine": True},
        ))

        # Residual add 2
        graph.add_node(IRNode(
            op="vadd", name=f"{b}_residual2",
            inputs=[f"{b}_fc2", f"{b}_residual1"],
            output_shape=(seq_len, embed_dim),
        ))

        prev_output = f"{b}_residual2"

    # --- Final LayerNorm ---
    graph.add_node(IRNode(
        op="layernorm", name="final_ln",
        inputs=[prev_output, f"{prefix_root}.layernorm.weight", f"{prefix_root}.layernorm.bias"],
        output_shape=(seq_len, embed_dim),
    ))

    # --- CLS token extraction ---
    graph.add_node(IRNode(
        op="cls_extract", name="cls_extract",
        inputs=["final_ln"],
        output_shape=(1, embed_dim),
        attrs={"comment": "Extract row 0 (CLS token)"},
    ))

    # --- Classifier head ---
    graph.add_node(IRNode(
        op="matmul", name="classifier",
        inputs=["cls_extract", "classifier.weight"],
        output_shape=(1, num_classes),
        attrs={"bias": "classifier.bias"},
    ))

    return graph


def extract_deit_tiny() -> IRGraph:
    """Build IR graph for DeiT-tiny-patch16-224 (legacy entry point)."""
    return extract_vit_graph(ModelConfig.deit_tiny())
