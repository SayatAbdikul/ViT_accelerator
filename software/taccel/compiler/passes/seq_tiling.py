"""Sequence-tiling IR rewrite.

When the residual tensor ``[seq_len_pad, embed_dim]`` no longer fits in ABUF
(true for ViT-B/16: 208 × 768 = 156 KB vs the 128 KB ABUF), the only way to
proceed without growing the RTL is to keep the residual in DRAM and stream
it through ABUF one sequence-tile at a time. This pass is the compile-time
half of that contract; the codegen half is the new ``tile_load`` /
``tile_save`` / ``init_residual_tile`` / ``concat_heads_tile`` IR ops.

Design constraints (FPGA cost):
  * **No new opcodes.** Tiles are emitted as a flat sequence of existing
    ISA ops (matmul, layernorm, vadd, …) with smaller dimensions; the
    "tile loop" is unrolled at compile time. The control unit, the
    systolic array, the SFU, the helper engine — none change.
  * **No new SRAM.** ABUF/WBUF/ACCUM stay at their current sizes. Tiling
    only changes *how* we use them, not how much we provision.
  * **Bounded DRAM staging.** A handful of fixed-size DRAM regions
    (two ping-ponged residuals + per-head Q/K/V/AttnV staging) backstop
    the residual stream. They are reused across the depth of the encoder.

Rewrite shape for one transformer block (T = num tiles, h = head index):

    # Phase A — per-tile pre-attention
    for t in 0..T:
        res_tile_t  = tile_load (residual_src_dram[t])
        ln1_tile_t  = layernorm  (res_tile_t,        ln1_w, ln1_b)
        for h in heads:
            q_tile_t_h = matmul (ln1_tile_t, Q_w_h, +Q_b_h)
            tile_save (q_tile_t_h → Q_stage_h[t])
            k_tile_t_h = matmul (ln1_tile_t, K_w_h, +K_b_h)
            tile_save (k_tile_t_h → K_stage_h[t])
            v_tile_t_h = matmul (ln1_tile_t, V_w_h, +V_b_h)
            tile_save (v_tile_t_h → V_stage_h[t])

    # Phase B — per-head full-seq attention
    for h in heads:
        q_h  = tile_load (Q_stage_h,  full)
        k_h  = tile_load (K_stage_h,  full)
        v_h  = tile_load (V_stage_h,  full)
        qkt  = matmul_qkt    (q_h, k_h)
        sc   = scale_mul     (qkt)
        sm   = softmax       (sc)
        av   = matmul_attn_v (sm, v_h)
        tile_save (av → AttnV_stage_h)

    # Phase C — per-tile post-attention
    for t in 0..T:
        cat_t   = concat_heads_tile(AttnV_stage_*, t)
        op_t    = matmul (cat_t, out_w, +out_b)
        skip_t  = tile_load (residual_src_dram[t])
        r1_t    = vadd      (op_t, skip_t)
        ln2_t   = layernorm (r1_t, ln2_w, ln2_b)
        fc1_t   = matmul    (ln2_t, fc1_w, +fc1_b, strip_mine)
        gelu_t  = gelu      (fc1_t, inline_with=fc1_t)
        fc2_t   = matmul    (gelu_t, fc2_w, +fc2_b, strip_mine)
        r2_t    = vadd      (fc2_t, r1_t)
        tile_save (r2_t → residual_dst_dram[t])

The two residual DRAM regions ``residual_dram_A`` and ``residual_dram_B``
are ping-ponged across blocks: block ``i`` reads from one and writes to
the other; block ``i+1`` swaps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..ir import IRGraph, IRNode
from ...model_config import ModelConfig
from .memory_estimate import decide_seq_tiling, TilingDecision
from .memory_estimate_w8a32 import decide_seq_tiling_w8a32


# ── DRAM staging region names. Codegen lazily allocates them on first
# reference via ``mem.alloc_dram_temp``. Reused across all encoder blocks
# (each block consumes and produces its own per-head staging within itself).
_RESIDUAL_A = "__seq_tile_residual_A"
_RESIDUAL_B = "__seq_tile_residual_B"


def _q_stage(h: int) -> str:
    return f"__seq_tile_Q_h{h}"


def _k_stage(h: int) -> str:
    return f"__seq_tile_K_h{h}"


def _v_stage(h: int) -> str:
    return f"__seq_tile_V_h{h}"


def _av_stage(h: int) -> str:
    return f"__seq_tile_AV_h{h}"


def seq_tiling_pass(graph: IRGraph, cfg: ModelConfig, ctx: Dict[str, Any]) -> IRGraph:
    """Rewrite ``graph`` to use sequence-tiled execution when needed.

    DeiT-tiny passes through unchanged (the policy says no tiling). ViT-B
    and any model whose ``[seq_len_pad, embed_dim]`` activation exceeds
    ``ABUF_SIZE/3`` gets the rewrite described in the module docstring.
    """
    decision = ctx.get("seq_tiling_decision") or decide_seq_tiling(cfg)
    ctx["seq_tiling_decision"] = decision
    if not decision.needs_tiling:
        return graph

    tile_rows = decision.tile_rows
    num_tiles = decision.num_tiles
    seq = cfg.seq_len_pad
    embed = cfg.embed_dim
    head_dim = cfg.head_dim
    num_heads = cfg.num_heads
    mlp_dim = cfg.mlp_dim
    num_classes = cfg.num_classes
    depth = cfg.depth
    prefix_root = cfg.module_prefix
    ctx["seq_tile_rows"] = tile_rows
    ctx["seq_num_tiles"] = num_tiles

    new = IRGraph()

    # Per-tile row range. The last tile may be short if seq_len_pad isn't
    # a multiple of tile_rows; codegen's existing pad-aware ops handle
    # the trailing partial tile via output_shape.
    def tile_row_count(t: int) -> int:
        start = t * tile_rows
        end = min(start + tile_rows, seq)
        return max(end - start, 0)

    def tile_byte_offset(t: int, cols: int) -> int:
        return t * tile_rows * cols

    # ── Phase 0: Initialise residual_A from CLS + patches + pos_embed.
    # One "init_residual_tile" node per tile. Codegen knows how to:
    #   t=0: load CLS to ABUF row 0, load patches rows [0:tile_rows-1] to
    #        ABUF rows 1..tile_rows, load pos_embed rows [0:tile_rows] to
    #        WBUF, VADD, then STORE to residual_A[0].
    #   t>0: load patches rows [t*tile_rows-1:(t+1)*tile_rows-1] to ABUF,
    #        load pos_embed rows [t*tile_rows:(t+1)*tile_rows] to WBUF,
    #        VADD, STORE to residual_A[t].
    for t in range(num_tiles):
        new.add_node(IRNode(
            op="init_residual_tile",
            name=f"init_residual_tile{t}",
            inputs=[
                f"{prefix_root}.embeddings.cls_token",
                f"{prefix_root}.embeddings.position_embeddings",
            ],
            output_shape=(tile_row_count(t), embed),
            attrs={
                "tile_idx": t,
                "tile_rows": tile_rows,
                "logical_rows": tile_row_count(t),
                "dst_dram": _RESIDUAL_A,
                "dst_offset_bytes": tile_byte_offset(t, embed),
                "total_dram_bytes": seq * embed,
            },
        ))

    # ── Encoder blocks. residual_A and residual_B ping-pong: block i reads
    # from src_dram and writes to dst_dram.
    for block_idx in range(depth):
        src_dram = _RESIDUAL_A if block_idx % 2 == 0 else _RESIDUAL_B
        dst_dram = _RESIDUAL_B if block_idx % 2 == 0 else _RESIDUAL_A
        prefix = f"{prefix_root}.encoder.layer.{block_idx}"
        b = f"block{block_idx}"

        # Phase A — per-tile pre-attention.
        for t in range(num_tiles):
            rows = tile_row_count(t)
            res_in = f"{b}_tile{t}_res_in"
            new.add_node(IRNode(
                op="tile_load",
                name=res_in,
                inputs=[],
                output_shape=(rows, embed),
                attrs={
                    "src_dram": src_dram,
                    "src_offset_bytes": tile_byte_offset(t, embed),
                    "total_dram_bytes": seq * embed,
                    "tile_rows": tile_rows,
                },
            ))

            ln1_name = f"{b}_tile{t}_ln1"
            new.add_node(IRNode(
                op="layernorm",
                name=ln1_name,
                inputs=[
                    res_in,
                    f"{prefix}.layernorm_before.weight",
                    f"{prefix}.layernorm_before.bias",
                ],
                output_shape=(rows, embed),
            ))

            for h in range(num_heads):
                for proj_name, stage_fn in (
                    ("query", _q_stage),
                    ("key", _k_stage),
                    ("value", _v_stage),
                ):
                    proj_node = f"{b}_tile{t}_head{h}_{proj_name}"
                    new.add_node(IRNode(
                        op="matmul",
                        name=proj_node,
                        inputs=[
                            ln1_name,
                            f"{prefix}.attention.attention.{proj_name}.weight_h{h}",
                        ],
                        output_shape=(rows, head_dim),
                        attrs={
                            "bias": f"{prefix}.attention.attention.{proj_name}.bias_h{h}",
                        },
                    ))
                    new.add_node(IRNode(
                        op="tile_save",
                        name=f"{proj_node}_save",
                        inputs=[proj_node],
                        output_shape=(rows, head_dim),
                        attrs={
                            "dst_dram": stage_fn(h),
                            "dst_offset_bytes": tile_byte_offset(t, head_dim),
                            "total_dram_bytes": seq * head_dim,
                        },
                    ))

        # Phase B — per-head full-sequence attention.
        for h in range(num_heads):
            q_name = f"{b}_head{h}_query"
            k_name = f"{b}_head{h}_key"
            v_name = f"{b}_head{h}_value"

            new.add_node(IRNode(
                op="tile_load", name=q_name,
                inputs=[], output_shape=(seq, head_dim),
                attrs={
                    "src_dram": _q_stage(h),
                    "src_offset_bytes": 0,
                    "total_dram_bytes": seq * head_dim,
                    "tile_rows": seq,
                },
            ))
            new.add_node(IRNode(
                op="tile_load", name=k_name,
                inputs=[], output_shape=(seq, head_dim),
                attrs={
                    "src_dram": _k_stage(h),
                    "src_offset_bytes": 0,
                    "total_dram_bytes": seq * head_dim,
                    "tile_rows": seq,
                },
            ))
            new.add_node(IRNode(
                op="tile_load", name=v_name,
                inputs=[], output_shape=(seq, head_dim),
                attrs={
                    "src_dram": _v_stage(h),
                    "src_offset_bytes": 0,
                    "total_dram_bytes": seq * head_dim,
                    "tile_rows": seq,
                },
            ))

            qkt_name = f"{b}_head{h}_qkt"
            new.add_node(IRNode(
                op="matmul_qkt", name=qkt_name,
                inputs=[q_name, k_name],
                output_shape=(seq, seq),
                attrs={"head_idx": h, "transpose_b": True},
            ))
            scale_name = f"{b}_head{h}_scale"
            new.add_node(IRNode(
                op="scale_mul", name=scale_name,
                inputs=[qkt_name],
                output_shape=(seq, seq),
                attrs={"scale": head_dim ** -0.5},
            ))
            softmax_name = f"{b}_head{h}_softmax"
            new.add_node(IRNode(
                op="softmax", name=softmax_name,
                inputs=[scale_name],
                output_shape=(seq, seq),
            ))
            attnv_name = f"{b}_head{h}_attn_v"
            new.add_node(IRNode(
                op="matmul_attn_v", name=attnv_name,
                inputs=[softmax_name, v_name],
                output_shape=(seq, head_dim),
                attrs={"head_idx": h},
            ))
            new.add_node(IRNode(
                op="tile_save",
                name=f"{attnv_name}_save",
                inputs=[attnv_name],
                output_shape=(seq, head_dim),
                attrs={
                    "dst_dram": _av_stage(h),
                    "dst_offset_bytes": 0,
                    "total_dram_bytes": seq * head_dim,
                },
            ))

        # Phase C — per-tile post-attention.
        for t in range(num_tiles):
            rows = tile_row_count(t)
            concat_name = f"{b}_tile{t}_concat"
            new.add_node(IRNode(
                op="concat_heads_tile",
                name=concat_name,
                inputs=[_av_stage(h) for h in range(num_heads)],
                output_shape=(rows, embed),
                attrs={
                    "tile_idx": t,
                    "tile_rows": tile_rows,
                    "logical_rows": rows,
                    "head_dim": head_dim,
                    "num_heads": num_heads,
                    "av_stage_total_bytes": seq * head_dim,
                },
            ))

            out_proj_name = f"{b}_tile{t}_out_proj"
            new.add_node(IRNode(
                op="matmul",
                name=out_proj_name,
                inputs=[concat_name, f"{prefix}.attention.output.dense.weight"],
                output_shape=(rows, embed),
                attrs={"bias": f"{prefix}.attention.output.dense.bias"},
            ))

            # Reload the residual skip for this tile (it was already consumed
            # in Phase A; ABUF doesn't hold it across the per-head attention).
            skip_name = f"{b}_tile{t}_res_skip"
            new.add_node(IRNode(
                op="tile_load", name=skip_name,
                inputs=[], output_shape=(rows, embed),
                attrs={
                    "src_dram": src_dram,
                    "src_offset_bytes": tile_byte_offset(t, embed),
                    "total_dram_bytes": seq * embed,
                    "tile_rows": tile_rows,
                },
            ))

            residual1_name = f"{b}_tile{t}_residual1"
            new.add_node(IRNode(
                op="vadd", name=residual1_name,
                inputs=[out_proj_name, skip_name],
                output_shape=(rows, embed),
            ))

            ln2_name = f"{b}_tile{t}_ln2"
            new.add_node(IRNode(
                op="layernorm", name=ln2_name,
                inputs=[
                    residual1_name,
                    f"{prefix}.layernorm_after.weight",
                    f"{prefix}.layernorm_after.bias",
                ],
                output_shape=(rows, embed),
            ))

            fc1_name = f"{b}_tile{t}_fc1"
            gelu_name = f"{b}_tile{t}_gelu"
            new.add_node(IRNode(
                op="matmul", name=fc1_name,
                inputs=[ln2_name, f"{prefix}.intermediate.dense.weight"],
                output_shape=(rows, mlp_dim),
                attrs={
                    "bias": f"{prefix}.intermediate.dense.bias",
                    "strip_mine": True,
                    "inline_gelu": gelu_name,
                },
            ))
            new.add_node(IRNode(
                op="gelu", name=gelu_name,
                inputs=[fc1_name],
                output_shape=(rows, mlp_dim),
                attrs={"inline_with": fc1_name},
            ))

            fc2_name = f"{b}_tile{t}_fc2"
            new.add_node(IRNode(
                op="matmul", name=fc2_name,
                inputs=[gelu_name, f"{prefix}.output.dense.weight"],
                output_shape=(rows, embed),
                attrs={
                    "bias": f"{prefix}.output.dense.bias",
                    "strip_mine": True,
                },
            ))

            residual2_name = f"{b}_tile{t}_residual2"
            new.add_node(IRNode(
                op="vadd", name=residual2_name,
                inputs=[fc2_name, residual1_name],
                output_shape=(rows, embed),
            ))

            new.add_node(IRNode(
                op="tile_save",
                name=f"{residual2_name}_save",
                inputs=[residual2_name],
                output_shape=(rows, embed),
                attrs={
                    "dst_dram": dst_dram,
                    "dst_offset_bytes": tile_byte_offset(t, embed),
                    "total_dram_bytes": seq * embed,
                },
            ))

    # ── Final stage: load only tile 0 of the last residual (CLS lives at
    # row 0), run final LN on that tile, extract row 0, classifier.
    last_dram = _RESIDUAL_A if depth % 2 == 0 else _RESIDUAL_B
    final_tile = "final_tile0_res"
    new.add_node(IRNode(
        op="tile_load", name=final_tile,
        inputs=[], output_shape=(tile_row_count(0), embed),
        attrs={
            "src_dram": last_dram,
            "src_offset_bytes": 0,
            "total_dram_bytes": seq * embed,
            "tile_rows": tile_rows,
        },
    ))

    final_ln = "final_ln"
    new.add_node(IRNode(
        op="layernorm", name=final_ln,
        inputs=[
            final_tile,
            f"{prefix_root}.layernorm.weight",
            f"{prefix_root}.layernorm.bias",
        ],
        output_shape=(tile_row_count(0), embed),
    ))

    new.add_node(IRNode(
        op="cls_extract", name="cls_extract",
        inputs=[final_ln],
        output_shape=(1, embed),
        attrs={"comment": "Extract row 0 (CLS token) from tile 0"},
    ))

    new.add_node(IRNode(
        op="matmul", name="classifier",
        inputs=["cls_extract", "classifier.weight"],
        output_shape=(1, num_classes),
        attrs={"bias": "classifier.bias"},
    ))

    return new


def seq_tiling_pass_w8a32(graph: IRGraph, cfg: ModelConfig, ctx: Dict[str, Any]) -> IRGraph:
    """Sequence-tiling pass tuned for the W8A32 path.

    Same IR rewrite as :func:`seq_tiling_pass`, but the policy comes from
    :func:`decide_seq_tiling_w8a32` (4× per-element bytes). Tiling is
    triggered for both DeiT-tiny and ViT-B in W8A32 mode because an FP32
    residual exceeds the 128 KB ABUF on both.
    """
    decision = decide_seq_tiling_w8a32(cfg)
    ctx["seq_tiling_decision"] = decision
    return seq_tiling_pass(graph, cfg, ctx)


__all__ = ["seq_tiling_pass", "seq_tiling_pass_w8a32"]
