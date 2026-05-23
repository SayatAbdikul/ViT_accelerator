"""M1.3 / M2 — ViT-B/16 end-to-end compile.

Documents what M1 unlocked, what M2 (sequence tiling) added, and where M3
must pick up.

M1 deliverable: the compiler is model-agnostic — a ``ModelConfig`` flows
through ``Compiler`` → ``CodeGenerator`` → IR builder without any hardcoded
DeiT-tiny constant being load-bearing.

M2 deliverable: sequence tiling. The seq_tiling pass rewrites every encoder
block so the [seq_len_pad, embed_dim] residual stream is staged through
DRAM rather than held whole in ABUF. ViT-B's 208×768 = 156 KB no longer
exceeds the 128 KB ABUF; per-tile activations are ≤ 24 KB. The compile
now advances past the residual barrier and reaches the next bottleneck:
the wide out_proj / FC1 / FC2 weight matrices (≥ 576 KB) exceed the
256 KB WBUF. That is the M3 entry point — weight-side N-strip mining,
distinct from M2's activation-side tiling.
"""
from __future__ import annotations

import pytest
import torch

from taccel.compiler.compiler import Compiler
from taccel.compiler.graph_extract import extract_vit_graph
from taccel.model_config import ModelConfig


def _synthetic_state_dict(cfg: ModelConfig):
    torch.manual_seed(0)

    def t(*shape):
        return torch.randn(*shape) * 0.1

    sd = {
        f"{cfg.module_prefix}.embeddings.cls_token": t(1, 1, cfg.embed_dim),
        f"{cfg.module_prefix}.embeddings.position_embeddings": t(1, cfg.seq_len, cfg.embed_dim),
        f"{cfg.module_prefix}.layernorm.weight": t(cfg.embed_dim),
        f"{cfg.module_prefix}.layernorm.bias": t(cfg.embed_dim),
        "classifier.weight": t(cfg.num_classes, cfg.embed_dim),
        "classifier.bias": t(cfg.num_classes),
    }
    for i in range(cfg.depth):
        p = f"{cfg.module_prefix}.encoder.layer.{i}"
        sd[f"{p}.layernorm_before.weight"] = t(cfg.embed_dim)
        sd[f"{p}.layernorm_before.bias"] = t(cfg.embed_dim)
        sd[f"{p}.layernorm_after.weight"] = t(cfg.embed_dim)
        sd[f"{p}.layernorm_after.bias"] = t(cfg.embed_dim)
        for proj in ["query", "key", "value"]:
            sd[f"{p}.attention.attention.{proj}.weight"] = t(cfg.embed_dim, cfg.embed_dim)
            sd[f"{p}.attention.attention.{proj}.bias"] = t(cfg.embed_dim)
        sd[f"{p}.attention.output.dense.weight"] = t(cfg.embed_dim, cfg.embed_dim)
        sd[f"{p}.attention.output.dense.bias"] = t(cfg.embed_dim)
        sd[f"{p}.intermediate.dense.weight"] = t(cfg.mlp_dim, cfg.embed_dim)
        sd[f"{p}.intermediate.dense.bias"] = t(cfg.mlp_dim)
        sd[f"{p}.output.dense.weight"] = t(cfg.embed_dim, cfg.mlp_dim)
        sd[f"{p}.output.dense.bias"] = t(cfg.embed_dim)
    return sd


def test_vit_base_ir_graph_builds():
    """The IR builder is model-agnostic: ViT-B yields a valid IRGraph."""
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    # 12 blocks × (12 heads × 6 + 8 non-head ops) + 5 framing nodes.
    # Anything in the 1000-1200 range is fine — the exact count is incidental.
    assert 1000 < len(graph) < 1200
    assert graph.get_node("classifier").output_shape == (1, cfg.num_classes)


def test_deit_tiny_compiler_with_explicit_cfg_matches_default():
    """Passing ModelConfig.deit_tiny() explicitly must match the default
    (no-cfg) compile — proves the cfg pathway is consistent."""
    cfg = ModelConfig.deit_tiny()
    sd = _synthetic_state_dict(cfg)

    prog_default = Compiler().compile(sd)
    prog_explicit = Compiler(cfg).compile(sd)

    assert prog_default.insn_count == prog_explicit.insn_count
    assert prog_default.instructions == prog_explicit.instructions
    assert prog_default.data == prog_explicit.data


def test_vit_base_compile_hits_known_m3_boundary():
    """After M2 sequence tiling, ViT-B advances past the ABUF residual
    barrier and reaches the *next* boundary: WBUF cannot hold the wide
    out_proj / FC1 / FC2 weight matrices (each ≥ 576 KB, vs 256 KB WBUF).
    M3 weight-side N-strip mining will close that gap. Until then the
    failure is loud and immediate."""
    cfg = ModelConfig.vit_base()
    sd = _synthetic_state_dict(cfg)
    with pytest.raises(MemoryError, match=r"Cannot allocate \d+B.*buffer 1"):
        Compiler(cfg).compile(sd)


def test_vit_base_seq_tiling_pass_activates():
    """The seq_tiling pass populates the pass context with its decision
    when invoked from compile(). For ViT-B the policy demands tiling
    (residual > ABUF/3); for DeiT-T it does not."""
    from taccel.compiler.passes.memory_estimate import decide_seq_tiling

    vit_b = decide_seq_tiling(ModelConfig.vit_base())
    assert vit_b.needs_tiling
    # tile_rows must align with SYS_DIM=16 and yield more than 1 tile.
    assert vit_b.tile_rows % 16 == 0
    assert vit_b.num_tiles >= 2
    # Per-tile bytes must comfortably clear ABUF/4 (32 KB for the current
    # 128 KB ABUF) so the LN output + per-head Q/K/V can coexist.
    assert vit_b.per_tile_bytes <= 32 * 1024

    deit_t = decide_seq_tiling(ModelConfig.deit_tiny())
    assert not deit_t.needs_tiling
    assert deit_t.num_tiles == 1


def test_vit_base_ir_after_tiling_has_dma_stage_nodes():
    """The rewritten IR includes init_residual_tile, tile_load, tile_save,
    and concat_heads_tile ops — the four new node kinds M2 introduced."""
    from taccel.compiler.passes import run_passes

    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    rewritten = run_passes(graph, cfg, {})

    op_kinds = {n.op for n in rewritten}
    assert "init_residual_tile" in op_kinds
    assert "tile_load" in op_kinds
    assert "tile_save" in op_kinds
    assert "concat_heads_tile" in op_kinds
    # The original full-seq cls_prepend / pos_embed_add are gone (replaced
    # by per-tile init_residual_tile nodes).
    assert "cls_prepend" not in op_kinds
    assert "pos_embed_add" not in op_kinds
