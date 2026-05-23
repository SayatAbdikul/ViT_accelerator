"""M1.3 — ViT-B/16 end-to-end compile.

Documents what M1 unlocked and where M2 must pick up.

M1 deliverable: the compiler is model-agnostic — a ``ModelConfig`` flows
through ``Compiler`` → ``CodeGenerator`` → IR builder without any hardcoded
DeiT-tiny constant being load-bearing.

M2 deliverable (deferred): activation tiling so the [seq_len_pad, embed_dim]
INT8 tensor fits in the 128 KB ABUF. ViT-B's 208×768 = 156 KB exceeds the
ABUF budget; today the codegen fails with a clean MemoryError at the very
first allocation. This test pins that failure mode so a future regression
(e.g. activation tiling silently corrupting layouts) is detectable.
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


def test_vit_base_compile_hits_known_m2_boundary():
    """ViT-B's [208, 768] INT8 activation tensor is 156 KB and cannot fit in
    the 128 KB ABUF. M2 must add sequence/feature tiling. Until then the
    failure is loud and immediate."""
    cfg = ModelConfig.vit_base()
    sd = _synthetic_state_dict(cfg)
    with pytest.raises(MemoryError, match=r"Cannot allocate \d+B"):
        Compiler(cfg).compile(sd)
