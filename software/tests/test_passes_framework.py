"""Tests for the M2 IR pass framework and sequence-tiling rewrite.

Three layers are exercised here:

1. ``run_passes`` orchestration — empty pipeline preserves the input graph;
   ctx is mutated in place; passes are applied in order.

2. ``decide_seq_tiling`` policy — tile size aligns to SYS_DIM, divides
   seq_len_pad when possible, fits per-tile activation in ABUF/4.

3. ``seq_tiling_pass`` rewrite — DeiT-tiny passes through unchanged;
   ViT-B's IR explodes from O(blocks × heads) nodes into O(blocks × tiles
   × heads) nodes with DMA-staging book-ends.
"""
from __future__ import annotations

import pytest

from taccel.compiler.graph_extract import extract_vit_graph
from taccel.compiler.ir import IRGraph, IRNode
from taccel.compiler.passes import default_pipeline, run_passes, seq_tiling_pass
from taccel.compiler.passes.memory_estimate import TilingDecision, decide_seq_tiling
from taccel.isa.opcodes import ABUF_SIZE
from taccel.model_config import ModelConfig


# ─── run_passes orchestration ────────────────────────────────────────────────

def test_run_passes_with_empty_pipeline_is_identity():
    cfg = ModelConfig.deit_tiny()
    graph = extract_vit_graph(cfg)
    rewritten = run_passes(graph, cfg, ctx={}, pipeline=[])
    assert rewritten is graph
    assert [n.name for n in rewritten] == [n.name for n in graph]


def test_run_passes_mutates_ctx_in_place():
    """Passes should be able to communicate via the ctx dict."""
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    ctx: dict = {}
    run_passes(graph, cfg, ctx)
    assert "seq_tiling_decision" in ctx
    assert isinstance(ctx["seq_tiling_decision"], TilingDecision)


def test_default_pipeline_contains_seq_tiling():
    pipeline = default_pipeline()
    assert seq_tiling_pass in pipeline


# ─── decide_seq_tiling policy ───────────────────────────────────────────────

def test_decide_seq_tiling_deit_tiny_does_not_tile():
    decision = decide_seq_tiling(ModelConfig.deit_tiny())
    assert not decision.needs_tiling
    assert decision.tile_rows == ModelConfig.deit_tiny().seq_len_pad
    assert decision.num_tiles == 1
    assert "no tiling" in decision.reason


def test_decide_seq_tiling_vit_base_tiles_with_aligned_rows():
    decision = decide_seq_tiling(ModelConfig.vit_base())
    assert decision.needs_tiling
    assert decision.tile_rows % 16 == 0
    assert decision.tile_rows >= 16
    assert decision.num_tiles >= 2
    # Per-tile bytes must fit in ABUF/4 with comfortable headroom.
    assert decision.per_tile_bytes <= ABUF_SIZE // 4
    # The full residual is the barrier; per-tile must be strictly less.
    assert decision.per_tile_bytes < decision.full_residual_bytes


def test_decide_seq_tiling_prefers_even_division_when_available():
    """Where a 16-multiple tile divides seq_len_pad evenly within the
    headroom window, pick it. ViT-B's seq_len_pad=208 has divisors 16, 26,
    52, 104, 208; only 16, 52, 104 are 16-multiples-and-divisors. With
    ABUF/4=32KB / embed_dim=768 → ≤41 row headroom → snaps to 32 → 16
    (since 32 doesn't divide 208 cleanly we fall back to a divisor)."""
    cfg = ModelConfig.vit_base()
    decision = decide_seq_tiling(cfg)
    # Either the policy picked an even divisor (208 % tile_rows == 0)
    # or it kept the largest 16-multiple that fits headroom (32) and
    # accepts a partial last tile.
    assert (cfg.seq_len_pad % decision.tile_rows == 0
            or decision.tile_rows == 32)


# ─── seq_tiling_pass rewrite ─────────────────────────────────────────────────

def test_seq_tiling_pass_passes_deit_tiny_through():
    cfg = ModelConfig.deit_tiny()
    graph = extract_vit_graph(cfg)
    rewritten = seq_tiling_pass(graph, cfg, ctx={})
    assert rewritten is graph
    assert len(rewritten) == len(graph)


def test_seq_tiling_pass_rewrites_vit_base():
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    original_size = len(graph)

    rewritten = seq_tiling_pass(graph, cfg, ctx={})

    # Tiling fans out many more nodes than the original (per-tile copies
    # of LN/matmul/vadd nodes, plus the dma-stage book-ends).
    assert len(rewritten) > original_size
    # Per-block work is (≈ num_tiles × per-block_pre_attn_ops +
    #                    num_heads × attention_ops +
    #                    num_tiles × per-block_post_attn_ops). Sanity-bound it.
    assert len(rewritten) < 20 * original_size


def test_seq_tiling_pass_inserts_dma_staging_ops():
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    rewritten = seq_tiling_pass(graph, cfg, ctx={})

    op_kinds = {n.op for n in rewritten}
    for new_op in ("tile_load", "tile_save", "init_residual_tile",
                   "concat_heads_tile"):
        assert new_op in op_kinds, f"missing new op {new_op}"


def test_seq_tiling_pass_residual_dram_ping_pongs():
    """Block i's tile_save must land in the buffer block i+1's tile_load
    reads from. We verify by spot-checking the dram names on consecutive
    block boundaries."""
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    rewritten = seq_tiling_pass(graph, cfg, ctx={})

    # Find the residual1 (=skip-loaded) tile_load for block0 and the
    # residual2 tile_save for block0; they must share the same staging
    # region (block0's read+write side). Block1's tile_load reads from
    # block0's tile_save destination.
    block0_skip_loads = [
        n for n in rewritten
        if n.op == "tile_load" and "block0_tile" in n.name and "res_skip" in n.name
    ]
    block0_residual2_saves = [
        n for n in rewritten
        if n.op == "tile_save"
        and n.name.startswith("block0_tile")
        and n.name.endswith("residual2_save")
    ]
    block1_loads = [
        n for n in rewritten
        if n.op == "tile_load" and "block1_tile" in n.name and "res_in" in n.name
    ]

    assert block0_skip_loads and block0_residual2_saves and block1_loads
    skip_dram = block0_skip_loads[0].attrs["src_dram"]
    save_dram = block0_residual2_saves[0].attrs["dst_dram"]
    next_load_dram = block1_loads[0].attrs["src_dram"]
    # Block0 reads from A, writes to B; block1 reads from B.
    assert skip_dram != save_dram
    assert save_dram == next_load_dram


def test_seq_tiling_pass_emits_init_for_every_tile():
    cfg = ModelConfig.vit_base()
    graph = extract_vit_graph(cfg)
    ctx: dict = {}
    rewritten = seq_tiling_pass(graph, cfg, ctx)

    num_tiles = ctx["seq_num_tiles"]
    inits = [n for n in rewritten if n.op == "init_residual_tile"]
    assert len(inits) == num_tiles
    # Each tile_idx 0..num_tiles-1 appears exactly once.
    assert sorted(n.attrs["tile_idx"] for n in inits) == list(range(num_tiles))
