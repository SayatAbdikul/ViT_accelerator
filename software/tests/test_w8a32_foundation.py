"""Phase 1 foundation smoke tests for the W8A32 path."""
from __future__ import annotations

import numpy as np
import pytest

from taccel.golden_model import memory as mem
from taccel.golden_model.state_w8a32 import MachineStateW8A32
from taccel.compiler.passes.memory_estimate_w8a32 import decide_seq_tiling_w8a32
from taccel.isa.opcodes import ABUF_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF
from taccel.model_config import ModelConfig


def test_machine_state_w8a32_instantiates():
    state = MachineStateW8A32()
    assert len(state.abuf) == ABUF_SIZE
    # ¼ as many FP32 elements as INT8 bytes.
    assert state.abuf_view_fp32.shape == (ABUF_SIZE // 4,)
    # Accumulator stays 16K (64KB / 4 bytes either way).
    assert state.accum_view_fp32.shape == (16384,)


def test_read_write_fp32_tile_roundtrips_on_abuf():
    state = MachineStateW8A32()
    data = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) * 0.5
    mem.write_fp32_tile(state, BUF_ABUF, 0, data)
    got = mem.read_fp32_tile(state, BUF_ABUF, 0, 16, 16)
    assert np.array_equal(data, got)


def test_read_write_fp32_tile_roundtrips_on_wbuf():
    state = MachineStateW8A32()
    data = (np.random.default_rng(0).standard_normal((16, 32))).astype(np.float32)
    mem.write_fp32_tile(state, BUF_WBUF, 0, data)
    got = mem.read_fp32_tile(state, BUF_WBUF, 0, 16, 32)
    assert np.array_equal(data, got)


def test_read_write_fp32_tile_roundtrips_on_accum():
    state = MachineStateW8A32()
    data = (np.random.default_rng(1).standard_normal((16, 16)) * 100).astype(np.float32)
    mem.write_fp32_tile(state, BUF_ACCUM, 0, data)
    got = mem.read_fp32_tile(state, BUF_ACCUM, 0, 16, 16)
    assert np.array_equal(data, got)


def test_decide_seq_tiling_w8a32_deit_tiny_tiles():
    """DeiT-tiny residual = 208*192*4 = 156 KB > 128 KB ABUF → tile."""
    cfg = ModelConfig.deit_tiny()
    decision = decide_seq_tiling_w8a32(cfg)
    assert decision.needs_tiling
    assert decision.tile_rows % 16 == 0
    # Per-tile FP32 residual must fit comfortably (≤ ABUF/4 = 32 KB).
    assert decision.per_tile_bytes <= ABUF_SIZE // 4


def test_decide_seq_tiling_w8a32_vit_base_picks_small_tile():
    """ViT-B residual = 208*768*4 = 624 KB → very tight tiling."""
    cfg = ModelConfig.vit_base()
    decision = decide_seq_tiling_w8a32(cfg)
    assert decision.needs_tiling
    assert decision.tile_rows == 16  # only multiples-of-16 that fit
    assert decision.num_tiles == (cfg.seq_len_pad + 15) // 16


def test_decide_seq_tiling_w8a32_consistency_with_per_tile_bytes():
    cfg = ModelConfig.vit_base()
    decision = decide_seq_tiling_w8a32(cfg)
    # per_tile_bytes must equal tile_rows × embed_dim × 4.
    assert decision.per_tile_bytes == decision.tile_rows * cfg.embed_dim * 4
