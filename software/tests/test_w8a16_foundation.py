"""Phase 1 foundation smoke tests for the W8A16 path."""
from __future__ import annotations

import numpy as np
import pytest

from taccel.golden_model import memory as mem
from taccel.golden_model.state_w8a16 import MachineStateW8A16
from taccel.compiler.passes.memory_estimate_w8a16 import decide_seq_tiling_w8a16
from taccel.isa.opcodes import ABUF_SIZE, BUF_ABUF, BUF_ACCUM, BUF_WBUF
from taccel.model_config import ModelConfig


def test_machine_state_w8a16_instantiates():
    state = MachineStateW8A16()
    assert len(state.abuf) == ABUF_SIZE
    # ½ as many FP16 elements as INT8 bytes (2 bytes per FP16).
    assert state.abuf_view_fp16.shape == (ABUF_SIZE // 2,)
    # Accumulator stays 16K FP32 elements (mixed-precision standard).
    assert state.accum_view_fp32.shape == (16384,)


def test_read_write_fp16_tile_roundtrips_on_abuf():
    state = MachineStateW8A16()
    data = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) * np.float16(0.5)
    mem.write_fp16_tile(state, BUF_ABUF, 0, data)
    got = mem.read_fp16_tile(state, BUF_ABUF, 0, 16, 16)
    assert np.array_equal(data, got)


def test_read_write_fp16_tile_roundtrips_on_wbuf():
    state = MachineStateW8A16()
    data = (np.random.default_rng(0).standard_normal((16, 32))).astype(np.float16)
    mem.write_fp16_tile(state, BUF_WBUF, 0, data)
    got = mem.read_fp16_tile(state, BUF_WBUF, 0, 16, 32)
    assert np.array_equal(data, got)


def test_fp16_tile_on_accum_is_rejected():
    """ACCUM is FP32 in W8A16 — reading FP16 from it should raise."""
    state = MachineStateW8A16()
    arr = np.zeros((4, 4), dtype=np.float16)
    with pytest.raises(ValueError, match="ACCUM is FP32"):
        mem.read_fp16_tile(state, BUF_ACCUM, 0, 4, 4)
    with pytest.raises(ValueError, match="ACCUM is FP32"):
        mem.write_fp16_tile(state, BUF_ACCUM, 0, arr)


def test_decide_seq_tiling_w8a16_deit_tiny_tiles():
    """DeiT-tiny residual = 208*192*2 = 79.9 KB; full residual fits in 128KB
    ABUF, but the trigger (ABUF/3 = 42.7 KB) still forces the tiler to land
    on tile_rows=16 because the FC1 output [16, 768]*2 = 24 KB caps it."""
    cfg = ModelConfig.deit_tiny()
    decision = decide_seq_tiling_w8a16(cfg)
    assert decision.needs_tiling
    assert decision.tile_rows % 16 == 0
    # Per-tile FP16 residual must fit comfortably under ABUF/4.
    assert decision.per_tile_bytes <= ABUF_SIZE // 4


def test_decide_seq_tiling_w8a16_vit_base_picks_small_tile():
    """ViT-B FP16 residual = 208*768*2 = 312 KB → still needs tiling."""
    cfg = ModelConfig.vit_base()
    decision = decide_seq_tiling_w8a16(cfg)
    assert decision.needs_tiling
    assert decision.tile_rows % 16 == 0


def test_decide_seq_tiling_w8a16_consistency_with_per_tile_bytes():
    cfg = ModelConfig.vit_base()
    decision = decide_seq_tiling_w8a16(cfg)
    # per_tile_bytes must equal tile_rows × embed_dim × 2.
    assert decision.per_tile_bytes == decision.tile_rows * cfg.embed_dim * 2


def test_compiler_mode_literal_accepts_w8a16():
    """Phase 1 only extends the mode literal; full compile lives in Phase 3."""
    from taccel.compiler.compiler import Compiler
    c = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a16")
    assert c.mode == "w8a16"
