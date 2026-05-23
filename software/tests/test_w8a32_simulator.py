"""Phase 2 unit tests: per-op W8A32 simulator handlers.

Each test drives one instruction through ``SimulatorW8A32`` with hand-written
ABUF/WBUF state and compares against a NumPy oracle within ``rtol=1e-5``.

The buffers are addressed in 16-byte units; for an FP32 tile of shape
(M, N) the WBUF needs ``M * N * 4`` bytes (4 bytes per FP32 element).
"""
from __future__ import annotations

import numpy as np
import pytest

from taccel.golden_model import memory as mem
from taccel.golden_model.simulator_w8a32 import SimulatorW8A32
from taccel.golden_model.state_w8a32 import MachineStateW8A32
from taccel.isa.instructions import (
    ConfigTileInsn, DequantAddInsn, GeluInsn, LayernormInsn, MatmulInsn,
    RequantInsn, RequantPcInsn, ScaleMulInsn, SoftmaxInsn, SoftmaxAttnVInsn,
    VaddInsn,
)
from taccel.isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF
from taccel.utils import fp32_prim_ref as fpr


def _make_sim(M_tiles=1, N_tiles=1, K_tiles=1):
    sim = SimulatorW8A32()
    sim.state.tile_config = (M_tiles - 1, N_tiles - 1, K_tiles - 1)
    return sim


# ── MATMUL ────────────────────────────────────────────────────────────


def test_matmul_w8a32_fp32_act_fp32_weight():
    """FP32 act × FP32 weight → FP32 ACCUM. W8A32 stores dequant FP32 weights."""
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    M, N, K = 16, 16, 16
    rng = np.random.default_rng(42)
    act = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.1

    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, act)
    mem.write_fp32_tile(sim.state, BUF_WBUF, 0, w)

    insn = MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=0,
    )
    sim._execute(insn)

    expected = act @ w
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, M, N)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_matmul_w8a32_accumulate_flag():
    """flags=1 → dst += src1 @ src2."""
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    rng = np.random.default_rng(1)
    initial_accum = (rng.standard_normal((16, 16)) * 100).astype(np.float32)
    act = rng.standard_normal((16, 16)).astype(np.float32) * 0.1
    w = rng.standard_normal((16, 16)).astype(np.float32) * 0.05

    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, initial_accum)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, act)
    mem.write_fp32_tile(sim.state, BUF_WBUF, 0, w)

    insn = MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=1,
    )
    sim._execute(insn)

    expected = initial_accum + act @ w
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ── VADD ──────────────────────────────────────────────────────────────


def test_vadd_w8a32_abuf_path_fp32_elementwise_add():
    sim = _make_sim()
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 16)).astype(np.float32)
    b = rng.standard_normal((16, 16)).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, a)
    # b at offset 64 units (1024 bytes = 16×16×4).
    mem.write_fp32_tile(sim.state, BUF_ABUF, 64, b)
    insn = VaddInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=64,
        dst_buf=BUF_ABUF, dst_off=128,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 128, 16, 16)
    np.testing.assert_allclose(got, a + b, rtol=1e-6, atol=1e-6)


def test_vadd_w8a32_accum_path_broadcast_bias():
    sim = _make_sim()
    rng = np.random.default_rng(3)
    accum = (rng.standard_normal((16, 16)) * 10).astype(np.float32)
    bias = (rng.standard_normal((1, 16)) * 2).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp32_tile(sim.state, BUF_WBUF, 0, bias)
    insn = VaddInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    np.testing.assert_allclose(got, accum + bias, rtol=1e-6, atol=1e-6)


# ── REQUANT / REQUANT_PC ──────────────────────────────────────────────


def test_requant_w8a32_is_fp32_scale_no_clip():
    """REQUANT in W8A32 = FP32 ACCUM × scalar scale → FP32 ABUF, no clip."""
    sim = _make_sim()
    rng = np.random.default_rng(4)
    accum = (rng.standard_normal((16, 16)) * 1000).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    sim.state.scale_regs[0] = np.float16(0.5)

    insn = RequantInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,  # unused
        dst_buf=BUF_ABUF, dst_off=0,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 0, 16, 16)
    expected = (accum * np.float32(np.float16(0.5))).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # Sanity: values that would have saturated in W8A8 (|x|>127) pass through.
    assert np.max(np.abs(got)) > 100


def test_requant_pc_w8a32_per_channel_scales_fp32_out():
    sim = _make_sim()
    rng = np.random.default_rng(5)
    accum = (rng.standard_normal((16, 32)) * 50).astype(np.float32)
    pc_scales = (rng.uniform(0.01, 0.1, size=32)).astype(np.float16)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    # scale table goes in WBUF as packed FP16 (64 bytes = 4 units).
    mem.write_bytes(sim.state, BUF_WBUF, 0, pc_scales.tobytes())

    insn = RequantPcInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=0,
    )
    # tile_config wants N=32 → n_tiles=2.
    sim.state.tile_config = (0, 1, 0)
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 0, 16, 32)
    expected = (accum * pc_scales.astype(np.float32)[None, :]).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


# ── DEQUANT_ADD ──────────────────────────────────────────────────────


def test_dequant_add_w8a32_fp32_fused_residual():
    sim = _make_sim()
    rng = np.random.default_rng(6)
    accum = (rng.standard_normal((16, 16)) * 100).astype(np.float32)
    skip = (rng.standard_normal((16, 16))).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, skip)
    sim.state.scale_regs[0] = np.float16(0.01)
    sim.state.scale_regs[1] = np.float16(1.0)
    insn = DequantAddInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)
    expected = (accum * np.float32(np.float16(0.01))
                + skip * np.float32(np.float16(1.0))).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ── SCALE_MUL ────────────────────────────────────────────────────────


def test_scale_mul_w8a32_abuf_fp32():
    sim = _make_sim()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 16)).astype(np.float32) * 2.0
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, x)
    sim.state.scale_regs[0] = np.float16(0.25)
    insn = ScaleMulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,  # unused
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)
    expected = (x * np.float32(np.float16(0.25))).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


# ── LAYERNORM ────────────────────────────────────────────────────────


def test_layernorm_w8a32_matches_numpy_oracle():
    sim = _make_sim()
    rng = np.random.default_rng(8)
    x = (rng.standard_normal((16, 16)) * 3.0).astype(np.float32)
    gamma = (rng.uniform(0.5, 1.5, size=16)).astype(np.float16)
    beta = (rng.standard_normal(16) * 0.1).astype(np.float16)

    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, x)
    # Pack gamma | beta as FP16 into WBUF.
    gb_bytes = gamma.tobytes() + beta.tobytes()
    mem.write_bytes(sim.state, BUF_WBUF, 0, gb_bytes)

    insn = LayernormInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)

    # Match the same FP32 primitive order the W8A32 implementation uses.
    gamma_fp32 = fpr.fp32_from_fp16_arr(np.frombuffer(gamma.tobytes(), dtype=np.uint16))
    beta_fp32 = fpr.fp32_from_fp16_arr(np.frombuffer(beta.tobytes(), dtype=np.uint16))
    mean = fpr.fp32_mean_rows(x)[:, None]
    var = fpr.fp32_var_rows(x, mean)[:, None]
    denom = np.sqrt((var + np.float32(1e-6)).astype(np.float32), dtype=np.float32)
    x_norm = ((x - mean) / denom).astype(np.float32)
    expected = (x_norm * gamma_fp32 + beta_fp32).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ── SOFTMAX ──────────────────────────────────────────────────────────


def test_softmax_w8a32_matches_numpy_softmax_on_abuf_input():
    sim = _make_sim()
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((16, 16)) * 3.0).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, x)
    insn = SoftmaxInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)
    # Row-wise softmax over last axis with numerical stabilization.
    x_shifted = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x_shifted.astype(np.float32))
    expected = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got.sum(axis=-1), np.ones(16), rtol=1e-5, atol=1e-5)


# ── GELU ─────────────────────────────────────────────────────────────


def test_gelu_w8a32_matches_fp32_gelu_oracle():
    sim = _make_sim()
    rng = np.random.default_rng(10)
    x = (rng.standard_normal((16, 16)) * 2.0).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, x)
    insn = GeluInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)
    expected = fpr.fp32_gelu_arr(x).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


# ── SOFTMAX_ATTNV ────────────────────────────────────────────────────


def test_softmax_attnv_w8a32_fp32_endtoend():
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    rng = np.random.default_rng(11)
    qkt = (rng.standard_normal((16, 16)) * 0.5).astype(np.float32)
    v = rng.standard_normal((16, 16)).astype(np.float32) * 0.2
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, qkt)
    mem.write_fp32_tile(sim.state, BUF_ABUF, 0, v)
    insn = SoftmaxAttnVInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=64,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ABUF, 64, 16, 16)
    sm_oracle = np.exp(qkt - qkt.max(axis=-1, keepdims=True))
    sm_oracle = sm_oracle / sm_oracle.sum(axis=-1, keepdims=True)
    expected = (sm_oracle.astype(np.float32) @ v).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
