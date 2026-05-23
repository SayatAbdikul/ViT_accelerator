"""Phase 2 unit tests: per-op W8A16 simulator handlers.

Each test drives one instruction through ``SimulatorW8A16`` with hand-
written ABUF/WBUF state and compares against a NumPy oracle. Tolerances
are looser than the W8A32 suite because FP16 endpoints round each value
to ~3 decimal digits of precision; reductions and the multiply-add proper
still happen in FP32 internally, so accuracy holds where the oracle
itself stays in FP32.
"""
from __future__ import annotations

import numpy as np
import pytest

from taccel.golden_model import memory as mem
from taccel.golden_model.simulator_w8a16 import SimulatorW8A16
from taccel.golden_model.state_w8a16 import MachineStateW8A16
from taccel.isa.instructions import (
    ConfigTileInsn, DequantAddInsn, GeluInsn, LayernormInsn, MatmulInsn,
    RequantInsn, RequantPcInsn, ScaleMulInsn, SoftmaxInsn, SoftmaxAttnVInsn,
    VaddInsn,
)
from taccel.isa.opcodes import BUF_ABUF, BUF_ACCUM, BUF_WBUF
from taccel.utils import fp32_prim_ref as fpr


def _make_sim(M_tiles=1, N_tiles=1, K_tiles=1):
    sim = SimulatorW8A16()
    sim.state.tile_config = (M_tiles - 1, N_tiles - 1, K_tiles - 1)
    return sim


# ── MATMUL ────────────────────────────────────────────────────────────


def test_matmul_w8a16_fp16_act_fp16_weight_fp32_accum():
    """FP16 act × FP16 weight → FP32 ACCUM (widen-on-read; FP32 internal mul)."""
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    M, N, K = 16, 16, 16
    rng = np.random.default_rng(42)
    act = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)

    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, act)
    mem.write_fp16_tile(sim.state, BUF_WBUF, 0, w)

    insn = MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=0,
    )
    sim._execute(insn)

    expected = act.astype(np.float32) @ w.astype(np.float32)
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, M, N)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_matmul_w8a16_accumulate_flag():
    """flags=1 → dst (FP32 ACCUM) += src1 @ src2."""
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    rng = np.random.default_rng(1)
    initial_accum = (rng.standard_normal((16, 16)) * 100).astype(np.float32)
    act = (rng.standard_normal((16, 16)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((16, 16)) * 0.05).astype(np.float16)

    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, initial_accum)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, act)
    mem.write_fp16_tile(sim.state, BUF_WBUF, 0, w)

    insn = MatmulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
        flags=1,
    )
    sim._execute(insn)

    expected = initial_accum + act.astype(np.float32) @ w.astype(np.float32)
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ── VADD ──────────────────────────────────────────────────────────────


def test_vadd_w8a16_abuf_path_fp16_elementwise_add():
    sim = _make_sim()
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 16)).astype(np.float16)
    b = rng.standard_normal((16, 16)).astype(np.float16)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, a)
    # b at offset 32 units (512 bytes = 16×16×2).
    mem.write_fp16_tile(sim.state, BUF_ABUF, 32, b)
    insn = VaddInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=32,
        dst_buf=BUF_ABUF, dst_off=64,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 64, 16, 16)
    # FP16 narrow on output → match (a+b).astype(np.float16) exactly.
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


def test_vadd_w8a16_accum_path_broadcast_bias_fp16_row():
    """Attention-mask path: src1=ACCUM (FP32), src2=WBUF (FP16 row)."""
    sim = _make_sim()
    rng = np.random.default_rng(3)
    accum = (rng.standard_normal((16, 16)) * 10).astype(np.float32)
    bias = (rng.standard_normal((1, 16)) * 2).astype(np.float16)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_tile(sim.state, BUF_WBUF, 0, bias)
    insn = VaddInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
    )
    sim._execute(insn)
    got = mem.read_fp32_tile(sim.state, BUF_ACCUM, 0, 16, 16)
    expected = (accum + bias.astype(np.float32)).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ── REQUANT / REQUANT_PC ──────────────────────────────────────────────


def test_requant_w8a16_is_fp16_narrow_no_clip():
    """REQUANT in W8A16 = FP32 ACCUM × scalar scale → ABUF FP16, no clip."""
    sim = _make_sim()
    rng = np.random.default_rng(4)
    accum = (rng.standard_normal((16, 16)) * 1000).astype(np.float32)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    sim.state.scale_regs[0] = np.float16(0.5)

    insn = RequantInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=0,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 0, 16, 16)
    expected = (accum * np.float32(np.float16(0.5))).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


# ── DEQUANT_ADD ──────────────────────────────────────────────────────


def test_dequant_add_w8a16_fp16_skip_fp16_out():
    sim = _make_sim()
    rng = np.random.default_rng(6)
    accum = (rng.standard_normal((16, 16)) * 100).astype(np.float32)
    skip = (rng.standard_normal((16, 16))).astype(np.float16)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, accum)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, skip)
    sim.state.scale_regs[0] = np.float16(0.01)
    sim.state.scale_regs[1] = np.float16(1.0)
    insn = DequantAddInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)
    expected = (accum * np.float32(np.float16(0.01))
                + skip.astype(np.float32) * np.float32(np.float16(1.0))).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


# ── SCALE_MUL ────────────────────────────────────────────────────────


def test_scale_mul_w8a16_abuf_fp16():
    sim = _make_sim()
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((16, 16)) * 2.0).astype(np.float16)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, x)
    sim.state.scale_regs[0] = np.float16(0.25)
    insn = ScaleMulInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)
    expected = (x.astype(np.float32) * np.float32(np.float16(0.25))).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


# ── LAYERNORM ────────────────────────────────────────────────────────


def test_layernorm_w8a16_matches_fp32_oracle_with_fp16_endpoints():
    sim = _make_sim()
    rng = np.random.default_rng(8)
    x_fp16 = (rng.standard_normal((16, 16)) * 3.0).astype(np.float16)
    gamma = (rng.uniform(0.5, 1.5, size=16)).astype(np.float16)
    beta = (rng.standard_normal(16) * 0.1).astype(np.float16)

    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, x_fp16)
    gb_bytes = gamma.tobytes() + beta.tobytes()
    mem.write_bytes(sim.state, BUF_WBUF, 0, gb_bytes)

    insn = LayernormInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)

    # Oracle uses the same primitive order; input widens FP16→FP32 first
    # (same as the simulator does), and we narrow the final answer.
    x = x_fp16.astype(np.float32)
    gamma_fp32 = fpr.fp32_from_fp16_arr(np.frombuffer(gamma.tobytes(), dtype=np.uint16))
    beta_fp32 = fpr.fp32_from_fp16_arr(np.frombuffer(beta.tobytes(), dtype=np.uint16))
    mean = fpr.fp32_mean_rows(x)[:, None]
    var = fpr.fp32_var_rows(x, mean)[:, None]
    denom = np.sqrt((var + np.float32(1e-6)).astype(np.float32), dtype=np.float32)
    x_norm = ((x - mean) / denom).astype(np.float32)
    expected = (x_norm * gamma_fp32 + beta_fp32).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


# ── SOFTMAX ──────────────────────────────────────────────────────────


def test_softmax_w8a16_matches_numpy_softmax_with_fp16_endpoints():
    sim = _make_sim()
    rng = np.random.default_rng(9)
    x_fp16 = (rng.standard_normal((16, 16)) * 3.0).astype(np.float16)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, x_fp16)
    insn = SoftmaxInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)

    x = x_fp16.astype(np.float32)
    x_shifted = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x_shifted.astype(np.float32))
    expected = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float16)
    # Looser tolerance: FP16 narrow at the end loses ~3 digits of precision.
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(np.float32),
                                rtol=5e-3, atol=5e-4)
    # Row probabilities still sum to ~1 after FP16 narrow.
    np.testing.assert_allclose(got.astype(np.float32).sum(axis=-1),
                                np.ones(16), rtol=5e-3, atol=5e-4)


# ── GELU ─────────────────────────────────────────────────────────────


def test_gelu_w8a16_matches_fp32_gelu_oracle_with_fp16_endpoints():
    sim = _make_sim()
    rng = np.random.default_rng(10)
    x_fp16 = (rng.standard_normal((16, 16)) * 2.0).astype(np.float16)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, x_fp16)
    insn = GeluInsn(
        src1_buf=BUF_ABUF, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)
    expected = fpr.fp32_gelu_arr(x_fp16.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(got, expected)


# ── SOFTMAX_ATTNV ────────────────────────────────────────────────────


def test_softmax_attnv_w8a16_fp32_qkt_fp16_v_fp16_out():
    sim = _make_sim(M_tiles=1, N_tiles=1, K_tiles=1)
    rng = np.random.default_rng(11)
    qkt = (rng.standard_normal((16, 16)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((16, 16)) * 0.2).astype(np.float16)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, qkt)
    mem.write_fp16_tile(sim.state, BUF_ABUF, 0, v)
    insn = SoftmaxAttnVInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_ABUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=32,
        sreg=0,
    )
    sim._execute(insn)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 32, 16, 16)
    sm_oracle = np.exp(qkt - qkt.max(axis=-1, keepdims=True))
    sm_oracle = sm_oracle / sm_oracle.sum(axis=-1, keepdims=True)
    expected = (sm_oracle.astype(np.float32) @ v.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(np.float32),
                                rtol=5e-3, atol=5e-4)


# ── Attention-mask underflow ─────────────────────────────────────────


def test_attention_mask_minus_65504_underflows_softmax_to_zero():
    """The W8A16 attention mask uses -65504 (FP16 minimum) in padded columns.

    After VADD broadcasts it into the FP32 ACCUM and SOFTMAX runs, the
    masked columns should be bit-exact zero — the FP16 -65504 widens to
    FP32 -65504 which under exp() underflows to 0.0.
    """
    sim = _make_sim()
    rng = np.random.default_rng(12)

    # 16 query rows × 16 key columns; mask the last 4 columns.
    qkt = (rng.standard_normal((16, 16)) * 1.0).astype(np.float32)
    mask_row = np.zeros(16, dtype=np.float16)
    mask_row[12:] = np.float16(-65504.0)
    mem.write_fp32_tile(sim.state, BUF_ACCUM, 0, qkt)
    mem.write_fp16_tile(sim.state, BUF_WBUF, 0, mask_row.reshape(1, 16))

    # VADD broadcasts the mask row into ACCUM.
    vadd = VaddInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ACCUM, dst_off=0,
    )
    sim._execute(vadd)

    sm = SoftmaxInsn(
        src1_buf=BUF_ACCUM, src1_off=0,
        src2_buf=BUF_WBUF, src2_off=0,
        dst_buf=BUF_ABUF, dst_off=0,
        sreg=0,
    )
    sim._execute(sm)
    got = mem.read_fp16_tile(sim.state, BUF_ABUF, 0, 16, 16)

    # Masked columns should be exactly zero post-softmax (exp underflow).
    assert np.all(got[:, 12:] == np.float16(0)), (
        f"Masked columns leaked probability mass: max = "
        f"{float(np.max(got[:, 12:].astype(np.float32))):.6e}"
    )
    # Unmasked rows should still sum to ~1.
    row_sums = got.astype(np.float32).sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(16), rtol=5e-3, atol=5e-4)
