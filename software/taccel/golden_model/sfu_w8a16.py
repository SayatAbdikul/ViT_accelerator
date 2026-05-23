"""W8A16 SFU ops: FP16 ABUF endpoints, FP32-internal math.

These mirror :mod:`sfu_w8a32` op-by-op. The only structural difference
is the dtype on the ABUF side: activations arrive as FP16 (widened to
FP32 on read) and are written back as FP16 (narrowed from FP32 on
write). All reductions / mean / var / exp / erf / gelu stay in FP32 —
identical to ``fp32_prim_ref`` semantics — so the SFU's numerical
behaviour is the FP32 reference operating on FP16-rounded inputs.

ACCUM stays FP32 (matmul accumulates in FP32 in W8A16), so any read
from ACCUM here uses ``read_fp32_tile`` not ``read_fp16_tile``.

Scale registers and the W8A8 INT8 dequant prelude / requant epilogue
remain absent on this path, matching the W8A32 contract.
"""
from __future__ import annotations

import numpy as np

from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..utils import fp32_prim_ref as fpr

CYCLE_PER_ELEMENT = 2


def _read_act_fp32(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read activations as FP32 from any buffer.

    ABUF / WBUF hold FP16 in W8A16 — widen on read.
    ACCUM holds FP32 — direct read, no widen needed.
    """
    if buf_id == BUF_ACCUM:
        return memory.read_fp32_tile(state, buf_id, offset_units, rows, cols)
    return memory.read_fp16_tile(state, buf_id, offset_units, rows, cols).astype(np.float32)


def _write_act_fp16(state, buf_id: int, offset_units: int, data: np.ndarray):
    """Write FP32 activations to ABUF/WBUF as FP16 (narrowing cast)."""
    memory.write_fp16_tile(state, buf_id, offset_units, data.astype(np.float16))


def execute_layernorm_w8a16(state, insn):
    """LayerNorm with FP16 endpoints, FP32 internal reductions."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N)

    gb_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[:N * 2], dtype=np.uint16))
    beta = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[N * 2:], dtype=np.uint16))

    eps = np.float32(1e-6)
    mean = fpr.fp32_mean_rows(x)[:, None]
    var = fpr.fp32_var_rows(x, mean)[:, None]
    denom = np.sqrt((var + eps).astype(np.float32), dtype=np.float32)
    x_norm = ((x - mean).astype(np.float32) / denom).astype(np.float32)
    x_out = ((x_norm * gamma).astype(np.float32) + beta).astype(np.float32)

    _write_act_fp16(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_w8a16(state, insn):
    """Softmax row-wise. src1 may be ABUF (FP16) or ACCUM (FP32)."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N)
    x_shifted = (x - x.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_x = fpr.fp32_exp_arr(x_shifted)
    denom = fpr.fp32_sum_rows(exp_x)[:, None]
    x_out = (exp_x / denom).astype(np.float32)

    _write_act_fp16(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu_w8a16(state, insn):
    """GELU via fp32_prim_ref.fp32_gelu_arr. FP16 endpoints."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N)
    x_out = fpr.fp32_gelu_arr(x).astype(np.float32)

    _write_act_fp16(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_attnv_w8a16(state, insn):
    """Fused softmax(QK^T) @ V. QK^T arrives from ACCUM (FP32); V from ABUF (FP16)."""
    from .simulator import ConfigError, IllegalBufferError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")
    if insn.src1_buf != BUF_ACCUM:
        raise IllegalBufferError(insn.src1_buf)
    if insn.src2_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.src2_buf)
    if insn.dst_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.dst_buf)

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1
    M = m_tiles * 16
    N = n_tiles * 16
    K = k_tiles * 16

    qkt = _read_act_fp32(state, BUF_ACCUM, insn.src1_off, M, K)
    v = _read_act_fp32(state, insn.src2_buf, insn.src2_off, K, N)

    qkt_shifted = (qkt - qkt.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_qkt = fpr.fp32_exp_arr(qkt_shifted)
    softmax = (exp_qkt / fpr.fp32_sum_rows(exp_qkt)[:, None]).astype(np.float32)
    attn_v = np.matmul(softmax, v, dtype=np.float32)

    _write_act_fp16(state, insn.dst_buf, insn.dst_off, attn_v)
    state.cycle_count += (M * K * CYCLE_PER_ELEMENT) + (m_tiles * n_tiles * k_tiles * 16) + (M * N)
    return None
