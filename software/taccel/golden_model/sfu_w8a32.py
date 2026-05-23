"""W8A32 SFU ops: pure FP32 LayerNorm / Softmax / GELU / SoftmaxAttnV.

These mirror :mod:`sfu` op-by-op, with two structural simplifications:

1. **No dequant prelude.** Activations arrive as FP32 in ABUF (or FP32
   from the ACCUM view); the ``inp.astype(np.float32) * in_scale`` step
   that bridges INT8 inputs in the W8A8 SFU is removed.

2. **No requant epilogue.** Results are written back as FP32 via
   :func:`memory.write_fp32_tile`; there is no ``fp32_quantize_i8_arr``
   clip/round to INT8.

The FP32 math itself (mean/var/exp/erf/gelu) is identical, so the
``fp32_prim_ref`` primitives still apply when we want bit-equivalence
with a future W8A32 RTL implementation.
"""
from __future__ import annotations

import numpy as np

from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..utils import fp32_prim_ref as fpr

CYCLE_PER_ELEMENT = 2


def _read_act_fp32(state, buf_id: int, offset_units: int, rows: int, cols: int) -> np.ndarray:
    """Read activations as FP32 from any buffer (ABUF/WBUF/ACCUM)."""
    return memory.read_fp32_tile(state, buf_id, offset_units, rows, cols)


def execute_layernorm_w8a32(state, insn):
    """LayerNorm in FP32: x_out = ((x - mean) / sqrt(var + eps)) * gamma + beta.

    src1 = FP32 input, src2 = FP16-packed gamma|beta, dst = FP32 output.
    Scale registers are ignored (no INT8 bridge to dequant across).
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N).astype(np.float32)

    gb_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[:N * 2], dtype=np.uint16))
    beta = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[N * 2:], dtype=np.uint16))

    eps = np.float32(1e-6)
    mean = fpr.fp32_mean_rows(x)[:, None]
    var = fpr.fp32_var_rows(x, mean)[:, None]
    denom = np.sqrt((var + eps).astype(np.float32), dtype=np.float32)
    x_norm = ((x - mean).astype(np.float32) / denom).astype(np.float32)
    x_out = ((x_norm * gamma).astype(np.float32) + beta).astype(np.float32)

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_w8a32(state, insn):
    """Softmax in FP32, row-wise. src1 may be ABUF or ACCUM (both FP32 in W8A32)."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N).astype(np.float32)
    x_shifted = (x - x.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_x = fpr.fp32_exp_arr(x_shifted)
    denom = fpr.fp32_sum_rows(exp_x)[:, None]
    x_out = (exp_x / denom).astype(np.float32)

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu_w8a32(state, insn):
    """GELU in FP32 via fp32_prim_ref.fp32_gelu_arr (A&S 7.1.26)."""
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    x = _read_act_fp32(state, insn.src1_buf, insn.src1_off, M, N).astype(np.float32)
    x_out = fpr.fp32_gelu_arr(x).astype(np.float32)

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, x_out)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_attnv_w8a32(state, insn):
    """Fused softmax(QK^T) @ V in pure FP32. No INT8 bracketing."""
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

    qkt = _read_act_fp32(state, BUF_ACCUM, insn.src1_off, M, K).astype(np.float32)
    v = _read_act_fp32(state, insn.src2_buf, insn.src2_off, K, N).astype(np.float32)

    qkt_shifted = (qkt - qkt.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_qkt = fpr.fp32_exp_arr(qkt_shifted)
    softmax = (exp_qkt / fpr.fp32_sum_rows(exp_qkt)[:, None]).astype(np.float32)
    attn_v = np.matmul(softmax, v, dtype=np.float32)

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, attn_v)
    state.cycle_count += (M * K * CYCLE_PER_ELEMENT) + (m_tiles * n_tiles * k_tiles * 16) + (M * N)
    return None
