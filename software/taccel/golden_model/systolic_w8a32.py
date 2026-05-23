"""W8A32 matmul: FP32 activations × FP32 weights → FP32 accumulator.

In the W8A32 path, weights are stored in DRAM as **dequantized FP32**
(INT8-quantized then multiplied back by per-channel scales), so the
systolic datapath only ever sees FP32 × FP32. This mirrors the
fake-quant ceiling at ``software/taccel/quantizer/fake_quant.py`` —
weight rounding error is baked into the FP32 bit pattern — and avoids a
post-MATMUL REQUANT_PC step entirely.

The matmul also covers QKT (``Q × K^T``, src2 is an FP32 activation in
WBUF after a transpose) and AttnV (``softmax × V``, src2 is V in ABUF as
FP32). Both work transparently because src2 is always read as FP32.

ACCUM remains an INT32-typed ndarray of 16K entries; the W8A32 path
reinterprets the underlying bytes as FP32 via ``state.accum.view(float32)``,
so the byte size and shape are unchanged.
"""
from __future__ import annotations

import numpy as np

from . import memory

TILE = 16
CYCLE_COST = 16


def execute_matmul_w8a32(state, insn):
    """W8A32 MATMUL: FP32 activations × FP32 weights → FP32 ACCUM.

    flags[0] = 0: dst = src1 @ src2       (overwrite)
    flags[0] = 1: dst += src1 @ src2      (accumulate)
    """
    from .simulator import ConfigError

    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1
    M = m_tiles * TILE
    N = n_tiles * TILE
    K = k_tiles * TILE

    accumulate = bool(insn.flags & 1)

    src1 = memory.read_fp32_tile(state, insn.src1_buf, insn.src1_off, M, K)
    src2 = memory.read_fp32_tile(state, insn.src2_buf, insn.src2_off, K, N)

    if accumulate:
        dst = memory.read_fp32_tile(state, insn.dst_buf, insn.dst_off, M, N)
    else:
        dst = np.zeros((M, N), dtype=np.float32)

    dst += np.matmul(src1, src2, dtype=np.float32)

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, dst)
    state.cycle_count += m_tiles * n_tiles * k_tiles * CYCLE_COST
