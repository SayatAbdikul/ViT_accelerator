"""W8A16 matmul: FP16 × FP16 → FP32 accumulator.

In the W8A16 path, weights and activations live in their respective
buffers as FP16 (weights are pre-dequantized at compile time, just
like W8A32 but with a trailing FP16 cast). The systolic datapath
widens both operands to FP32 for the multiply-accumulate, which is
the standard mixed-precision convention.

ACCUM is FP32 in W8A16 (byte-identical to W8A32's reinterpretation
of the INT32 array), so accumulate-mode reads and writes go through
``read_fp32_tile`` / ``write_fp32_tile`` exactly as in W8A32.

This also covers QKT (``Q × K^T``, src2 is FP16 in WBUF after a
transpose) and AttnV (``softmax × V``, src2 is V in ABUF as FP16) —
both transparent because src2 is always read as FP16 here.
"""
from __future__ import annotations

import numpy as np

from . import memory

TILE = 16
CYCLE_COST = 16


def execute_matmul_w8a16(state, insn):
    """W8A16 MATMUL: FP16 acts × FP16 weights → FP32 ACCUM.

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

    # FP16 in ABUF/WBUF — widen to FP32 for the multiply.
    src1 = memory.read_fp16_tile(state, insn.src1_buf, insn.src1_off, M, K).astype(np.float32)
    src2 = memory.read_fp16_tile(state, insn.src2_buf, insn.src2_off, K, N).astype(np.float32)

    if accumulate:
        dst = memory.read_fp32_tile(state, insn.dst_buf, insn.dst_off, M, N)
    else:
        dst = np.zeros((M, N), dtype=np.float32)

    # Sequential K-loop accumulator, matching the RTL systolic PE's per-cycle
    # FP32 mul + FP32 add for k = 0..K-1. np.matmul reorders the reduction
    # under BLAS and would diverge from RTL at the bit level. Each iteration
    # below is one element-wise FP32 multiply (RNE) followed by one
    # element-wise FP32 add (RNE), so dst[i,j] evolves as
    # ((((dst[i,j] + a[i,0]*b[0,j]) + a[i,1]*b[1,j]) + ...) + a[i,K-1]*b[K-1,j]).
    for k in range(K):
        dst += src1[:, k:k+1] * src2[k:k+1, :]

    memory.write_fp32_tile(state, insn.dst_buf, insn.dst_off, dst)
    state.cycle_count += m_tiles * n_tiles * k_tiles * CYCLE_COST
