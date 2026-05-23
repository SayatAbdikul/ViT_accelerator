"""Golden model simulator for the W8A16 path.

Reuses the W8A8 :class:`Simulator` for byte-mover ops (LOAD/STORE/BUF_COPY,
SET_*, CONFIG_TILE, NOP/HALT/SYNC), and overrides every dtype-sensitive
op so activations flow as FP16 on the ABUF / WBUF side and FP32 on the
ACCUM side:

* MATMUL → :func:`systolic_w8a16.execute_matmul_w8a16` (FP16 × FP16 → FP32)
* LAYERNORM/SOFTMAX/GELU/SOFTMAX_ATTNV → :mod:`sfu_w8a16` (FP16 endpoints,
  FP32 reductions)
* REQUANT/REQUANT_PC → FP32 scale-multiply, narrow back to FP16 (no clip)
* DEQUANT_ADD → FP32 scaled add (ACCUM + FP16-widened skip) → FP16 narrow
* SCALE_MUL → FP32 scale-multiply with FP16 narrow on ABUF dst
* VADD → FP16 + FP16 → FP16 (ABUF) or FP32 + FP32 row broadcast → FP32 (ACCUM)
* BUF_COPY transpose → FP16-element transpose (2 bytes/element)

The REQUANT family opcodes never appear in W8A16 programs (the compile
path does not emit them), but providing soft no-clip semantics keeps the
simulator robust to test fixtures that exercise them.

Trace capture is disabled here — same convention as :mod:`simulator_w8a32`.
"""
from __future__ import annotations

import numpy as np

from ..isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from . import memory as mem
from .simulator import Simulator, ConfigError, IllegalBufferError
from .state_w8a16 import MachineStateW8A16
from .systolic_w8a16 import execute_matmul_w8a16
from .sfu_w8a16 import (
    execute_layernorm_w8a16,
    execute_softmax_w8a16,
    execute_gelu_w8a16,
    execute_softmax_attnv_w8a16,
)

UNIT = 16


class SimulatorW8A16(Simulator):
    """Simulator for the W8A16 path.

    Constructs a :class:`MachineStateW8A16` by default if none is supplied.
    """

    def __init__(self, state: MachineStateW8A16 = None):
        super().__init__(state or MachineStateW8A16())

    # ── dispatch overrides ────────────────────────────────────────────

    def _execute(self, insn):
        op = insn.opcode

        if op == Opcode.MATMUL:
            execute_matmul_w8a16(self.state, insn)
        elif op == Opcode.LAYERNORM:
            execute_layernorm_w8a16(self.state, insn)
        elif op == Opcode.SOFTMAX:
            execute_softmax_w8a16(self.state, insn)
        elif op == Opcode.GELU:
            execute_gelu_w8a16(self.state, insn)
        elif op == Opcode.SOFTMAX_ATTNV:
            execute_softmax_attnv_w8a16(self.state, insn)
        elif op == Opcode.REQUANT:
            self._exec_requant_w8a16(insn)
        elif op == Opcode.REQUANT_PC:
            self._exec_requant_pc_w8a16(insn)
        elif op == Opcode.DEQUANT_ADD:
            self._exec_dequant_add_w8a16(insn)
        elif op == Opcode.SCALE_MUL:
            self._exec_scale_mul_w8a16(insn)
        elif op == Opcode.VADD:
            self._exec_vadd_w8a16(insn)
        elif op == Opcode.BUF_COPY and insn.transpose:
            # FP16-element transpose (BUF_COPY with transpose=1). The W8A8
            # version transposes individual INT8 bytes which would scramble
            # FP16 layouts; this override treats each 2 bytes as one element.
            self._exec_buf_copy_transpose_fp16(insn)
        else:
            # Byte movers (incl. flat BUF_COPY), CONFIG_TILE, SET_*, NOP, HALT,
            # SYNC — reuse W8A8. Flat byte copies preserve FP16 bit patterns.
            super()._execute(insn)

    # ── trace capture: no-op on this fork (out of scope, mirrors W8A32) ──

    def _capture_trace_events(self, pc: int):
        return

    # ── helpers: FP16 read/write on ABUF/WBUF; FP32 on ACCUM ────────────

    def _read_act_fp32(self, buf_id: int, off: int, M: int, N: int) -> np.ndarray:
        """Read a tile as FP32 from any buffer (FP16 widen on ABUF/WBUF)."""
        if buf_id == BUF_ACCUM:
            return mem.read_fp32_tile(self.state, buf_id, off, M, N)
        return mem.read_fp16_tile(self.state, buf_id, off, M, N).astype(np.float32)

    def _write_act(self, buf_id: int, off: int, data: np.ndarray):
        """Write a tile to ABUF/WBUF as FP16 or to ACCUM as FP32."""
        if buf_id == BUF_ACCUM:
            mem.write_fp32_tile(self.state, buf_id, off, data.astype(np.float32))
        else:
            mem.write_fp16_tile(self.state, buf_id, off, data.astype(np.float16))

    # ── FP16 reinterpretations of the dtype-sensitive ops ──────────────

    def _exec_requant_w8a16(self, insn):
        """ACCUM FP32 × scalar FP16 → ABUF FP16 (no clip)."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        scale = np.float32(self.state.scale_regs[insn.sreg])
        src = self._read_act_fp32(insn.src1_buf, insn.src1_off, M, N)
        result = (src * scale).astype(np.float32)
        self._write_act(insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_requant_pc_w8a16(self, insn):
        """ACCUM FP32 × per-channel FP16 vector → ABUF FP16 (no clip)."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        if insn.src1_buf != BUF_ACCUM:
            raise IllegalBufferError(insn.src1_buf)
        if insn.src2_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.src2_buf)
        if insn.dst_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.dst_buf)

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        src = mem.read_fp32_tile(self.state, BUF_ACCUM, insn.src1_off, M, N)
        scale_bytes = mem.read_bytes(self.state, insn.src2_buf, insn.src2_off, N * 2)
        scales = np.frombuffer(scale_bytes, dtype=np.float16).astype(np.float32).reshape(1, N)
        result = (src * scales).astype(np.float32)
        self._write_act(insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_dequant_add_w8a16(self, insn):
        """ACCUM FP32 × accum_scale + skip FP16 widen × skip_scale → ABUF FP16."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")
        if insn.src1_buf != BUF_ACCUM:
            raise IllegalBufferError(insn.src1_buf)
        if insn.src2_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.src2_buf)
        if insn.dst_buf == BUF_ACCUM:
            raise IllegalBufferError(insn.dst_buf)
        if insn.sreg >= 15:
            raise ConfigError("DEQUANT_ADD sreg+1 out of range")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        accum_scale = np.float32(self.state.scale_regs[insn.sreg])
        skip_scale = np.float32(self.state.scale_regs[insn.sreg + 1])
        accum = mem.read_fp32_tile(self.state, BUF_ACCUM, insn.src1_off, M, N)
        skip = self._read_act_fp32(insn.src2_buf, insn.src2_off, M, N)
        result = (accum * accum_scale + skip * skip_scale).astype(np.float32)
        self._write_act(insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_scale_mul_w8a16(self, insn):
        """FP32-internal × scale → narrow on ABUF dst, stay FP32 on ACCUM dst."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        scale = np.float32(self.state.scale_regs[insn.sreg])
        src = self._read_act_fp32(insn.src1_buf, insn.src1_off, M, N)
        result = (src * scale).astype(np.float32)
        self._write_act(insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_buf_copy_transpose_fp16(self, insn):
        """BUF_COPY transpose with FP16-element granularity.

        Each source row spans ``total_bytes / src_row_count`` bytes which
        we reinterpret as ``cols = bytes // 2`` FP16 elements. Output is
        the element-wise transpose written as FP16 bytes.
        """
        total_bytes = insn.length * UNIT
        src_row_count = insn.src_rows * 16
        if src_row_count == 0 or total_bytes == 0:
            return
        byte_cols = total_bytes // src_row_count
        elem_cols = byte_cols // 2
        if elem_cols == 0:
            return
        src_data = mem.read_bytes(self.state, insn.src_buf, insn.src_off, total_bytes)
        src_array = np.frombuffer(src_data, dtype=np.float16).reshape(src_row_count, elem_cols)
        dst_array = np.ascontiguousarray(src_array.T)
        mem.write_bytes(self.state, insn.dst_buf, insn.dst_off, dst_array.tobytes())
        self.state.cycle_count += insn.length

    def _exec_vadd_w8a16(self, insn):
        """VADD: two paths.

        * ABUF source — elementwise FP16 + FP16 → FP16 (widen, add, narrow).
        * ACCUM source — FP32 + FP16-row-widen-to-FP32 broadcast → FP32 ACCUM.
          This is the attention-mask broadcast path: the WBUF mask row is
          FP16 (-65504 in padded columns), widened to FP32 on read, added
          into FP32 ACCUM. The padded columns end up at FP32(-65504), which
          underflows exp() to zero in the downstream softmax.
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        if insn.src1_buf == BUF_ABUF:
            src1 = mem.read_fp16_tile(self.state, insn.src1_buf, insn.src1_off, M, N).astype(np.float32)
            src2 = mem.read_fp16_tile(self.state, insn.src2_buf, insn.src2_off, M, N).astype(np.float32)
            result = (src1 + src2).astype(np.float32)
            self._write_act(insn.dst_buf, insn.dst_off, result)
        elif insn.src1_buf == BUF_ACCUM:
            src1 = mem.read_fp32_tile(self.state, BUF_ACCUM, insn.src1_off, M, N)
            src2_row = mem.read_fp16_tile(self.state, insn.src2_buf, insn.src2_off, 1, N).astype(np.float32)
            result = (src1 + np.tile(src2_row, (M, 1))).astype(np.float32)
            self._write_act(insn.dst_buf, insn.dst_off, result)
        else:
            raise IllegalBufferError(insn.src1_buf)

        self.state.cycle_count += M * N
