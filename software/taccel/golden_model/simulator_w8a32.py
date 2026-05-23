"""Golden model simulator for the W8A32 path.

Reuses the W8A8 :class:`Simulator` for byte-mover ops (LOAD/STORE/BUF_COPY,
SET_*, CONFIG_TILE, NOP/HALT/SYNC), and overrides every dtype-sensitive
op so activations flow as FP32 end-to-end:

* MATMUL → :func:`systolic_w8a32.execute_matmul_w8a32`
* LAYERNORM/SOFTMAX/GELU/SOFTMAX_ATTNV → :mod:`sfu_w8a32`
* REQUANT/REQUANT_PC → FP32 scale-multiply (no clip-to-INT8 epilogue)
* DEQUANT_ADD → FP32 scaled add (both operands FP32)
* SCALE_MUL → FP32 scale-multiply (no clip)
* VADD → FP32 elementwise add (ABUF) or FP32 broadcast bias add (ACCUM)

Trace capture is disabled by default — the W8A32 trace surface is
out of scope for this fork; ``program.trace_manifest`` is empty so the
trace machinery is a structural no-op.
"""
from __future__ import annotations

import numpy as np

from ..isa.opcodes import Opcode, BUF_ABUF, BUF_WBUF, BUF_ACCUM
from . import memory as mem
from .simulator import Simulator, ConfigError, IllegalBufferError
from .state_w8a32 import MachineStateW8A32
from .systolic_w8a32 import execute_matmul_w8a32
from .sfu_w8a32 import (
    execute_layernorm_w8a32,
    execute_softmax_w8a32,
    execute_gelu_w8a32,
    execute_softmax_attnv_w8a32,
)

UNIT = 16


class SimulatorW8A32(Simulator):
    """Simulator for the W8A32 path.

    Constructs a :class:`MachineStateW8A32` by default if none is supplied.
    """

    def __init__(self, state: MachineStateW8A32 = None):
        super().__init__(state or MachineStateW8A32())

    # ── dispatch overrides ────────────────────────────────────────────

    def _execute(self, insn):
        op = insn.opcode

        if op == Opcode.MATMUL:
            execute_matmul_w8a32(self.state, insn)
        elif op == Opcode.LAYERNORM:
            execute_layernorm_w8a32(self.state, insn)
        elif op == Opcode.SOFTMAX:
            execute_softmax_w8a32(self.state, insn)
        elif op == Opcode.GELU:
            execute_gelu_w8a32(self.state, insn)
        elif op == Opcode.SOFTMAX_ATTNV:
            execute_softmax_attnv_w8a32(self.state, insn)
        elif op == Opcode.REQUANT:
            self._exec_requant_w8a32(insn)
        elif op == Opcode.REQUANT_PC:
            self._exec_requant_pc_w8a32(insn)
        elif op == Opcode.DEQUANT_ADD:
            self._exec_dequant_add_w8a32(insn)
        elif op == Opcode.SCALE_MUL:
            self._exec_scale_mul_w8a32(insn)
        elif op == Opcode.VADD:
            self._exec_vadd_w8a32(insn)
        elif op == Opcode.BUF_COPY and insn.transpose:
            # FP32-element transpose (BUF_COPY with transpose=1). The W8A8
            # version transposes individual INT8 bytes which would scramble
            # FP32 layouts; this override treats each 4 bytes as one element.
            self._exec_buf_copy_transpose_fp32(insn)
        else:
            # Byte movers (incl. flat BUF_COPY), CONFIG_TILE, SET_*, NOP, HALT,
            # SYNC — reuse W8A8. Flat byte copies preserve FP32 bit patterns
            # because the source bytes (whether stored in ACCUM as int32 or
            # ABUF/WBUF as raw bytes) are identical to the FP32 little-endian
            # encoding the destination expects.
            super()._execute(insn)

    # ── trace capture: no-op on this fork (W8A32 trace is out of scope) ───

    def _capture_trace_events(self, pc: int):
        return

    # ── FP32 reinterpretations of the dtype-sensitive ops ─────────────

    def _exec_requant_w8a32(self, insn):
        """ACCUM FP32 × scalar FP16 → ABUF FP32 (no clip)."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        scale = np.float32(self.state.scale_regs[insn.sreg])
        src = mem.read_fp32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
        result = (src * scale).astype(np.float32)
        mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_requant_pc_w8a32(self, insn):
        """ACCUM FP32 × per-channel FP16 vector → ABUF FP32 (no clip).

        src1 must be ACCUM. src2 points at a packed FP16 scale table with
        N entries, one per output column; broadcast over all M rows.
        """
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
        mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_dequant_add_w8a32(self, insn):
        """ACCUM FP32 × accum_scale + skip FP32 × skip_scale → ABUF FP32 (no clip)."""
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
        skip = mem.read_fp32_tile(self.state, insn.src2_buf, insn.src2_off, M, N)
        result = (accum * accum_scale + skip * skip_scale).astype(np.float32)
        mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_scale_mul_w8a32(self, insn):
        """FP32 × scale → FP32 (no clip). Works on ABUF or ACCUM."""
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        scale = np.float32(self.state.scale_regs[insn.sreg])
        src = mem.read_fp32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
        result = (src * scale).astype(np.float32)
        mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        self.state.cycle_count += M * N

    def _exec_buf_copy_transpose_fp32(self, insn):
        """BUF_COPY transpose with FP32-element granularity.

        ``length`` is in 16-byte units, ``src_rows`` in 16-row units. Each
        source row spans ``total_bytes / src_row_count`` bytes which we
        reinterpret as ``cols = bytes // 4`` FP32 elements. Output is the
        element-wise transpose written as FP32 bytes.
        """
        total_bytes = insn.length * UNIT
        src_row_count = insn.src_rows * 16
        if src_row_count == 0 or total_bytes == 0:
            return
        byte_cols = total_bytes // src_row_count
        elem_cols = byte_cols // 4
        if elem_cols == 0:
            return
        src_data = mem.read_bytes(self.state, insn.src_buf, insn.src_off, total_bytes)
        src_array = np.frombuffer(src_data, dtype=np.float32).reshape(src_row_count, elem_cols)
        dst_array = np.ascontiguousarray(src_array.T)
        mem.write_bytes(self.state, insn.dst_buf, insn.dst_off, dst_array.tobytes())
        self.state.cycle_count += insn.length

    def _exec_vadd_w8a32(self, insn):
        """VADD in FP32. Two paths:
          - ABUF source: elementwise FP32 + FP32 → FP32.
          - ACCUM source: FP32 + FP32 bias-row broadcast → FP32 ACCUM.
        No INT8 saturation (no clip) — overflow simply rolls into FP32 inf.
        """
        if self.state.tile_config is None:
            raise ConfigError("CONFIG_TILE not set")

        m_tiles = self.state.tile_config[0] + 1
        n_tiles = self.state.tile_config[1] + 1
        M = m_tiles * 16
        N = n_tiles * 16

        if insn.src1_buf == BUF_ABUF:
            src1 = mem.read_fp32_tile(self.state, insn.src1_buf, insn.src1_off, M, N)
            src2 = mem.read_fp32_tile(self.state, insn.src2_buf, insn.src2_off, M, N)
            result = (src1 + src2).astype(np.float32)
            mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        elif insn.src1_buf == BUF_ACCUM:
            src1 = mem.read_fp32_tile(self.state, BUF_ACCUM, insn.src1_off, M, N)
            src2_row = mem.read_fp32_tile(self.state, insn.src2_buf, insn.src2_off, 1, N)
            result = (src1 + np.tile(src2_row, (M, 1))).astype(np.float32)
            mem.write_fp32_tile(self.state, insn.dst_buf, insn.dst_off, result)
        else:
            raise IllegalBufferError(insn.src1_buf)

        self.state.cycle_count += M * N
