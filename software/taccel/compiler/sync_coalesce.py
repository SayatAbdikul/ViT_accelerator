"""Peephole pass that drops DMA-mask SYNCs the RTL already auto-fences.

W8A16/W8A32 codegen emits ``SYNC(mask=0b001)`` after every LOAD/STORE.
Most of these are redundant: the RTL control unit auto-stalls many
consumers on ``dma_busy`` at issue. But not all — and the cases that
look removable on paper are not all safe in practice.

What the RTL actually guarantees (``rtl/src/control_unit.sv``):

* Helper consumers (``OP_BUF_COPY`` line 287, ``OP_SCALE_MUL`` /
  ``OP_VADD`` line 321) and SFU consumers (``OP_SOFTMAX`` /
  ``OP_LAYERNORM`` / ``OP_GELU`` line 366) **stall at issue** on
  ``dma_busy``. A SYNC(0b001) immediately before them is a no-op —
  they cannot dispatch until DMA drains regardless.
* ``OP_MATMUL`` (line 340) does **not** check ``dma_busy``; it would
  dispatch ahead of an in-flight LOAD and read stale ABUF/WBUF. A
  SYNC(0b001) before MATMUL is load-bearing.
* ``OP_LOAD`` / ``OP_STORE`` (line 297) do **not** check
  ``dma_busy``. The DMA engine in ``rtl/src/dma_engine.sv`` has **no
  command queue** — it only accepts a ``dispatch`` pulse in ``D_IDLE``
  (line 187). Issuing LOAD#2 while LOAD#1 is still in flight silently
  drops LOAD#2's dispatch pulse, ABUF gets stale data, and the
  bit-exact gate fails. A SYNC(0b001) between adjacent DMA ops is
  load-bearing and **must not** be removed.

This pass therefore drops the DMA bit only when the next engine op is
a helper or SFU op (which auto-fences). For ``OP_MATMUL``, another
``OP_LOAD/STORE``, ``OP_HALT``, or end-of-stream, the SYNC is kept
verbatim. Non-DMA SYNCs (mask = 0b010 / 0b100) are untouched.

Correctness boundary: the pass only strips bits whose hazard the RTL
will independently re-enforce at issue time. The bit-exact parity
gate is the verification.
"""
from __future__ import annotations

from typing import List

from ..isa.opcodes import Opcode
from ..isa.instructions import Instruction, SyncInsn


# Instructions that touch no engine — pure issue-stage state writes.
# These can sit between a DMA op and its consumer without changing
# whether the DMA fence is needed.
_CONTROL_ONLY_OPCODES = frozenset({
    Opcode.NOP,
    Opcode.SET_ADDR_LO,
    Opcode.SET_ADDR_HI,
    Opcode.SET_SCALE,
    Opcode.CONFIG_TILE,
})


# DMA-mask bit in the SYNC resource_mask. Matches
# ``rtl/src/include/taccel_pkg.sv`` ``SYNC_DMA_BIT``.
_DMA_MASK_BIT = 0b001


# Engine ops whose ``S_ISSUE`` branch in ``control_unit.sv`` checks
# ``dma_busy`` and stalls until DMA drains. A SYNC(0b001) immediately
# before any of these is redundant — the issue stage will not dispatch
# them while DMA is in flight regardless of whether the SYNC was
# emitted.
_AUTO_DMA_FENCED = frozenset({
    Opcode.BUF_COPY,
    Opcode.SCALE_MUL,
    Opcode.VADD,
    Opcode.SOFTMAX,
    Opcode.LAYERNORM,
    Opcode.GELU,
})


def _next_engine_op(insns: List[Instruction], start: int) -> Opcode | None:
    """Return the opcode of the next instruction that actually runs on
    an engine, skipping control-only ops. ``None`` if the program ends
    in control-only instructions (e.g. trailing HALT only)."""
    for j in range(start, len(insns)):
        op = insns[j].opcode
        if op in _CONTROL_ONLY_OPCODES:
            continue
        return op
    return None


def coalesce_dma_syncs(insns: List[Instruction]) -> List[Instruction]:
    """Return a new instruction list with redundant DMA-bit syncs
    stripped.

    Rules:
    * SYNC with ``mask & 0b001`` and the next engine op is
      OP_LOAD/OP_STORE/anything-other-than-OP_MATMUL → clear the DMA
      bit.
    * SYNC reduced to ``mask == 0`` → drop the instruction.
    * SYNC with bits outside the DMA bit (e.g. ``0b010`` for systolic
      or ``0b100`` for SFU) is preserved verbatim.

    The relative order of every retained instruction is unchanged.
    """
    out: List[Instruction] = []
    for i, insn in enumerate(insns):
        if not isinstance(insn, SyncInsn) or not (insn.resource_mask & _DMA_MASK_BIT):
            out.append(insn)
            continue

        next_op = _next_engine_op(insns, i + 1)
        # Drop the DMA bit only when the next engine op is one that
        # the RTL control unit will *itself* stall on ``dma_busy``
        # before dispatching. Adjacent OP_LOAD/STORE specifically must
        # keep the SYNC: the DMA engine has no command queue, so a
        # second dispatch pulse arriving while DMA is in flight is
        # silently dropped (``rtl/src/dma_engine.sv`` line 187).
        if next_op not in _AUTO_DMA_FENCED:
            out.append(insn)
            continue

        new_mask = insn.resource_mask & ~_DMA_MASK_BIT
        if new_mask == 0:
            continue  # SYNC becomes a no-op; drop it
        out.append(SyncInsn(resource_mask=new_mask))

    return out
