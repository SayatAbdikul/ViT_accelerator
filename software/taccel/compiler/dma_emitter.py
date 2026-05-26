"""DMA addressing helpers for the TACCEL ISA.

Every DRAM access in TACCEL needs a 56-bit byte address split across two
A-type instructions (SET_ADDR_LO writes bits [27:0], SET_ADDR_HI writes
bits [55:28]). This module owns that split.

Two helpers live here:

* :func:`set_addr` — stateless, always emits both halves. Kept for
  callers that want unconditional setup (e.g. tests).

* :class:`AddrPlanner` — stateful, caches the current value of each of
  the four address registers across a single compile. Two complementary
  optimisations come out of that:

  1. **SET_ADDR_HI elision.** When the new address's high 28 bits match
     what the register already holds, the HI write is skipped. For
     DeiT-tiny (whole DRAM image < 256 MB → high 28 bits are always 0)
     this eliminates 100% of HI writes.

  2. **dram_off walking.** The M-type LOAD/STORE encoding already has a
     16-bit ``dram_off`` field that the DMA engine adds to the address
     register (scaled ×16 bytes). When the requested address is within
     ``[cached, cached + 1 MB)`` of the same register's cached base, we
     emit no SET_ADDR at all — the LOAD/STORE encodes the byte delta in
     ``dram_off`` directly. RTL ``dma_engine.sv`` and the golden
     ``dma.py`` both compute the address as ``base + dram_off*16``, so
     the only change is on the compiler side.

A tracing-the-stream measurement on DeiT-tiny W8A16 found 314,050
SET_ADDR_HI emissions, 100% of them redundant, and 312,441 LOADs all
issued with ``dram_off=0`` — i.e. both optimisations had room to cut
the entire pair. Combined they shrink the program by ~45%.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ..isa.instructions import Instruction, SetAddrLoInsn, SetAddrHiInsn


# M-type ``dram_off`` is 16 bits, scaled by 16 bytes → 1 MB walkable
# window above whatever ``addr_reg`` currently holds.
_DRAM_OFF_MAX = 0xFFFF
_DRAM_OFF_UNIT = 16
_WINDOW_BYTES = (_DRAM_OFF_MAX + 1) * _DRAM_OFF_UNIT  # 1,048,576


def _walking_disabled() -> bool:
    """Diagnostic knob: ``TACCEL_DISABLE_DRAM_OFF=1`` forces the planner
    to never use the M-type ``dram_off`` walking path (A2). Address
    register caching (A1) still applies. Used to bisect whether the
    nonzero ``dram_off`` path has a hidden RTL-side cost."""
    return os.environ.get("TACCEL_DISABLE_DRAM_OFF", "").lower() in (
        "1", "true", "yes",
    )


def set_addr(addr_reg: int, byte_addr: int) -> List[Instruction]:
    """Emit SET_ADDR_LO + SET_ADDR_HI to set a 56-bit DRAM address.

    Stateless: emits both halves unconditionally. Codegen should prefer
    :class:`AddrPlanner` to elide redundant writes; this helper is kept
    for tests and ad-hoc users that want the no-state behavior.
    """
    lo = byte_addr & 0xFFFFFFF
    hi = (byte_addr >> 28) & 0xFFFFFFF
    return [
        SetAddrLoInsn(addr_reg=addr_reg, imm28=lo),
        SetAddrHiInsn(addr_reg=addr_reg, imm28=hi),
    ]


class AddrPlanner:
    """Cache of the four 56-bit address registers as the compiler thinks
    the program will leave them.

    The planner state must mirror what the RTL/golden will have in
    ``addr_regs[r]`` at the program point where the next access is
    issued. Since the only writers of ``addr_regs`` are SET_ADDR_LO and
    SET_ADDR_HI, and both are issued exclusively through this planner
    (everywhere except :func:`set_addr` callers, which the codegen
    avoids), the cache is exact.

    Cache value semantics:

    * ``None`` — the planner has not written this register yet. RTL
      reset value is 0, but we conservatively emit both halves on first
      use so we don't rely on that.
    * ``int`` — the exact 56-bit value the register will hold when the
      next instruction in the stream executes.
    """

    def __init__(self) -> None:
        self._cached: List[Optional[int]] = [None, None, None, None]

    def reset(self) -> None:
        """Drop all cached state.

        Codegen creates one planner per compile so this is rarely needed;
        provided for tests that reuse a planner across synthetic
        compiles.
        """
        self._cached = [None, None, None, None]

    def current(self, addr_reg: int) -> Optional[int]:
        """Return the cached 56-bit value of ``addr_reg`` (or ``None``
        if it has not been written by the planner)."""
        return self._cached[addr_reg]

    def plan_access(
        self, addr_reg: int, byte_addr: int
    ) -> Tuple[List[Instruction], int]:
        """Plan a DMA access to ``byte_addr`` via ``addr_reg``.

        Returns ``(setup_insns, dram_off)``. The caller emits
        ``setup_insns`` (zero, one, or two A-type instructions) and then
        a LOAD/STORE with ``addr_reg=addr_reg`` and the returned
        ``dram_off`` in its M-type ``dram_off`` field.

        Guarantees on success:

        * ``addr_regs[addr_reg] + dram_off * 16 == byte_addr`` once
          ``setup_insns`` execute.
        * ``0 <= dram_off <= 0xFFFF``.
        * No emitted SetAddr*Insn whose ``imm28`` matches the cache.

        Raises :class:`ValueError` if ``byte_addr`` is negative, exceeds
        56 bits, or is not 16-byte aligned (the M-type cannot represent
        a non-16-byte delta).
        """
        if not 0 <= addr_reg < 4:
            raise ValueError(f"addr_reg must be 0-3, got {addr_reg}")
        if byte_addr < 0 or byte_addr.bit_length() > 56:
            raise ValueError(
                f"byte_addr {byte_addr:#x} exceeds 56-bit DRAM range"
            )
        if byte_addr & (_DRAM_OFF_UNIT - 1):
            raise ValueError(
                f"byte_addr {byte_addr:#x} must be 16-byte aligned for "
                f"M-type dram_off encoding"
            )

        cached = self._cached[addr_reg]

        # Walking window: the cache already holds a base within 1 MB
        # below byte_addr → no SET_ADDR needed, encode the delta in
        # dram_off. Skipped when ``TACCEL_DISABLE_DRAM_OFF=1`` so the
        # planner reduces to pure address-register caching (A1 only).
        if cached is not None and not _walking_disabled():
            delta = byte_addr - cached
            if 0 <= delta < _WINDOW_BYTES:
                # 16-byte alignment was checked above; cached is also a
                # plan_access result so it's aligned too — delta divides
                # cleanly.
                return [], delta // _DRAM_OFF_UNIT

        # Miss: write enough of addr_reg to make byte_addr representable
        # with dram_off=0 (i.e. write the full 56-bit value).
        lo = byte_addr & 0xFFFFFFF
        hi = (byte_addr >> 28) & 0xFFFFFFF

        if cached is None:
            cached_lo: Optional[int] = None
            cached_hi: Optional[int] = None
        else:
            cached_lo = cached & 0xFFFFFFF
            cached_hi = (cached >> 28) & 0xFFFFFFF

        insns: List[Instruction] = []
        if cached_lo != lo:
            insns.append(SetAddrLoInsn(addr_reg=addr_reg, imm28=lo))
        if cached_hi != hi:
            insns.append(SetAddrHiInsn(addr_reg=addr_reg, imm28=hi))

        self._cached[addr_reg] = byte_addr
        return insns, 0
