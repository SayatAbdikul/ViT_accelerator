"""Unit tests for :mod:`taccel.compiler.dma_emitter`.

The :class:`AddrPlanner` is load-bearing for compile-time correctness:
its cache must mirror what the RTL / golden simulator will have in
``addr_regs[r]`` at the program point where the next instruction
executes. If the cache drifts, the next LOAD's effective byte address
diverges from what the programmer intended and the bit-exact parity
gate fails.

These tests pin down the contract:

* First access to any addr_reg writes both halves.
* Same address again writes nothing.
* Walking forward within 1 MB writes nothing and returns nonzero
  ``dram_off``.
* Stepping outside the window writes whichever halves changed.
* The cache is per-register (writes to R1 don't affect R0).
"""
from __future__ import annotations

import pytest

from taccel.compiler.dma_emitter import AddrPlanner, set_addr
from taccel.isa.instructions import SetAddrLoInsn, SetAddrHiInsn
from taccel.isa.opcodes import Opcode


def _kinds(insns):
    """Return the opcode list for the emitted setup instructions."""
    return [type(i).__name__ for i in insns]


def test_first_access_writes_both_halves():
    p = AddrPlanner()
    insns, dram_off = p.plan_access(0, 0x12345_6789_0)  # 36-bit addr
    assert _kinds(insns) == ["SetAddrLoInsn", "SetAddrHiInsn"]
    assert dram_off == 0


def test_repeat_same_address_writes_nothing():
    p = AddrPlanner()
    p.plan_access(0, 0x1000)
    insns, dram_off = p.plan_access(0, 0x1000)
    assert insns == []
    assert dram_off == 0


def test_walk_forward_uses_dram_off_no_setup():
    p = AddrPlanner()
    p.plan_access(0, 0x1_0000)  # base
    insns, dram_off = p.plan_access(0, 0x1_0010)  # +16 bytes
    assert insns == []
    assert dram_off == 1
    insns, dram_off = p.plan_access(0, 0x1_0100)
    assert insns == []
    assert dram_off == 16


def test_walk_at_window_edge():
    p = AddrPlanner()
    p.plan_access(0, 0)
    # 1 MB - 16 bytes is the last representable offset (dram_off = 0xFFFF).
    insns, dram_off = p.plan_access(0, 0xFFFF * 16)
    assert insns == []
    assert dram_off == 0xFFFF


def test_walk_beyond_window_emits_setup():
    p = AddrPlanner()
    p.plan_access(0, 0)
    # 1 MB exactly is outside the 16-bit dram_off range, so we must
    # write a new base. Since hi == 0 stayed the same, only LO is
    # written.
    insns, dram_off = p.plan_access(0, 1 << 20)
    assert _kinds(insns) == ["SetAddrLoInsn"]
    assert dram_off == 0


def test_backwards_step_emits_setup():
    # Stepping back below the base means dram_off would need to be
    # negative — not representable. We re-base.
    p = AddrPlanner()
    p.plan_access(0, 0x1_0000)
    insns, dram_off = p.plan_access(0, 0)
    assert _kinds(insns) == ["SetAddrLoInsn"]
    assert dram_off == 0


def test_high_bits_change_writes_both_halves():
    p = AddrPlanner()
    # First access to a 36-bit address with both halves nonzero so
    # subsequent moves that touch HI also have a different LO.
    p.plan_access(0, 0x1234_5670)  # LO = 0x1234567, HI = 0
    # Now jump 30 bits up: LO becomes 0x1234567 (unchanged), HI becomes 4.
    # Verify we cover the "HI changes alongside LO" path by picking a
    # destination whose LO differs from the cached one too.
    insns, dram_off = p.plan_access(0, (4 << 28) | 0x89AB_CD0)
    assert _kinds(insns) == ["SetAddrLoInsn", "SetAddrHiInsn"]
    assert dram_off == 0


def test_high_bits_unchanged_writes_only_lo():
    p = AddrPlanner()
    p.plan_access(0, 0)
    # Stay within the lower 28 bits — HI stays 0, only LO changes
    # (and since the delta exceeds 1 MB the dram_off path can't
    # absorb it).
    insns, dram_off = p.plan_access(0, 0x800_0000)  # 128 MB, < 2^28
    assert _kinds(insns) == ["SetAddrLoInsn"]
    assert dram_off == 0


def test_per_register_state_isolated():
    p = AddrPlanner()
    p.plan_access(0, 0x1000)
    insns, dram_off = p.plan_access(1, 0x1000)
    # R1 was uncached → emits both halves even though R0 holds the
    # same value.
    assert _kinds(insns) == ["SetAddrLoInsn", "SetAddrHiInsn"]
    assert dram_off == 0


def test_emitted_imm28_matches_address():
    p = AddrPlanner()
    addr = 0xDEAD_BEEF_0  # 36 bits
    insns, _ = p.plan_access(2, addr)
    lo, hi = insns
    assert isinstance(lo, SetAddrLoInsn) and isinstance(hi, SetAddrHiInsn)
    assert lo.addr_reg == 2 and hi.addr_reg == 2
    assert lo.imm28 == (addr & 0xFFFFFFF)
    assert hi.imm28 == (addr >> 28) & 0xFFFFFFF


def test_byte_addr_must_be_16_aligned():
    p = AddrPlanner()
    with pytest.raises(ValueError, match="16-byte aligned"):
        p.plan_access(0, 0x1008)


def test_byte_addr_56_bit_ceiling():
    p = AddrPlanner()
    with pytest.raises(ValueError, match="56-bit"):
        p.plan_access(0, 1 << 56)


def test_addr_reg_range():
    p = AddrPlanner()
    with pytest.raises(ValueError, match="addr_reg"):
        p.plan_access(4, 0)


def test_legacy_set_addr_still_works():
    """The stateless helper is kept; it should always emit both halves."""
    insns = set_addr(1, 0x123_0000)
    assert _kinds(insns) == ["SetAddrLoInsn", "SetAddrHiInsn"]
    assert insns[0].imm28 == 0x123_0000 & 0xFFFFFFF
    assert insns[1].imm28 == 0


def test_reset_drops_state():
    p = AddrPlanner()
    p.plan_access(0, 0x1000)
    p.reset()
    insns, _ = p.plan_access(0, 0x1000)
    assert _kinds(insns) == ["SetAddrLoInsn", "SetAddrHiInsn"]
