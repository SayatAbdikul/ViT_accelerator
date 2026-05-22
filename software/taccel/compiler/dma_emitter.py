"""DMA addressing helpers for the TACCEL ISA.

Every DRAM access in TACCEL needs a 56-bit byte address split across two
A-type instructions (SET_ADDR_LO writes bits [27:0], SET_ADDR_HI writes
bits [55:28]). This module owns that split so callers say
``set_addr(addr_reg, byte_addr)`` instead of inlining the bit-shift idiom.
"""
from typing import List

from ..isa.instructions import Instruction, SetAddrLoInsn, SetAddrHiInsn


def set_addr(addr_reg: int, byte_addr: int) -> List[Instruction]:
    """Emit SET_ADDR_LO + SET_ADDR_HI to set a 56-bit DRAM address."""
    lo = byte_addr & 0xFFFFFFF
    hi = (byte_addr >> 28) & 0xFFFFFFF
    return [
        SetAddrLoInsn(addr_reg=addr_reg, imm28=lo),
        SetAddrHiInsn(addr_reg=addr_reg, imm28=hi),
    ]
