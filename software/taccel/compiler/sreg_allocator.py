"""Scale-register allocator for the TACCEL ISA.

The ISA has 16 scale registers (S0..S15), but the codegen needs three
disjoint allocation domains so the streams never overwrite each other:

  - Singles  : odd  registers (1, 3, 5, 7, 9, 11, 13) — used by REQUANT,
               SCALE_MUL, and any other op that consumes one scale.
  - Pairs    : even registers (0, 2, 4, 6, 8, 10, 12) — used by SFU ops
               that need (in_scale, out_scale) on consecutive sregs.
  - Quads    : the four registers (0, 4, 8, 12) — used by SOFTMAX_ATTNV
               which needs four consecutive sregs.

Each domain wraps around independently. Scale registers are set immediately
before use, so wrapping is safe.

Extracting this state out of CodeGenerator means the allocation policy lives
in one place and can be unit-tested in isolation; future ISA variants can
swap pools by subclassing or by passing different lists.
"""
from typing import List


# Default register pools. Defined module-level so callers can override.
ODD_POOL: List[int] = [1, 3, 5, 7, 9, 11, 13]
PAIR_POOL: List[int] = [0, 2, 4, 6, 8, 10, 12]
QUAD_POOL: List[int] = [0, 4, 8, 12]


class SRegAllocator:
    """Round-robin allocator for the three sreg pools used by codegen."""

    def __init__(
        self,
        odd_pool: List[int] = ODD_POOL,
        pair_pool: List[int] = PAIR_POOL,
        quad_pool: List[int] = QUAD_POOL,
    ) -> None:
        self._odd_pool = list(odd_pool)
        self._pair_pool = list(pair_pool)
        self._quad_pool = list(quad_pool)
        self._next_single = 0
        self._next_pair = 0
        self._next_quad = 0

    def alloc_single(self) -> int:
        """Allocate one scale register from the odd pool."""
        reg = self._odd_pool[self._next_single % len(self._odd_pool)]
        self._next_single = (self._next_single + 1) % len(self._odd_pool)
        return reg

    def alloc_pair(self) -> int:
        """Allocate a consecutive pair (reg, reg+1) from the even pool.

        Returns the lower (even) register; caller uses reg and reg+1.
        """
        reg = self._pair_pool[self._next_pair % len(self._pair_pool)]
        self._next_pair = (self._next_pair + 1) % len(self._pair_pool)
        return reg

    def alloc_quad(self) -> int:
        """Allocate four consecutive scale registers (reg, reg+1, reg+2, reg+3)."""
        reg = self._quad_pool[self._next_quad % len(self._quad_pool)]
        self._next_quad = (self._next_quad + 1) % len(self._quad_pool)
        return reg

    def reset(self) -> None:
        """Reset all three pool indices to 0."""
        self._next_single = 0
        self._next_pair = 0
        self._next_quad = 0
