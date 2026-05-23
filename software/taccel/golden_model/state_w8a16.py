"""Machine state for the W8A16 golden model simulator.

W8A16 mode keeps ABUF activations as FP16 (2 bytes/element, 2× the
capacity of W8A32 in elements) while the accumulator stays FP32 — the
matmul engine widens both FP16 inputs to FP32 internally and emits
FP32 results into ACCUM, matching the standard mixed-precision
convention.

WBUF still holds INT8 weights + FP16 per-channel scales for the W8A8
path; on the W8A16 path WBUF receives FP16 dequant weights pre-baked
at compile time (per-channel scale already applied), mirroring the
W8A32 strategy. The byte budgets in `MachineState` are unchanged —
only the element interpretation flips.
"""
from __future__ import annotations

import numpy as np

from .state import MachineState


class MachineStateW8A16(MachineState):
    """W8A16 machine state.

    The underlying storage is identical to the W8A8 ``MachineState`` —
    same ABUF/WBUF byte sizes, same ACCUM 16384-element int32 array. The
    difference is purely interpretive:

    * ABUF / WBUF byte capacity holds 2× as many FP16 elements as the
      W8A32 path's FP32 elements (64K FP16 elements in 128 KB ABUF).
    * ACCUM is still viewed as FP32 (same byte layout as int32; matmul
      accumulates in FP32 even though both operands are FP16).
    """

    @property
    def abuf_view_fp16(self) -> np.ndarray:
        """Return the activation buffer as a flat FP16 view (64K elements)."""
        return np.frombuffer(self.abuf, dtype=np.float16)

    @property
    def accum_view_fp32(self) -> np.ndarray:
        """Return the accumulator as a flat FP32 view (16K elements).

        FP16 × FP16 → FP32 is the standard mixed-precision accumulator
        convention; ACCUM is byte-identical to the W8A32 path's view.
        """
        return self.accum.view(np.float32)
