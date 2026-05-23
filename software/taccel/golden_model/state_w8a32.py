"""Machine state for the W8A32 golden model simulator.

W8A32 mode reinterprets activation buffers and the accumulator as FP32
storage. The byte budgets are identical to W8A8 — only the element
count changes (¼ as many FP32 activations as INT8 activations fit in
the same ABUF byte capacity).

WBUF still holds INT8 weights + FP16 per-channel scales; the matmul
engine dequantizes weights at multiply time rather than via a separate
REQUANT_PC op.
"""
from __future__ import annotations

import numpy as np

from .state import MachineState


class MachineStateW8A32(MachineState):
    """W8A32 machine state.

    The underlying storage is identical to the W8A8 ``MachineState`` —
    same ABUF/WBUF byte sizes, same ACCUM 16384-element int32 array. The
    difference is purely interpretive: callers read/write ABUF and ACCUM
    via :func:`memory.read_fp32_tile` / :func:`memory.write_fp32_tile`,
    which view the underlying bytes as float32.
    """

    @property
    def abuf_view_fp32(self) -> np.ndarray:
        """Return the activation buffer as a flat FP32 view (32K elements)."""
        return np.frombuffer(self.abuf, dtype=np.float32)

    @property
    def accum_view_fp32(self) -> np.ndarray:
        """Return the accumulator as a flat FP32 view (16K elements)."""
        return self.accum.view(np.float32)
