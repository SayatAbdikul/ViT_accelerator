"""SET_SCALE instruction emission helpers.

CodeGenerator constructs SET_SCALE instructions in 25+ places, almost
always with the same boilerplate:

    SetScaleInsn(sreg=X, src_mode=0, imm16=_fp16_to_uint16(value))

These helpers collapse that boilerplate. The SFU dual-scale convention
(consecutive sregs for in_scale and out_scale) gets its own helper since
it appears repeatedly.

All values are encoded as FP16 bit patterns stored little-endian — the
simulator round-trips them through ``np.frombuffer(..., dtype=np.float16)``
to recover the value, so byte order is load-bearing.
"""
from typing import List

import numpy as np

from ..isa.instructions import SetScaleInsn


def fp16_to_uint16(val: float) -> int:
    """Convert an FP32 value to its FP16 bit pattern as little-endian uint16."""
    fp16 = np.float16(val)
    return int(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0])


def emit_scale(sreg: int, value: float) -> SetScaleInsn:
    """One SET_SCALE on ``sreg`` with ``value`` packed as FP16."""
    return SetScaleInsn(sreg=sreg, src_mode=0, imm16=fp16_to_uint16(value))


def emit_scale_pair(sreg: int, in_val: float, out_val: float) -> List[SetScaleInsn]:
    """SFU dual-scale convention: SET_SCALE on ``sreg`` (in) and ``sreg+1`` (out)."""
    return [emit_scale(sreg, in_val), emit_scale(sreg + 1, out_val)]
