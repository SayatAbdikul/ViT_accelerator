from functools import partial

from .quantize import dequantize_tensor, quantize_tensor
from .scales import ScalePropagator

# Canonical W8A16 weight-quantization entry point: per-channel symmetric INT8
# on the [out, in] view of every Linear/Conv2d weight. This is the same scheme
# that ``fake_quant.apply_weight_quantization`` invokes (see fake_quant.py:39),
# and the same scheme ``Compiler.compile_w8a16`` uses for in-program dequant.
# Exported by name so the W8A16 path has a single import surface; downstream
# code should call ``W8A16_QUANTIZE(tensor)`` rather than retyping the
# ``per_channel=True`` kwarg at each call site.
W8A16_QUANTIZE = partial(quantize_tensor, per_channel=True)

# W8A32 uses the identical per-channel INT8 weight quantization scheme — the
# activation precision (FP32 vs FP16) does not affect the weight-side
# contract. Aliased separately so the W8A32 ceiling tooling can import its
# mode-named entry point.
W8A32_QUANTIZE = W8A16_QUANTIZE
