from functools import partial

from .quantize import (
    adaround_greedy,
    dequantize_tensor,
    quantize_weights,
    quantize_tensor,
    quantize_tensor_clipped,
)
from .scales import ScalePropagator
from .calibrate import calibrate_model, CalibrationResult, collect_layer_inputs
from .smooth_quant import compute_smooth_factors, apply_smooth_quant
from .twin_uniform import (
    quantize_dequant_gelu_twin,
    quantize_dequant_softmax_twin,
)
from .hessian_guided import (
    gelu_fc2_hessian_diag,
    softmax_attn_v_hessian_diag,
    weighted_quant_error_score,
)

# Canonical W8A32 weight-quantization entry point: per-channel symmetric INT8
# on the [out, in] view of every Linear/Conv2d weight. This is the same scheme
# that ``fake_quant.apply_weight_quantization`` invokes (see fake_quant.py:39),
# and the same scheme ``Compiler.compile_w8a32`` uses for in-program dequant.
# Exported by name so the W8A32 path has a single import surface; downstream
# code should call ``W8A32_QUANTIZE(tensor)`` rather than retyping the
# ``per_channel=True`` kwarg at each call site.
W8A32_QUANTIZE = partial(quantize_tensor, per_channel=True)

# W8A16 uses the identical per-channel INT8 weight quantization scheme — the
# activation precision (FP32 vs FP16) does not affect the weight-side
# contract. Aliased separately so the W8A16 codegen / tooling can import its
# mode-named entry point and so future divergence (e.g. clip-search-driven
# per-channel scales tuned for FP16-narrowed dequant weights) has a clean
# place to land.
W8A16_QUANTIZE = W8A32_QUANTIZE
