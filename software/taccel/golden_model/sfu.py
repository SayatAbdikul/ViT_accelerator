"""Special Function Units: LayerNorm, Softmax, GELU (FP32 internal, INT8 I/O).

Precision spec (matches RTL target)
------------------------------------
- Scale registers store FP16 values, widened to FP32 for arithmetic.
- All SFU internal operations use FP32 (dequant, reduction, exp/erf, requant).
- Rounding convention: **round-half-to-even** (IEEE 754 default, NumPy default).
  This applies to REQUANT, SCALE_MUL, and all SFU requantization paths.
  RTL must implement the same rounding mode, otherwise results may differ
  by ±1 LSB on tie values (e.g. 0.5 → 0 with banker's vs. 1 with away).
- Clip to INT8 after rounding: [-128, 127].

GELU erf() implementation
--------------------------
The golden model calls ``fp32_prim_ref.fp32_gelu_arr``, which implements
the Abramowitz & Stegun 7.1.26 polynomial in FP32 with the same operation
order and rounding points as the synthesizable RTL ``fp32_gelu_bits``.
This makes RTL ≡ golden bit-exact by construction; the exact-logit gate at
``software/tools/batch_compare_rtl_golden.py:141`` is the load-bearing
invariant.

``_erf_poly()`` below is the reference Python expression of the same
polynomial used in fp32_prim_ref and RTL.  It is exercised directly by
``software/tests/test_golden_model.py::test_erf_poly_matches_scipy_for_int8``
and ``test_erf_poly_max_fp32_error`` as a precision check against scipy;
it is not on the execute_gelu path.
"""
import numpy as np
from . import memory
from ..isa.opcodes import BUF_ACCUM
from ..utils import fp32_prim_ref as fpr
from ..utils.int8_ops import clip_int8

CYCLE_PER_ELEMENT = 2


def _erf_poly(x: np.ndarray) -> np.ndarray:
    """Polynomial approximation of erf(x) for RTL implementation reference.

    Abramowitz & Stegun formula 7.1.26 — max |error| < 5e-7 in FP32
    (~1.5e-7 in FP64).  Uses only FMA + exp — no erf hardware needed.

    RTL implementation: 5 FMA + 1 exp + 1 reciprocal per element.
    """
    a1 = np.float32(0.254829592)
    a2 = np.float32(-0.284496736)
    a3 = np.float32(1.421413741)
    a4 = np.float32(-1.453152027)
    a5 = np.float32(1.061405429)
    p = np.float32(0.3275911)

    sign = np.sign(x)
    x_abs = np.abs(x)
    t = np.float32(1.0) / (np.float32(1.0) + p * x_abs)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    y = np.float32(1.0) - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * np.exp(-(x_abs * x_abs))
    return sign * y


def _get_dual_scales(state, sreg: int):
    """Return (in_scale, out_scale) as FP32 from consecutive scale registers.

    Scale registers hold FP16. Widening to FP32 preserves the exact FP16 value
    without adding precision — this is the RTL behaviour (FP16 reg → FP32 datapath).
    """
    from .simulator import ConfigError
    if sreg >= 15:
        raise ConfigError("SFU sreg+1 out of range")
    in_scale  = np.float32(state.scale_regs[sreg])
    out_scale = np.float32(state.scale_regs[sreg + 1])
    return in_scale, out_scale


def _get_quad_scales(state, sreg: int):
    """Return four consecutive FP32 scales from scale registers."""
    from .simulator import ConfigError
    if sreg >= 13:
        raise ConfigError("SFU sreg+3 out of range")
    return tuple(np.float32(state.scale_regs[sreg + idx]) for idx in range(4))


def _runtime_twin_spec(state, kind: str):
    specs = getattr(state, "runtime_twin_specs", {}) or {}
    spec = specs.get(int(getattr(state, "current_pc", -1)))
    if not spec or spec.get("kind") != kind or spec.get("mode") != "paper_exact":
        return None
    return spec


def execute_layernorm(state, insn):
    """LayerNorm: dequant INT8 → FP32, normalize, requant → INT8.

    src1 = input activations, src2 = gamma/beta (FP16 packed), dst = output.
    All arithmetic in FP32; gamma/beta widened from FP16.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)

    # Read gamma, beta — stored as FP16, widen to FP32 via the shared twin
    # (reproduces the RTL fp32_from_fp16 inf/NaN -> ±65504 clamp).
    gb_bytes = memory.read_bytes(state, insn.src2_buf, insn.src2_off, N * 4)
    gamma = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[:N * 2], dtype=np.uint16))
    beta  = fpr.fp32_from_fp16_arr(np.frombuffer(gb_bytes[N * 2:], dtype=np.uint16))

    # Dequantize: INT8 × FP32(in_scale) → FP32 (elementwise == RTL fp32_mul)
    x = (inp.astype(np.float32) * in_scale).astype(np.float32)

    # Normalize in FP32. Mean/var are SEQUENTIAL left folds matching the RTL
    # FSM element order (sfu_engine.sv:696-711), NOT numpy pairwise sums.
    eps = np.float32(1e-6)
    mean = fpr.fp32_mean_rows(x)[:, None]
    var = fpr.fp32_var_rows(x, mean)[:, None]
    denom = np.sqrt((var + eps).astype(np.float32), dtype=np.float32)
    x_norm = ((x - mean).astype(np.float32) / denom).astype(np.float32)
    x_out = ((x_norm * gamma).astype(np.float32) + beta).astype(np.float32)

    result = fpr.fp32_quantize_i8_arr(x_out, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax(state, insn):
    """Softmax: dequant (INT8 or INT32) → FP32, softmax along last dim, requant → INT8.

    Numerically stable: subtract row-max before exp.
    All arithmetic in FP32.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    if insn.src1_buf == BUF_ACCUM:
        # C1 path: consume raw INT32 QKT accumulators directly.
        inp_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, N)
        x = inp_i32.astype(np.float32) * in_scale
    else:
        inp_i8 = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
        x = inp_i8.astype(np.float32) * in_scale

    # Numerically stable softmax in FP32. row-max is order-independent;
    # exp() and the exp-sum SEQUENTIAL fold match RTL (sfu_engine.sv:677-690).
    x = x.astype(np.float32)
    x_shifted = (x - x.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_x = fpr.fp32_exp_arr(x_shifted)
    denom = fpr.fp32_sum_rows(exp_x)[:, None]
    x_out = (exp_x / denom).astype(np.float32)

    result = fpr.fp32_quantize_i8_arr(x_out, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_gelu(state, insn):
    """GELU: dequant INT8 → FP32, GELU activation, requant → INT8.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    All arithmetic in FP32 via fp32_prim_ref.fp32_gelu_arr (A&S 7.1.26
    polynomial), bit-exact to the RTL fp32_gelu_bits primitive.
    """
    from .simulator import ConfigError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    M = m_tiles * 16
    N = n_tiles * 16

    in_scale, out_scale = _get_dual_scales(state, insn.sreg)

    if insn.src1_buf == BUF_ACCUM:
        inp_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, N)
        x = inp_i32.astype(np.float32) * in_scale
    else:
        inp = memory.read_int8_tile(state, insn.src1_buf, insn.src1_off, M, N)
        # Dequantize: INT8 × FP32(in_scale) → FP32
        x = inp.astype(np.float32) * in_scale

    # GELU in FP32 via the shared A&S 7.1.26 poly twin (== RTL fp32_gelu_bits;
    # the old scipy.special.erf path is not hardware-realizable).
    x = x.astype(np.float32)
    x_out = fpr.fp32_gelu_arr(x)

    result = fpr.fp32_quantize_i8_arr(x_out, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)
    state.cycle_count += M * N * CYCLE_PER_ELEMENT


def execute_softmax_attnv(state, insn):
    """Fused SOFTMAX_ATTNV.

    Reads QKT from ACCUM and V from INT8 SRAM, computes softmax(QKT) in FP32,
    immediately multiplies by dequantized V in FP32, and requantizes the final
    attn@V output to INT8. For diagnostics, it also returns a virtual INT8
    softmax tile using a trace-only scale in sreg+3.
    """
    from .simulator import ConfigError, IllegalBufferError
    if state.tile_config is None:
        raise ConfigError("CONFIG_TILE not set")
    if insn.src1_buf != BUF_ACCUM:
        raise IllegalBufferError(insn.src1_buf)
    if insn.src2_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.src2_buf)
    if insn.dst_buf == BUF_ACCUM:
        raise IllegalBufferError(insn.dst_buf)

    m_tiles = state.tile_config[0] + 1
    n_tiles = state.tile_config[1] + 1
    k_tiles = state.tile_config[2] + 1
    M = m_tiles * 16
    N = n_tiles * 16
    K = k_tiles * 16

    qkt_in_scale, v_scale, out_scale, softmax_trace_scale = _get_quad_scales(state, insn.sreg)

    qkt_i32 = memory.read_int32_tile(state, BUF_ACCUM, insn.src1_off, M, K)
    v_i8 = memory.read_int8_tile(state, insn.src2_buf, insn.src2_off, K, N)

    qkt = (qkt_i32.astype(np.float32) * qkt_in_scale).astype(np.float32)
    v = (v_i8.astype(np.float32) * v_scale).astype(np.float32)

    qkt_shifted = (qkt - qkt.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_qkt = fpr.fp32_exp_arr(qkt_shifted)
    softmax = (exp_qkt / fpr.fp32_sum_rows(exp_qkt)[:, None]).astype(np.float32)
    # attn@V as a SEQUENTIAL FP32 fold over k (matches the RTL accumulator),
    # not numpy's pairwise matmul.
    prod = (softmax[:, :, None] * v[None, :, :]).astype(np.float32)
    attn_v = np.add.accumulate(prod, axis=1, dtype=np.float32)[:, -1, :].astype(np.float32)

    result = fpr.fp32_quantize_i8_arr(attn_v, out_scale)
    memory.write_int8_tile(state, insn.dst_buf, insn.dst_off, result)

    softmax_i8 = fpr.fp32_quantize_i8_arr(softmax, softmax_trace_scale)

    state.cycle_count += (M * K * CYCLE_PER_ELEMENT) + (m_tiles * n_tiles * k_tiles * 16) + (M * N)
    return {
        "softmax": {
            "raw": softmax_i8,
            "dtype": "int8",
            "scale": float(softmax_trace_scale),
            "sat": int(np.count_nonzero((softmax_i8 == 127) | (softmax_i8 == -128))),
            "zero": int(np.count_nonzero(softmax_i8 == 0)),
            "total": int(softmax_i8.size),
        }
    }
