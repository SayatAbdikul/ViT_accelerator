"""Deterministic FP32 primitive reference (the Python twin).

Single source of numeric truth shared by:
  * RTL  : rtl/src/include/fp32_prim_pkg.sv  (fp32_*_bits)
  * C++  : rtl/verilator/test_fp32_prims.cpp (oracle)
  * Golden: software/taccel/golden_model/sfu.py (after Phase 2 repoint)

Spec: rtl/src/include/ARITH_CONTRACT.md.

div / sqrt are IEEE-754 binary32 correctly-rounded (numpy float32 ops are
round-to-nearest-ties-to-even and bit-identical to C `(float)(a/b)` / `sqrtf`
and to a correctly-rounded RTL Newton step). exp / erf / gelu are *co-defined*
(a fixed polynomial — they are NOT libm-equal; that is the whole point: RTL
cannot reproduce libm, so the golden is repointed onto this exact algorithm).
Reductions are sequential left folds that match the RTL FSM element order
(sfu_engine.sv:677-710), NOT numpy pairwise summation.
"""
from __future__ import annotations

import numpy as np

f32 = np.float32

# ─── bit <-> value helpers ────────────────────────────────────────────────────


def bits_to_f32(u: int) -> np.float32:
    return np.frombuffer(np.uint32(u & 0xFFFFFFFF).tobytes(), dtype=np.float32)[0]


def f32_to_bits(x) -> int:
    return int(np.frombuffer(f32(x).tobytes(), dtype=np.uint32)[0])


# ─── scalar core ops (correctly-rounded binary32) ─────────────────────────────


def fp32_add(a, b) -> np.float32:
    return f32(a) + f32(b)


def fp32_sub(a, b) -> np.float32:
    return f32(a) - f32(b)


def fp32_mul(a, b) -> np.float32:
    return f32(a) * f32(b)


def fp32_div(a, b) -> np.float32:
    """Correctly-rounded binary32 division (matches C (float)(a/b))."""
    a = f32(a)
    b = f32(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return f32(a) / f32(b)


def fp32_sqrt(x) -> np.float32:
    """Correctly-rounded binary32 sqrt (matches C sqrtf)."""
    x = f32(x)
    with np.errstate(invalid="ignore"):
        return np.sqrt(x, dtype=np.float32)


# ─── exp : co-defined range-reduced degree-7 Horner (Taylor) ──────────────────
# x = k*ln2 + r,  |r| <= ln2/2 ;  exp(x) = 2^k * P(r).
# Constants frozen as the exact FP32 values (see ARITH_CONTRACT.md). The RTL
# uses the identical bit patterns and identical Horner order.

_LOG2E = f32(1.4426950408889634)          # 0x3FB8AA3B
_LN2_HI = f32(0.693145751953125)          # 0x3F317200  (exact, 11 low bits 0)
_LN2_LO = f32(1.428606765330187e-06)      # 0x35BFBE8E  (ln2 - ln2_hi)
# 1/n! for n = 7..0  (Horner high->low)
_EXP_C = [f32(1.0 / 5040.0), f32(1.0 / 720.0), f32(1.0 / 120.0),
          f32(1.0 / 24.0), f32(1.0 / 6.0), f32(0.5), f32(1.0), f32(1.0)]

_F32_INF = np.float32(np.inf)


def _rint_half_even(x: np.float32) -> int:
    """Round binary32 to nearest integer, ties to even (matches RTL fp32 rint)."""
    x = f32(x)
    fl = np.float32(np.floor(x))
    frac = f32(x - fl)
    n = int(fl)
    if frac > f32(0.5):
        return n + 1
    if frac < f32(0.5):
        return n
    return n + 1 if (n & 1) else n


def fp32_exp(x) -> np.float32:
    x = f32(x)
    if np.isnan(x):
        return f32(np.nan)
    # Saturate well outside binary32 range (exp overflows ~88.7, underflows ~-87.3)
    if x > f32(88.8):
        return _F32_INF
    if x < f32(-103.0):
        return f32(0.0)
    k = _rint_half_even(fp32_mul(x, _LOG2E))
    kf = f32(np.float32(k))
    # Cody-Waite: r = (x - k*ln2_hi) - k*ln2_lo
    r = fp32_sub(fp32_sub(x, fp32_mul(kf, _LN2_HI)), fp32_mul(kf, _LN2_LO))
    p = _EXP_C[0]
    for c in _EXP_C[1:]:
        p = fp32_add(fp32_mul(p, r), c)
    # multiply by 2^k via exponent field (ldexp), staying in binary32
    out = f32(np.ldexp(np.float64(p), k))
    if not np.isfinite(out):
        return _F32_INF
    return f32(out)


# ─── erf / gelu : A&S 7.1.26, identical structure to sfu.py:_erf_poly ─────────
#
# Constants are co-defined with rtl/src/include/fp32_prim_pkg.sv (FP32_ERF_*).
# Some literals (A3, A4, P) round to different FP32 bit patterns than the
# RTL's bit-exact constants; we must use bits_to_f32 to lock the bytes,
# otherwise fp32_gelu / fp32_erf diverge from RTL by 1–2 FP16 ULPs (caught
# in rtl/cocotb/test_sfu.py during Phase 3).

_ERF_A1 = bits_to_f32(0x3E827906)
_ERF_A2 = bits_to_f32(0xBE91A98E)
_ERF_A3 = bits_to_f32(0x3FB5D78E)
_ERF_A4 = bits_to_f32(0xBFBA0005)
_ERF_A5 = bits_to_f32(0x3F87DC22)
_ERF_P = bits_to_f32(0x3EA7B9D2)
_INV_SQRT2 = bits_to_f32(0x3F3504F3)


def fp32_erf(x) -> np.float32:
    x = f32(x)
    s = f32(np.sign(x))
    a = f32(np.abs(x))
    t = fp32_div(f32(1.0), fp32_add(f32(1.0), fp32_mul(_ERF_P, a)))
    t2 = fp32_mul(t, t)
    t3 = fp32_mul(t2, t)
    t4 = fp32_mul(t3, t)
    t5 = fp32_mul(t4, t)
    poly = fp32_add(
        fp32_add(
            fp32_add(
                fp32_add(fp32_mul(_ERF_A1, t), fp32_mul(_ERF_A2, t2)),
                fp32_mul(_ERF_A3, t3)),
            fp32_mul(_ERF_A4, t4)),
        fp32_mul(_ERF_A5, t5))
    e = fp32_exp(f32(-fp32_mul(a, a)))
    y = fp32_sub(f32(1.0), fp32_mul(poly, e))
    return fp32_mul(s, y)


def fp32_gelu(x) -> np.float32:
    x = f32(x)
    e = fp32_erf(fp32_mul(x, _INV_SQRT2))
    return fp32_mul(fp32_mul(x, f32(0.5)), fp32_add(f32(1.0), e))


# ─── FP16 -> FP32 widen (reproduces legacy fp16_to_real, incl. inf clamp) ─────


def fp32_from_fp16(h: int) -> np.float32:
    h &= 0xFFFF
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    if e == 0 and f == 0:
        return bits_to_f32((s << 31))
    if e == 0:                                   # subnormal: f * 2^-24, exact
        k = f.bit_length() - 1
        frac = (f - (1 << k)) << (23 - k)
        return bits_to_f32((s << 31) | ((k + 103) << 23) | frac)
    if e == 0x1F:                                # inf/NaN -> clamp ±65504.0
        return bits_to_f32(0xC77FE000 if s else 0x477FE000)
    return bits_to_f32((s << 31) | ((e + 112) << 23) | (f << 13))


# ─── INT8 quantize : div then round-half-even then clip ───────────────────────


def fp32_quantize_i8(v, s) -> int:
    s = f32(s)
    if s == f32(0.0):
        return 0
    q = _rint_half_even(fp32_div(v, s))
    if q < -128:
        return -128
    if q > 127:
        return 127
    return q


# ─── sequential FP32 reductions (match RTL fold order) ────────────────────────


def fp32_max_seq(row) -> np.float32:
    row = np.asarray(row, dtype=np.float32)
    m = f32(row[0])
    for i in range(1, row.shape[0]):
        if row[i] > m:
            m = f32(row[i])
    return m


def fp32_sum_seq(row) -> np.float32:
    row = np.asarray(row, dtype=np.float32)
    acc = f32(0.0)
    for v in row:
        acc = fp32_add(acc, v)
    return acc


def fp32_mean_seq(row) -> np.float32:
    row = np.asarray(row, dtype=np.float32)
    return fp32_div(fp32_sum_seq(row), f32(np.float32(row.shape[0])))


def fp32_var_seq(row, mean=None) -> np.float32:
    row = np.asarray(row, dtype=np.float32)
    if mean is None:
        mean = fp32_mean_seq(row)
    acc = f32(0.0)
    for v in row:
        d = fp32_sub(v, mean)
        acc = fp32_add(acc, fp32_mul(d, d))
    return fp32_div(acc, f32(np.float32(row.shape[0])))


# ─── vectorized variants (bit-equal to the scalar fns; for the golden model) ──
# Each elementary numpy float32 op is RNE-correctly-rounded, so a vectorized
# expression with the identical op order is bit-identical to the scalar fold.
# np.add.accumulate(.,dtype=f32)[...,-1] IS the sequential left fold;
# np.maximum is order-independent; np.rint is ties-to-even. Verified in the
# self-check below against the scalar functions.


def fp32_sum_rows(x2d) -> np.ndarray:
    x2d = np.asarray(x2d, dtype=np.float32)
    return np.add.accumulate(x2d, axis=-1, dtype=np.float32)[..., -1]


def fp32_mean_rows(x2d) -> np.ndarray:
    x2d = np.asarray(x2d, dtype=np.float32)
    n = f32(np.float32(x2d.shape[-1]))
    return (fp32_sum_rows(x2d) / n).astype(np.float32)


def fp32_var_rows(x2d, mean_col) -> np.ndarray:
    x2d = np.asarray(x2d, dtype=np.float32)
    d = (x2d - mean_col).astype(np.float32)
    sq = (d * d).astype(np.float32)
    n = f32(np.float32(x2d.shape[-1]))
    return (np.add.accumulate(sq, axis=-1, dtype=np.float32)[..., -1] / n).astype(np.float32)


def fp32_exp_arr(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = (x * _LOG2E).astype(np.float32)
    k = np.rint(m).astype(np.int64)                       # ties to even
    kf = k.astype(np.float32)
    r = ((x - (kf * _LN2_HI).astype(np.float32)).astype(np.float32)
         - (kf * _LN2_LO).astype(np.float32)).astype(np.float32)
    p = np.full(x.shape, _EXP_C[0], dtype=np.float32)
    for c in _EXP_C[1:]:
        p = ((p * r).astype(np.float32) + c).astype(np.float32)
    with np.errstate(over="ignore"):
        res = np.ldexp(p.astype(np.float64), k).astype(np.float32)
    inf = np.float32(np.inf)
    res = np.where(np.isfinite(res) | np.isnan(res), res, inf).astype(np.float32)
    # guard precedence mirrors the scalar fp32_exp if-chain (low->high priority)
    res = np.where(k > 300, inf, res)
    res = np.where(k < -160, np.float32(0.0), res)
    minf = ~np.isfinite(m)
    res = np.where(minf & ~np.isnan(m), np.where(m < 0, np.float32(0.0), inf), res)
    res = np.where(np.isnan(m), np.float32(np.nan), res)
    res = np.where(np.isneginf(x), np.float32(0.0), res)
    res = np.where(np.isposinf(x), inf, res)
    res = np.where(np.isnan(x), np.float32(np.nan), res)
    return res.astype(np.float32)


def fp32_erf_arr(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    s = np.where(x < 0, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    a = np.abs(x).astype(np.float32)
    t = (f32(1.0) / (f32(1.0) + (_ERF_P * a).astype(np.float32)).astype(np.float32)).astype(np.float32)
    t2 = (t * t).astype(np.float32)
    t3 = (t2 * t).astype(np.float32)
    t4 = (t3 * t).astype(np.float32)
    t5 = (t4 * t).astype(np.float32)
    poly = ((((( _ERF_A1 * t).astype(np.float32) + (_ERF_A2 * t2).astype(np.float32)).astype(np.float32)
              + (_ERF_A3 * t3).astype(np.float32)).astype(np.float32)
             + (_ERF_A4 * t4).astype(np.float32)).astype(np.float32)
            + (_ERF_A5 * t5).astype(np.float32)).astype(np.float32)
    e = fp32_exp_arr((-(a * a).astype(np.float32)).astype(np.float32))
    y = (f32(1.0) - (poly * e).astype(np.float32)).astype(np.float32)
    out = (s * y).astype(np.float32)
    return np.where(x == 0, np.float32(0.0), out).astype(np.float32)


def fp32_gelu_arr(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    e = fp32_erf_arr((x * _INV_SQRT2).astype(np.float32))
    return ((x * f32(0.5)).astype(np.float32) * (f32(1.0) + e).astype(np.float32)).astype(np.float32)


def fp32_quantize_i8_arr(v, s) -> np.ndarray:
    """Elementwise quantize. v: f32 array, s: scalar f32 scale."""
    v = np.asarray(v, dtype=np.float32)
    s = f32(s)
    if s == f32(0.0):
        return np.zeros(v.shape, dtype=np.int8)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        qf = (v / s).astype(np.float32)
    q = np.rint(qf).astype(np.int64)                      # ties to even
    q = np.where(np.isnan(qf), 0, q)
    q = np.where(np.isposinf(qf), 127, q)
    q = np.where(np.isneginf(qf), -128, q)
    return np.clip(q, -128, 127).astype(np.int8)


_FP16_LUT = None


def fp32_from_fp16_arr(u16) -> np.ndarray:
    """Elementwise FP16(bits)->FP32 with the legacy ±65504 inf/NaN clamp."""
    global _FP16_LUT
    if _FP16_LUT is None:
        _FP16_LUT = np.array([fp32_from_fp16(h) for h in range(65536)],
                             dtype=np.float32)
    return _FP16_LUT[np.asarray(u16, dtype=np.uint16)]


if __name__ == "__main__":  # lightweight self-check
    rng = np.random.default_rng(0)
    a = rng.standard_normal(20000).astype(np.float32) * f32(40.0)
    b = rng.standard_normal(20000).astype(np.float32) * f32(40.0)
    b[b == 0] = f32(1.0)
    # div / sqrt must equal numpy float32 (correctly rounded) exactly
    assert all(f32_to_bits(fp32_div(x, y)) == f32_to_bits(f32(x) / f32(y))
               for x, y in zip(a[:2000], b[:2000])), "div != numpy f32"
    pa = np.abs(a[:2000])
    assert all(f32_to_bits(fp32_sqrt(x)) == f32_to_bits(np.sqrt(f32(x), dtype=np.float32))
               for x in pa), "sqrt != numpy f32"
    # exp / gelu accuracy vs libm (co-defined, but must be close)
    xs = np.linspace(-10, 10, 4001, dtype=np.float32)
    ee = np.array([float(fp32_exp(x)) for x in xs])
    assert np.max(np.abs(ee - np.exp(xs.astype(np.float64)))
                  / np.maximum(np.exp(xs.astype(np.float64)), 1e-30)) < 5e-6, "exp rel err"
    gg = np.array([float(fp32_gelu(x)) for x in xs])
    from scipy.special import erf as _erf
    gref = xs.astype(np.float64) * 0.5 * (1 + _erf(xs.astype(np.float64) / np.sqrt(2)))
    assert np.max(np.abs(gg - gref)) < 2e-5, "gelu abs err"
    # fp16 widen spot checks
    assert f32_to_bits(fp32_from_fp16(0x3C00)) == f32_to_bits(f32(1.0))
    assert f32_to_bits(fp32_from_fp16(0x7C00)) == 0x477FE000  # +inf -> 65504
    assert f32_to_bits(fp32_from_fp16(0xFC00)) == 0xC77FE000  # -inf -> -65504
    assert fp32_quantize_i8(f32(0.5), f32(1.0)) == 0          # ties to even
    assert fp32_quantize_i8(f32(1.5), f32(1.0)) == 2
    assert fp32_quantize_i8(f32(1000.0), f32(1.0)) == 127

    # vectorized variants must be BIT-identical to the scalar twin
    xs2 = rng.standard_normal(50000).astype(np.float32) * f32(25.0)
    ev = fp32_exp_arr(xs2)
    es = np.array([fp32_exp(x) for x in xs2], dtype=np.float32)
    assert np.array_equal(ev.view(np.uint32), es.view(np.uint32)), "exp_arr != scalar"
    gv = fp32_gelu_arr(xs2)
    gs = np.array([fp32_gelu(x) for x in xs2], dtype=np.float32)
    assert np.array_equal(gv.view(np.uint32), gs.view(np.uint32)), "gelu_arr != scalar"
    sv = f32(2.0)
    qv = fp32_quantize_i8_arr(xs2, sv)
    qs = np.array([fp32_quantize_i8(x, sv) for x in xs2], dtype=np.int8)
    assert np.array_equal(qv, qs), "quantize_arr != scalar"
    # sequential fold equivalence (accumulate == left fold)
    rows = rng.standard_normal((64, 208)).astype(np.float32) * f32(6.0)
    sr = fp32_sum_rows(rows)
    for i in range(rows.shape[0]):
        assert f32_to_bits(sr[i]) == f32_to_bits(fp32_sum_seq(rows[i])), "sum_rows fold"
    mr = fp32_mean_rows(rows)
    vr = fp32_var_rows(rows, mr[:, None])
    for i in range(rows.shape[0]):
        assert f32_to_bits(vr[i]) == f32_to_bits(fp32_var_seq(rows[i], mr[i])), "var_rows"
    assert f32_to_bits(fp32_from_fp16_arr(np.uint16(0x7C00))) == 0x477FE000
    print("fp32_prim_ref self-check OK (scalar + vectorized bit-identical)")
