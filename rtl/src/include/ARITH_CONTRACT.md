# TACCEL FP32 Arithmetic Contract (normative)

This file is the single source of truth for every FP32 primitive used in the
synthesizable SFU / blocking-helper datapath. Three implementations MUST stay
bit-identical to this spec:

1. RTL — `rtl/src/include/fp32_prim_pkg.sv` (`fp32_*_bits` functions)
2. Python twin — `software/taccel/utils/fp32_prim_ref.py`
3. C++ oracle — `rtl/verilator/test_fp32_prims.cpp`

Sign-off invariant: `golden_logits == rtl_logits` (exact integer equality,
`software/tools/batch_compare_rtl_golden.py:141`). There is **no** ULP/1-LSB
tolerance in sign-off. Do not weaken that line.

## Format

IEEE-754 binary32. `fp32_t = logic[31:0]` = `{sign[31], exp[30:23], frac[22:0]}`,
exponent bias 127. Round-to-nearest-ties-to-even (RNE) everywhere, implemented by
the existing `fp32_pack_rounded` (guard/round/sticky, `fp32_prim_pkg.sv:202`).
NaN is flushed to the canonical quiet NaN `0x7FC00000`.

## Primitives

### fp32_from_fp16_bits(h16) -> fp32_t
Replaces `fp16_to_real`+`pow2_int` (`sfu_engine.sv:211`,
`blocking_helper_engine.sv:317`). FP16 = `{s[15], e[14:10], f[9:0]}`, bias 15.
- `e==0, f==0` → `{s, 31'd0}` (±0).
- `e==0, f!=0` (subnormal) → value `= f · 2^-24`, exact in FP32:
  `k = msb_index(f)` (0..9); result `= {s, 8'(k+103), (f - (1<<k)) << (23-k)}`.
- `e==31` (inf/NaN) → **clamp to ±65504.0** (`s ? 0xC77FE000 : 0x477FE000`).
  This reproduces the legacy `fp16_to_real` clamp at `sfu_engine.sv:226-227`;
  it does NOT emit true Inf/NaN. Load-bearing — do not "fix".
- otherwise (normal) → exact widen: `{s, 8'(e + 112), {f, 13'd0}}`.

### fp32_div_bits(a, b) -> fp32_t
Correctly-rounded RNE quotient — bit-identical to `(float)((float)a/(float)b)`
and to numpy `np.float32(a)/np.float32(b)`. Implementation: 8-bit reciprocal
seed on `b`'s mantissa → 2 Newton–Raphson iterations in ≥32-bit fixed precision
→ form `q`, compute residual `r = a − q·b` in extended precision, adjust `q` by
one ULP toward the correctly-rounded result, then `fp32_pack_rounded`. Special
cases: NaN→QNaN; `x/0` (x≠0)→signed Inf; `0/0`,`Inf/Inf`→QNaN; `Inf/finite`→
signed Inf; `finite/Inf`→signed 0; sign = `a.sign ^ b.sign`.

### fp32_sqrt_bits(x) -> fp32_t
Correctly-rounded RNE — bit-identical to `sqrtf(x)` / `np.sqrt(np.float32(x))`.
`x<0`→QNaN; `x` is +0/−0→same signed zero; `+Inf`→`+Inf`; NaN→QNaN. Even/odd
exponent split, NR on `1/sqrt(m)` for `m∈[1,4)` → `q`, residual
`r = x − q·q` → ±1 ULP correction → `fp32_pack_rounded`.

### fp32_exp_bits(x) -> fp32_t
Co-defined (NOT libm-equal). Range reduction `x = k·ln2 + r`,
`k = round(x · log2e)`, `r = x − k·ln2_hi − k·ln2_lo` (Cody–Waite split,
`|r| ≤ ln2/2`). `exp(r)` by fixed Horner minimax poly of degree 6 evaluated
with `fp32_mul_bits`/`fp32_add_bits` in this exact order. `exp(x) = 2^k · P(r)`
by adding `k` to the result exponent field (overflow→`+Inf`, underflow→`+0`).
Coefficients + `log2e`, `ln2_hi`, `ln2_lo` as FP32 bit patterns are listed in
`fp32_prim_pkg.sv` next to the function and mirrored verbatim in the twin.

### fp32_erf_bits(x) / fp32_gelu_bits(x) -> fp32_t
`erf` = Abramowitz & Stegun 7.1.26, identical structure to
`software/taccel/golden_model/sfu.py:_erf_poly` (lines 38-61):
`s=sign(x); a=|x|; t=1/(1+P·a); y=1 − (A1·t+A2·t²+A3·t³+A4·t⁴+A5·t⁵)·exp(−a²);
erf = s·y`. Constants `ERF_A1..A5, ERF_P` from `taccel_pkg.sv:191-196`. `exp`
uses `fp32_exp_bits`. `gelu(x) = x · 0.5 · (1 + erf(x · invsqrt2))`,
`invsqrt2 = fp32(1/√2) = 0x3F3504F3`. Evaluation order fixed; see twin.

### fp32_quantize_i8_bits(v, s) -> signed int (clamped [-128,127])
- `s == 0` (`fp32_is_zero`) → `0`.
- else `q_f = fp32_div_bits(v, s)`; integer round-half-to-even of `q_f` using a
  **bit-exact FP32 floor** (decode exponent, mask fractional mantissa bits — NOT
  `$floor` on `real`): `n = floor(q_f); frac = q_f − n;
  q = (frac>0.5)?n+1 : (frac<0.5)?n : (n&1)?n+1:n`; clamp `[-128,127]`.
  Matches `testbench.h:36-46` and `np.clip(np.round(np.float32(v)/np.float32(s)),-128,127)`.

## Reductions (sequential FP32 fold — load-bearing)

RTL accumulates element-by-element left-to-right (`sfu_engine.sv:677-683`
exp-sum, `:696-700` mean-sum, `:703-710` var-sum). numpy `.sum()/.var()` use
pairwise/SIMD order and will NOT match. The twin MUST provide and the golden
MUST use:
- `fp32_max_seq(row[0..n))` : `m=row[0]; for i in 1..n: if row[i]>m: m=row[i]`.
- `fp32_sum_seq(row[0..n))` : `acc=+0.0; for i in 0..n: acc=fp32_add(acc,row[i])`.
- `fp32_mean(row,n)`  = `fp32_div(fp32_sum_seq(row), fp32_from_int(n))`.
- `fp32_var_seq(row,n,mean)` = `fp32_div(Σ_seq fp32_mul(d,d), n)`, `d=fp32_sub(x,mean)`.

LayerNorm `eps = fp32(1e-6) = 0x358637BD`. `n` enters arithmetic as the FP32 of
the integer element count.
