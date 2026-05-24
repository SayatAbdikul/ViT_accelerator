# TACCEL FP32 Arithmetic Contract (normative)

This file is the single source of truth for every FP32 primitive used in the
synthesizable systolic / SFU / blocking-helper datapath under the **W8A16**
precision mode (the shipping software contract). Three implementations MUST
stay bit-identical to this spec:

1. RTL — `rtl/src/include/fp32_prim_pkg.sv` (`fp32_*_bits` functions)
2. Python twin — `software/taccel/utils/fp32_prim_ref.py`
3. C++ oracle — `rtl/verilator/test_fp32_prims.cpp`

**Sign-off invariant:** `golden_logits == rtl_logits` (exact integer equality
across the 20-image frozen benchmark). There is **no** ULP/1-LSB tolerance in
sign-off. The driver tool is `software/tools/batch_compare_rtl_golden.py`
(restored from commit `ac72b2e` and re-ported to W8A16 during Phase 6 of the
RTL fork — see `/home/user/.claude/plans/now-create-a-comprehensive-cheerful-lampson.md`).

## W8A16 datapath contract

| Buffer | Element type | Reader interpretation |
|---|---|---|
| ABUF  | FP16 (2 bytes/elem, 8 elems / 128-bit row) | systolic A-input, SFU input/output, helper VADD/SCALE_MUL endpoints |
| WBUF  | FP16 dequant weights (per-channel INT8 dequanted at compile time, then narrowed to FP16) | systolic B-input, helper bias broadcast |
| ACCUM | FP32 (4 bytes/elem, 4 elems / 128-bit row) | systolic drain, helper SCALE_MUL src, SFU mask add path |

The compiler (`software/taccel/compiler/codegen_w8a16.py`) emits **no** INT8
arithmetic opcodes — `OP_REQUANT`, `OP_REQUANT_PC`, `OP_DEQUANT_ADD`,
`OP_SOFTMAX_ATTNV` are never produced. The W8A16 RTL decode unit treats
these as `FAULT_UNSUPPORTED_OP` (the encodings remain valid ISA — this
is forward-compatible if a future INT8 datapath returns).

The **attention mask** is FP16 `-65504.0` (loaded as a normal FP16 row from
DRAM by the codegen-emitted `LOAD`, routed through `VADD` into the FP32
ACCUM). After widen to FP32 and inside the FP32-internal softmax,
`exp(FP32(-65504))` underflows to FP32 `+0.0` bit-exactly, producing the
same masking semantics as the previous W8A32 path's `-1e9` mask.

## Format

IEEE-754 binary32. `fp32_t = logic[31:0]` = `{sign[31], exp[30:23], frac[22:0]}`,
exponent bias 127. Round-to-nearest-ties-to-even (RNE) everywhere, implemented by
the existing `fp32_pack_rounded` (guard/round/sticky, `fp32_prim_pkg.sv`).
NaN is flushed to the canonical quiet NaN `0x7FC00000`.

IEEE-754 binary16 (FP16): `{sign[15], exp[14:10], frac[9:0]}`, bias 15.
Max normal = 65504. Min normal = 2^-14. Min subnormal = 2^-24.

## Systolic PE: FP16-widen → FP32 MAC (load-bearing)

The W8A16 systolic PE accepts two FP16 inputs and accumulates into a
32-bit register interpreted as FP32. Per-cycle datapath:

```
fp32_t a32  = fp32_from_fp16_bits(a_in16);
fp32_t b32  = fp32_from_fp16_bits(b_in16);
fp32_t prod = fp32_mul_bits(a32, b32);
fp32_t acc' = fp32_add_bits(acc, prod);
```

Both `fp32_mul_bits` and `fp32_add_bits` are correctly-rounded RNE binary32.
The MAC composition is **not** a fused FMA — it rounds twice (mul → RNE,
then add → RNE), matching numpy `np.float32(a)*np.float32(b)` followed by
`acc += ...`. The golden model
(`software/taccel/golden_model/systolic_w8a16.py`) replicates this exactly
with a sequential K-loop accumulator; do not switch the golden back to
`np.matmul`, which uses BLAS-dependent reduction order and will diverge
at the bit level.

## Reductions (sequential FP32 fold — load-bearing)

RTL accumulates element-by-element left-to-right (`sfu_engine.sv` exp-sum,
mean-sum, var-sum; `systolic_pe.sv` K-loop MAC). numpy `.sum()/.var()` or
`np.matmul` use pairwise/SIMD order and will NOT match. The twin MUST provide
and the golden MUST use:
- `fp32_max_seq(row[0..n))` : `m=row[0]; for i in 1..n: if row[i]>m: m=row[i]`.
- `fp32_sum_seq(row[0..n))` : `acc=+0.0; for i in 0..n: acc=fp32_add(acc,row[i])`.
- `fp32_mean(row,n)`  = `fp32_div(fp32_sum_seq(row), fp32_from_int(n))`.
- `fp32_var_seq(row,n,mean)` = `fp32_div(Σ_seq fp32_mul(d,d), n)`, `d=fp32_sub(x,mean)`.
- **MATMUL K-loop:** `dst[i,j] = ((((dst[i,j] + a[i,0]*b[0,j]) + a[i,1]*b[1,j]) + ...) + a[i,K-1]*b[K-1,j])`, where each `*` and `+` is one RNE FP32 op.

LayerNorm `eps = fp32(1e-6) = 0x358637BD`. `n` enters arithmetic as the FP32 of
the integer element count.

## Primitives

### fp32_from_fp16_bits(h16) -> fp32_t
FP16-to-FP32 widening used for every ABUF/WBUF input on the FP16 datapath
(`systolic_pe.sv`, `sfu_engine.sv`, `blocking_helper_engine.sv`).
FP16 = `{s[15], e[14:10], f[9:0]}`, bias 15.
- `e==0, f==0` → `{s, 31'd0}` (±0).
- `e==0, f!=0` (subnormal) → value `= f · 2^-24`, exact in FP32:
  `k = msb_index(f)` (0..9); result `= {s, 8'(k+103), (f - (1<<k)) << (23-k)}`.
- `e==31` (inf/NaN) → **clamp to ±65504.0** (`s ? 0xC77FE000 : 0x477FE000`).
  This reproduces the legacy `fp16_to_real` clamp and matches the
  `fp32_to_fp16_bits` round-trip behavior (FP16 inf → FP32 ±65504 → FP16
  ±65504, i.e. one ULP below ±inf). The W8A16 codegen never emits FP16
  inf or NaN on the hot path, so this clamp is benign — load-bearing only
  for the round-trip identity. Do not "fix" without revisiting the
  `fp32_to_fp16_bits` test.
- otherwise (normal) → exact widen: `{s, 8'(e + 112), {f, 13'd0}}`.

### fp32_to_fp16_bits(f32) -> logic[15:0]   *(new for W8A16)*
FP32-to-FP16 narrowing used on every FP16-bound write — SFU ABUF write-back,
helper VADD output, helper SCALE_MUL output. Matches
`numpy.float32(...).astype(np.float16)` byte-for-byte under default casting.
- FP32 NaN → canonical FP16 QNaN `{sign, 5'h1F, 10'h200}`.
- FP32 ±inf → FP16 ±inf `{sign, 5'h1F, 10'd0}`.
- FP32 zero or denormal → FP16 ±0 (FP32 denormals underflow to FP16 zero).
- Finite FP32 with unbiased exp > 15 → FP16 ±inf (overflow; matches numpy).
- Finite FP32 with unbiased exp in [-14, 15] → FP16 normal: shift the
  24-bit mantissa (implicit 1 + 23-bit frac) right by 13 with RNE; if the
  rounded value carries into the next binade, increment exponent (or
  overflow to ±inf at the top boundary).
- Finite FP32 with unbiased exp < -14 → FP16 subnormal: shift right by
  `13 + (-14 - exp_unbiased)` with RNE; if the rounded value becomes
  0x400, promote to the smallest FP16 normal `{sign, 5'd1, 10'd0}`.

Coverage gate (`rtl/verilator/test_fp32_prims.cpp`): all 65,536 FP16
patterns round-trip through `fp32_from_fp16_bits` → `fp32_to_fp16_bits`
to themselves for every finite value, and to `±0x7BFF` (FP16 max-normal)
for every FP16 ±inf and NaN — the latter is by design, because
`fp32_from_fp16_bits` clamps FP16 ±inf and NaN to FP32 ±65504. Plus 19
hand-picked boundary cases (zero/inf/NaN/overflow/underflow/smallest
subnormal/smallest normal) and 1,000,000 random stress patterns.

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

The attention-mask underflow contract relies on `exp(FP32(-65504))` returning
bit-exact FP32 `+0.0` — the range reduction produces `k = -94567` (well below
the underflow threshold), the result-exponent add saturates to underflow,
and the function returns `+0.0`. Pinned by the Phase 3 cocotb test.

### fp32_erf_bits(x) / fp32_gelu_bits(x) -> fp32_t
`erf` = Abramowitz & Stegun 7.1.26, identical structure to the corresponding
`software/taccel/utils/fp32_prim_ref.py` reference:
`s=sign(x); a=|x|; t=1/(1+P·a); y=1 − (A1·t+A2·t²+A3·t³+A4·t⁴+A5·t⁵)·exp(−a²);
erf = s·y`. Constants `ERF_A1..A5, ERF_P` baked next to the function. `exp`
uses `fp32_exp_bits`. `gelu(x) = x · 0.5 · (1 + erf(x · invsqrt2))`,
`invsqrt2 = fp32(1/√2) = 0x3F3504F3`. Evaluation order fixed; see twin.

### fp32_quantize_i8_bits(v, s) -> signed int (clamped [-128,127])
**Unused in the W8A16 datapath** (no INT8 outputs). Kept for forward
compatibility if a future RTL revision re-introduces INT8 endpoints. Spec
unchanged: `s == 0` → `0`; else `q_f = fp32_div_bits(v, s)`; integer
round-half-to-even of `q_f` using a bit-exact FP32 floor; clamp `[-128,127]`.
