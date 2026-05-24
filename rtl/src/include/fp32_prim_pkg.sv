// Synthesizable FP32 primitives for the TACCEL SFU and blocking-helper
// engines. Every operation here is implemented with integer arithmetic on
// IEEE-754 binary32 bit patterns — no SV `real`, no DPI-C, no vendor FP IP
// — so the entire RTL/src tree synthesises onto FPGA. The grep gate at
// `rtl/verilator/Makefile`'s SYNTH_BUILD target enforces this.
//
// The normative spec for every algorithm, constant, and intermediate
// rounding point is `rtl/src/include/ARITH_CONTRACT.md`. The Python twin at
// `software/taccel/utils/fp32_prim_ref.py` mirrors this package bit-for-bit
// so the golden model is RTL-equivalent by construction (exact integer
// logits — see software/tools/batch_compare_rtl_golden.py:141).
//
// File layout — two sections separated by the TRANSCENDENTAL banner below:
//   1. BASIC ARITH (Phase 0–2, stable):   pack/decode helpers, predicates,
//        fp32_round/add/sub/mul_bits, integer ↔ FP32 widening, FP16 → FP32.
//   2. TRANSCENDENTAL & QUANTIZE (Phase 3–4, may grow for Phase 5
//        pipelining): fp32_div_bits, sqrt, exp, erf, gelu, quantize_i8,
//        and their constants (FP32_ONE, FP32_EC*, FP32_ERF_*).

`ifndef FP32_PRIM_PKG_SV
`define FP32_PRIM_PKG_SV

package fp32_prim_pkg;

  // =====================================================================
  // SECTION 1: BASIC ARITH (Phase 0–2, stable)
  //   Types, predicates, IEEE-754 pack/decode helpers, and the four basic
  //   ops (round/add/sub/mul). Everything here is correctly rounded.
  // =====================================================================

  typedef logic [31:0] fp32_t;
  typedef logic [63:0] fp64_t;
  typedef longint unsigned u64_t;
  typedef longint signed   s64_t;

  localparam fp32_t FP32_QNAN_BITS = 32'h7fc0_0000;
  localparam fp32_t FP32_POS_INF   = 32'h7f80_0000;
  localparam fp32_t FP32_NEG_INF   = 32'hff80_0000;

  function automatic bit fp32_is_nan(input fp32_t bits);
    fp32_is_nan = (bits[30:23] == 8'hff) && (bits[22:0] != 23'd0);
  endfunction

  function automatic bit fp32_is_inf(input fp32_t bits);
    fp32_is_inf = (bits[30:23] == 8'hff) && (bits[22:0] == 23'd0);
  endfunction

  function automatic bit fp32_is_zero(input fp32_t bits);
    fp32_is_zero = (bits[30:0] == 31'd0);
  endfunction

  function automatic longint unsigned fp32_round_shift_right(
      input longint unsigned value,
      input int shift
  );
    longint unsigned quotient;
    longint unsigned remainder;
    longint unsigned half;
    longint unsigned mask;
    begin
      if (shift <= 0) begin
        fp32_round_shift_right = value << (-shift);
      end else if (shift >= 63) begin
        quotient = 0;
        remainder = value;
        half = 64'h8000_0000_0000_0000;
        fp32_round_shift_right =
            (remainder > half || (remainder == half && quotient[0])) ? 64'd1 : 64'd0;
      end else begin
        quotient = value >> shift;
        mask = (64'd1 << shift) - 64'd1;
        remainder = value & mask;
        half = 64'd1 << (shift - 1);
        if (remainder > half || (remainder == half && quotient[0])) begin
          fp32_round_shift_right = quotient + 64'd1;
        end else begin
          fp32_round_shift_right = quotient;
        end
      end
    end
  endfunction

  function automatic longint unsigned fp32_shift_right_sticky(
      input longint unsigned value,
      input int shift
  );
    longint unsigned shifted;
    longint unsigned mask;
    begin
      if (shift <= 0) begin
        fp32_shift_right_sticky = value << (-shift);
      end else if (shift >= 64) begin
        fp32_shift_right_sticky = (value != 64'd0) ? 64'd1 : 64'd0;
      end else begin
        shifted = value >> shift;
        mask = (64'd1 << shift) - 64'd1;
        fp32_shift_right_sticky = shifted | (((value & mask) != 64'd0) ? 64'd1 : 64'd0);
      end
    end
  endfunction

  function automatic int fp32_msb_index(input longint unsigned value);
    int idx;
    begin
      fp32_msb_index = -1;
      for (idx = 63; idx >= 0; idx--) begin
        if (value[idx] && fp32_msb_index < 0) begin
          fp32_msb_index = idx;
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_pack_rounded(
      input bit sign_b,
      input int exp_unbiased,
      input longint unsigned ext_sig
  );
    longint unsigned mant24_u;
    logic [22:0] frac_bits;
    logic [7:0] exp_bits;
    bit guard_b;
    bit round_b;
    bit sticky_b;
    begin
      if (ext_sig == 64'd0) begin
        fp32_pack_rounded = {sign_b, 31'd0};
      end else begin
        while ((exp_unbiased > -126) && (ext_sig < 64'h0000_0000_0400_0000)) begin
          ext_sig = ext_sig << 1;
          exp_unbiased--;
        end

        if (exp_unbiased < -126) begin
          ext_sig = fp32_shift_right_sticky(ext_sig, -126 - exp_unbiased);
          exp_unbiased = -126;
        end

        guard_b = ext_sig[2];
        round_b = ext_sig[1];
        sticky_b = ext_sig[0];
        mant24_u = ext_sig >> 3;
        if (guard_b && (round_b || sticky_b || mant24_u[0])) begin
          mant24_u = mant24_u + 64'd1;
        end

        if (mant24_u == 64'd0) begin
          fp32_pack_rounded = {sign_b, 31'd0};
        end else begin
          if (mant24_u >= 64'h0000_0000_0100_0000) begin
            mant24_u = fp32_shift_right_sticky(mant24_u, 1);
            exp_unbiased++;
          end

          if (exp_unbiased > 127) begin
            fp32_pack_rounded = sign_b ? FP32_NEG_INF : FP32_POS_INF;
          end else if (exp_unbiased <= -126) begin
            if (mant24_u >= 64'h0000_0000_0080_0000) begin
              exp_bits = 8'd1;
              frac_bits = mant24_u[22:0];
              fp32_pack_rounded = {sign_b, exp_bits, frac_bits};
            end else begin
              frac_bits = mant24_u[22:0];
              fp32_pack_rounded = {sign_b, 8'd0, frac_bits};
            end
          end else begin
            exp_bits = 8'(exp_unbiased + 127);
            frac_bits = mant24_u[22:0];
            fp32_pack_rounded = {sign_b, exp_bits, frac_bits};
          end
        end
      end
    end
  endfunction

  function automatic void fp32_decode_finite(
      input fp32_t bits,
      output bit sign_b,
      output int exp_unbiased,
      output longint unsigned mant24
  );
    begin
      sign_b = bits[31];
      if (bits[30:23] == 8'd0) begin
        exp_unbiased = -126;
        mant24 = {41'd0, bits[22:0]};
      end else begin
        exp_unbiased = int'(bits[30:23]) - 127;
        mant24 = {40'd0, 1'b1, bits[22:0]};
      end
    end
  endfunction

  function automatic fp32_t fp32_addsub_bits(
      input fp32_t lhs_bits,
      input fp32_t rhs_bits,
      input bit subtract_b
  );
    fp32_t rhs_eff_bits;
    bit sign_a;
    bit sign_b;
    int exp_a;
    int exp_b;
    int exp_r;
    longint unsigned mant_a;
    longint unsigned mant_b;
    longint unsigned ext_a;
    longint unsigned ext_b;
    longint unsigned mag_r;
    bit sign_r;
    begin
      rhs_eff_bits = rhs_bits ^ {subtract_b, 31'd0};

      if (fp32_is_nan(lhs_bits) || fp32_is_nan(rhs_eff_bits)) begin
        fp32_addsub_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits) && fp32_is_inf(rhs_eff_bits)) begin
        fp32_addsub_bits = (lhs_bits[31] == rhs_eff_bits[31])
            ? {lhs_bits[31], 8'hff, 23'd0}
            : FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits)) begin
        fp32_addsub_bits = {lhs_bits[31], 8'hff, 23'd0};
      end else if (fp32_is_inf(rhs_eff_bits)) begin
        fp32_addsub_bits = {rhs_eff_bits[31], 8'hff, 23'd0};
      end else if (fp32_is_zero(lhs_bits) && fp32_is_zero(rhs_eff_bits)) begin
        fp32_addsub_bits = {
          (lhs_bits[31] && rhs_eff_bits[31]),
          31'd0
        };
      end else begin
        fp32_decode_finite(lhs_bits, sign_a, exp_a, mant_a);
        fp32_decode_finite(rhs_eff_bits, sign_b, exp_b, mant_b);
        ext_a = mant_a << 3;
        ext_b = mant_b << 3;

        if (exp_a > exp_b) begin
          ext_b = fp32_shift_right_sticky(ext_b, exp_a - exp_b);
          exp_r = exp_a;
        end else if (exp_b > exp_a) begin
          ext_a = fp32_shift_right_sticky(ext_a, exp_b - exp_a);
          exp_r = exp_b;
        end else begin
          exp_r = exp_a;
        end

        if (sign_a == sign_b) begin
          mag_r = ext_a + ext_b;
          sign_r = sign_a;
          if (mag_r >= 64'h0000_0000_0800_0000) begin
            mag_r = fp32_shift_right_sticky(mag_r, 1);
            exp_r++;
          end
        end else if (ext_a > ext_b) begin
          mag_r = ext_a - ext_b;
          sign_r = sign_a;
        end else if (ext_b > ext_a) begin
          mag_r = ext_b - ext_a;
          sign_r = sign_b;
        end else begin
          mag_r = 64'd0;
          sign_r = 1'b0;
        end

        fp32_addsub_bits = fp32_pack_rounded(sign_r, exp_r, mag_r);
      end
    end
  endfunction

  function automatic fp32_t fp32_round_bits(input fp32_t value_bits);
    if (fp32_is_nan(value_bits)) begin
      fp32_round_bits = FP32_QNAN_BITS;
    end else begin
      fp32_round_bits = value_bits;
    end
  endfunction

  function automatic fp32_t fp32_add_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    fp32_add_bits = fp32_addsub_bits(lhs_bits, rhs_bits, 1'b0);
  endfunction

  function automatic fp32_t fp32_sub_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    fp32_sub_bits = fp32_addsub_bits(lhs_bits, rhs_bits, 1'b1);
  endfunction

  function automatic fp32_t fp32_mul_bits(input fp32_t lhs_bits, input fp32_t rhs_bits);
    bit sign_a;
    bit sign_b;
    bit sign_r;
    int exp_a;
    int exp_b;
    int exp_r;
    int msb_idx;
    longint unsigned mant_a;
    longint unsigned mant_b;
    longint unsigned product;
    longint unsigned ext_sig;
    begin
      sign_r = lhs_bits[31] ^ rhs_bits[31];

      if (fp32_is_nan(lhs_bits) || fp32_is_nan(rhs_bits)) begin
        fp32_mul_bits = FP32_QNAN_BITS;
      end else if ((fp32_is_inf(lhs_bits) && fp32_is_zero(rhs_bits))
          || (fp32_is_zero(lhs_bits) && fp32_is_inf(rhs_bits))) begin
        fp32_mul_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(lhs_bits) || fp32_is_inf(rhs_bits)) begin
        fp32_mul_bits = sign_r ? FP32_NEG_INF : FP32_POS_INF;
      end else if (fp32_is_zero(lhs_bits) || fp32_is_zero(rhs_bits)) begin
        fp32_mul_bits = {sign_r, 31'd0};
      end else begin
        fp32_decode_finite(lhs_bits, sign_a, exp_a, mant_a);
        fp32_decode_finite(rhs_bits, sign_b, exp_b, mant_b);
        sign_r = sign_a ^ sign_b;
        product = mant_a * mant_b;
        msb_idx = fp32_msb_index(product);
        exp_r = exp_a + exp_b - 46 + msb_idx;
        if (msb_idx > 26) begin
          ext_sig = fp32_shift_right_sticky(product, msb_idx - 26);
        end else begin
          ext_sig = product << (26 - msb_idx);
        end
        fp32_mul_bits = fp32_pack_rounded(sign_r, exp_r, ext_sig);
      end
    end
  endfunction

  // =====================================================================
  // SECTION 2: TRANSCENDENTAL & QUANTIZE (Phase 3–4, may grow for Phase 5)
  //   fp32_div_bits, fp32_sqrt_bits, fp32_exp_bits, fp32_erf_bits,
  //   fp32_gelu_bits, fp32_quantize_i8_bits, plus their constants. These
  //   are NOT correctly rounded — they're polynomial / Newton-Raphson
  //   approximations co-defined with software/taccel/utils/fp32_prim_ref.py
  //   so RTL ≡ golden by construction. Spec:
  //   rtl/src/include/ARITH_CONTRACT.md. The 208-element combinational
  //   loops that call these will be serialised in Phase 5; algorithms here
  //   may be re-shaped (Horner ↔ tree, etc.) for timing closure.
  // =====================================================================

  localparam fp32_t FP32_ONE     = 32'h3F800000;
  localparam fp32_t FP32_HALF    = 32'h3F000000;
  localparam fp32_t FP32_NEG_ONE = 32'hBF800000;
  localparam fp32_t FP32_LOG2E   = 32'h3FB8AA3B;
  localparam fp32_t FP32_LN2_HI  = 32'h3F317200;
  localparam fp32_t FP32_LN2_LO  = 32'h35BFBE8E;
  localparam fp32_t FP32_INVSQRT2= 32'h3F3504F3;
  // exp(r) Horner coefficients high->low : 1/5040,1/720,1/120,1/24,1/6,1/2,1,1
  localparam fp32_t FP32_EC0 = 32'h39500D01;
  localparam fp32_t FP32_EC1 = 32'h3AB60B61;
  localparam fp32_t FP32_EC2 = 32'h3C088889;
  localparam fp32_t FP32_EC3 = 32'h3D2AAAAB;
  localparam fp32_t FP32_EC4 = 32'h3E2AAAAB;
  localparam fp32_t FP32_EC5 = 32'h3F000000;
  localparam fp32_t FP32_EC6 = 32'h3F800000;
  localparam fp32_t FP32_EC7 = 32'h3F800000;
  // erf (Abramowitz & Stegun 7.1.26) — same bits as taccel_pkg ERF_*
  localparam fp32_t FP32_ERF_A1 = 32'h3E827906;
  localparam fp32_t FP32_ERF_A2 = 32'hBE91A98E;
  localparam fp32_t FP32_ERF_A3 = 32'h3FB5D78E;
  localparam fp32_t FP32_ERF_A4 = 32'hBFBA0005;
  localparam fp32_t FP32_ERF_A5 = 32'h3F87DC22;
  localparam fp32_t FP32_ERF_P  = 32'h3EA7B9D2;

  // floor(sqrt(x)) for a 64-bit unsigned operand (bitwise digit recurrence).
  function automatic longint unsigned fp32_isqrt64(input longint unsigned x);
    longint unsigned res;
    longint unsigned bit_;
    begin
      res  = 64'd0;
      bit_ = 64'h4000_0000_0000_0000; // 1 << 62
      while (bit_ > x) bit_ = bit_ >> 2;
      while (bit_ != 64'd0) begin
        if (x >= res + bit_) begin
          x   = x - (res + bit_);
          res = (res >> 1) + bit_;
        end else begin
          res = res >> 1;
        end
        bit_ = bit_ >> 2;
      end
      fp32_isqrt64 = res;
    end
  endfunction

  // Round-and-pack an arbitrary positive fixed-point value M * 2^s into fp32
  // with RNE. extra_sticky folds any below-LSB residue (e.g. a division
  // remainder) into the sticky bit. Caller guarantees M > 0.
  function automatic fp32_t fp32_pack_from_fixed(
      input bit sign_b,
      input longint unsigned m_in,
      input int s_in,
      input bit extra_sticky
  );
    int msb;
    longint unsigned mn;
    begin
      msb = fp32_msb_index(m_in);
      if (msb > 26) begin
        mn = fp32_shift_right_sticky(m_in, msb - 26);
      end else begin
        mn = m_in << (26 - msb);
      end
      if (extra_sticky) mn = mn | 64'd1;
      fp32_pack_from_fixed = fp32_pack_rounded(sign_b, s_in + msb, mn);
    end
  endfunction

  // Signed integer -> fp32 (RNE for magnitudes that exceed 24 bits).
  function automatic fp32_t fp32_from_int(input longint signed v);
    longint unsigned mag;
    begin
      if (v == 0) begin
        fp32_from_int = 32'd0;
      end else begin
        mag = v[63] ? u64_t'(-v) : u64_t'(v);
        fp32_from_int = fp32_pack_from_fixed(v[63], mag, 0, 1'b0);
      end
    end
  endfunction

  // Signed 8/32-bit integer -> fp32 (RNE for magnitudes > 24 bits, exact below).
  function automatic fp32_t fp32_from_i8(input logic signed [7:0] v);
    fp32_from_i8 = fp32_from_int(s64_t'(v));
  endfunction

  function automatic fp32_t fp32_from_i32(input logic signed [31:0] v);
    fp32_from_i32 = fp32_from_int(s64_t'(v));
  endfunction

  // Ordered FP32 greater-than for finite operands (no-NaN paths in the SFU).
  function automatic bit fp32_gt(input fp32_t a, input fp32_t b);
    logic        sa, sb;
    logic [30:0] aa, bb;
    begin
      sa = a[31]; sb = b[31];
      aa = a[30:0]; bb = b[30:0];
      if ((aa == 31'd0) && (bb == 31'd0)) fp32_gt = 1'b0;            // ±0 == ±0
      else if (sa != sb)                  fp32_gt = !sa;             // +x > -y
      else if (!sa)                       fp32_gt = (aa > bb);       // both +
      else                                fp32_gt = (aa < bb);       // both -
    end
  endfunction

  function automatic fp32_t fp32_from_fp16_bits(input logic [15:0] h);
    bit         s;
    logic [4:0] e;
    logic [9:0] f;
    int         k;
    u64_t       fr;
    begin
      s = h[15];
      e = h[14:10];
      f = h[9:0];
      if ((e == 5'd0) && (f == 10'd0)) begin
        fp32_from_fp16_bits = {s, 31'd0};
      end else if (e == 5'd0) begin
        k  = fp32_msb_index({54'd0, f});
        fr = (u64_t'(f) - (64'd1 << k)) << (23 - k);
        fp32_from_fp16_bits = {s, 8'(k + 103), fr[22:0]};
      end else if (e == 5'h1F) begin
        fp32_from_fp16_bits = s ? 32'hC77FE000 : 32'h477FE000;
      end else begin
        fp32_from_fp16_bits = {s, 8'(int'(e) + 112), f, 13'd0};
      end
    end
  endfunction

  // FP32 → FP16 narrowing with IEEE-754 round-to-nearest-even (RNE).
  //   - Finite values outside the FP16 normal range overflow to ±inf
  //     (matches numpy.float32(...).astype(np.float16) under default
  //     'unsafe' casting; W8A16 codegen pre-clamps the attention mask
  //     to -65504 before writing FP16, so overflow does not arise on
  //     the inference hot path).
  //   - FP32 NaN → canonical FP16 QNaN ({sign, 5'h1F, 10'h200}).
  //   - FP32 denormals underflow to FP16 ±0.
  //   - FP16 subnormals are produced via the same RNE shift; rounding
  //     up into the smallest FP16 normal is handled explicitly.
  function automatic logic [15:0] fp32_to_fp16_bits(input fp32_t f);
    bit          s;
    logic [7:0]  e8;
    logic [22:0] f23;
    int          e_unb;
    u64_t        mant24;
    u64_t        rounded;
    int          shift_amt;
    logic [9:0]  frac10;
    begin
      s   = f[31];
      e8  = f[30:23];
      f23 = f[22:0];

      if (e8 == 8'hFF && f23 != 23'd0) begin
        fp32_to_fp16_bits = {s, 5'h1F, 10'h200};         // NaN
      end else if (e8 == 8'hFF) begin
        fp32_to_fp16_bits = {s, 5'h1F, 10'd0};           // ±inf
      end else if (e8 == 8'd0) begin
        fp32_to_fp16_bits = {s, 15'd0};                   // FP32 zero/denormal → ±0
      end else begin
        e_unb  = int'(e8) - 127;
        mant24 = {40'd0, 1'b1, f23};                      // implicit 1 + 23-bit frac, zero-extended to 64 b
        if (e_unb > 15) begin
          fp32_to_fp16_bits = {s, 5'h1F, 10'd0};         // overflow → ±inf
        end else if (e_unb >= -14) begin
          // Normal FP16: shift 24-bit mantissa right by 13 with RNE so
          // 11 significant bits remain (implicit 1 + 10 frac).
          rounded = fp32_round_shift_right(mant24, 13);
          if (rounded == 64'h800) begin
            // Rounded up into the next binade.
            if (e_unb + 1 > 15) begin
              fp32_to_fp16_bits = {s, 5'h1F, 10'd0};
            end else begin
              fp32_to_fp16_bits = {s, 5'(e_unb + 1 + 15), 10'd0};
            end
          end else begin
            frac10 = rounded[9:0];
            fp32_to_fp16_bits = {s, 5'(e_unb + 15), frac10};
          end
        end else begin
          // FP16 subnormal: e16=0, frac = mant24 shifted right enough
          // that the implicit 1 is absorbed into the fraction. Shift
          // amount is 13 (normal alignment) plus (-14 - e_unb) extra
          // bits to make up for the missing exponent room.
          shift_amt = 13 + (-14 - e_unb);
          rounded   = fp32_round_shift_right(mant24, shift_amt);
          if (rounded == 64'h400) begin
            // Subnormal rounded up into the smallest FP16 normal.
            fp32_to_fp16_bits = {s, 5'd1, 10'd0};
          end else begin
            frac10 = rounded[9:0];
            fp32_to_fp16_bits = {s, 5'd0, frac10};
          end
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_div_bits(input fp32_t a, input fp32_t b);
    bit sign_r;
    bit sa, sb;
    int ea, eb;
    longint unsigned ma, mb, p, q, r;
    begin
      sign_r = a[31] ^ b[31];
      if (fp32_is_nan(a) || fp32_is_nan(b)) begin
        fp32_div_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(a) && fp32_is_inf(b)) begin
        fp32_div_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(a)) begin
        fp32_div_bits = sign_r ? FP32_NEG_INF : FP32_POS_INF;
      end else if (fp32_is_inf(b)) begin
        fp32_div_bits = {sign_r, 31'd0};
      end else if (fp32_is_zero(b)) begin
        fp32_div_bits = fp32_is_zero(a)
            ? FP32_QNAN_BITS
            : (sign_r ? FP32_NEG_INF : FP32_POS_INF);
      end else if (fp32_is_zero(a)) begin
        fp32_div_bits = {sign_r, 31'd0};
      end else begin
        fp32_decode_finite(a, sa, ea, ma);
        fp32_decode_finite(b, sb, eb, mb);
        p = ma << 28;
        q = p / mb;
        r = p % mb;
        fp32_div_bits = fp32_pack_from_fixed(sign_r, q, ea - eb - 28, (r != 64'd0));
      end
    end
  endfunction

  function automatic fp32_t fp32_sqrt_bits(input fp32_t x);
    bit sx;
    int ex, e2, sh;
    longint unsigned mx, t, tsc, ssr, rs;
    begin
      if (fp32_is_nan(x)) begin
        fp32_sqrt_bits = FP32_QNAN_BITS;
      end else if (fp32_is_zero(x)) begin
        fp32_sqrt_bits = {x[31], 31'd0};
      end else if (x[31]) begin
        fp32_sqrt_bits = FP32_QNAN_BITS;            // sqrt of negative
      end else if (fp32_is_inf(x)) begin
        fp32_sqrt_bits = FP32_POS_INF;
      end else begin
        fp32_decode_finite(x, sx, ex, mx);
        e2 = ex - 23;
        if ((e2 & 1) == 1) begin
          t  = mx << 1;
          sh = (e2 - 1) / 2;
        end else begin
          t  = mx;
          sh = e2 / 2;
        end
        tsc = t << 38;                              // sqrt scales by 2^19
        ssr = fp32_isqrt64(tsc);
        rs  = tsc - (ssr * ssr);
        fp32_sqrt_bits =
            fp32_pack_from_fixed(1'b0, ssr, sh - 19, (rs != 64'd0));
      end
    end
  endfunction

  // Round fp32 to nearest integer, ties to even. Returns a saturating
  // sentinel (|.| = 200) when the magnitude is already outside INT8 range.
  function automatic longint signed fp32_rint_i64(input fp32_t a);
    bit sx;
    int ex, nfrac;
    longint unsigned mx, ip, half, rem, mask;
    longint signed res;
    begin
      if (fp32_is_zero(a)) begin
        fp32_rint_i64 = 64'sd0;
      end else if (fp32_is_nan(a)) begin
        fp32_rint_i64 = 64'sd0;
      end else if (fp32_is_inf(a)) begin
        res = s64_t'(64'd1 << 62);                   // overflow sentinel
        fp32_rint_i64 = a[31] ? -res : res;
      end else begin
        fp32_decode_finite(a, sx, ex, mx);
        if (ex >= 62) begin
          res = s64_t'(64'd1 << 62);                 // > INT range -> sentinel
          fp32_rint_i64 = sx ? -res : res;
        end else if (ex >= 23) begin
          res = s64_t'(mx << (ex - 23));             // exact integer, no frac
          fp32_rint_i64 = sx ? -res : res;
        end else if (ex < -1) begin
          fp32_rint_i64 = 64'sd0;                    // |a| < 0.5
        end else begin
          nfrac = 23 - ex;                           // ex in [-1..22] -> [1..24]
          ip   = mx >> nfrac;
          half = 64'd1 << (nfrac - 1);
          mask = (64'd1 << nfrac) - 64'd1;
          rem  = mx & mask;
          if (rem > half) ip = ip + 64'd1;
          else if (rem == half) ip = ip + (ip & 64'd1);
          res = s64_t'(ip);
          fp32_rint_i64 = sx ? -res : res;
        end
      end
    end
  endfunction

  function automatic longint signed fp32_quantize_i8_bits(
      input fp32_t v, input fp32_t s
  );
    fp32_t qf;
    longint signed q;
    begin
      if (fp32_is_zero(s)) begin
        fp32_quantize_i8_bits = 64'sd0;
      end else begin
        qf = fp32_div_bits(v, s);
        if (fp32_is_nan(qf)) begin
          fp32_quantize_i8_bits = 64'sd0;
        end else if (fp32_is_inf(qf)) begin
          fp32_quantize_i8_bits = qf[31] ? -64'sd128 : 64'sd127;
        end else begin
          q = fp32_rint_i64(qf);
          if (q < -64'sd128) q = -64'sd128;
          else if (q > 64'sd127) q = 64'sd127;
          fp32_quantize_i8_bits = q;
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_exp_bits(input fp32_t x);
    fp32_t m, kf, r, p;
    bit sp;
    int ep, kk;
    longint unsigned mp;
    longint signed k64;
    begin
      if (fp32_is_nan(x)) begin
        fp32_exp_bits = FP32_QNAN_BITS;
      end else if (fp32_is_inf(x)) begin
        fp32_exp_bits = x[31] ? 32'd0 : FP32_POS_INF;
      end else begin
        m = fp32_mul_bits(x, FP32_LOG2E);
        if (fp32_is_inf(m) || fp32_is_nan(m)) begin
          fp32_exp_bits = (fp32_is_nan(m) || m[31]) ?
              (fp32_is_nan(m) ? FP32_QNAN_BITS : 32'd0) : FP32_POS_INF;
        end else begin
          k64 = fp32_rint_i64(m);
          if (k64 > 64'sd300)  fp32_exp_bits = FP32_POS_INF;
          else if (k64 < -64'sd160) fp32_exp_bits = 32'd0;
          else begin
            kk  = int'(k64);
            kf  = fp32_from_int(k64);
            r   = fp32_sub_bits(
                      fp32_sub_bits(x, fp32_mul_bits(kf, FP32_LN2_HI)),
                      fp32_mul_bits(kf, FP32_LN2_LO));
            p = FP32_EC0;
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC1);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC2);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC3);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC4);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC5);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC6);
            p = fp32_add_bits(fp32_mul_bits(p, r), FP32_EC7);
            fp32_decode_finite(p, sp, ep, mp);
            fp32_exp_bits = fp32_pack_from_fixed(1'b0, mp, (ep - 23) + kk, 1'b0);
          end
        end
      end
    end
  endfunction

  function automatic fp32_t fp32_erf_bits(input fp32_t x);
    bit    neg;
    fp32_t a, t, t2, t3, t4, t5, poly, aa, e, y;
    begin
      if (fp32_is_zero(x)) begin
        fp32_erf_bits = 32'd0;
      end else begin
        neg = x[31];
        a   = {1'b0, x[30:0]};
        t   = fp32_div_bits(FP32_ONE,
                  fp32_add_bits(FP32_ONE, fp32_mul_bits(FP32_ERF_P, a)));
        t2  = fp32_mul_bits(t, t);
        t3  = fp32_mul_bits(t2, t);
        t4  = fp32_mul_bits(t3, t);
        t5  = fp32_mul_bits(t4, t);
        poly = fp32_add_bits(
                 fp32_add_bits(
                   fp32_add_bits(
                     fp32_add_bits(fp32_mul_bits(FP32_ERF_A1, t),
                                   fp32_mul_bits(FP32_ERF_A2, t2)),
                     fp32_mul_bits(FP32_ERF_A3, t3)),
                   fp32_mul_bits(FP32_ERF_A4, t4)),
                 fp32_mul_bits(FP32_ERF_A5, t5));
        aa  = fp32_mul_bits(a, a);
        e   = fp32_exp_bits({~aa[31], aa[30:0]});
        y   = fp32_sub_bits(FP32_ONE, fp32_mul_bits(poly, e));
        fp32_erf_bits = neg ? fp32_mul_bits(FP32_NEG_ONE, y) : y;
      end
    end
  endfunction

  function automatic fp32_t fp32_gelu_bits(input fp32_t x);
    fp32_t e;
    begin
      e = fp32_erf_bits(fp32_mul_bits(x, FP32_INVSQRT2));
      fp32_gelu_bits =
          fp32_mul_bits(fp32_mul_bits(x, FP32_HALF),
                        fp32_add_bits(FP32_ONE, e));
    end
  endfunction

endpackage

`endif
