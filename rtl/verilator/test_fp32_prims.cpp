// FP32 primitive unit bench. Verifies the synthesizable fp32_*_bits package
// (rtl/src/include/fp32_prim_pkg.sv) against self-contained oracles:
//   * round/add/sub/mul/div/sqrt/quantize/from_fp16 : correctly-rounded or
//     exact-spec  ->  C `float` / integer oracle, bit-exact.
//   * exp/erf/gelu : co-defined polynomial (NOT libm) -> identical-constant
//     C `float` poly. Each elementary op is correctly-rounded binary32 in
//     both the SV package and here, so results are bit-identical by
//     construction. Spec: rtl/src/include/ARITH_CONTRACT.md.
#pragma STDC FP_CONTRACT OFF

#include "Vtb_fp32_prim.h"
#include "verilated.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

enum Op : uint8_t {
    OP_ROUND = 0, OP_ADD = 1, OP_SUB = 2, OP_MUL = 3,
    OP_DIV = 4, OP_SQRT = 5, OP_EXP = 6, OP_ERF = 7,
    OP_GELU = 8, OP_FROM_FP16 = 9, OP_QUANT = 10,
};

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
float bits_float(uint32_t bits) {
    float value = 0.0f;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
bool is_nan_bits(uint32_t bits) {
    return ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
}
bool bits_match(uint32_t got, uint32_t expected) {
    if (is_nan_bits(expected)) return is_nan_bits(got);
    return got == expected;
}

// frozen FP32 constants (identical bits to fp32_prim_ref.py / the SV package)
const float LOG2E   = bits_float(0x3FB8AA3Bu);
const float LN2_HI  = bits_float(0x3F317200u);
const float LN2_LO  = bits_float(0x35BFBE8Eu);
const float INVSQR2 = bits_float(0x3F3504F3u);
const float EC[8] = {
    bits_float(0x39500D01u), bits_float(0x3AB60B61u), bits_float(0x3C088889u),
    bits_float(0x3D2AAAABu), bits_float(0x3E2AAAABu), bits_float(0x3F000000u),
    bits_float(0x3F800000u), bits_float(0x3F800000u)};
const float A1 = bits_float(0x3E827906u), A2 = bits_float(0xBE91A98Eu);
const float A3 = bits_float(0x3FB5D78Eu), A4 = bits_float(0xBFBA0005u);
const float A5 = bits_float(0x3F87DC22u), ERP = bits_float(0x3EA7B9D2u);
const float F_ONE = 1.0f, F_HALF = 0.5f;

long rhe_f(float x) {  // round-half-to-even of a binary32 to integer
    float fl = std::floor(x);
    float frac = x - fl;
    long n = (long)fl;
    if (frac > 0.5f) return n + 1;
    if (frac < 0.5f) return n;
    return (n & 1L) ? n + 1 : n;
}

float exp_core(float x) {
    if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
    if (std::isinf(x)) return x < 0 ? 0.0f : std::numeric_limits<float>::infinity();
    float m = x * LOG2E;
    if (std::isinf(m) || std::isnan(m))
        return std::isnan(m) ? std::numeric_limits<float>::quiet_NaN()
                             : (m < 0 ? 0.0f : std::numeric_limits<float>::infinity());
    long k = rhe_f(m);
    if (k > 300) return std::numeric_limits<float>::infinity();
    if (k < -160) return 0.0f;
    float kf = (float)k;
    float r = (x - kf * LN2_HI) - kf * LN2_LO;
    float p = EC[0];
    for (int i = 1; i < 8; ++i) p = p * r + EC[i];
    return std::ldexp(p, (int)k);
}
float erf_core(float x) {
    if (x == 0.0f) return 0.0f;
    float s = x < 0 ? -1.0f : 1.0f;
    float a = std::fabs(x);
    float t = F_ONE / (F_ONE + ERP * a);
    float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    float poly = ((((A1 * t) + (A2 * t2)) + (A3 * t3)) + (A4 * t4)) + (A5 * t5);
    float e = exp_core(-(a * a));
    float y = F_ONE - (poly * e);
    return s * y;
}
float gelu_core(float x) {
    float e = erf_core(x * INVSQR2);
    return (x * F_HALF) * (F_ONE + e);
}

uint32_t round_ref(uint32_t a) { return is_nan_bits(a) ? 0x7fc00000u : a; }
uint32_t add_ref(uint32_t a, uint32_t b) { return float_bits(bits_float(a) + bits_float(b)); }
uint32_t sub_ref(uint32_t a, uint32_t b) { return float_bits(bits_float(a) - bits_float(b)); }
uint32_t mul_ref(uint32_t a, uint32_t b) { return float_bits(bits_float(a) * bits_float(b)); }
uint32_t div_ref(uint32_t a, uint32_t b) { return float_bits(bits_float(a) / bits_float(b)); }
uint32_t sqrt_ref(uint32_t a) { return float_bits(std::sqrt(bits_float(a))); }
uint32_t exp_ref(uint32_t a) { return float_bits(exp_core(bits_float(a))); }
uint32_t erf_ref(uint32_t a) { return float_bits(erf_core(bits_float(a))); }
uint32_t gelu_ref(uint32_t a) { return float_bits(gelu_core(bits_float(a))); }

uint32_t from_fp16_ref(uint16_t h) {  // mirrors fp32_from_fp16_bits exactly
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, f = h & 0x3FFu;
    if (e == 0 && f == 0) return s << 31;
    if (e == 0) {
        int k = 31 - __builtin_clz(f);
        return (s << 31) | (uint32_t)((k + 103) << 23) |
               (uint32_t)(((f - (1u << k)) << (23 - k)) & 0x7FFFFFu);
    }
    if (e == 0x1F) return s ? 0xC77FE000u : 0x477FE000u;
    return (s << 31) | ((e + 112u) << 23) | (f << 13);
}
int quant_ref(uint32_t vb, uint32_t sb) {
    float s = bits_float(sb);
    if (s == 0.0f) return 0;
    float qf = bits_float(vb) / s;
    if (std::isnan(qf)) return 0;
    if (std::isinf(qf)) return qf < 0 ? -128 : 127;
    long q = rhe_f(qf);
    if (q < -128) return -128;
    if (q > 127) return 127;
    return (int)q;
}

uint32_t evalr(Vtb_fp32_prim &tb, Op op, uint32_t a, uint32_t b = 0) {
    tb.op = op; tb.a_bits = a; tb.b_bits = b; tb.eval();
    return tb.result_bits;
}

uint32_t g_rng = 0x12345678u;
uint32_t nxt() { g_rng = 1664525u * g_rng + 1013904223u; return g_rng; }

} // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vtb_fp32_prim tb;
    int failures = 0;
    auto fail = [&](const std::string &n, uint32_t a, uint32_t b, uint32_t g, uint32_t e) {
        if (failures < 20)
            std::cerr << "[FAIL] " << n << " a=0x" << std::hex << a << " b=0x" << b
                      << " got=0x" << g << " exp=0x" << e << std::dec << "\n";
        failures++;
    };
    auto chk = [&](const std::string &n, Op op, uint32_t a, uint32_t b, uint32_t e) {
        uint32_t got = evalr(tb, op, a, b);
        if (!bits_match(got, e)) fail(n, a, b, got, e);
    };

    // ── existing add/sub/mul/round coverage ───────────────────────────────
    const std::vector<uint32_t> rt = {
        float_bits(0.0f), float_bits(-0.0f), float_bits(1.0f), float_bits(-1.0f),
        float_bits(0.15625f), float_bits(-13.75f), float_bits(65504.0f),
        0x00000001u, 0x007fffffu, 0x00800000u, 0x7f7fffffu, 0x7f800000u, 0xff800000u};
    for (uint32_t v : rt) chk("roundtrip", OP_ROUND, v, 0, round_ref(v));

    struct BC { const char *n; uint32_t a, b; };
    const std::vector<BC> cases = {
        {"simple", float_bits(1.0f), float_bits(2.0f)},
        {"negative", float_bits(-1.5f), float_bits(0.25f)},
        {"cancel", float_bits(1.0e20f), float_bits(-1.0e20f)},
        {"tie-ish", float_bits(1.0f), 0x33800000u},
        {"subnormal", 0x00000001u, float_bits(2.0f)},
        {"normal_min", 0x00800000u, float_bits(0.5f)},
        {"signed_zero", float_bits(-0.0f), float_bits(3.0f)},
        {"overflow", float_bits(3.4e38f), float_bits(2.0f)},
        {"inf", 0x7f800000u, float_bits(1.0f)}};
    for (const auto &tc : cases) {
        chk(std::string(tc.n) + "_add", OP_ADD, tc.a, tc.b, add_ref(tc.a, tc.b));
        chk(std::string(tc.n) + "_sub", OP_SUB, tc.a, tc.b, sub_ref(tc.a, tc.b));
        chk(std::string(tc.n) + "_mul", OP_MUL, tc.a, tc.b, mul_ref(tc.a, tc.b));
    }
    for (int i = 0; i < 256; ++i) {
        uint32_t a = nxt() & 0x7effffffu, b = nxt() & 0x7effffffu;
        chk("rnd_add", OP_ADD, a, b, add_ref(a, b));
        chk("rnd_sub", OP_SUB, a, b, sub_ref(a, b));
        chk("rnd_mul", OP_MUL, a, b, mul_ref(a, b));
    }
    chk("inf-inf", OP_SUB, 0x7f800000u, 0x7f800000u, 0x7fc00000u);
    chk("0*inf", OP_MUL, float_bits(0.0f), 0x7f800000u, 0x7fc00000u);

    // ── div: correctly-rounded, bit-exact vs C float ─────────────────────
    const std::vector<BC> dc = {
        {"d1", float_bits(1.0f), float_bits(3.0f)},
        {"d2", float_bits(-7.0f), float_bits(2.0f)},
        {"d3", float_bits(1.0f), float_bits(0.0f)},
        {"d4", float_bits(0.0f), float_bits(0.0f)},
        {"d5", float_bits(5.0f), 0x7f800000u},
        {"d6", 0x7f800000u, float_bits(2.0f)},
        {"d7", 0x00000001u, float_bits(2.0f)},
        {"d8", float_bits(192.0f), float_bits(7.0f)}};
    for (auto &c : dc) chk(c.n, OP_DIV, c.a, c.b, div_ref(c.a, c.b));
    for (int i = 0; i < 400000; ++i) {
        uint32_t a = (nxt() & 0x807fffffu) | ((96u + (nxt() % 64u)) << 23);
        uint32_t b = (nxt() & 0x807fffffu) | ((96u + (nxt() % 64u)) << 23);
        chk("rnd_div", OP_DIV, a, b, div_ref(a, b));
    }

    // ── sqrt: correctly-rounded, bit-exact vs C sqrtf ────────────────────
    for (uint32_t v : {float_bits(0.0f), float_bits(-0.0f), float_bits(1.0f),
                       float_bits(2.0f), float_bits(0.25f), float_bits(1e-6f),
                       float_bits(65504.0f), 0x7f800000u, float_bits(-4.0f)})
        chk("sqrt_dir", OP_SQRT, v, 0, sqrt_ref(v));
    for (int i = 0; i < 400000; ++i) {
        uint32_t a = (nxt() & 0x007fffffu) | ((80u + (nxt() % 80u)) << 23); // +finite
        chk("rnd_sqrt", OP_SQRT, a, 0, sqrt_ref(a));
    }

    // ── from_fp16: exact integer rewiring incl. ±65504 inf/NaN clamp ─────
    for (uint32_t h = 0; h < 65536; ++h)
        chk("fp16", OP_FROM_FP16, h, 0, from_fp16_ref((uint16_t)h));

    // ── quantize_i8: div + round-half-even + clip ────────────────────────
    {
        struct QC { uint32_t v, s; };
        const std::vector<QC> qc = {
            {float_bits(0.5f), float_bits(1.0f)}, {float_bits(1.5f), float_bits(1.0f)},
            {float_bits(2.5f), float_bits(1.0f)}, {float_bits(-0.5f), float_bits(1.0f)},
            {float_bits(1000.f), float_bits(1.0f)}, {float_bits(-1000.f), float_bits(1.0f)},
            {float_bits(1.0f), float_bits(0.0f)}, {float_bits(63.7f), float_bits(0.5f)}};
        for (auto &c : qc) {
            tb.op = OP_QUANT; tb.a_bits = c.v; tb.b_bits = c.s; tb.eval();
            int got = (int)(int32_t)tb.q_i32, exp = quant_ref(c.v, c.s);
            if (got != exp) { fail("quant_dir", c.v, c.s, got, exp); }
        }
        for (int i = 0; i < 300000; ++i) {
            uint32_t v = (nxt() & 0x807fffffu) | ((110u + (nxt() % 30u)) << 23);
            uint32_t s = (nxt() & 0x007fffffu) | ((118u + (nxt() % 12u)) << 23);
            if ((nxt() & 0x3F) == 0) s = float_bits(0.0f);
            tb.op = OP_QUANT; tb.a_bits = v; tb.b_bits = s; tb.eval();
            int got = (int)(int32_t)tb.q_i32, exp = quant_ref(v, s);
            if (got != exp) fail("rnd_quant", v, s, got, exp);
        }
    }

    // ── exp / erf / gelu: co-defined poly, bit-exact vs C poly oracle ────
    for (int i = 0; i <= 360000; ++i) {       // dense sweep over [-90, 90]
        float x = -90.0f + (180.0f * (float)i / 360000.0f);
        uint32_t xb = float_bits(x);
        chk("exp_swp", OP_EXP, xb, 0, exp_ref(xb));
    }
    for (int i = 0; i <= 200000; ++i) {       // erf/gelu over [-8, 8]
        float x = -8.0f + (16.0f * (float)i / 200000.0f);
        uint32_t xb = float_bits(x);
        chk("erf_swp", OP_ERF, xb, 0, erf_ref(xb));
        chk("gelu_swp", OP_GELU, xb, 0, gelu_ref(xb));
    }
    for (uint32_t v : {float_bits(0.0f), float_bits(-0.0f), 0x7f800000u, 0xff800000u,
                       0x7fc00000u, float_bits(88.7f), float_bits(-103.5f)}) {
        chk("exp_edge", OP_EXP, v, 0, exp_ref(v));
        chk("erf_edge", OP_ERF, v, 0, erf_ref(v));
        chk("gelu_edge", OP_GELU, v, 0, gelu_ref(v));
    }
    for (int i = 0; i < 200000; ++i) {        // random finite stress for exp/gelu
        uint32_t a = (nxt() & 0x807fffffu) | ((118u + (nxt() % 8u)) << 23);
        chk("rnd_exp", OP_EXP, a, 0, exp_ref(a));
        chk("rnd_gelu", OP_GELU, a, 0, gelu_ref(a));
    }

    if (failures != 0) {
        std::cerr << failures << " FP32 primitive checks failed\n";
        return 1;
    }
    std::cout << "FP32 primitive tests passed\n";
    return 0;
}
