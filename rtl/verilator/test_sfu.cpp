// Verilator tests for the W8A16 SFU engine.
//
// FP16 ABUF endpoints, FP32 ACCUM endpoints, FP32-internal math. These
// tests act as a smoke gate on the SFU dispatch path and verify the
// FP16 output is close to a C++ float32 oracle (a few ULPs of FP16 max
// across LayerNorm/Softmax/GELU). The load-bearing bit-exact gate lives
// in rtl/cocotb/test_sfu.py — it compares against fp32_prim_ref which
// is RNE-equal to fp32_prim_pkg by construction.

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "verilated.h"
#include "include/testbench.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

static int tests_run  = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
    printf("PASS: %s\n", name); tests_pass++; tests_run++; } while(0)
#define TEST_FAIL(name, msg) do { \
    fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); } while(0)

using tbutil::SimHarness;
using tbutil::sram_write_bytes;
using tbutil::sram_read_bytes;
constexpr int BUF_ABUF_ID  = tbutil::BUF_ABUF_ID;
constexpr int BUF_WBUF_ID  = tbutil::BUF_WBUF_ID;
constexpr int BUF_ACCUM_ID = tbutil::BUF_ACCUM_ID;

namespace {

// ───── FP16 ↔ FP32 helpers (match numpy.float{16,32}.astype semantics) ─────

uint32_t float_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

float bits_float(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t f = h & 0x3FFu;
    if (e == 0 && f == 0)
        return bits_float(s << 31);
    if (e == 0) {
        // subnormal
        int k = 31 - __builtin_clz(f);
        uint32_t bits = (s << 31) | (uint32_t)((k + 103) << 23) |
                       (uint32_t)(((f - (1u << k)) << (23 - k)) & 0x7FFFFFu);
        return bits_float(bits);
    }
    if (e == 0x1F) {
        // Matches fp32_from_fp16_bits's ±65504 clamp on inf/NaN.
        return bits_float(s ? 0xC77FE000u : 0x477FE000u);
    }
    return bits_float((s << 31) | ((e + 112u) << 23) | (f << 13));
}

uint16_t fp32_to_fp16(float v) {
    uint32_t f = float_bits(v);
    uint32_t s   = (f >> 31) & 1u;
    uint32_t e8  = (f >> 23) & 0xFFu;
    uint32_t f23 = f & 0x7FFFFFu;
    if (e8 == 0xFFu && f23 != 0u)
        return (uint16_t)((s << 15) | (0x1Fu << 10) | 0x200u);
    if (e8 == 0xFFu)
        return (uint16_t)((s << 15) | (0x1Fu << 10));
    if (e8 == 0u)
        return (uint16_t)(s << 15);
    int e_unb = (int)e8 - 127;
    uint64_t mant24 = (1ULL << 23) | (uint64_t)f23;
    auto rshift_rne = [](uint64_t value, int shift) -> uint64_t {
        if (shift <= 0) return value << (-shift);
        if (shift >= 63) {
            uint64_t half = 1ULL << 63;
            return (value > half) ? 1ULL : 0ULL;
        }
        uint64_t q = value >> shift;
        uint64_t mask = (1ULL << shift) - 1ULL;
        uint64_t r = value & mask;
        uint64_t half = 1ULL << (shift - 1);
        if (r > half || (r == half && (q & 1ULL))) return q + 1ULL;
        return q;
    };
    if (e_unb > 15) return (uint16_t)((s << 15) | (0x1Fu << 10));
    if (e_unb >= -14) {
        uint64_t rounded = rshift_rne(mant24, 13);
        if (rounded == 0x800ULL) {
            if (e_unb + 1 > 15) return (uint16_t)((s << 15) | (0x1Fu << 10));
            return (uint16_t)((s << 15) | (uint32_t)((e_unb + 1 + 15) << 10));
        }
        return (uint16_t)((s << 15) | (uint32_t)((e_unb + 15) << 10) |
                          (uint32_t)(rounded & 0x3FFu));
    }
    int shift_amt = 13 + (-14 - e_unb);
    uint64_t rounded = rshift_rne(mant24, shift_amt);
    if (rounded == 0x400ULL) return (uint16_t)((s << 15) | (1u << 10));
    return (uint16_t)((s << 15) | (uint32_t)(rounded & 0x3FFu));
}

std::vector<uint8_t> pack_fp16_row_major(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h = fp32_to_fp16(v[i]);
        out[i * 2 + 0] = uint8_t(h & 0xFF);
        out[i * 2 + 1] = uint8_t((h >> 8) & 0xFF);
    }
    return out;
}

std::vector<uint8_t> pack_fp32_row_major(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * 4);
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t b = float_bits(v[i]);
        for (int k = 0; k < 4; ++k)
            out[i * 4 + k] = uint8_t((b >> (k * 8)) & 0xFF);
    }
    return out;
}

std::vector<uint16_t> unpack_fp16_bytes(const std::vector<uint8_t>& bytes) {
    std::vector<uint16_t> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = uint16_t(bytes[i * 2]) | (uint16_t(bytes[i * 2 + 1]) << 8);
    return out;
}

// Compare two FP16 byte vectors, allowing up to `max_ulps` integer
// difference in the FP16 representation (treating sign-magnitude bits
// monotonically — only used here where values share a sign).
void expect_close_fp16(const char* name, const std::vector<uint8_t>& got_bytes,
                       const std::vector<uint8_t>& exp_bytes, int max_ulps) {
    if (got_bytes.size() != exp_bytes.size())
        TEST_FAIL(name, "size mismatch");
    auto got = unpack_fp16_bytes(got_bytes);
    auto exp = unpack_fp16_bytes(exp_bytes);
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == exp[i]) continue;
        // Map to lexicographic order via sign-flip trick for FP16.
        auto map_key = [](uint16_t b) -> int32_t {
            if (b & 0x8000u) return -int32_t(b & 0x7FFFu);
            return int32_t(b);
        };
        int32_t diff = std::abs(map_key(got[i]) - map_key(exp[i]));
        if (diff > max_ulps) {
            std::fprintf(stderr,
                         "%s: lane %zu got=0x%04x (%g) exp=0x%04x (%g) ulp_diff=%d max=%d\n",
                         name, i, got[i], double(fp16_to_fp32(got[i])),
                         exp[i], double(fp16_to_fp32(exp[i])), diff, max_ulps);
            TEST_FAIL(name, "fp16 mismatch beyond tolerance");
        }
    }
}

// ───── float32 SFU oracles (match RTL semantics; not bit-exact) ─────

std::vector<float> softmax_oracle(const std::vector<float>& x, int M, int N) {
    std::vector<float> y(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        float row_max = x[size_t(r) * size_t(N)];
        for (int c = 1; c < N; ++c)
            row_max = std::max(row_max, x[size_t(r) * size_t(N) + size_t(c)]);
        float sum = 0.0f;
        std::vector<float> exps((size_t)N);
        for (int c = 0; c < N; ++c) {
            exps[size_t(c)] = std::exp(x[size_t(r) * size_t(N) + size_t(c)] - row_max);
            sum += exps[size_t(c)];
        }
        for (int c = 0; c < N; ++c)
            y[size_t(r) * size_t(N) + size_t(c)] = exps[size_t(c)] / sum;
    }
    return y;
}

std::vector<float> layernorm_oracle(const std::vector<float>& x,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& beta,
                                    int M, int N) {
    std::vector<float> y(size_t(M) * size_t(N));
    const float eps = 1.0e-6f;
    for (int r = 0; r < M; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c)
            sum += x[size_t(r) * size_t(N) + size_t(c)];
        float mean = sum / float(N);
        float vsum = 0.0f;
        for (int c = 0; c < N; ++c) {
            float d = x[size_t(r) * size_t(N) + size_t(c)] - mean;
            vsum += d * d;
        }
        float var = vsum / float(N);
        float denom = std::sqrt(var + eps);
        for (int c = 0; c < N; ++c) {
            float xn = (x[size_t(r) * size_t(N) + size_t(c)] - mean) / denom;
            y[size_t(r) * size_t(N) + size_t(c)] =
                xn * gamma[size_t(c)] + beta[size_t(c)];
        }
    }
    return y;
}

std::vector<float> gelu_oracle(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x[i];
        y[i] = v * 0.5f * (1.0f + std::erf(v * inv_sqrt2));
    }
    return y;
}

std::vector<uint8_t> fp16_bytes_from_fp32(const std::vector<float>& v) {
    return pack_fp16_row_major(v);
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
    if (!dut->done || dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
}

void test_softmax_abuf() {
    const char* name = "softmax_abuf_fp16";
    const int M = 16, N = 32;
    const int src_off_u = 0;
    const int dst_off_u = 256;
    const int out_rows = (M * N * 2) / 16;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::vector<float> x(size_t(M) * size_t(N));
    for (auto& v : x) v = float(fp16_to_fp32(fp32_to_fp16(ud(rng))));  // FP16-snap

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(src_off_u) * 16u,
                     pack_fp16_row_major(x));

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SOFTMAX(BUF_ABUF_ID, src_off_u, BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto exp_f32 = softmax_oracle(x, M, N);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    expect_close_fp16(name, got, exp, /*max_ulps=*/4);
    TEST_PASS(name);
}

void test_softmax_accum_fp32() {
    const char* name = "softmax_accum_fp32";
    const int M = 16, N = 32;
    const int src_off_u = 0;
    const int dst_off_u = 384;
    const int out_rows = (M * N * 2) / 16;

    std::mt19937 rng(22);
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    std::vector<float> x(size_t(M) * size_t(N));
    for (auto& v : x) v = ud(rng);

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, size_t(src_off_u) * 16u,
                     pack_fp32_row_major(x));

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SOFTMAX(BUF_ACCUM_ID, src_off_u, BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto exp_f32 = softmax_oracle(x, M, N);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    expect_close_fp16(name, got, exp, /*max_ulps=*/4);
    TEST_PASS(name);
}

void test_softmax_attention_mask() {
    const char* name = "softmax_attention_mask_fp16";
    const int M = 16, N = 16;
    const int src_off_u = 0;
    const int dst_off_u = 512;
    const int out_rows = (M * N * 2) / 16;

    std::mt19937 rng(33);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    std::vector<float> x(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float v = (c < N / 2)
                        ? float(fp16_to_fp32(fp32_to_fp16(ud(rng))))
                        : -65504.0f;
            x[size_t(r) * size_t(N) + size_t(c)] = v;
        }
    }

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(src_off_u) * 16u,
                     pack_fp16_row_major(x));

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SOFTMAX(BUF_ABUF_ID, src_off_u, BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto got16 = unpack_fp16_bytes(got);
    for (int r = 0; r < M; ++r) {
        for (int c = N / 2; c < N; ++c) {
            uint16_t bits = got16[size_t(r) * size_t(N) + size_t(c)];
            if (bits != 0x0000u) {
                std::fprintf(stderr,
                             "%s: masked lane (r=%d c=%d) not FP16 +0, got=0x%04x\n",
                             name, r, c, bits);
                TEST_FAIL(name, "attention mask did not underflow to zero");
            }
        }
    }
    auto exp_f32 = softmax_oracle(x, M, N);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    expect_close_fp16(name, got, exp, /*max_ulps=*/4);
    TEST_PASS(name);
}

void test_layernorm_abuf() {
    const char* name = "layernorm_abuf_fp16";
    const int M = 16, N = 64;
    const int src_off_u = 0;
    const int gb_off_u = 0;
    const int dst_off_u = 768;
    const int out_rows = (M * N * 2) / 16;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    std::uniform_real_distribution<float> ug(0.5f, 1.5f);
    std::uniform_real_distribution<float> ub(-0.3f, 0.3f);
    std::vector<float> x(size_t(M) * size_t(N));
    std::vector<float> gamma((size_t)N);
    std::vector<float> beta((size_t)N);
    for (auto& v : x) v = float(fp16_to_fp32(fp32_to_fp16(ud(rng))));
    for (auto& v : gamma) v = float(fp16_to_fp32(fp32_to_fp16(ug(rng))));
    for (auto& v : beta) v = float(fp16_to_fp32(fp32_to_fp16(ub(rng))));

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(src_off_u) * 16u,
                     pack_fp16_row_major(x));
    auto gb_bytes = pack_fp16_row_major(gamma);
    auto beta_bytes = pack_fp16_row_major(beta);
    gb_bytes.insert(gb_bytes.end(), beta_bytes.begin(), beta_bytes.end());
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, size_t(gb_off_u) * 16u, gb_bytes);

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::LAYERNORM(BUF_ABUF_ID, src_off_u, BUF_WBUF_ID, gb_off_u,
                        BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto exp_f32 = layernorm_oracle(x, gamma, beta, M, N);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    expect_close_fp16(name, got, exp, /*max_ulps=*/8);
    TEST_PASS(name);
}

void test_gelu_abuf() {
    const char* name = "gelu_abuf_fp16";
    const int M = 16, N = 16;
    const int src_off_u = 0;
    const int dst_off_u = 1024;
    const int out_rows = (M * N * 2) / 16;

    std::vector<float> x(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            x[size_t(r) * size_t(N) + size_t(c)] =
                float(fp16_to_fp32(fp32_to_fp16(
                    (float(c) - 8.0f) * 0.25f + float(r) * 0.0625f)));

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, size_t(src_off_u) * 16u,
                     pack_fp16_row_major(x));

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::GELU(BUF_ABUF_ID, src_off_u, BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto exp_f32 = gelu_oracle(x);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    // GELU uses an Abramowitz & Stegun polynomial approximation; allow a few
    // extra ULPs vs std::erf.
    expect_close_fp16(name, got, exp, /*max_ulps=*/16);
    TEST_PASS(name);
}

void test_gelu_accum_fp32() {
    const char* name = "gelu_accum_fp32";
    const int M = 16, N = 16;
    const int src_off_u = 0;
    const int dst_off_u = 1280;
    const int out_rows = (M * N * 2) / 16;

    std::vector<float> x(size_t(M) * size_t(N));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            x[size_t(r) * size_t(N) + size_t(c)] =
                (float(c) - 8.0f) * 0.5f + float(r) * 0.125f;

    SimHarness s;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, size_t(src_off_u) * 16u,
                     pack_fp32_row_major(x));

    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::GELU(BUF_ACCUM_ID, src_off_u, BUF_ABUF_ID, dst_off_u, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(1500000);
    expect_clean_halt(name, s.dut.get());

    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(dst_off_u) * 16u, size_t(out_rows) * 16u);
    auto exp_f32 = gelu_oracle(x);
    auto exp = fp16_bytes_from_fp32(exp_f32);
    expect_close_fp16(name, got, exp, /*max_ulps=*/16);
    TEST_PASS(name);
}

void test_softmax_attnv_unsupported() {
    const char* name = "softmax_attnv_unsupported";
    SimHarness s;
    s.load({
        insn::CONFIG_TILE(1, 1, 1),
        insn::SOFTMAX_ATTNV(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0,
                            BUF_WBUF_ID, 256, 0, 0),
        insn::SYNC(0b100),
        insn::HALT(),
    });
    s.run(200000);
    if (!s.dut->fault) {
        TEST_FAIL(name, "expected fault for legacy SOFTMAX_ATTNV");
    }
    TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    printf("--- W8A16 SFU verilator tests ---\n");
    test_softmax_abuf();
    test_softmax_accum_fp32();
    test_softmax_attention_mask();
    test_layernorm_abuf();
    test_gelu_abuf();
    test_gelu_accum_fp32();
    test_softmax_attnv_unsupported();
    printf("%d / %d tests passed\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
