// Verilator tests for the W8A16 blocking helper engine.
//
// Operations under test:
//   * BUF_COPY  — flat copy and FP16-element-aware transpose.
//   * VADD      — FP16 + FP16 → FP16 (ABUF) and FP32 + FP16-broadcast → FP32
//                 (ACCUM bias / attention-mask).
//   * SCALE_MUL — FP32 × FP16-widened scale, narrowed to FP16 on ABUF dst or
//                 kept FP32 on ACCUM dst.
//
// Bit-exactness: every arithmetic step uses standard IEEE binary32 in both
// the C++ oracle and the RTL (fp32_prim_pkg's mul/add are correctly rounded),
// and the FP16 narrow uses RNE in both. Expect 0 ULPs on FP16 outputs and
// exact 32-bit equality on FP32 ACCUM outputs.
//
// The load-bearing bit-exact gate against fp32_prim_ref lives in
// rtl/cocotb/test_helpers.py.

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
#include <memory>
#include <random>
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

// ───── FP16 ↔ FP32 helpers (match RTL fp32_to_fp16_bits / fp32_from_fp16_bits) ─────

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
        int k = 31 - __builtin_clz(f);
        uint32_t bits = (s << 31) | (uint32_t)((k + 103) << 23) |
                       (uint32_t)(((f - (1u << k)) << (23 - k)) & 0x7FFFFFu);
        return bits_float(bits);
    }
    if (e == 0x1F) {
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

std::vector<uint8_t> pack_fp16(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) {
        uint16_t h = fp32_to_fp16(v[i]);
        out[i * 2 + 0] = uint8_t(h & 0xFF);
        out[i * 2 + 1] = uint8_t((h >> 8) & 0xFF);
    }
    return out;
}

std::vector<uint8_t> pack_fp32(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * 4);
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t b = float_bits(v[i]);
        for (int k = 0; k < 4; ++k)
            out[i * 4 + k] = uint8_t((b >> (k * 8)) & 0xFF);
    }
    return out;
}

void expect_bytes_equal(const char* name, const std::vector<uint8_t>& got,
                        const std::vector<uint8_t>& exp) {
    if (got.size() != exp.size())
        TEST_FAIL(name, "size mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            fprintf(stderr, "%s: first byte mismatch at idx=%zu got=0x%02x exp=0x%02x\n",
                    name, i, got[i], exp[i]);
            TEST_FAIL(name, "byte mismatch");
        }
    }
}

void expect_clean_halt(const char* name, Vtaccel_top* dut) {
    if (!dut->done || dut->fault)
        TEST_FAIL(name, "did not halt cleanly");
}

// FP16-snap a float so the oracle works on the same FP16 patterns as the RTL.
float fp16_snap(float v) { return fp16_to_fp32(fp32_to_fp16(v)); }

// ───── BUF_COPY ──────────────────────────────────────────────────────────

void test_buf_copy_flat_interbuffer() {
    const char* name = "buf_copy_flat_interbuffer";
    SimHarness s;
    std::vector<uint8_t> src(48);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = uint8_t((0x20 + 7 * i) & 0xFF);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, src);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 10, 3, 0, 0),
        insn::HALT(),
    });
    s.run();
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 10 * 16, src.size());
    expect_bytes_equal(name, got, src);
    TEST_PASS(name);
}

void test_buf_copy_overlap_compaction() {
    const char* name = "buf_copy_overlap_compaction";
    SimHarness s;
    std::vector<uint8_t> bytes(6 * 16);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = uint8_t((0x51 + 11 * i) & 0xFF);
    auto expected = bytes;
    std::memmove(expected.data() + 16, expected.data() + 32, 48);
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes);

    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 2, BUF_ABUF_ID, 1, 3, 0, 0),
        insn::HALT(),
    });
    s.run();
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID, 0, bytes.size());
    expect_bytes_equal(name, got, expected);
    TEST_PASS(name);
}

void test_buf_copy_transpose_fp16_square() {
    const char* name = "buf_copy_transpose_fp16_square";
    SimHarness s;
    constexpr int rows = 16;
    constexpr int cols = 16;
    std::vector<float> src_f((size_t)rows * cols);
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    for (auto& v : src_f) v = fp16_snap(ud(rng));

    // Element transpose at FP16 granularity.
    std::vector<float> dst_f((size_t)cols * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst_f[size_t(c) * rows + r] = src_f[size_t(r) * cols + c];

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, pack_fp16(src_f));
    int length_units = (rows * cols * 2) / 16;
    int src_rows_field = rows / 16;
    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0,
                       length_units, src_rows_field, 1),
        insn::HALT(),
    });
    s.run(200000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, dst_f.size() * 2);
    expect_bytes_equal(name, got, pack_fp16(dst_f));
    TEST_PASS(name);
}

void test_buf_copy_transpose_fp16_rect() {
    const char* name = "buf_copy_transpose_fp16_rect";
    SimHarness s;
    constexpr int rows = 32;
    constexpr int cols = 16;
    std::vector<float> src_f((size_t)rows * cols);
    std::mt19937 rng(202);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    for (auto& v : src_f) v = fp16_snap(ud(rng));

    std::vector<float> dst_f((size_t)cols * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst_f[size_t(c) * rows + r] = src_f[size_t(r) * cols + c];

    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, pack_fp16(src_f));
    int length_units = (rows * cols * 2) / 16;
    int src_rows_field = rows / 16;
    s.load({
        insn::BUF_COPY(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0,
                       length_units, src_rows_field, 1),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_WBUF_ID, 0, dst_f.size() * 2);
    expect_bytes_equal(name, got, pack_fp16(dst_f));
    TEST_PASS(name);
}

// ───── VADD ──────────────────────────────────────────────────────────────

void test_vadd_fp16_abuf() {
    const char* name = "vadd_fp16_abuf";
    SimHarness s;
    constexpr int M = 16, N = 16;
    std::mt19937 rng(303);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::vector<float> a((size_t)M * N), b((size_t)M * N), expected((size_t)M * N);
    for (auto& v : a) v = fp16_snap(ud(rng));
    for (auto& v : b) v = fp16_snap(ud(rng));
    for (size_t i = 0; i < a.size(); ++i) expected[i] = a[i] + b[i];

    int rows_units = (M * N * 2) / 16;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, pack_fp16(a));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, pack_fp16(b));

    int out_off = rows_units;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::VADD(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, out_off, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(out_off) * 16, size_t(rows_units) * 16);
    expect_bytes_equal(name, got, pack_fp16(expected));
    TEST_PASS(name);
}

void test_vadd_accum_bias_broadcast() {
    const char* name = "vadd_accum_bias_broadcast";
    SimHarness s;
    constexpr int M = 16, N = 16;
    std::mt19937 rng(404);
    std::uniform_real_distribution<float> ud_acc(-5.0f, 5.0f);
    std::uniform_real_distribution<float> ud_bias(-1.0f, 1.0f);
    std::vector<float> accum((size_t)M * N), bias((size_t)N), expected((size_t)M * N);
    for (auto& v : accum) v = ud_acc(rng);
    for (auto& v : bias) v = fp16_snap(ud_bias(rng));
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            expected[size_t(r) * N + c] = accum[size_t(r) * N + c] + bias[size_t(c)];

    int accum_rows = (M * N * 4) / 16;
    int bias_rows  = (N * 2) / 16;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_fp32(accum));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, pack_fp16(bias));

    int out_off = 256;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, out_off, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ACCUM_ID,
                               size_t(out_off) * 16, size_t(accum_rows) * 16);
    (void)bias_rows;
    expect_bytes_equal(name, got, pack_fp32(expected));
    TEST_PASS(name);
}

void test_vadd_attention_mask_fp16() {
    const char* name = "vadd_attention_mask_fp16";
    SimHarness s;
    constexpr int M = 16, N = 16;
    std::vector<float> accum((size_t)M * N, 0.0f);
    std::vector<float> mask((size_t)N, 0.0f);
    for (int c = N / 2; c < N; ++c) mask[size_t(c)] = -65504.0f;

    std::vector<float> expected((size_t)M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            expected[size_t(r) * N + c] = accum[size_t(r) * N + c] + mask[size_t(c)];

    int accum_rows = (M * N * 4) / 16;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_fp32(accum));
    sram_write_bytes(s.dut.get(), BUF_WBUF_ID, 0, pack_fp16(mask));

    int out_off = 384;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::VADD(BUF_ACCUM_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, out_off, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ACCUM_ID,
                               size_t(out_off) * 16, size_t(accum_rows) * 16);
    expect_bytes_equal(name, got, pack_fp32(expected));
    TEST_PASS(name);
}

// ───── SCALE_MUL ─────────────────────────────────────────────────────────

void test_scale_mul_abuf_fp16() {
    const char* name = "scale_mul_abuf_fp16";
    SimHarness s;
    constexpr int M = 16, N = 16;
    const float scale_f = -0.5f;
    const uint16_t scale_bits = fp32_to_fp16(scale_f);
    std::mt19937 rng(505);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    std::vector<float> x((size_t)M * N), expected((size_t)M * N);
    for (auto& v : x) v = fp16_snap(ud(rng));
    const float scale = fp16_to_fp32(scale_bits);
    for (size_t i = 0; i < x.size(); ++i) expected[i] = x[i] * scale;

    int rows_units = (M * N * 2) / 16;
    sram_write_bytes(s.dut.get(), BUF_ABUF_ID, 0, pack_fp16(x));

    int out_off = 128;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SET_SCALE(2, scale_bits, 0),
        insn::SCALE_MUL(BUF_ABUF_ID, 0, BUF_ABUF_ID, out_off, 2, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(out_off) * 16, size_t(rows_units) * 16);
    expect_bytes_equal(name, got, pack_fp16(expected));
    TEST_PASS(name);
}

void test_scale_mul_accum_fp32() {
    const char* name = "scale_mul_accum_fp32";
    SimHarness s;
    constexpr int M = 16, N = 16;
    const float scale_f = 3.0f;
    const uint16_t scale_bits = fp32_to_fp16(scale_f);
    std::mt19937 rng(606);
    std::uniform_real_distribution<float> ud(-10.0f, 10.0f);
    std::vector<float> x((size_t)M * N), expected((size_t)M * N);
    for (auto& v : x) v = ud(rng);
    const float scale = fp16_to_fp32(scale_bits);
    for (size_t i = 0; i < x.size(); ++i) expected[i] = x[i] * scale;

    int rows_units = (M * N * 4) / 16;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_fp32(x));

    int out_off = 256;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SET_SCALE(3, scale_bits, 0),
        insn::SCALE_MUL(BUF_ACCUM_ID, 0, BUF_ACCUM_ID, out_off, 3, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ACCUM_ID,
                               size_t(out_off) * 16, size_t(rows_units) * 16);
    expect_bytes_equal(name, got, pack_fp32(expected));
    TEST_PASS(name);
}

void test_scale_mul_accum_to_abuf_narrow() {
    const char* name = "scale_mul_accum_to_abuf_narrow";
    SimHarness s;
    constexpr int M = 16, N = 16;
    const uint16_t scale_one = fp32_to_fp16(1.0f);
    std::mt19937 rng(707);
    std::uniform_real_distribution<float> ud(-100.0f, 100.0f);
    std::vector<float> x((size_t)M * N), expected((size_t)M * N);
    for (auto& v : x) v = ud(rng);
    for (size_t i = 0; i < x.size(); ++i) expected[i] = x[i];

    int accum_rows = (M * N * 4) / 16;
    int abuf_rows  = (M * N * 2) / 16;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_fp32(x));

    int out_off = 320;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SET_SCALE(4, scale_one, 0),
        insn::SCALE_MUL(BUF_ACCUM_ID, 0, BUF_ABUF_ID, out_off, 4, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    (void)accum_rows;
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(out_off) * 16, size_t(abuf_rows) * 16);
    expect_bytes_equal(name, got, pack_fp16(expected));
    TEST_PASS(name);
}

void test_scale_mul_accum_to_abuf_nonunit() {
    const char* name = "scale_mul_accum_to_abuf_nonunit";
    SimHarness s;
    constexpr int M = 16, N = 16;
    const uint16_t scale_bits = fp32_to_fp16(0.25f);
    const float scale = fp16_to_fp32(scale_bits);
    std::mt19937 rng(808);
    std::uniform_real_distribution<float> ud(-50.0f, 50.0f);
    std::vector<float> x((size_t)M * N), expected((size_t)M * N);
    for (auto& v : x) v = ud(rng);
    for (size_t i = 0; i < x.size(); ++i) expected[i] = x[i] * scale;

    int abuf_rows = (M * N * 2) / 16;
    sram_write_bytes(s.dut.get(), BUF_ACCUM_ID, 0, pack_fp32(x));

    int out_off = 448;
    s.load({
        insn::CONFIG_TILE(M / 16, N / 16, 1),
        insn::SET_SCALE(5, scale_bits, 0),
        insn::SCALE_MUL(BUF_ACCUM_ID, 0, BUF_ABUF_ID, out_off, 5, 0),
        insn::HALT(),
    });
    s.run(300000);
    expect_clean_halt(name, s.dut.get());
    auto got = sram_read_bytes(s.dut.get(), BUF_ABUF_ID,
                               size_t(out_off) * 16, size_t(abuf_rows) * 16);
    expect_bytes_equal(name, got, pack_fp16(expected));
    TEST_PASS(name);
}

// ───── Dropped opcodes raise FAULT_UNSUPPORTED_OP ────────────────────────

void expect_fault_program(const char* name, const std::vector<uint64_t>& prog,
                          uint32_t expected_fault_code, int timeout = 5000) {
    SimHarness s;
    s.load(prog);
    s.run(timeout);
    if (s.dut->fault != 1) TEST_FAIL(name, "fault did not assert");
    if (s.dut->done == 1) TEST_FAIL(name, "done should remain low under fault");
    if (s.dut->fault_code != expected_fault_code)
        TEST_FAIL(name, "unexpected fault code");
    TEST_PASS(name);
}

void test_requant_unsupported() {
    expect_fault_program("requant_unsupported",
        { insn::CONFIG_TILE(1, 1, 1),
          insn::SET_SCALE(0, 0x3C00, 0),
          insn::REQUANT(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, 0, 0),
          insn::HALT() }, 0x6, 5000);
}

void test_requant_pc_unsupported() {
    expect_fault_program("requant_pc_unsupported",
        { insn::CONFIG_TILE(1, 1, 1),
          insn::REQUANT_PC(BUF_ACCUM_ID, 0, BUF_WBUF_ID, 0, BUF_ABUF_ID, 0, 0, 0),
          insn::HALT() }, 0x6, 5000);
}

void test_dequant_add_unsupported() {
    expect_fault_program("dequant_add_unsupported",
        { insn::CONFIG_TILE(1, 1, 1),
          insn::SET_SCALE(0, 0x3C00, 0),
          insn::SET_SCALE(1, 0x3C00, 0),
          insn::DEQUANT_ADD(BUF_ACCUM_ID, 0, BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, 0, 0),
          insn::HALT() }, 0x6, 5000);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    printf("--- W8A16 helper engine verilator tests ---\n");
    test_buf_copy_flat_interbuffer();
    test_buf_copy_overlap_compaction();
    test_buf_copy_transpose_fp16_square();
    test_buf_copy_transpose_fp16_rect();
    test_vadd_fp16_abuf();
    test_vadd_accum_bias_broadcast();
    test_vadd_attention_mask_fp16();
    test_scale_mul_abuf_fp16();
    test_scale_mul_accum_fp32();
    test_scale_mul_accum_to_abuf_narrow();
    test_scale_mul_accum_to_abuf_nonunit();
    test_requant_unsupported();
    test_requant_pc_unsupported();
    test_dequant_add_unsupported();
    printf("\n%d / %d tests passed\n", tests_pass, tests_run);
    if (tests_pass != tests_run) std::exit(1);
    return 0;
}
