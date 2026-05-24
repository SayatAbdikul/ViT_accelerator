// Verilator top-level MATMUL tests for the W8A16 systolic datapath.
//
// Inputs are FP16 (encoded via fp32_to_fp16); the FP32 oracle uses the
// sequential K-loop reduction defined in systolic_test_utils.h, matching
// the per-PE FP32 MAC order in software/taccel/golden_model/systolic_w8a16.py.
// ACCUM is read back as FP32 bit patterns and compared exactly.

#include "Vtaccel_top.h"
#include "verilated.h"
#include "include/systolic_test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace systolic_test;

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

static void expect_clean_halt(const char* name, Vtaccel_top* dut) {
  if (!dut->done || dut->fault)
    TEST_FAIL(name, "did not halt cleanly");
}

static void test_matmul_identity() {
  const char* name = "matmul_identity_16x16";
  Sim s;

  uint16_t a[16][16] = {};
  uint16_t eye[16][16] = {};
  uint32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = fp32_to_fp16(static_cast<float>((i * 3 + j) & 0x7F));
      eye[i][j] = fp32_to_fp16((i == j) ? 1.0f : 0.0f);
    }
  }

  prepare_logical_16x16(s.dram, prog, a, eye, 0x100000, 0x110000);
  matmul_fp_ref<16, 16, 16>(a, eye, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(600000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "identity"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_ones() {
  const char* name = "matmul_ones_16x16";
  Sim s;

  uint16_t a[16][16] = {};
  uint16_t b[16][16] = {};
  uint32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = fp32_to_fp16(1.0f);
      b[i][j] = fp32_to_fp16(1.0f);
    }
  }

  prepare_logical_16x16(s.dram, prog, a, b, 0x120000, 0x130000);
  matmul_fp_ref<16, 16, 16>(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(600000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "ones"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_accumulate_flag() {
  const char* name = "matmul_accumulate_flag";
  Sim s;

  uint16_t a[16][16] = {};
  uint16_t b[16][16] = {};
  uint32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = fp32_to_fp16(1.0f);
      b[i][j] = fp32_to_fp16((i == j) ? 2.0f : 0.0f);
    }
  }

  prepare_logical_16x16(s.dram, prog, a, b, 0x140000, 0x150000);
  matmul_fp_ref<16, 16, 16>(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 1));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 1));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(800000);
  expect_clean_halt(name, s.dut.get());

  // After two non-clearing MATMULs into the same destination tile, the
  // accumulator holds 2 * exp (FP32 doubling is exact for these values).
  uint32_t exp2[16][16] = {};
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      exp2[i][j] = float_to_bits(bits_to_float(exp[i][j]) * 2.0f);

  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp2, "accumulate"))
    TEST_FAIL(name, "accumulate mismatch");
  TEST_PASS(name);
}

static void test_matmul_multitile_2x2x2() {
  const char* name = "matmul_multitile_2x2x2";
  Sim s;

  uint16_t a[32][32] = {};
  uint16_t b[32][32] = {};
  uint32_t exp[32][32] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      // Values in [-5,5] -- exactly representable in FP16, and 32-K
      // products fit comfortably in FP32 without rounding artifacts.
      float av = static_cast<float>(((i * 7 + j * 5 + 3) % 11) - 5);
      float bv = static_cast<float>(((i * 3 + j * 9 + 1) % 13) - 6);
      a[i][j] = fp32_to_fp16(av);
      b[i][j] = fp32_to_fp16(bv);
    }
  }

  prepare_logical_32x32(s.dram, prog, a, b, 0x160000, 0x180000);
  matmul_fp_ref<32, 32, 32>(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(2, 2, 2));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<32, 32>(s.dut.get(), 0, exp, "multitile"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_random_regression_16x16() {
  const char* name = "matmul_random_regression_16x16";
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (int tc = 0; tc < 4; ++tc) {
    Sim s;
    uint16_t a[16][16] = {};
    uint16_t b[16][16] = {};
    uint32_t exp[16][16] = {};
    std::vector<uint64_t> prog;

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        a[i][j] = fp32_to_fp16(dist(rng));
        b[i][j] = fp32_to_fp16(dist(rng));
      }
    }

    prepare_logical_16x16(s.dram, prog, a, b, 0x1C0000 + tc * 0x4000, 0x1C2000 + tc * 0x4000);
    matmul_fp_ref<16, 16, 16>(a, b, exp);

    prog.push_back(insn::CONFIG_TILE(1, 1, 1));
    prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
    prog.push_back(insn::SYNC(0b010));
    prog.push_back(insn::HALT());

    s.load_program(prog);
    s.run(600000);
    expect_clean_halt(name, s.dut.get());
    if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "random"))
      TEST_FAIL(name, "ACCUM mismatch");
  }

  TEST_PASS(name);
}

static void test_matmul_k4_boundary_stress() {
  const char* name = "matmul_k4_boundary_stress";
  Sim s;

  uint16_t a[16][64] = {};
  uint16_t b[64][16] = {};
  uint32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  // K=64 sequence of +1/-1 products -- exact in FP32 even with 64 terms;
  // verifies multi-K-tile streaming (4 K-tiles per output tile).
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 64; ++k)
      a[i][k] = fp32_to_fp16(((i + k) & 1) ? 1.0f : -1.0f);
  }
  for (int k = 0; k < 64; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = fp32_to_fp16(((k * 7 + j) & 1) ? -1.0f : 1.0f);
  }

  prepare_logical_16x64x16(s.dram, prog, a, b, 0x200000, 0x210000);
  matmul_fp_ref<16, 16, 64>(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 4));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1200000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "k4"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_matmul_identity();
  test_matmul_ones();
  test_matmul_accumulate_flag();
  test_matmul_multitile_2x2x2();
  test_matmul_random_regression_16x16();
  test_matmul_k4_boundary_stress();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
