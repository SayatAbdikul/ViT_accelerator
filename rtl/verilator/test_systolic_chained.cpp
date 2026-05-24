// Top-level chained-mode MATMUL tests (W8A16 datapath).
//
// SYSTOLIC_ARCH_MODE=1 (chained) is the shipping default. test_systolic.cpp
// already exercises chained mode (it builds with the same default).  This
// file used to also pin a fine-grained cycle-level schedule trace (lane
// progression, SRAM read interleaving) against a per-PE INT8 reference;
// the new W8A16 cadence is multi-cycle per K-step so that trace was deleted
// rather than retrofitted. The bit-exact MATMUL gates below are the
// load-bearing chained-mode coverage; the schedule structure is covered
// by the array-level unit test in test_systolic_array_chained.cpp.

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

static void test_matmul_identity_chained() {
  const char* name = "matmul_chained_identity";
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
  s.run(800000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "chained-identity"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

static void test_matmul_random_chained() {
  const char* name = "matmul_chained_random";
  std::mt19937 rng(24680);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (int tc = 0; tc < 2; ++tc) {
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

    prepare_logical_16x16(s.dram, prog, a, b, 0x160000 + tc * 0x4000, 0x162000 + tc * 0x4000);
    matmul_fp_ref<16, 16, 16>(a, b, exp);

    prog.push_back(insn::CONFIG_TILE(1, 1, 1));
    prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
    prog.push_back(insn::SYNC(0b010));
    prog.push_back(insn::HALT());

    s.load_program(prog);
    s.run(800000);
    expect_clean_halt(name, s.dut.get());
    if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "chained-random"))
      TEST_FAIL(name, "ACCUM mismatch");
  }

  TEST_PASS(name);
}

static void test_matmul_k4_chained() {
  const char* name = "matmul_chained_k4_boundary_stress";
  Sim s;

  uint16_t a[16][64] = {};
  uint16_t b[64][16] = {};
  uint32_t exp[16][16] = {};
  std::vector<uint64_t> prog;

  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 64; ++k)
      a[i][k] = fp32_to_fp16(((i + k) & 1) ? 1.0f : -1.0f);
  for (int k = 0; k < 64; ++k)
    for (int j = 0; j < 16; ++j)
      b[k][j] = fp32_to_fp16(((k * 7 + j) & 1) ? -1.0f : 1.0f);

  prepare_logical_16x64x16(s.dram, prog, a, b, 0x180000, 0x190000);
  matmul_fp_ref<16, 16, 64>(a, b, exp);

  prog.push_back(insn::CONFIG_TILE(1, 1, 4));
  prog.push_back(insn::MATMUL(BUF_ABUF_ID, 0, BUF_WBUF_ID, 0, BUF_ACCUM_ID, 0, 0, 0));
  prog.push_back(insn::SYNC(0b010));
  prog.push_back(insn::HALT());

  s.load_program(prog);
  s.run(1500000);
  expect_clean_halt(name, s.dut.get());
  if (!check_accum_bits<16, 16>(s.dut.get(), 0, exp, "chained-k4"))
    TEST_FAIL(name, "ACCUM mismatch");
  TEST_PASS(name);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_matmul_identity_chained();
  test_matmul_random_chained();
  test_matmul_k4_chained();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
