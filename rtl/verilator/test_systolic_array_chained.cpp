// Unit-level chained systolic-array tests (W8A16 datapath).
//
// The systolic_array module now takes 256-bit a/b_row_data ports holding
// 16 FP16 lanes each. Each PE widens FP16 to FP32 and accumulates an FP32
// MAC (RNE FP32 mul, RNE FP32 add -- NOT a fused FMA).  This test drives
// the array directly and checks the FP32 accumulator bit-by-bit against
// a C++ reference doing the same widen/mul/add sequence on standard
// IEEE-754 binary32 floats (which is RNE on default compiler flags).

#include "Vsystolic_array.h"
#include "verilated.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

static int tests_run = 0;
static int tests_pass = 0;

#define TEST_PASS(name) do { \
  std::printf("PASS: %s\n", name); tests_pass++; tests_run++; \
} while (0)

#define TEST_FAIL(name, msg) do { \
  std::fprintf(stderr, "FAIL: %s - %s\n", name, msg); std::exit(1); \
} while (0)

namespace {

constexpr int SYS_DIM = 16;
constexpr int CHAIN_FLUSH_CYCLES = 2 * (SYS_DIM - 1);
constexpr int CHAIN_TOTAL_STEPS = SYS_DIM + CHAIN_FLUSH_CYCLES;

using Row = std::array<uint16_t, SYS_DIM>;          // 16 FP16 lanes
using AccMatrix = std::array<std::array<uint32_t, SYS_DIM>, SYS_DIM>;  // FP32 bit patterns

inline uint32_t float_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}

inline float bits_float(uint32_t b) {
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
}

// FP16 -> FP32 widen matching fp32_from_fp16_bits in fp32_prim_pkg.sv:
// finite values widen exactly; FP16 inf/NaN clamps to FP32 +/-65504.
// We only use finite normal FP16 in these tests so the clamp path is
// untouched; the helper is still written for completeness.
float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t(h) >> 15) & 0x1;
  uint32_t exp5 = (uint32_t(h) >> 10) & 0x1F;
  uint32_t frac10 = uint32_t(h) & 0x3FF;
  uint32_t out;
  if (exp5 == 0 && frac10 == 0) {
    out = sign << 31;
  } else if (exp5 == 0) {
    // Subnormal: value = frac10 * 2^-24.
    int k = 9;
    while ((frac10 & (1u << k)) == 0) --k;
    uint32_t mant = (frac10 - (1u << k)) << (23 - k);
    uint32_t exp32 = uint32_t(k + 103);
    out = (sign << 31) | (exp32 << 23) | mant;
  } else if (exp5 == 0x1F) {
    // FP16 inf/NaN -> FP32 +/-65504.
    out = sign ? 0xC77FE000u : 0x477FE000u;
  } else {
    // Normal FP16 -> normal FP32: exp += 112, frac << 13.
    uint32_t exp32 = exp5 + 112;
    uint32_t mant = frac10 << 13;
    out = (sign << 31) | (exp32 << 23) | mant;
  }
  return bits_float(out);
}

// FP32 -> FP16 with RNE matching fp32_to_fp16_bits in fp32_prim_pkg.sv
// (used only for building input vectors from floats).
uint16_t fp32_to_fp16(float f) {
  uint32_t b = float_bits(f);
  uint32_t sign = (b >> 31) & 0x1;
  uint32_t exp8 = (b >> 23) & 0xFF;
  uint32_t frac23 = b & 0x7FFFFF;

  if (exp8 == 0xFF) {
    if (frac23 != 0) return uint16_t((sign << 15) | (0x1F << 10) | 0x200);
    return uint16_t((sign << 15) | (0x1F << 10));
  }
  if (exp8 == 0 || (exp8 < 113)) {
    if (exp8 == 0) return uint16_t(sign << 15);
    int e_unb = int(exp8) - 127;
    if (e_unb < -24) return uint16_t(sign << 15);
    uint32_t mant24 = (1u << 23) | frac23;
    int rshift = 13 + (-14 - e_unb);
    uint64_t val = uint64_t(mant24);
    uint32_t lost_mask = (1u << rshift) - 1;
    uint64_t lost = val & lost_mask;
    uint64_t half = uint64_t(1) << (rshift - 1);
    uint64_t rounded = val >> rshift;
    bool tie = (lost == half);
    if (lost > half || (tie && (rounded & 1))) rounded += 1;
    if (rounded == 0x400) return uint16_t((sign << 15) | (1 << 10));
    return uint16_t((sign << 15) | uint16_t(rounded));
  }
  int e_unb = int(exp8) - 127;
  if (e_unb > 15) return uint16_t((sign << 15) | (0x1F << 10));
  uint32_t mant24 = (1u << 23) | frac23;
  int rshift = 13;
  uint32_t lost_mask = (1u << rshift) - 1;
  uint32_t lost = mant24 & lost_mask;
  uint32_t half = 1u << (rshift - 1);
  uint32_t rounded = mant24 >> rshift;
  bool tie = (lost == half);
  if (lost > half || (tie && (rounded & 1))) rounded += 1;
  uint16_t exp5_out;
  uint16_t frac10_out;
  if (rounded == 0x800) {
    exp5_out = uint16_t(e_unb + 15 + 1);
    frac10_out = 0;
  } else {
    exp5_out = uint16_t(e_unb + 15);
    frac10_out = uint16_t(rounded & 0x3FF);
  }
  if (exp5_out >= 0x1F) return uint16_t((sign << 15) | (0x1F << 10));
  return uint16_t((sign << 15) | (exp5_out << 10) | frac10_out);
}

// Build an FP16 row from a small-integer float row (exact representation).
Row build_row(const std::array<float, SYS_DIM>& xs) {
  Row r{};
  for (int i = 0; i < SYS_DIM; ++i) r[i] = fp32_to_fp16(xs[i]);
  return r;
}

// Reference model of the chained array: FP32 accumulators, FP16 forwarded
// operands (held as uint16_t bit patterns, widened to FP32 inside the PE).
struct ChainedArrayRef {
  std::array<std::array<uint16_t, SYS_DIM - 1>, SYS_DIM> a_skew{};
  std::array<std::array<uint16_t, SYS_DIM - 1>, SYS_DIM> b_skew{};
  std::array<std::array<uint16_t, SYS_DIM>, SYS_DIM> a_out{};
  std::array<std::array<uint16_t, SYS_DIM>, SYS_DIM> b_out{};
  AccMatrix acc{};

  void reset() {
    for (int i = 0; i < SYS_DIM; ++i) {
      for (int j = 0; j < SYS_DIM; ++j) {
        a_out[i][j] = 0;
        b_out[i][j] = 0;
        acc[i][j] = 0;
      }
      for (int s = 0; s < SYS_DIM - 1; ++s) {
        a_skew[i][s] = 0;
        b_skew[i][s] = 0;
      }
    }
  }

  void step(const Row& a_row, const Row& b_row, bool step_en, bool clear_acc) {
    Row a_edge{};
    Row b_edge{};
    std::array<std::array<uint16_t, SYS_DIM>, SYS_DIM> pe_a_in{};
    std::array<std::array<uint16_t, SYS_DIM>, SYS_DIM> pe_b_in{};
    auto next_a_out = a_out;
    auto next_b_out = b_out;
    auto next_acc = acc;
    auto next_a_skew = a_skew;
    auto next_b_skew = b_skew;

    for (int i = 0; i < SYS_DIM; ++i) {
      a_edge[i] = (i == 0) ? a_row[i] : a_skew[i][i - 1];
      b_edge[i] = (i == 0) ? b_row[i] : b_skew[i][i - 1];
    }

    for (int i = 0; i < SYS_DIM; ++i) {
      for (int j = 0; j < SYS_DIM; ++j) {
        pe_a_in[i][j] = (j == 0) ? a_edge[i] : a_out[i][j - 1];
        pe_b_in[i][j] = (i == 0) ? b_edge[j] : b_out[i - 1][j];
        next_a_out[i][j] = pe_a_in[i][j];
        next_b_out[i][j] = pe_b_in[i][j];

        if (clear_acc) {
          next_acc[i][j] = 0;
        } else if (step_en) {
          float a32 = fp16_to_fp32(pe_a_in[i][j]);
          float b32 = fp16_to_fp32(pe_b_in[i][j]);
          float prod = a32 * b32;
          float acc_cur = bits_float(acc[i][j]);
          float acc_new = acc_cur + prod;
          next_acc[i][j] = float_bits(acc_new);
        }
      }
    }

    if (clear_acc) {
      for (int i = 0; i < SYS_DIM; ++i) {
        for (int s = 0; s < SYS_DIM - 1; ++s) {
          next_a_skew[i][s] = 0;
          next_b_skew[i][s] = 0;
        }
      }
    } else if (step_en) {
      for (int i = 0; i < SYS_DIM; ++i) {
        next_a_skew[i][0] = a_row[i];
        next_b_skew[i][0] = b_row[i];
        for (int s = 1; s < SYS_DIM - 1; ++s) {
          next_a_skew[i][s] = a_skew[i][s - 1];
          next_b_skew[i][s] = b_skew[i][s - 1];
        }
      }
    }

    a_out = next_a_out;
    b_out = next_b_out;
    acc = next_acc;
    a_skew = next_a_skew;
    b_skew = next_b_skew;
  }
};

// Pack 16 FP16 lanes (256 bits) into the Verilator port. The DUT port is
// declared as logic [SYS_DIM*16-1:0]; Verilator emits this as a VlWide
// array of 32-bit words (lane 0 in the low bits).
void set_row_data(const Row& row, VlWide<8>& port) {
  for (int word = 0; word < 8; ++word) {
    uint32_t packed = 0;
    for (int half = 0; half < 2; ++half) {
      int lane = word * 2 + half;
      packed |= uint32_t(row[lane]) << (half * 16);
    }
    port[word] = packed;
  }
}

void drive_rows(Vsystolic_array* dut, const Row& a_row, const Row& b_row) {
  set_row_data(a_row, dut->a_row_data);
  set_row_data(b_row, dut->b_row_data);
}

void tick(Vsystolic_array* dut) {
  dut->clk = 0;
  dut->eval();
  dut->clk = 1;
  dut->eval();
}

void reset(Vsystolic_array* dut, ChainedArrayRef& ref) {
  Row zero{};
  dut->rst_n = 0;
  dut->step_en = 0;
  dut->clear_acc = 0;
  drive_rows(dut, zero, zero);
  for (int i = 0; i < 4; ++i)
    tick(dut);
  dut->rst_n = 1;
  dut->step_en = 0;
  dut->clear_acc = 1;
  tick(dut);
  ref.reset();
  dut->clear_acc = 0;
  dut->step_en = 0;
  tick(dut);
}

void compare_acc(const char* name, Vsystolic_array* dut, const ChainedArrayRef& ref, int cycle) {
  for (int i = 0; i < SYS_DIM; ++i) {
    for (int j = 0; j < SYS_DIM; ++j) {
      int idx = i * SYS_DIM + j;
      uint32_t got = static_cast<uint32_t>(dut->acc_flat[idx]);
      uint32_t exp = ref.acc[i][j];
      if (got != exp) {
        std::fprintf(stderr,
                     "%s cycle=%d pe=(%d,%d) got=0x%08x exp=0x%08x (%g vs %g)\n",
                     name, cycle, i, j, got, exp,
                     double(bits_float(got)), double(bits_float(exp)));
        TEST_FAIL(name, "chained array mismatch");
      }
    }
  }
}

void step_and_check(const char* name, Vsystolic_array* dut, ChainedArrayRef& ref,
                    const Row& a_row, const Row& b_row, int cycle,
                    bool step_en = true, bool clear_acc = false) {
  dut->rst_n = 1;
  dut->step_en = step_en ? 1 : 0;
  dut->clear_acc = clear_acc ? 1 : 0;
  drive_rows(dut, a_row, b_row);
  tick(dut);
  ref.step(a_row, b_row, step_en, clear_acc);
  compare_acc(name, dut, ref, cycle);
}

void test_single_impulse_timing() {
  const char* name = "systolic_array_chained_single_impulse";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  Row a_row{};
  Row b_row{};
  constexpr int target_row = 7;
  constexpr int target_col = 11;
  const float a_val = 3.0f;
  const float b_val = 5.0f;
  int first_nonzero_cycle = -1;

  a_row[target_row] = fp32_to_fp16(a_val);
  b_row[target_col] = fp32_to_fp16(b_val);
  step_and_check(name, &dut, ref, a_row, b_row, 0);

  for (int cycle = 1; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
    uint32_t got = static_cast<uint32_t>(dut.acc_flat[target_row * SYS_DIM + target_col]);
    if ((first_nonzero_cycle < 0) && (got != 0))
      first_nonzero_cycle = cycle;
  }

  if (first_nonzero_cycle != (target_row + target_col)) {
    std::fprintf(stderr, "first nonzero cycle got=%d exp=%d\n",
                 first_nonzero_cycle, target_row + target_col);
    TEST_FAIL(name, "unexpected chained arrival timing");
  }
  uint32_t final_bits = static_cast<uint32_t>(dut.acc_flat[target_row * SYS_DIM + target_col]);
  if (bits_float(final_bits) != a_val * b_val)
    TEST_FAIL(name, "final impulse value mismatch");

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

void test_identity_stream() {
  const char* name = "systolic_array_chained_identity_stream";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  float a[16][16] = {};
  float eye[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<float>((i * 3 + j) & 0x7F);   // exact in FP16/FP32
      eye[i][j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = fp32_to_fp16(a[lane][cycle]);
      b_row[lane] = fp32_to_fp16(eye[cycle][lane]);
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  for (int i = 0; i < SYS_DIM; ++i) {
    for (int j = 0; j < SYS_DIM; ++j) {
      uint32_t got = static_cast<uint32_t>(dut.acc_flat[i * SYS_DIM + j]);
      float exp = (i == j) ? a[i][j] : a[i][j];  // result of A * I
      uint32_t exp_bits = float_bits(exp);
      if (got != exp_bits) {
        std::fprintf(stderr, "identity result mismatch i=%d j=%d got=0x%08x exp=0x%08x\n",
                     i, j, got, exp_bits);
        TEST_FAIL(name, "identity stream mismatch");
      }
    }
  }

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

void test_small_integer_stream() {
  const char* name = "systolic_array_chained_small_int_stream";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  // Inputs in [-8, +7] so 16-element sums of products stay exactly
  // representable in float32 (no rounding effects to mask the bit-exact
  // gate). Stresses both signs and the full 16-lane width.
  float a[16][16] = {};
  float b[16][16] = {};

  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 16; ++k)
      a[i][k] = static_cast<float>(((i + k) & 0x0F) - 8);
  }
  for (int k = 0; k < 16; ++k) {
    for (int j = 0; j < 16; ++j)
      b[k][j] = static_cast<float>(((k * 3 + j * 5) & 0x0F) - 8);
  }

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = fp32_to_fp16(a[lane][cycle]);
      b_row[lane] = fp32_to_fp16(b[cycle][lane]);
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  TEST_PASS(name);
}

void test_random_seeded() {
  const char* name = "systolic_array_chained_random_seeded";
  Vsystolic_array dut;
  ChainedArrayRef ref;
  reset(&dut, ref);

  std::mt19937 rng(123456);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  float a[16][16] = {};
  float b[16][16] = {};

  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 16; ++k)
      a[i][k] = dist(rng);
  for (int k = 0; k < 16; ++k)
    for (int j = 0; j < 16; ++j)
      b[k][j] = dist(rng);

  for (int cycle = 0; cycle < SYS_DIM; ++cycle) {
    Row a_row{};
    Row b_row{};
    for (int lane = 0; lane < SYS_DIM; ++lane) {
      a_row[lane] = fp32_to_fp16(a[lane][cycle]);
      b_row[lane] = fp32_to_fp16(b[cycle][lane]);
    }
    step_and_check(name, &dut, ref, a_row, b_row, cycle);
  }

  for (int cycle = SYS_DIM; cycle < CHAIN_TOTAL_STEPS; ++cycle) {
    Row zero{};
    step_and_check(name, &dut, ref, zero, zero, cycle);
  }

  Row zero{};
  step_and_check(name, &dut, ref, zero, zero, CHAIN_TOTAL_STEPS);
  TEST_PASS(name);
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  test_single_impulse_timing();
  test_identity_stream();
  test_small_integer_stream();
  test_random_seeded();

  std::printf("\n%d / %d tests passed\n", tests_pass, tests_run);
  return (tests_pass == tests_run) ? 0 : 1;
}
