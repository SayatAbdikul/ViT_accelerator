#pragma once

// Shared verilator-test helpers for the W8A16 systolic datapath.
// A 16-byte SRAM row now holds 8 FP16 lanes (not 16 INT8 lanes); a logical
// 16-lane row is therefore two SRAM rows. ACCUM holds 4 FP32 elements per
// 16-byte row, unchanged from before (only the bit interpretation flipped
// from INT32 to FP32).

#include "Vtaccel_top.h"
#include "Vtaccel_top___024root.h"
#include "testbench.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace systolic_test {

constexpr int BUF_ABUF_ID  = 0;
constexpr int BUF_WBUF_ID  = 1;
constexpr int BUF_ACCUM_ID = 2;

constexpr int SYS_DIM = 16;

inline uint32_t float_to_bits(float f) { uint32_t b; std::memcpy(&b, &f, sizeof(b)); return b; }
inline float    bits_to_float(uint32_t b) { float f; std::memcpy(&f, &b, sizeof(f)); return f; }

// FP32->FP16 RNE narrowing mirroring fp32_to_fp16_bits in fp32_prim_pkg.sv
// (and numpy.float32(x).astype(np.float16)). Used to encode test inputs.
inline uint16_t fp32_to_fp16(float f) {
  uint32_t b = float_to_bits(f);
  uint32_t sign = (b >> 31) & 0x1;
  uint32_t exp8 = (b >> 23) & 0xFF;
  uint32_t frac23 = b & 0x7FFFFF;
  if (exp8 == 0xFF) {
    if (frac23 != 0) return uint16_t((sign << 15) | (0x1F << 10) | 0x200);
    return uint16_t((sign << 15) | (0x1F << 10));
  }
  if (exp8 == 0 || exp8 < 113) {
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
  uint32_t lost = mant24 & 0x1FFF;
  uint32_t half = 0x1000;
  uint32_t rounded = mant24 >> 13;
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

// FP16->FP32 widen matching fp32_from_fp16_bits (legacy contract: FP16
// inf/NaN clamp to FP32 +/-65504). Test inputs avoid that path so this
// is just exact widening in practice.
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t(h) >> 15) & 0x1;
  uint32_t exp5 = (uint32_t(h) >> 10) & 0x1F;
  uint32_t frac10 = uint32_t(h) & 0x3FF;
  uint32_t out;
  if (exp5 == 0 && frac10 == 0) {
    out = sign << 31;
  } else if (exp5 == 0) {
    int k = 9; while ((frac10 & (1u << k)) == 0) --k;
    uint32_t mant = (frac10 - (1u << k)) << (23 - k);
    out = (sign << 31) | ((uint32_t(k + 103)) << 23) | mant;
  } else if (exp5 == 0x1F) {
    out = sign ? 0xC77FE000u : 0x477FE000u;
  } else {
    out = (sign << 31) | ((exp5 + 112) << 23) | (frac10 << 13);
  }
  return bits_to_float(out);
}

struct Sim {
  std::unique_ptr<Vtaccel_top> dut;
  AXI4SlaveModel dram;

  Sim() : dut(std::make_unique<Vtaccel_top>()), dram(16 * 1024 * 1024) {
    do_reset(dut.get());
  }

  void load_program(const std::vector<uint64_t>& prog) {
    dram.write_program(prog);
  }

  void run(int timeout = 300000) {
    dut->start = 1;
    tick(dut.get(), dram);
    dut->start = 0;
    run_until_halt(dut.get(), dram, timeout);
  }
};

inline void append_set_addr(std::vector<uint64_t>& prog, int reg, uint64_t addr) {
  prog.push_back(insn::SET_ADDR_LO(reg, static_cast<uint32_t>(addr & 0x0FFFFFFFULL)));
  prog.push_back(insn::SET_ADDR_HI(reg, static_cast<uint32_t>((addr >> 28) & 0x0FFFFFFFULL)));
}

inline void append_load_sync(std::vector<uint64_t>& prog, int reg, uint64_t addr,
                             int buf_id, int sram_off, int xfer_len) {
  append_set_addr(prog, reg, addr);
  prog.push_back(insn::LOAD(buf_id, sram_off, xfer_len, reg, 0));
  prog.push_back(insn::SYNC(0b001));
}

// Pack an MxK FP16 matrix in row-major order: row r at byte offset r * K * 2.
// One logical row is K * 2 bytes; SRAM rows are 16 bytes each so each logical
// row spans K/8 SRAM rows.
template <int M, int K>
std::vector<uint8_t> flatten_fp16_rowmajor(const uint16_t (&m)[M][K]) {
  std::vector<uint8_t> out(M * K * 2);
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < K; ++c) {
      uint16_t v = m[r][c];
      out[(r * K + c) * 2 + 0] = uint8_t(v & 0xFF);
      out[(r * K + c) * 2 + 1] = uint8_t((v >> 8) & 0xFF);
    }
  }
  return out;
}

inline void write_dram_bytes(AXI4SlaveModel& dram, uint64_t addr, const std::vector<uint8_t>& bytes) {
  dram.write_bytes(addr, bytes.data(), bytes.size());
}

inline void prepare_logical_16x16(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                  const uint16_t (&a)[16][16], const uint16_t (&b)[16][16],
                                  uint64_t a_addr, uint64_t b_addr,
                                  int abuf_off = 0, int wbuf_off = 0) {
  write_dram_bytes(dram, a_addr, flatten_fp16_rowmajor<16, 16>(a));
  write_dram_bytes(dram, b_addr, flatten_fp16_rowmajor<16, 16>(b));
  // FP16: 16 elems x 2 bytes = 32 bytes per logical row = 2 SRAM rows.
  // 16 rows x 2 = 32 16-byte transfer units per matrix.
  append_load_sync(prog, 0, a_addr, BUF_ABUF_ID, abuf_off, (16 * 16 * 2) / 16);
  append_load_sync(prog, 1, b_addr, BUF_WBUF_ID, wbuf_off, (16 * 16 * 2) / 16);
}

inline void prepare_logical_32x32(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                  const uint16_t (&a)[32][32], const uint16_t (&b)[32][32],
                                  uint64_t a_base, uint64_t b_base,
                                  int abuf_off = 0, int wbuf_off = 0) {
  write_dram_bytes(dram, a_base, flatten_fp16_rowmajor<32, 32>(a));
  write_dram_bytes(dram, b_base, flatten_fp16_rowmajor<32, 32>(b));
  append_load_sync(prog, 0, a_base, BUF_ABUF_ID, abuf_off, (32 * 32 * 2) / 16);
  append_load_sync(prog, 1, b_base, BUF_WBUF_ID, wbuf_off, (32 * 32 * 2) / 16);
}

inline void prepare_logical_16x64x16(AXI4SlaveModel& dram, std::vector<uint64_t>& prog,
                                     const uint16_t (&a)[16][64], const uint16_t (&b)[64][16],
                                     uint64_t a_base, uint64_t b_base,
                                     int abuf_off = 0, int wbuf_off = 0) {
  write_dram_bytes(dram, a_base, flatten_fp16_rowmajor<16, 64>(a));
  write_dram_bytes(dram, b_base, flatten_fp16_rowmajor<64, 16>(b));
  append_load_sync(prog, 0, a_base, BUF_ABUF_ID, abuf_off, (16 * 64 * 2) / 16);
  append_load_sync(prog, 1, b_base, BUF_WBUF_ID, wbuf_off, (64 * 16 * 2) / 16);
}

// ACCUM is 4 FP32 elements per 16-byte row. Tile layout per the controller's
// drain logic: row(i,j) of an MxN tile lives at SRAM row = dst_off + i*4 + j/4,
// lane = j%4. Address arithmetic is identical to the previous INT32 layout
// (only bit interpretation changes from INT32 to FP32).
inline uint32_t read_accum_bits(Vtaccel_top* dut, int dst_off, int i, int j) {
  auto* r = dut->rootp;
  int grp = j / 4;
  int lane = j % 4;
  int row = dst_off + i * 4 + grp;
  return r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
}

inline uint32_t read_accum_bits_32x32(Vtaccel_top* dut, int off, int i, int j) {
  int grp = j / 4;
  int lane = j % 4;
  int row = off + i * 8 + grp;
  auto* r = dut->rootp;
  return r->taccel_top__DOT__u_sram__DOT__u_accum__DOT__mem[row][lane];
}

// Sequential K-loop FP32 oracle (matches systolic_pe's per-cycle widen+mul+add).
// FP16 inputs are widened, multiplied, then added to the running float
// accumulator one K at a time in increasing K order. This is the SAME order
// as software/taccel/golden_model/systolic_w8a16.py.
template <int M, int N, int K>
void matmul_fp_ref(const uint16_t (&a)[M][K], const uint16_t (&b)[K][N], uint32_t (&c)[M][N]) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a32 = fp16_to_fp32(a[i][k]);
        float b32 = fp16_to_fp32(b[k][j]);
        acc = acc + a32 * b32;
      }
      c[i][j] = float_to_bits(acc);
    }
  }
}

template <int M, int N>
bool check_accum_bits(Vtaccel_top* dut, int dst_off, const uint32_t (&exp)[M][N],
                     const char* tag) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      uint32_t got = (M == 32) ? read_accum_bits_32x32(dut, dst_off, i, j)
                               : read_accum_bits(dut, dst_off, i, j);
      if (got != exp[i][j]) {
        std::fprintf(stderr, "%s mismatch i=%d j=%d got=0x%08x exp=0x%08x (%g vs %g)\n",
                     tag, i, j, got, exp[i][j],
                     double(bits_to_float(got)), double(bits_to_float(exp[i][j])));
        return false;
      }
    }
  }
  return true;
}

}  // namespace systolic_test
