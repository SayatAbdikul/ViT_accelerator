"""cocotb tests for the W8A16 blocking helper engine.

The helper engine supports:
  * BUF_COPY  — flat byte copy and FP16-element transpose.
  * VADD      — FP16 + FP16 → FP16 (ABUF mode) and FP32 + FP16-broadcast →
                FP32 (ACCUM bias / attention-mask mode).
  * SCALE_MUL — FP32-internal × FP16 scale, narrowed to FP16 when dst=ABUF,
                kept FP32 when dst=ACCUM.

Removed in W8A16 (raises FAULT_UNSUPPORTED_OP): REQUANT (0x0B),
REQUANT_PC (0x11), DEQUANT_ADD (0x13).

Bit-exact oracles use fp32_prim_ref to match RTL fp32_prim_pkg byte-for-byte.

Requires SIM=verilator: fp32_prim_pkg uses constructs Icarus iverilog cannot
parse.
"""

import os
import sys

import cocotb
import numpy as np

# Make the repo's `software/` package importable when cocotb runs from rtl/cocotb.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from software.taccel.utils import fp32_prim_ref as fpr  # noqa: E402

from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, SET_SCALE, LOAD, STORE,
    BUF_COPY, SCALE_MUL, VADD, REQUANT, REQUANT_PC, DEQUANT_ADD,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)
from utils.testbench import set_addr, setup_test, wait_halt


def _fp16_bytes(arr) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float16).tobytes()


def _fp32_bytes(arr) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _assert_bytes_equal(got: bytes, exp: bytes, tag: str):
    if got != exp:
        diff_idx = next(i for i in range(min(len(got), len(exp))) if got[i] != exp[i])
        ctx_lo = max(0, diff_idx - 4)
        ctx_hi = min(len(got), diff_idx + 8)
        raise AssertionError(
            f"{tag}: first mismatch at byte {diff_idx} "
            f"(got=0x{got[diff_idx]:02x} exp=0x{exp[diff_idx]:02x}); "
            f"got[{ctx_lo}:{ctx_hi}]={got[ctx_lo:ctx_hi].hex()} "
            f"exp[{ctx_lo}:{ctx_hi}]={exp[ctx_lo:ctx_hi].hex()}"
        )


def _fp16_add_oracle(a_fp16: np.ndarray, b_fp16: np.ndarray) -> np.ndarray:
    """Match RTL VADD ABUF: widen each to FP32, add, narrow to FP16."""
    a_b = a_fp16.astype(np.float32)
    b_b = b_fp16.astype(np.float32)
    return (a_b + b_b).astype(np.float16)


def _fp32_bias_add_oracle(accum_fp32: np.ndarray, bias_fp16: np.ndarray) -> np.ndarray:
    """Match RTL VADD ACCUM: FP32 + FP16-widened-broadcast → FP32."""
    return (accum_fp32 + bias_fp16.astype(np.float32)).astype(np.float32)


def _fp16_scale_oracle(x_fp16: np.ndarray, scale_fp16: np.float16) -> np.ndarray:
    """ABUF→ABUF SCALE_MUL: widen, mul, narrow."""
    return (x_fp16.astype(np.float32) * np.float32(scale_fp16)).astype(np.float16)


def _fp32_scale_oracle(x_fp32: np.ndarray, scale_fp16: np.float16) -> np.ndarray:
    """ACCUM→ACCUM SCALE_MUL: stays FP32 (scale widened from FP16)."""
    return (x_fp32 * np.float32(scale_fp16)).astype(np.float32)


def _accum_to_abuf_oracle(x_fp32: np.ndarray, scale_fp16: np.float16) -> np.ndarray:
    """ACCUM→ABUF SCALE_MUL: FP32-mul then FP16 narrow."""
    return (x_fp32 * np.float32(scale_fp16)).astype(np.float16)


# ─── BUF_COPY ──────────────────────────────────────────────────────────────

@cocotb.test()
async def test_buf_copy_flat_roundtrip(dut):
    """Flat BUF_COPY preserves bytes between buffers."""
    src_addr = 0x10000
    dst_addr = 0x12000
    src = bytes((0x20 + 7 * i) & 0xFF for i in range(48))
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 3, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 8, 3, 0, 0),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, 8, 3, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: src})
    await wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(src)]) == src


@cocotb.test()
async def test_buf_copy_overlap_roundtrip(dut):
    """Backward memmove for overlapping in-place compaction."""
    src_addr = 0x14000
    dst_addr = 0x15000
    src = bytearray((0x51 + 11 * i) & 0xFF for i in range(96))
    expected = bytearray(src)
    expected[16:64] = expected[32:80]
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, 6, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 2, BUF_ABUF, 1, 3, 0, 0),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, 0, 6, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: bytes(src)})
    await wait_halt(dut)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    assert bytes(dram.mem[dst_addr:dst_addr + len(expected)]) == bytes(expected)


@cocotb.test()
async def test_buf_copy_transpose_fp16_roundtrip(dut):
    """FP16-element transpose: [16, 16] FP16 → [16, 16] FP16.

    The codegen emits BUF_COPY transpose to make K^T from K (FP16); the RTL
    must treat 16-byte SRAM rows as 8 FP16 elements and transpose at element
    granularity, not byte granularity.
    """
    src_addr = 0x18000
    dst_addr = 0x19000
    rows, cols = 16, 16  # both multiples of 8 elements
    rng = np.random.default_rng(101)
    src_arr = rng.uniform(-3.0, 3.0, (rows, cols)).astype(np.float16)
    expected = np.ascontiguousarray(src_arr.T)
    src_bytes = src_arr.tobytes()
    # length in 16-byte units = total_bytes / 16; src_rows in 16-row units.
    length_units = (rows * cols * 2) // 16
    src_rows_field = rows // 16
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, length_units, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 0, length_units, src_rows_field, 1),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, 0, length_units, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: src_bytes})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        expected.tobytes(),
        "transpose-fp16",
    )


@cocotb.test()
async def test_buf_copy_transpose_fp16_rect(dut):
    """Rectangular FP16 transpose: [32, 16] → [16, 32]."""
    src_addr = 0x1B000
    dst_addr = 0x1C000
    rows, cols = 32, 16
    rng = np.random.default_rng(202)
    src_arr = rng.uniform(-2.0, 2.0, (rows, cols)).astype(np.float16)
    expected = np.ascontiguousarray(src_arr.T)
    src_bytes = src_arr.tobytes()
    length_units = (rows * cols * 2) // 16
    src_rows_field = rows // 16
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, length_units, 0, 0),
        SYNC(0b001),
        BUF_COPY(BUF_ABUF, 0, BUF_WBUF, 0, length_units, src_rows_field, 1),
        *set_addr(1, dst_addr),
        STORE(BUF_WBUF, 0, length_units, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: src_bytes})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        expected.tobytes(),
        "transpose-rect",
    )


# ─── VADD ──────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_vadd_fp16_abuf(dut):
    """VADD ABUF+WBUF → ABUF: FP16 + FP16 → FP16 (bit-exact)."""
    src_a_addr = 0x1D000
    src_b_addr = 0x1E000
    dst_addr = 0x1F000
    M, N = 16, 16
    rng = np.random.default_rng(303)
    a_fp16 = rng.uniform(-3.0, 3.0, (M, N)).astype(np.float16)
    b_fp16 = rng.uniform(-3.0, 3.0, (M, N)).astype(np.float16)
    expected = _fp16_add_oracle(a_fp16, b_fp16)

    # FP16 row count = M * N / 8.
    rows_units = (M * N * 2) // 16
    prog = [
        *set_addr(0, src_a_addr),
        LOAD(BUF_ABUF, 0, rows_units, 0, 0),
        SYNC(0b001),
        *set_addr(1, src_b_addr),
        LOAD(BUF_WBUF, 0, rows_units, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        VADD(BUF_ABUF, 0, BUF_WBUF, 0, BUF_ABUF, rows_units, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ABUF, rows_units, rows_units, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog,
                            dram_writes={src_a_addr: _fp16_bytes(a_fp16),
                                         src_b_addr: _fp16_bytes(b_fp16)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp16_bytes(expected),
        "vadd-fp16",
    )


@cocotb.test()
async def test_vadd_accum_bias_broadcast(dut):
    """VADD ACCUM+WBUF → ACCUM: FP32 + FP16-broadcast row → FP32 (bit-exact).

    The bias FP16 row spans 2 ACCUM 4-column chunks per M row; this exercises
    the c4_half_q half-row selection in H_VACC_WRITE.
    """
    accum_addr = 0x20000
    bias_addr = 0x21000
    dst_addr = 0x22000
    M, N = 16, 16
    rng = np.random.default_rng(404)
    accum_fp32 = rng.uniform(-5.0, 5.0, (M, N)).astype(np.float32)
    bias_fp16 = rng.uniform(-1.0, 1.0, (N,)).astype(np.float16)
    expected = _fp32_bias_add_oracle(accum_fp32, bias_fp16)

    accum_rows = (M * N * 4) // 16   # FP32 -> 4 lanes per 16-byte row
    bias_rows = (N * 2) // 16        # one FP16 row of N cols
    out_off = 256
    prog = [
        *set_addr(0, accum_addr),
        LOAD(BUF_ACCUM, 0, accum_rows, 0, 0),
        SYNC(0b001),
        *set_addr(1, bias_addr),
        LOAD(BUF_WBUF, 0, bias_rows, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        VADD(BUF_ACCUM, 0, BUF_WBUF, 0, BUF_ACCUM, out_off, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ACCUM, out_off, accum_rows, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog,
                            dram_writes={accum_addr: _fp32_bytes(accum_fp32),
                                         bias_addr: _fp16_bytes(bias_fp16)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp32_bytes(expected),
        "vadd-bias-fp32",
    )


@cocotb.test()
async def test_vadd_attention_mask_underflow(dut):
    """The W8A16 codegen broadcasts an FP16 mask row (-65504 in padded cols)
    via VADD ACCUM+WBUF. After downstream softmax the masked positions must
    bit-exactly underflow to FP16 +0.0. Here we just verify the FP32 ACCUM
    bit pattern carries -65504 (FP16 widened) into masked columns and 0.0
    elsewhere.
    """
    accum_addr = 0x23000
    bias_addr = 0x24000
    dst_addr = 0x25000
    M, N = 16, 16
    accum_fp32 = np.zeros((M, N), dtype=np.float32)
    mask_fp16 = np.zeros(N, dtype=np.float16)
    mask_fp16[N // 2:] = np.float16(-65504.0)
    expected = _fp32_bias_add_oracle(accum_fp32, mask_fp16)

    accum_rows = (M * N * 4) // 16
    bias_rows = (N * 2) // 16
    out_off = 384
    prog = [
        *set_addr(0, accum_addr),
        LOAD(BUF_ACCUM, 0, accum_rows, 0, 0),
        SYNC(0b001),
        *set_addr(1, bias_addr),
        LOAD(BUF_WBUF, 0, bias_rows, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        VADD(BUF_ACCUM, 0, BUF_WBUF, 0, BUF_ACCUM, out_off, 0),
        *set_addr(2, dst_addr),
        STORE(BUF_ACCUM, out_off, accum_rows, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog,
                            dram_writes={accum_addr: _fp32_bytes(accum_fp32),
                                         bias_addr: _fp16_bytes(mask_fp16)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp32_bytes(expected),
        "vadd-attn-mask",
    )


# ─── SCALE_MUL ─────────────────────────────────────────────────────────────

@cocotb.test()
async def test_scale_mul_abuf_fp16(dut):
    """SCALE_MUL ABUF → ABUF (FP16 widen × scale, narrow). 1-read-1-write."""
    src_addr = 0x26000
    dst_addr = 0x27000
    M, N = 16, 16
    scale_fp16 = np.float16(-0.5)
    scale_bits = int(scale_fp16.view(np.uint16))
    rng = np.random.default_rng(505)
    x_fp16 = rng.uniform(-2.0, 2.0, (M, N)).astype(np.float16)
    expected = _fp16_scale_oracle(x_fp16, scale_fp16)

    rows_units = (M * N * 2) // 16
    out_off = 128
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, 0, rows_units, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SET_SCALE(2, scale_bits),
        SCALE_MUL(BUF_ABUF, 0, BUF_ABUF, out_off, 2),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, out_off, rows_units, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp16_bytes(x_fp16)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp16_bytes(expected),
        "scale-mul-fp16",
    )


@cocotb.test()
async def test_scale_mul_accum_fp32(dut):
    """SCALE_MUL ACCUM → ACCUM (FP32 × FP16-widened scale). 1-read-1-write."""
    src_addr = 0x28000
    dst_addr = 0x29000
    M, N = 16, 16
    scale_fp16 = np.float16(3.0)
    scale_bits = int(scale_fp16.view(np.uint16))
    rng = np.random.default_rng(606)
    x_fp32 = rng.uniform(-10.0, 10.0, (M, N)).astype(np.float32)
    expected = _fp32_scale_oracle(x_fp32, scale_fp16)

    rows_units = (M * N * 4) // 16
    out_off = 256
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, rows_units, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SET_SCALE(3, scale_bits),
        SCALE_MUL(BUF_ACCUM, 0, BUF_ACCUM, out_off, 3),
        *set_addr(1, dst_addr),
        STORE(BUF_ACCUM, out_off, rows_units, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp32_bytes(x_fp32)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp32_bytes(expected),
        "scale-mul-fp32",
    )


@cocotb.test()
async def test_scale_mul_accum_to_abuf_narrow(dut):
    """SCALE_MUL ACCUM → ABUF: FP32 × scale → FP16 narrow (2-reads-1-write).

    This is the W8A16 _accum_to_abuf commit path with scale=1.0; the codegen
    uses sreg 15. The 2-read-1-write FSM combines 2 ACCUM rows (8 FP32
    columns) into 1 ABUF row (8 FP16 columns).
    """
    src_addr = 0x2A000
    dst_addr = 0x2B000
    M, N = 16, 16
    scale_fp16 = np.float16(1.0)
    scale_bits = int(scale_fp16.view(np.uint16))
    rng = np.random.default_rng(707)
    x_fp32 = rng.uniform(-100.0, 100.0, (M, N)).astype(np.float32)
    expected = _accum_to_abuf_oracle(x_fp32, scale_fp16)

    accum_rows = (M * N * 4) // 16
    abuf_rows = (M * N * 2) // 16
    out_off = 320
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, accum_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SET_SCALE(4, scale_bits),
        SCALE_MUL(BUF_ACCUM, 0, BUF_ABUF, out_off, 4),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, out_off, abuf_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp32_bytes(x_fp32)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp16_bytes(expected),
        "scale-mul-narrow",
    )


@cocotb.test()
async def test_scale_mul_accum_to_abuf_nonunit_scale(dut):
    """SCALE_MUL ACCUM → ABUF with scale ≠ 1.0 (covers the scale multiply
    in the narrowing path; ensures fp32_mul_bits is wired correctly)."""
    src_addr = 0x2C000
    dst_addr = 0x2D000
    M, N = 16, 16
    scale_fp16 = np.float16(0.25)
    scale_bits = int(scale_fp16.view(np.uint16))
    rng = np.random.default_rng(808)
    x_fp32 = rng.uniform(-50.0, 50.0, (M, N)).astype(np.float32)
    expected = _accum_to_abuf_oracle(x_fp32, scale_fp16)

    accum_rows = (M * N * 4) // 16
    abuf_rows = (M * N * 2) // 16
    out_off = 448
    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, 0, accum_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SET_SCALE(5, scale_bits),
        SCALE_MUL(BUF_ACCUM, 0, BUF_ABUF, out_off, 5),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, out_off, abuf_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp32_bytes(x_fp32)})
    await wait_halt(dut, max_cycles=400_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    _assert_bytes_equal(
        bytes(dram.mem[dst_addr:dst_addr + expected.nbytes]),
        _fp16_bytes(expected),
        "scale-mul-narrow-quarter",
    )


# ─── Dropped opcodes raise FAULT_UNSUPPORTED_OP ────────────────────────────

async def _expect_fault(dut, prog):
    await setup_test(dut, prog)
    await wait_halt(dut, max_cycles=8000)
    assert int(dut.fault.value) == 1
    assert int(dut.fault_code.value) == 0x6  # FAULT_UNSUPPORTED_OP


@cocotb.test()
async def test_requant_raises_fault(dut):
    """OP_REQUANT (0x0B) is unsupported in W8A16 RTL."""
    prog = [
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(0, 0x3C00),
        REQUANT(BUF_ACCUM, 0, BUF_ABUF, 0, 0),
        HALT(),
    ]
    await _expect_fault(dut, prog)


@cocotb.test()
async def test_requant_pc_raises_fault(dut):
    """OP_REQUANT_PC (0x11) is unsupported in W8A16 RTL."""
    prog = [
        CONFIG_TILE(1, 1, 1),
        REQUANT_PC(BUF_ACCUM, 0, BUF_WBUF, 0, BUF_ABUF, 0, 0),
        HALT(),
    ]
    await _expect_fault(dut, prog)


@cocotb.test()
async def test_dequant_add_raises_fault(dut):
    """OP_DEQUANT_ADD (0x13) is unsupported in W8A16 RTL."""
    prog = [
        CONFIG_TILE(1, 1, 1),
        SET_SCALE(0, 0x3C00),
        SET_SCALE(1, 0x3C00),
        DEQUANT_ADD(BUF_ACCUM, 0, BUF_ABUF, 0, BUF_WBUF, 0, 0),
        HALT(),
    ]
    await _expect_fault(dut, prog)
