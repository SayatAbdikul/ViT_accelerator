"""cocotb tests for the W8A16 SFU engine.

The SFU has FP16 ABUF endpoints (widen on read, narrow on write) and FP32
ACCUM endpoints (bit-reinterpretation, no scale mul). All internal math
remains FP32 via fp32_prim_pkg. These tests are byte-exact against the
software golden in software/taccel/golden_model/sfu_w8a16.py — every FP32
step in the oracle is a correctly-rounded numpy float32 operation that
mirrors RTL bit-by-bit (verified by software/tests/test_w8a16_simulator.py
and fp32_prim_ref's self-checks).

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

from utils.dram_model import DramModel
from utils.insn_builder import (
    HALT, SYNC, CONFIG_TILE, LOAD, STORE, SOFTMAX, LAYERNORM, GELU,
    BUF_ABUF, BUF_WBUF, BUF_ACCUM,
)
from utils.testbench import set_addr, setup_test, wait_halt


def _fp16_bytes(arr_fp16: np.ndarray) -> bytes:
    """Row-major little-endian bytes of an FP16 array (machine assumed LE)."""
    return np.ascontiguousarray(arr_fp16, dtype=np.float16).tobytes()


def _fp32_bytes(arr_fp32: np.ndarray) -> bytes:
    """Row-major little-endian bytes of an FP32 array."""
    return np.ascontiguousarray(arr_fp32, dtype=np.float32).tobytes()


def _softmax_oracle(x_fp32: np.ndarray) -> np.ndarray:
    """Match RTL: row-wise softmax in FP32, then narrow to FP16.

    fpr.fp32_exp_arr is correctly rounded vs scalar fp32_exp; sequential sum
    via fpr.fp32_sum_rows matches the RTL's left-fold reduction; FP32 div is
    IEEE binary32 (correctly rounded). FP32 -> FP16 via numpy astype matches
    fp32_to_fp16_bits (tested in rtl/verilator/test_fp32_prims.cpp).
    """
    x_shifted = (x_fp32 - x_fp32.max(axis=-1, keepdims=True)).astype(np.float32)
    exp_x = fpr.fp32_exp_arr(x_shifted)
    denom = fpr.fp32_sum_rows(exp_x)[:, None]
    y = (exp_x / denom).astype(np.float32)
    return y.astype(np.float16)


def _layernorm_oracle(x_fp32: np.ndarray, gamma_fp32: np.ndarray,
                      beta_fp32: np.ndarray) -> np.ndarray:
    eps = np.float32(1e-6)
    mean = fpr.fp32_mean_rows(x_fp32)[:, None]
    var = fpr.fp32_var_rows(x_fp32, mean)[:, None]
    denom = np.sqrt((var + eps).astype(np.float32), dtype=np.float32)
    x_norm = ((x_fp32 - mean).astype(np.float32) / denom).astype(np.float32)
    y = ((x_norm * gamma_fp32).astype(np.float32) + beta_fp32).astype(np.float32)
    return y.astype(np.float16)


def _gelu_oracle(x_fp32: np.ndarray) -> np.ndarray:
    return fpr.fp32_gelu_arr(x_fp32).astype(np.float16)


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


@cocotb.test()
async def test_softmax_abuf_small(dut):
    """SOFTMAX with FP16 ABUF input, FP16 ABUF output."""
    M, N = 16, 32  # n_tiles=2, fits in SFU_MAX_ROW_ELEMS
    src_addr = 0x30000
    dst_addr = 0x34000
    src_off_u = 0       # ABUF row offset (units of 16-byte rows)
    dst_off_u = 256

    rng = np.random.default_rng(11)
    x = rng.uniform(-3.0, 3.0, (M, N)).astype(np.float16)
    expected = _fp16_bytes(_softmax_oracle(x.astype(np.float32)))

    # FP16 row width = 8 elements -> N/8 rows per logical row, M total logical rows.
    in_rows = (M * N * 2) // 16
    out_rows = (M * N * 2) // 16

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SOFTMAX(BUF_ABUF, src_off_u, BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp16_bytes(x)})
    await wait_halt(dut, max_cycles=800_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "softmax-abuf")


@cocotb.test()
async def test_softmax_accum_fp32(dut):
    """SOFTMAX with FP32 ACCUM input, FP16 ABUF output."""
    M, N = 16, 32
    src_addr = 0x36000
    dst_addr = 0x3A000
    src_off_u = 0
    dst_off_u = 384

    rng = np.random.default_rng(22)
    x = rng.uniform(-4.0, 4.0, (M, N)).astype(np.float32)
    expected = _fp16_bytes(_softmax_oracle(x))

    in_rows = (M * N * 4) // 16  # FP32 -> 4 elements per 16-byte row
    out_rows = (M * N * 2) // 16

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SOFTMAX(BUF_ACCUM, src_off_u, BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp32_bytes(x)})
    await wait_halt(dut, max_cycles=800_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "softmax-accum")


@cocotb.test()
async def test_softmax_attention_mask_underflow(dut):
    """Attention masking: FP16 -65504.0 underflows exp() to FP32 +0.0.

    The W8A16 codegen emits -65504 as the FP16 attention-mask "-inf" sentinel.
    After widen to FP32 and exp(), masked positions must contribute exactly
    zero to the softmax denominator, and the resulting FP16 probability bit
    pattern must be 0x0000.
    """
    M, N = 16, 16
    src_addr = 0x3C000
    dst_addr = 0x3E000
    src_off_u = 0
    dst_off_u = 512

    # Each row: half real values, half masked. Masked positions should yield 0.
    x = np.zeros((M, N), dtype=np.float16)
    rng = np.random.default_rng(33)
    real_vals = rng.uniform(-2.0, 2.0, (M, N // 2)).astype(np.float16)
    x[:, :N // 2] = real_vals
    x[:, N // 2:] = np.float16(-65504.0)

    expected = _fp16_bytes(_softmax_oracle(x.astype(np.float32)))

    in_rows = (M * N * 2) // 16
    out_rows = in_rows

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        SOFTMAX(BUF_ABUF, src_off_u, BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp16_bytes(x)})
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "softmax-mask-underflow")

    # Spot-check: masked columns must be FP16 +0.0 bit pattern.
    got_fp16 = np.frombuffer(got, dtype=np.float16).reshape(M, N)
    masked = got_fp16[:, N // 2:].view(np.uint16)
    assert np.all(masked == 0), \
        f"masked columns not FP16 zero: got unique = {np.unique(masked).tolist()}"


@cocotb.test()
async def test_layernorm_abuf(dut):
    """LAYERNORM with FP16 ABUF input, FP16 gamma/beta in WBUF, FP16 ABUF output."""
    M, N = 16, 64  # n_tiles=4
    src_addr = 0x40000
    gb_addr = 0x44000
    dst_addr = 0x48000
    src_off_u = 0
    gb_off_u = 0
    dst_off_u = 768

    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, (M, N)).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, (N,)).astype(np.float16)
    beta = rng.uniform(-0.3, 0.3, (N,)).astype(np.float16)

    expected = _fp16_bytes(
        _layernorm_oracle(x.astype(np.float32),
                          gamma.astype(np.float32),
                          beta.astype(np.float32))
    )

    in_rows = (M * N * 2) // 16
    gb_bytes = _fp16_bytes(gamma) + _fp16_bytes(beta)
    gb_rows = len(gb_bytes) // 16
    out_rows = in_rows

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        *set_addr(1, gb_addr),
        LOAD(BUF_WBUF, gb_off_u, gb_rows, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        LAYERNORM(BUF_ABUF, src_off_u, BUF_WBUF, gb_off_u,
                  BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(2, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 2, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(
        dut, prog,
        dram_writes={src_addr: _fp16_bytes(x), gb_addr: gb_bytes},
    )
    await wait_halt(dut, max_cycles=900_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "layernorm-abuf")


@cocotb.test()
async def test_gelu_abuf(dut):
    """GELU with FP16 ABUF input, FP16 ABUF output."""
    M, N = 16, 16
    src_addr = 0x4C000
    dst_addr = 0x4E000
    src_off_u = 0
    dst_off_u = 1024

    x = np.array([[((j - 8) * 0.25 + i * 0.0625) for j in range(N)] for i in range(M)],
                 dtype=np.float16)
    expected = _fp16_bytes(_gelu_oracle(x.astype(np.float32)))

    in_rows = (M * N * 2) // 16
    out_rows = in_rows

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ABUF, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        GELU(BUF_ABUF, src_off_u, BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp16_bytes(x)})
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "gelu-abuf")


@cocotb.test()
async def test_gelu_accum_fp32(dut):
    """GELU with FP32 ACCUM input, FP16 ABUF output (commits FP32 -> FP16)."""
    M, N = 16, 16
    src_addr = 0x50000
    dst_addr = 0x52000
    src_off_u = 0
    dst_off_u = 1280

    x = np.array([[((j - 8) * 0.5 + i * 0.125) for j in range(N)] for i in range(M)],
                 dtype=np.float32)
    expected = _fp16_bytes(_gelu_oracle(x))

    in_rows = (M * N * 4) // 16  # 4 FP32 per 16-byte row
    out_rows = (M * N * 2) // 16

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, src_off_u, in_rows, 0, 0),
        SYNC(0b001),
        CONFIG_TILE(M // 16, N // 16, 1),
        GELU(BUF_ACCUM, src_off_u, BUF_ABUF, dst_off_u, sreg=0),
        SYNC(0b100),
        *set_addr(1, dst_addr),
        STORE(BUF_ABUF, dst_off_u, out_rows, 1, 0),
        SYNC(0b001),
        HALT(),
    ]
    dram = await setup_test(dut, prog, dram_writes={src_addr: _fp32_bytes(x)})
    await wait_halt(dut, max_cycles=600_000)
    assert int(dut.done.value) == 1 and int(dut.fault.value) == 0
    got = bytes(dram.mem[dst_addr:dst_addr + len(expected)])
    _assert_bytes_equal(got, expected, "gelu-accum")


@cocotb.test()
async def test_softmax_attnv_raises_fault(dut):
    """OP_SOFTMAX_ATTNV is unsupported in the W8A16 SFU.

    Until Phase 5 routes this through decode_unit's illegal-opcode path,
    the SFU itself rejects the dispatch with FAULT_UNSUPPORTED_OP.
    """
    from utils.insn_builder import SOFTMAX_ATTNV

    src_addr = 0x54000
    src_off_u = 0
    v_off_u = 128
    dst_off_u = 256

    prog = [
        *set_addr(0, src_addr),
        LOAD(BUF_ACCUM, src_off_u, 64, 0, 0),
        SYNC(0b001),
        *set_addr(1, src_addr + 0x1000),
        LOAD(BUF_ABUF, v_off_u, 16, 1, 0),
        SYNC(0b001),
        CONFIG_TILE(1, 1, 1),
        SOFTMAX_ATTNV(BUF_ACCUM, src_off_u, BUF_ABUF, v_off_u,
                      BUF_WBUF, dst_off_u, sreg=0),
        SYNC(0b100),
        HALT(),
    ]
    dram_writes = {
        src_addr: bytes(64 * 16),
        src_addr + 0x1000: bytes(16 * 16),
    }
    await setup_test(dut, prog, dram_writes=dram_writes)
    await wait_halt(dut, max_cycles=200_000)
    assert int(dut.fault.value) == 1, "expected fault for legacy SOFTMAX_ATTNV"
