"""cocotb chained-mode systolic tests (W8A16 datapath, Verilator only).

Chained mode is the shipping default (SYS_MODE_DEFAULT in taccel_pkg.sv)
and test_systolic.py already exercises it via the default makefile flags.
This file pins additional cases that specifically rely on the chained
flush window completing before drain.
"""

import cocotb
import numpy as np

from utils.dram_model import DramModel
from utils.insn_builder import HALT, SYNC, CONFIG_TILE, MATMUL, BUF_ACCUM
from utils.systolic_contract import prepare_fp16_logical, matmul_fp_ref
from utils.testbench import read_accum_fp32_16x16, read_accum_fp32_32x32, setup_test, wait_halt


def _assert_acc_bits_equal(got_fp32: np.ndarray, exp_fp32: np.ndarray, tag: str):
    got_bits = got_fp32.view(np.uint32)
    exp_bits = exp_fp32.view(np.uint32)
    if not np.array_equal(got_bits, exp_bits):
        diff = np.argwhere(got_bits != exp_bits)
        i, j = int(diff[0][0]), int(diff[0][1])
        raise AssertionError(
            f"{tag} first mismatch at ({i},{j}): got=0x{int(got_bits[i, j]):08x} "
            f"({float(got_fp32[i, j])}) exp=0x{int(exp_bits[i, j]):08x} ({float(exp_fp32[i, j])})"
        )


@cocotb.test()
async def test_matmul_identity_chained(dut):
    a = np.array([[float((i * 3 + j) & 0x7F) for j in range(16)] for i in range(16)],
                 dtype=np.float16)
    eye = np.eye(16, dtype=np.float16)
    exp = matmul_fp_ref(a, eye)

    prog = []
    dram = DramModel()
    prepare_fp16_logical(dram, prog, a, eye, 0x100000, 0x110000)
    prog.extend([
        CONFIG_TILE(1, 1, 1),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=900_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = read_accum_fp32_16x16(dut)
    _assert_acc_bits_equal(got, exp, "chained-identity")


@cocotb.test()
async def test_matmul_k4_boundary_stress_chained(dut):
    a = np.array([[1.0 if ((i + k) & 1) else -1.0 for k in range(64)] for i in range(16)],
                 dtype=np.float16)
    b = np.array([[-1.0 if ((k * 7 + j) & 1) else 1.0 for j in range(16)] for k in range(64)],
                 dtype=np.float16)
    exp = matmul_fp_ref(a, b)

    prog = []
    dram = DramModel()
    prepare_fp16_logical(dram, prog, a, b, 0x140000, 0x150000)
    prog.extend([
        CONFIG_TILE(1, 1, 4),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=1_600_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = read_accum_fp32_16x16(dut)
    _assert_acc_bits_equal(got, exp, "chained-k4")


@cocotb.test()
async def test_matmul_multitile_2x2x2_chained(dut):
    a = np.array([[float(((i * 7 + j * 5 + 3) % 11) - 5) for j in range(32)] for i in range(32)],
                 dtype=np.float16)
    b = np.array([[float(((i * 3 + j * 9 + 1) % 13) - 6) for j in range(32)] for i in range(32)],
                 dtype=np.float16)
    exp = matmul_fp_ref(a, b)

    prog = []
    dram = DramModel()
    prepare_fp16_logical(dram, prog, a, b, 0x160000, 0x180000)
    prog.extend([
        CONFIG_TILE(2, 2, 2),
        MATMUL(0, 0, 1, 0, BUF_ACCUM, 0, sreg=0, flags=0),
        SYNC(0b010),
        HALT(),
    ])
    await setup_test(dut, prog, dram=dram)
    await wait_halt(dut, max_cycles=1_500_000)

    assert int(dut.done.value) == 1
    assert int(dut.fault.value) == 0
    got = read_accum_fp32_32x32(dut)
    _assert_acc_bits_equal(got, exp, "chained-multitile")
