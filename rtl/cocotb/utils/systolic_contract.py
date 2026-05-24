"""Shared cocotb helpers for row-major MATMUL contract preparation.

The W8A16 datapath holds 8 FP16 elements per 16-byte SRAM row (down from
16 INT8 elements before). The INT8 helpers below remain for the helper /
SFU cocotb suites that still target the legacy INT8 helper engine; the
FP16 helpers are used by the systolic MATMUL tests.
"""

import numpy as np

from utils.insn_builder import LOAD, SET_ADDR_HI, SET_ADDR_LO, SYNC, BUF_ABUF, BUF_WBUF


SYS_DIM = 16


def set_addr_insns(reg: int, addr: int) -> list[int]:
    return [
        SET_ADDR_LO(reg, addr & 0x0FFFFFFF),
        SET_ADDR_HI(reg, (addr >> 28) & 0x0FFFFFFF),
    ]


def append_load_sync(prog: list[int], reg: int, addr: int, buf_id: int, sram_off: int, xfer_len: int) -> None:
    prog.extend(set_addr_insns(reg, addr))
    prog.append(LOAD(buf_id, sram_off, xfer_len, reg, 0))
    prog.append(SYNC(0b001))


# ---- Legacy INT8 layout (helpers / SFU engines, removed in Phase 4) ----

def flatten_16x16(mat: list[list[int]]) -> bytes:
    return bytes((row[col] & 0xFF) for row in mat for col in range(16))


def flatten_tile_32x32(mat: list[list[int]], row_base: int, col_base: int) -> bytes:
    return bytes((mat[row_base + r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_16x64(mat: list[list[int]], col_base: int) -> bytes:
    return bytes((mat[r][col_base + c] & 0xFF) for r in range(16) for c in range(16))


def flatten_tile_64x16(mat: list[list[int]], row_base: int) -> bytes:
    return bytes((mat[row_base + r][c] & 0xFF) for r in range(16) for c in range(16))


def prepare_logical_16x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_addr: int, b_addr: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    dram.write_bytes(a_addr, flatten_16x16(a))
    dram.write_bytes(b_addr, flatten_16x16(b))
    append_load_sync(prog, 0, a_addr, BUF_ABUF, abuf_off, (16 * 16) // 16)
    append_load_sync(prog, 1, b_addr, BUF_WBUF, wbuf_off, (16 * 16) // 16)


def prepare_logical_32x32(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                          a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    a_bytes = bytes((a[r][c] & 0xFF) for r in range(32) for c in range(32))
    b_bytes = bytes((b[r][c] & 0xFF) for r in range(32) for c in range(32))
    dram.write_bytes(a_base, a_bytes)
    dram.write_bytes(b_base, b_bytes)
    append_load_sync(prog, 0, a_base, BUF_ABUF, abuf_off, (32 * 32) // 16)
    append_load_sync(prog, 1, b_base, BUF_WBUF, wbuf_off, (32 * 32) // 16)


def prepare_logical_16x64x16(dram, prog: list[int], a: list[list[int]], b: list[list[int]],
                             a_base: int, b_base: int, abuf_off: int = 0, wbuf_off: int = 0) -> None:
    a_bytes = bytes((a[r][c] & 0xFF) for r in range(16) for c in range(64))
    b_bytes = bytes((b[r][c] & 0xFF) for r in range(64) for c in range(16))
    dram.write_bytes(a_base, a_bytes)
    dram.write_bytes(b_base, b_bytes)
    append_load_sync(prog, 0, a_base, BUF_ABUF, abuf_off, (16 * 64) // 16)
    append_load_sync(prog, 1, b_base, BUF_WBUF, wbuf_off, (64 * 16) // 16)


# ---- W8A16 FP16 layout (systolic MATMUL tests) ----

def _fp16_bytes_rowmajor(arr_fp16: np.ndarray) -> bytes:
    """Return arr (numpy fp16) flattened row-major as little-endian bytes."""
    return arr_fp16.astype(np.float16).tobytes()


def prepare_fp16_logical(dram, prog: list[int], a_fp16: np.ndarray, b_fp16: np.ndarray,
                         a_addr: int, b_addr: int,
                         abuf_off: int = 0, wbuf_off: int = 0) -> None:
    """Generic FP16 MATMUL prep. Shapes are read off the arrays.

    a is M x K FP16, b is K x N FP16, both little-endian. 8 FP16 lanes
    per 16-byte SRAM row -> xfer_len = elements*2/16.
    """
    a_fp16 = np.ascontiguousarray(a_fp16.astype(np.float16))
    b_fp16 = np.ascontiguousarray(b_fp16.astype(np.float16))
    dram.write_bytes(a_addr, _fp16_bytes_rowmajor(a_fp16))
    dram.write_bytes(b_addr, _fp16_bytes_rowmajor(b_fp16))
    a_bytes = a_fp16.size * 2
    b_bytes = b_fp16.size * 2
    append_load_sync(prog, 0, a_addr, BUF_ABUF, abuf_off, a_bytes // 16)
    append_load_sync(prog, 1, b_addr, BUF_WBUF, wbuf_off, b_bytes // 16)


def matmul_fp_ref(a_fp16: np.ndarray, b_fp16: np.ndarray) -> np.ndarray:
    """Sequential K-loop FP32 matmul matching the per-PE FP32 MAC order.

    Matches software/taccel/golden_model/systolic_w8a16.py and the per-PE
    widen+mul+add sequence in rtl/src/systolic/systolic_pe.sv.
    """
    a32 = a_fp16.astype(np.float16).astype(np.float32)
    b32 = b_fp16.astype(np.float16).astype(np.float32)
    M, K = a32.shape
    _K, N = b32.shape
    assert K == _K
    acc = np.zeros((M, N), dtype=np.float32)
    for k in range(K):
        acc = acc + a32[:, k:k + 1] * b32[k:k + 1, :]
    return acc
