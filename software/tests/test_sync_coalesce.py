"""Unit tests for :mod:`taccel.compiler.sync_coalesce`.

The pass strips redundant DMA-bit SYNCs and is the only place where
the W8A16/W8A32 codegen relaxes the original "SYNC after every LOAD"
discipline. The contract is precise:

* Drop the DMA bit when the next engine op is anything other than
  ``OP_MATMUL`` or ``OP_HALT`` (or absent entirely).
* Preserve ``SYNC`` instructions whose remaining mask is nonzero.
* Never reorder retained instructions; never add new ones.
"""
from __future__ import annotations

from taccel.compiler.sync_coalesce import coalesce_dma_syncs
from taccel.isa.instructions import (
    SyncInsn, LoadInsn, StoreInsn, MatmulInsn, BufCopyInsn, ScaleMulInsn,
    NopInsn, HaltInsn, SetAddrLoInsn, ConfigTileInsn,
)


def _ops(insns):
    return [type(i).__name__ for i in insns]


def test_sync_between_two_loads_is_kept():
    """The DMA engine has no command queue (``dma_engine.sv`` line 187);
    a second LOAD dispatched while the first is in flight has its
    pulse silently dropped. The SYNC between adjacent DMA ops is
    load-bearing and the pass must not remove it."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        LoadInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "LoadInsn"]
    assert out[1].resource_mask == 0b001


def test_sync_before_matmul_is_preserved():
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        MatmulInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "MatmulInsn"]
    assert out[1].resource_mask == 0b001


def test_sync_before_buf_copy_is_dropped():
    """BUF_COPY auto-stalls on dma_busy at issue (control_unit.sv:287),
    so the explicit DMA fence is redundant."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        BufCopyInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "BufCopyInsn"]


def test_sync_before_scale_mul_is_dropped():
    """Helper ops auto-stall on dma_busy (control_unit.sv:321)."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        ScaleMulInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "ScaleMulInsn"]


def test_sync_before_halt_is_preserved():
    """Defensive: keep the fence before HALT in case any program tail
    ever ends LOAD→HALT."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        HaltInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "HaltInsn"]


def test_sync_at_end_of_stream_is_preserved():
    """If the only thing after a SYNC is control-only (or nothing), it
    is treated like a final drain and kept verbatim."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        NopInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "NopInsn"]


def test_lookahead_skips_control_only_ops():
    """SET_ADDR/NOP/CONFIG_TILE between SYNC and consumer don't change
    whether the DMA fence is needed. With BUF_COPY as the next engine
    op (auto-fenced on dma_busy at issue) the SYNC is dropped despite
    the intervening control-only insns."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        SetAddrLoInsn(),
        ConfigTileInsn(),
        BufCopyInsn(),  # next real op auto-fences on dma_busy → drop
    ]
    out = coalesce_dma_syncs(insns)
    assert "SyncInsn" not in _ops(out)


def test_non_dma_sync_is_unchanged():
    """SYNCs with no DMA bit guard a different engine — leave them
    alone."""
    insns = [
        MatmulInsn(),
        SyncInsn(resource_mask=0b010),  # systolic
        ScaleMulInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["MatmulInsn", "SyncInsn", "ScaleMulInsn"]
    assert out[1].resource_mask == 0b010


def test_multi_mask_sync_before_auto_fence_strips_dma_bit():
    """A SYNC(0b101) before BUF_COPY (auto-fenced on dma_busy) drops
    only the DMA bit and keeps the SFU bit."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b101),  # DMA + SFU
        BufCopyInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "BufCopyInsn"]
    assert out[1].resource_mask == 0b100  # only SFU left


def test_sync_before_store_is_kept():
    """STORE has no command queue in the DMA engine; a STORE pulse
    arriving while a LOAD is mid-flight is dropped just like a second
    LOAD. The SYNC between LOAD→STORE is load-bearing."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        StoreInsn(),
    ]
    out = coalesce_dma_syncs(insns)
    assert _ops(out) == ["LoadInsn", "SyncInsn", "StoreInsn"]
    assert out[1].resource_mask == 0b001


def test_idempotent():
    """Running the pass twice gives the same result as running it once."""
    insns = [
        LoadInsn(),
        SyncInsn(resource_mask=0b001),
        BufCopyInsn(),
        SyncInsn(resource_mask=0b001),
        ScaleMulInsn(),
    ]
    once = coalesce_dma_syncs(insns)
    twice = coalesce_dma_syncs(once)
    assert _ops(once) == _ops(twice)
