"""Shared sequence-tiling decision dataclass.

Both :mod:`memory_estimate_w8a16` and :mod:`memory_estimate_w8a32` produce
``TilingDecision`` instances with mode-specific element-width math. The
dataclass itself is mode-agnostic and lives here so the two modes share a
single type identity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TilingDecision:
    """Result of the sequence-tiling policy decision.

    Attributes
    ----------
    needs_tiling
        True if the full residual exceeds the trigger threshold and the
        seq_tiling pass should rewrite the IR.
    tile_rows
        Rows per tile (multiple of 16). When ``needs_tiling`` is False
        this equals the full ``seq_len_pad``.
    num_tiles
        ``ceil(seq_len_pad / tile_rows)``. Equals 1 when ``needs_tiling``
        is False.
    full_residual_bytes
        Bytes the un-tiled residual stream would consume in ABUF.
    per_tile_bytes
        Bytes per tile after splitting.
    """

    needs_tiling: bool
    tile_rows: int
    num_tiles: int
    full_residual_bytes: int
    per_tile_bytes: int
