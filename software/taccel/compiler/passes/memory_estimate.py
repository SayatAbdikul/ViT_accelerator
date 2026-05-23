"""Static ABUF footprint estimator used to pick a tile size.

The codegen aggressively frees per-head sub-allocations, but the residual
stream ``[seq_len_pad, embed_dim]`` is the load-bearing tensor: if a single
copy of it does not fit in ABUF, no amount of scheduling helps and the
sequence-tiling pass must activate.

This module is the policy half of the decision; the rewrite half lives in
``seq_tiling.py``. They share the same arithmetic so the policy and the
generated code agree on what fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...isa.opcodes import ABUF_SIZE, ACCUM_SIZE
from ...model_config import ModelConfig


@dataclass(frozen=True)
class TilingDecision:
    """Outcome of the sequence-tiling policy."""

    needs_tiling: bool
    tile_rows: int            # per-tile sequence-row count after padding
    num_tiles: int            # ceil(seq_len_pad / tile_rows)
    full_residual_bytes: int  # seq_len_pad * embed_dim — the original allocation
    per_tile_bytes: int       # tile_rows * embed_dim — what tiling actually allocates

    @property
    def reason(self) -> str:
        if self.needs_tiling:
            return (
                f"residual {self.full_residual_bytes}B > ABUF/3 "
                f"({ABUF_SIZE // 3}B); tile={self.tile_rows} rows × "
                f"{self.num_tiles} tiles"
            )
        return f"residual {self.full_residual_bytes}B fits in ABUF; no tiling"


# Headroom rationale: ABUF must simultaneously hold (a) the input residual
# tile, (b) the LN1 output tile, and (c) per-head Q/K/V scratch, plus
# weight prefetch buffers. Picking the tile such that ONE copy of the
# tile-sized [tile_rows, embed_dim] tensor is ≤ ABUF/4 leaves comfortable
# room for the simultaneous live set after compaction. Empirically for
# ViT-B/16 with embed=768, tile=32 → 24KB ≤ 128KB/4 = 32KB. ✓
_HEADROOM_DIVISOR = 4

# Trigger: residual ≥ ABUF/3 means the existing one-shot allocation no
# longer leaves room for the LN output and at least one head's scratch.
# Below this we don't pay the DRAM-staging cost.
_TRIGGER_DIVISOR = 3


def decide_seq_tiling(cfg: ModelConfig) -> TilingDecision:
    """Decide whether to sequence-tile and at what granularity.

    The chosen tile_rows is the largest multiple of SYS_DIM=16 that:
    * keeps one tile-sized activation ≤ ABUF/_HEADROOM_DIVISOR,
    * keeps the matmul ACCUM intermediate (tile_rows × mlp_dim × 4 bytes,
      strip-mined) ≤ ACCUM_SIZE — already enforced by the existing
      MLP strip-mining loop, so this just picks a tile_rows that doesn't
      blow up the strip count needlessly.
    * divides seq_len_pad ideally (clean iteration); otherwise the last
      tile is short and the codegen emits a partial-tile path.
    """
    full_bytes = cfg.seq_len_pad * cfg.embed_dim
    trigger = ABUF_SIZE // _TRIGGER_DIVISOR
    if full_bytes <= trigger:
        return TilingDecision(
            needs_tiling=False,
            tile_rows=cfg.seq_len_pad,
            num_tiles=1,
            full_residual_bytes=full_bytes,
            per_tile_bytes=full_bytes,
        )

    headroom = ABUF_SIZE // _HEADROOM_DIVISOR
    # Pick the largest tile_rows that fits. Candidates in 16-step granularity.
    max_tile_rows = max(16, headroom // cfg.embed_dim)
    # Snap down to a multiple of 16 (SYS_DIM alignment).
    tile_rows = (max_tile_rows // 16) * 16
    tile_rows = max(tile_rows, 16)

    # Prefer a tile that divides seq_len_pad evenly when one is available
    # in the half-window around the headroom optimum. Even division removes
    # the partial-tile fast-path from codegen and keeps the program shorter.
    seq = cfg.seq_len_pad
    if seq % tile_rows != 0:
        candidates = [
            t for t in range(tile_rows, max(15, tile_rows // 2), -16)
            if seq % t == 0
        ]
        if candidates:
            tile_rows = candidates[0]

    num_tiles = (seq + tile_rows - 1) // tile_rows
    return TilingDecision(
        needs_tiling=True,
        tile_rows=tile_rows,
        num_tiles=num_tiles,
        full_residual_bytes=full_bytes,
        per_tile_bytes=tile_rows * cfg.embed_dim,
    )
