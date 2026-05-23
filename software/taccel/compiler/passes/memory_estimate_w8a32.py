"""Static ABUF footprint estimator — W8A32 (FP32 activations) variant.

Same policy as :mod:`memory_estimate` but with ``element_bytes = 4``: in
W8A32 the residual stream consumes 4× the bytes it does in W8A8. For
DeiT-tiny (197×192=37.8 KB → 151 KB) tiling activates immediately
since the full residual exceeds ABUF (128 KB). For ViT-B/16 the per-tile
budget tightens proportionally and ``tile_rows`` lands around 8.
"""
from __future__ import annotations

from typing import Optional

from ...isa.opcodes import ABUF_SIZE
from ...model_config import ModelConfig
from .memory_estimate import TilingDecision

_ELEMENT_BYTES = 4

# Headroom / trigger ratios are kept identical to the W8A8 policy
# (ABUF / 4 headroom; ABUF / 3 trigger). Because the *bytes* per element
# quadrupled, the *element* threshold is implicitly ¼ that of W8A8.
_HEADROOM_DIVISOR = 4
_TRIGGER_DIVISOR = 3


def decide_seq_tiling_w8a32(cfg: ModelConfig) -> TilingDecision:
    """Decide sequence-tiling policy for the W8A32 path."""
    full_bytes = cfg.seq_len_pad * cfg.embed_dim * _ELEMENT_BYTES
    trigger = ABUF_SIZE // _TRIGGER_DIVISOR
    if full_bytes <= trigger:
        return TilingDecision(
            needs_tiling=False,
            tile_rows=cfg.seq_len_pad,
            num_tiles=1,
            full_residual_bytes=full_bytes,
            per_tile_bytes=full_bytes,
        )

    # The FP32 FC1 / classifier output ``[tile_rows, mlp_dim] * 4`` is the
    # tightest constraint. With residual1, LN2 output and the FC1 output all
    # simultaneously live in ABUF post-LN2, we need the FC1 output alone to
    # be ≤ ABUF/2 to leave room for the other live tensors plus N-strip
    # spill workspace. Hence the ABUF/2 cap on ``per_tile_fc1_bytes``.
    seq = cfg.seq_len_pad
    fc1_bound = (ABUF_SIZE // 2) // (cfg.mlp_dim * _ELEMENT_BYTES)
    residual_bound = (ABUF_SIZE // _HEADROOM_DIVISOR) // (cfg.embed_dim * _ELEMENT_BYTES)
    max_tile_rows = max(16, min(fc1_bound, residual_bound))
    max_tile_rows = (max_tile_rows // 16) * 16

    # Prefer the largest multiple of 16 that divides ``seq`` exactly to avoid
    # a short trailing tile with separate codegen.
    candidates = [t for t in range(max_tile_rows, 0, -16) if seq % t == 0]
    tile_rows = candidates[0] if candidates else max_tile_rows
    num_tiles = (seq + tile_rows - 1) // tile_rows
    return TilingDecision(
        needs_tiling=True,
        tile_rows=tile_rows,
        num_tiles=num_tiles,
        full_residual_bytes=full_bytes,
        per_tile_bytes=tile_rows * cfg.embed_dim * _ELEMENT_BYTES,
    )
