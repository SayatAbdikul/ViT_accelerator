"""Static ABUF footprint estimator — W8A16 (FP16 activations) variant.

Same policy as :mod:`memory_estimate` but with ``element_bytes = 2``: in
W8A16 the residual stream consumes 2× the bytes it does in W8A8 (and ½
the bytes it does in W8A32). For DeiT-tiny (208 × 192 × 2 = 79.9 KB)
the full residual fits in 128 KB ABUF on its own, but the FC1 / classifier
output [tile, mlp_dim] × 2 still exceeds the per-tile ABUF/2 cap so the
tiler still triggers and lands on tile_rows=16 — matching the W8A32
shape on DeiT-tiny.

For ViT-B/16 the FP16 residual is 208 × 768 × 2 = 312 KB > 128 KB, so
sequence tiling is mandatory there as well.
"""
from __future__ import annotations

from typing import Optional

from ...isa.opcodes import ABUF_SIZE
from ...model_config import ModelConfig
from .memory_estimate import TilingDecision

_ELEMENT_BYTES = 2

# Headroom / trigger ratios mirror the W8A32 policy; only the
# *element* threshold halves because each FP16 element is 2 bytes
# (vs FP32's 4).
_HEADROOM_DIVISOR = 4
_TRIGGER_DIVISOR = 3


def decide_seq_tiling_w8a16(cfg: ModelConfig) -> TilingDecision:
    """Decide sequence-tiling policy for the W8A16 path."""
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

    # The FP16 FC1 / classifier output ``[tile_rows, mlp_dim] * 2`` is the
    # tightest constraint. With residual1, LN2 output and the FC1 output
    # all simultaneously live in ABUF post-LN2, the FC1 output alone must
    # be ≤ ABUF/2 to leave room for the other live tensors plus N-strip
    # spill workspace.
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
