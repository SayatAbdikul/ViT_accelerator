"""Compile-time memory budget reporter.

Given a ``ModelConfig``, report the peak per-buffer footprint that the
compiler would attempt to allocate. Useful for predicting the M2 sequence/
feature-tiling cost-of-entry for new ViT variants before running the full
compile pipeline.

Usage:
    python -m tools.profile_memory --model deit-tiny
    python -m tools.profile_memory --model vit-base
    python -m tools.profile_memory --model vit-base --mode w8a32

Modes:
    w8a8  — INT8 activations (1 byte/element); default.
    w8a32 — FP32 activations (4 bytes/element). ABUF capacity in *elements*
            is 4× smaller and the M2 seq-tiling policy kicks in earlier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo-relative imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE  # noqa: E402
from taccel.model_config import ModelConfig  # noqa: E402
from taccel.compiler.passes.memory_estimate import decide_seq_tiling  # noqa: E402
from taccel.compiler.passes.memory_estimate_w8a32 import (  # noqa: E402
    decide_seq_tiling_w8a32,
)


_MODELS = {
    "deit-tiny": ModelConfig.deit_tiny,
    "vit-base": ModelConfig.vit_base,
}


def _bytes_pretty(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def estimate_peak_abuf(cfg: ModelConfig, element_bytes: int = 1) -> dict:
    """Estimate ABUF allocations in bytes for one transformer block.

    The codegen aggressively frees per-head sub-allocations and the strip-mined
    MLP cleans up between strips, so the relevant question is *what is the
    largest single allocation we attempt?* — if that exceeds the buffer cap,
    no amount of clever ordering will help.

    ``element_bytes`` defaults to 1 (W8A8 INT8 activations). Pass 4 for the
    W8A32 path (FP32 activations) — every estimate scales linearly.
    """
    embed = cfg.embed_dim
    seq = cfg.seq_len_pad
    head_dim = cfg.head_dim

    residual = seq * embed * element_bytes
    qkv_one_head = 3 * seq * head_dim * element_bytes
    attn_scratch = seq * seq * element_bytes  # softmax row buffer
    mlp_strip = seq * 16 * element_bytes  # codegen strip-mines MLP at 16 cols/strip

    largest_single = max(residual, qkv_one_head, attn_scratch, mlp_strip)
    return {
        "residual": residual,
        "qkv_one_head": qkv_one_head,
        "attn_scratch": attn_scratch,
        "mlp_strip": mlp_strip,
        "largest_single_alloc": largest_single,
    }


def estimate_weight_dram(cfg: ModelConfig) -> int:
    """Estimate total DRAM weight footprint (INT8 weights + FP16 scales)."""
    per_block = (
        3 * cfg.embed_dim * cfg.embed_dim          # Q, K, V weights
        + cfg.embed_dim * cfg.embed_dim            # out_proj
        + cfg.embed_dim * cfg.mlp_dim              # FC1
        + cfg.mlp_dim * cfg.embed_dim              # FC2
        + 4 * cfg.embed_dim                        # 2 layernorms × (γ + β) — FP16, 2×
    )
    embedding = (
        cfg.embed_dim                              # cls token
        + cfg.seq_len * cfg.embed_dim              # pos embed
    )
    head = cfg.embed_dim * cfg.num_classes
    return per_block * cfg.depth + embedding + head


def report(cfg: ModelConfig, label: str, mode: str = "w8a8") -> None:
    element_bytes = 4 if mode == "w8a32" else 1
    print(f"=== {label} ({mode}) ===")
    print(f"  embed_dim={cfg.embed_dim}  heads={cfg.num_heads}  head_dim={cfg.head_dim}")
    print(f"  depth={cfg.depth}  mlp_dim={cfg.mlp_dim}")
    print(f"  seq_len={cfg.seq_len}  seq_len_pad={cfg.seq_len_pad}")
    print(f"  activation element width: {element_bytes} byte"
          f"{'s' if element_bytes != 1 else ''}")

    abuf = estimate_peak_abuf(cfg, element_bytes=element_bytes)
    print()
    print("  ABUF allocation sizes (single block):")
    for k in ["residual", "qkv_one_head", "attn_scratch", "mlp_strip"]:
        print(f"    {k:<22} {_bytes_pretty(abuf[k])}")
    print(f"    {'LARGEST_SINGLE':<22} {_bytes_pretty(abuf['largest_single_alloc'])}  "
          f"(ABUF cap = {_bytes_pretty(ABUF_SIZE)})")

    wbuf_dram = estimate_weight_dram(cfg)
    print()
    if mode == "w8a32":
        # FP32 weights live in DRAM (INT8 quantize → dequant at compile time),
        # so the DRAM footprint is 4× the INT8 number.
        print(f"  Weight DRAM footprint (FP32 dequant): {_bytes_pretty(wbuf_dram * 4)}")
    else:
        print(f"  Weight DRAM footprint (INT8): {_bytes_pretty(wbuf_dram)}")
    print(f"  WBUF capacity (streams from DRAM): {_bytes_pretty(WBUF_SIZE)}")
    print(f"  ACCUM capacity: {_bytes_pretty(ACCUM_SIZE)}")

    print()
    fits = abuf["largest_single_alloc"] <= ABUF_SIZE
    print(f"  ABUF fit (untiled): {'YES' if fits else 'NO — sequence tiling required'}")

    # Report M2 sequence-tiling policy for the selected mode.
    if mode == "w8a32":
        decision = decide_seq_tiling_w8a32(cfg)
    else:
        decision = decide_seq_tiling(cfg)
    if decision.needs_tiling:
        per_tile = decision.per_tile_bytes
        print(f"  → M2 seq tiling: tile={decision.tile_rows} rows × "
              f"{decision.num_tiles} tiles, per-tile residual = "
              f"{_bytes_pretty(per_tile)}")
    else:
        reason = getattr(decision, "reason", "fits untiled")
        print(f"  → M2 seq tiling: not required ({reason})")

    # Wide-weight WBUF check (M3 boundary).
    widest_weight = max(
        cfg.embed_dim * cfg.embed_dim,        # out_proj
        cfg.embed_dim * cfg.mlp_dim,          # FC1
        cfg.mlp_dim * cfg.embed_dim,          # FC2
    )
    wbuf_fit = widest_weight <= WBUF_SIZE
    print(f"  WBUF fit (widest weight {_bytes_pretty(widest_weight)}): "
          f"{'YES' if wbuf_fit else 'NO — M3 weight N-strip mining required'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(_MODELS.keys()),
        help="Model preset to report on (may be repeated). Default: all.",
    )
    parser.add_argument(
        "--mode",
        choices=["w8a8", "w8a32"],
        default="w8a8",
        help="Precision mode. w8a8 = INT8 activations (1 byte/element); "
             "w8a32 = FP32 activations (4 bytes/element), 4× tighter ABUF.",
    )
    args = parser.parse_args()

    models = args.model or sorted(_MODELS.keys())
    for name in models:
        report(_MODELS[name](), name, mode=args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
