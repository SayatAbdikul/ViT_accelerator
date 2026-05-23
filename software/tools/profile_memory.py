"""Compile-time memory budget reporter.

Given a ``ModelConfig``, report the peak per-buffer footprint that the
compiler would attempt to allocate. Useful for predicting the M2 sequence/
feature-tiling cost-of-entry for new ViT variants before running the full
compile pipeline.

Usage:
    python -m tools.profile_memory --model deit-tiny
    python -m tools.profile_memory --model vit-base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo-relative imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.isa.opcodes import ABUF_SIZE, WBUF_SIZE, ACCUM_SIZE  # noqa: E402
from taccel.model_config import ModelConfig  # noqa: E402


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


def estimate_peak_abuf(cfg: ModelConfig) -> dict:
    """Estimate ABUF allocations in INT8 bytes for one transformer block.

    The codegen aggressively frees per-head sub-allocations and the strip-mined
    MLP cleans up between strips, so the relevant question is *what is the
    largest single allocation we attempt?* — if that exceeds the buffer cap,
    no amount of clever ordering will help.
    """
    embed = cfg.embed_dim
    seq = cfg.seq_len_pad
    head_dim = cfg.head_dim

    residual = seq * embed
    qkv_one_head = 3 * seq * head_dim
    attn_scratch = seq * seq  # softmax row buffer (INT8 attention probs)
    mlp_strip = seq * 16  # codegen strip-mines MLP at 16 cols/strip

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


def report(cfg: ModelConfig, label: str) -> None:
    print(f"=== {label} ===")
    print(f"  embed_dim={cfg.embed_dim}  heads={cfg.num_heads}  head_dim={cfg.head_dim}")
    print(f"  depth={cfg.depth}  mlp_dim={cfg.mlp_dim}")
    print(f"  seq_len={cfg.seq_len}  seq_len_pad={cfg.seq_len_pad}")

    abuf = estimate_peak_abuf(cfg)
    print()
    print("  ABUF allocation sizes (single block):")
    for k in ["residual", "qkv_one_head", "attn_scratch", "mlp_strip"]:
        print(f"    {k:<22} {_bytes_pretty(abuf[k])}")
    print(f"    {'LARGEST_SINGLE':<22} {_bytes_pretty(abuf['largest_single_alloc'])}  "
          f"(ABUF cap = {_bytes_pretty(ABUF_SIZE)})")

    wbuf_dram = estimate_weight_dram(cfg)
    print()
    print(f"  Weight DRAM footprint (INT8): {_bytes_pretty(wbuf_dram)}")
    print(f"  WBUF capacity (streams from DRAM): {_bytes_pretty(WBUF_SIZE)}")
    print(f"  ACCUM capacity: {_bytes_pretty(ACCUM_SIZE)}")

    print()
    fits = abuf["largest_single_alloc"] <= ABUF_SIZE
    print(f"  ABUF fit: {'YES' if fits else 'NO — M2 sequence/feature tiling required'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(_MODELS.keys()),
        help="Model preset to report on (may be repeated). Default: all.",
    )
    args = parser.parse_args()

    models = args.model or sorted(_MODELS.keys())
    for name in models:
        report(_MODELS[name](), name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
