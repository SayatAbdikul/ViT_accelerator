#!/usr/bin/env python3
"""End-to-end W8A32 accuracy benchmark.

Runs DeiT-tiny through the real W8A32 toolchain (compiler + golden
simulator), in contrast to ``precision_sweep_w8a32.py`` which only
exercises the PyTorch fake-quant ceiling. The two should agree to within
≤ 1e-3 cosine on every image; any larger gap indicates a bug in the
W8A32 codegen / simulator (cf. the seq-padding attention leak fixed in
``codegen_w8a32._emit_qkt``).

Usage:
    python -m tools.benchmark_w8a32 --max-images 20

Per-image columns:
  cos_fp32 — cosine vs HF FP32 reference (the load-bearing accuracy gate
             from docs/precision_modes.md is ≥ 0.998 average).
  cos_fq   — cosine vs ``apply_weight_quantization`` reference (the
             fake-quant ceiling; W8A32 toolchain should be bit-equivalent
             to it within tight FP32 rounding tolerance, ≥ 0.999).
  argmax   — argmax(logits) for each of (W8A32, fp32, fake_quant).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.compiler.compiler import Compiler
from taccel.model_config import ModelConfig
from taccel.golden_model.simulator_w8a32 import SimulatorW8A32
from taccel.quantizer.fake_quant import apply_weight_quantization
from taccel.quantizer.quantize import quantize_tensor, dequantize_tensor


MODEL_NAME = "facebook/deit-tiny-patch16-224"

# Local frozen benchmark image cache shared with benchmark_fp32_vs_int8.py.
LOCAL_FROZEN_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images",
    "frozen_benchmark",
)
DEFAULT_IMAGE_IDS = [
    139, 285, 632, 724, 776, 785, 872, 1000, 1296, 1353,
    1503, 1761, 2006, 2153, 2473, 2685, 3501, 3845, 5037, 39769,
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _host_patch_embed(pixel_values, state_dict):
    """FP32 patch projection with the same fake-quant patch weights the
    in-program dequant pipeline uses for the rest of the network."""
    import torch
    import torch.nn.functional as F

    patch_w = state_dict[
        "vit.embeddings.patch_embeddings.projection.weight"
    ].numpy().astype(np.float32)
    patch_b = state_dict[
        "vit.embeddings.patch_embeddings.projection.bias"
    ].numpy().astype(np.float32)
    qw, scw = quantize_tensor(patch_w.reshape(patch_w.shape[0], -1), per_channel=True)
    patch_w_dq = dequantize_tensor(qw, scw).astype(np.float32).reshape(patch_w.shape)
    with torch.no_grad():
        patches = F.conv2d(
            pixel_values,
            torch.from_numpy(patch_w_dq),
            bias=torch.from_numpy(patch_b),
            stride=16,
        ).flatten(2).transpose(1, 2)[0].numpy().astype(np.float32)
    return patches


def _load_local_images(image_ids, image_root):
    from PIL import Image

    loaded = []
    missing = []
    for img_id in image_ids:
        path = os.path.join(image_root, f"{img_id:012d}.jpg")
        if not os.path.exists(path):
            missing.append(img_id)
            continue
        with Image.open(path) as img:
            loaded.append((img_id, img.convert("RGB")))
    if missing:
        raise SystemExit(
            f"Missing local frozen benchmark images for COCO ids: {missing}\n"
            "Run `python -m tools.benchmark_fp32_vs_int8 "
            "--populate-local-benchmark-cache` first."
        )
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end W8A32 accuracy benchmark.")
    parser.add_argument("--max-images", type=int, default=len(DEFAULT_IMAGE_IDS),
                        help="Number of frozen benchmark images to evaluate.")
    parser.add_argument("--image-dir", default=LOCAL_FROZEN_IMAGE_DIR,
                        help="Directory holding the local frozen benchmark cache.")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, ViTForImageClassification

    print(f"Loading {MODEL_NAME}...")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    fq_model, _ = apply_weight_quantization(model)
    fq_model.eval()

    print("Compiling DeiT-tiny in W8A32 mode (this happens once)...")
    state_dict = model.state_dict()
    compiler = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a32")
    program = compiler.compile_w8a32(state_dict)
    print(f"  {program.insn_count} instructions, "
          f"{len(program.data):,} bytes data")

    image_ids = DEFAULT_IMAGE_IDS[: args.max_images]
    images = _load_local_images(image_ids, args.image_dir)

    co = program.compiler_manifest["classifier_output"]
    cos_fp32_vals = []
    cos_fq_vals = []
    top1_fp32_match = 0
    top1_fq_match = 0

    print()
    print(f"  Running W8A32 simulator on {len(images)} frozen images...")
    print(f"  {'#':>3}  {'id':>6}  {'cos_fp32':>10}  {'cos_fq':>10}  "
          f"{'argmax(w8a32,fp32,fq)':>26}")
    for idx, (img_id, img) in enumerate(images, 1):
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            ref_fp32 = model(pixel_values=pixel_values).logits.numpy()[0]
            ref_fq = fq_model(pixel_values=pixel_values).logits.numpy()[0]

        patches = _host_patch_embed(pixel_values, state_dict)

        sim = SimulatorW8A32()
        sim.load_program(program)
        patch_bytes = patches.tobytes()
        sim.state.dram[
            program.input_offset:program.input_offset + len(patch_bytes)
        ] = patch_bytes
        sim.run(max_instructions=program.insn_count + 10)
        if not sim.state.halted:
            print(f"  WARNING: image {img_id} did not reach HALT")
            continue

        logits = np.frombuffer(
            sim.state.abuf,
            dtype=np.float32,
            count=co["N_pad"],
            offset=co["offset_bytes"],
        )[: co["logical_cols"]].copy()
        if not np.isfinite(logits).all():
            print(f"  WARNING: image {img_id} produced NaN/Inf logits")
            continue

        cos_fp32 = _cosine(logits, ref_fp32)
        cos_fq = _cosine(logits, ref_fq)
        cos_fp32_vals.append(cos_fp32)
        cos_fq_vals.append(cos_fq)
        am_w = int(np.argmax(logits))
        am_fp32 = int(np.argmax(ref_fp32))
        am_fq = int(np.argmax(ref_fq))
        top1_fp32_match += int(am_w == am_fp32)
        top1_fq_match += int(am_w == am_fq)

        print(f"  {idx:>3}  {img_id:>6}  {cos_fp32:>10.6f}  {cos_fq:>10.6f}  "
              f"({am_w:>4}, {am_fp32:>4}, {am_fq:>4})")

    n = len(cos_fp32_vals)
    if n == 0:
        print("\n  ERROR: no usable runs.")
        return 1

    arr_fp32 = np.array(cos_fp32_vals)
    arr_fq = np.array(cos_fq_vals)
    print()
    print(f"  ─── W8A32 (compile + simulate) results over {n} images ───")
    print(f"    cos_fp32 avg : {float(arr_fp32.mean()):.6f}")
    print(f"    cos_fp32 min : {float(arr_fp32.min()):.6f}")
    print(f"    cos_fq   avg : {float(arr_fq.mean()):.6f}")
    print(f"    cos_fq   min : {float(arr_fq.min()):.6f}")
    print(f"    top-1 vs FP32        : {top1_fp32_match}/{n} "
          f"({100.0 * top1_fp32_match / n:.0f}%)")
    print(f"    top-1 vs fake_quant  : {top1_fq_match}/{n} "
          f"({100.0 * top1_fq_match / n:.0f}%)")
    print()
    print("  Load-bearing gate (docs/precision_modes.md):")
    print(f"    cos_fp32 avg ≥ 0.998 : "
          f"{'PASS' if arr_fp32.mean() >= 0.998 else 'FAIL'}  "
          f"(measured {float(arr_fp32.mean()):.6f})")
    print(f"    cos_fq   avg ≥ 0.999 : "
          f"{'PASS' if arr_fq.mean() >= 0.999 else 'FAIL'}  "
          f"(measured {float(arr_fq.mean()):.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
