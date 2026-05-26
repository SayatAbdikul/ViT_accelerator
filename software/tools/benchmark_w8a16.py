#!/usr/bin/env python3
"""End-to-end W8A16 accuracy benchmark.

Runs DeiT-tiny through the real W8A16 toolchain (compiler + golden
simulator). FP16 activations rounded each tensor at the ABUF boundary
add a modest per-tensor noise vs the W8A32 path; the gates below absorb
that compounded across 12 transformer blocks.

Usage:
    python -m tools.benchmark_w8a16 --max-images 20

Per-image columns:
  cos_fp32 — cosine vs HF FP32 reference (load-bearing gate ≥ 0.997).
  cos_fq   — cosine vs ``apply_weight_quantization`` reference (load-
             bearing gate ≥ 0.998; one nine looser than W8A32's 0.999
             to absorb FP16 narrowing compounded over 12 blocks).
  argmax   — argmax(logits) for each of (W8A16, fp32, fake_quant).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.compiler.cache import load_or_compile
from taccel.model_config import ModelConfig
from taccel.golden_model.simulator_w8a16 import SimulatorW8A16
from taccel.quantizer.fake_quant import apply_weight_quantization
from taccel.quantizer.quantize import quantize_tensor, dequantize_tensor


MODEL_NAME = "facebook/deit-tiny-patch16-224"

# ImageNet normalization (same as AutoImageProcessor would apply).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_image(img):
    """Resize-shorter-side-to-256, center-crop 224, normalize. [1,3,224,224]."""
    import torch
    from PIL import Image as _Image

    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), _Image.BICUBIC)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)

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
    """FP32 patch projection with the fake-quant patch weights, narrowed
    to FP16 at the DMA boundary so the simulator sees the same dtype the
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
        ).flatten(2).transpose(1, 2)[0].numpy().astype(np.float16)
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


# Per-worker globals populated by _init_worker via fork inheritance. These
# would otherwise be re-pickled per task; with the "fork" start method they
# are simply inherited via copy-on-write from the parent process.
_W_MODEL = None
_W_FQ_MODEL = None
_W_STATE_DICT = None
_W_PROGRAM = None
_W_CO = None


def _init_worker(model, fq_model, state_dict, program, co):
    """ProcessPoolExecutor initializer — caches large objects in module
    globals on each worker so per-task pickling is just (img_id, img)."""
    global _W_MODEL, _W_FQ_MODEL, _W_STATE_DICT, _W_PROGRAM, _W_CO
    _W_MODEL = model
    _W_FQ_MODEL = fq_model
    _W_STATE_DICT = state_dict
    _W_PROGRAM = program
    _W_CO = co
    # Avoid OpenMP oversubscription: each worker gets a single torch thread.
    # With N workers × 1 thread each, the pool saturates the cores without
    # any worker thrashing.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_one_image(args):
    """Worker: returns (img_id, result_or_None_on_skip, reason_or_None).

    result_or_None is (cos_fp32, cos_fq, am_w, am_fp32, am_fq) on success.
    """
    img_id, img = args
    import torch
    from taccel.golden_model.simulator_w8a16 import SimulatorW8A16

    pixel_values = _preprocess_image(img)
    with torch.no_grad():
        ref_fp32 = _W_MODEL(pixel_values=pixel_values).logits.numpy()[0]
        ref_fq = _W_FQ_MODEL(pixel_values=pixel_values).logits.numpy()[0]

    patches = _host_patch_embed(pixel_values, _W_STATE_DICT)

    sim = SimulatorW8A16()
    sim.load_program(_W_PROGRAM)
    patch_bytes = patches.tobytes()
    sim.state.dram[
        _W_PROGRAM.input_offset:_W_PROGRAM.input_offset + len(patch_bytes)
    ] = patch_bytes
    sim.run(max_instructions=_W_PROGRAM.insn_count + 10)
    if not sim.state.halted:
        return img_id, None, "did not reach HALT"

    logits = np.frombuffer(
        sim.state.abuf,
        dtype=np.float16,
        count=_W_CO["N_pad"],
        offset=_W_CO["offset_bytes"],
    )[: _W_CO["logical_cols"]].astype(np.float32)
    if not np.isfinite(logits).all():
        return img_id, None, "produced NaN/Inf logits"

    cos_fp32 = _cosine(logits, ref_fp32)
    cos_fq = _cosine(logits, ref_fq)
    am_w = int(np.argmax(logits))
    am_fp32 = int(np.argmax(ref_fp32))
    am_fq = int(np.argmax(ref_fq))
    return img_id, (cos_fp32, cos_fq, am_w, am_fp32, am_fq), None


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end W8A16 accuracy benchmark.")
    parser.add_argument("--max-images", type=int, default=len(DEFAULT_IMAGE_IDS),
                        help="Number of frozen benchmark images to evaluate.")
    parser.add_argument("--image-dir", default=LOCAL_FROZEN_IMAGE_DIR,
                        help="Directory holding the local frozen benchmark cache.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker count; 0 (default) = nproc, "
                             "1 = sequential (skip the process pool).")
    args = parser.parse_args()

    import torch
    from transformers import ViTForImageClassification

    print(f"Loading {MODEL_NAME}...")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    fq_model, _ = apply_weight_quantization(model)
    fq_model.eval()

    print("Compiling DeiT-tiny in W8A16 mode (cached on disk; skip on hit)...")
    state_dict = model.state_dict()
    program = load_or_compile(
        ModelConfig.deit_tiny(), state_dict, mode="w8a16", verbose=True,
    )
    print(f"  {program.insn_count} instructions, "
          f"{len(program.data):,} bytes data")

    image_ids = DEFAULT_IMAGE_IDS[: args.max_images]
    images = _load_local_images(image_ids, args.image_dir)

    co = program.compiler_manifest["classifier_output"]
    cos_fp32_vals = []
    cos_fq_vals = []
    top1_fp32_match = 0
    top1_fq_match = 0

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(images)))

    print()
    print(f"  Running W8A16 simulator on {len(images)} frozen images "
          f"({workers} worker{'s' if workers != 1 else ''})...")
    print(f"  {'#':>3}  {'id':>6}  {'cos_fp32':>10}  {'cos_fq':>10}  "
          f"{'argmax(w8a16,fp32,fq)':>26}")

    # Collect by img_id so we can print in submission order regardless of
    # which worker finishes first.
    results: dict[int, tuple] = {}

    if workers == 1:
        # Sequential fallback: same code path as the worker, but no pool.
        _init_worker(model, fq_model, state_dict, program, co)
        for img_id, img in images:
            _, result, reason = _run_one_image((img_id, img))
            if result is None:
                print(f"  WARNING: image {img_id} {reason}")
            else:
                results[img_id] = result
    else:
        # fork: workers inherit model/fq_model/state_dict/program from
        # parent via COW. Spawn would re-import and re-load the model
        # per worker (~1-2s each) which would dominate the 20-image budget.
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model, fq_model, state_dict, program, co),
        ) as ex:
            futures = [ex.submit(_run_one_image, (img_id, img))
                       for img_id, img in images]
            for fut in as_completed(futures):
                img_id, result, reason = fut.result()
                if result is None:
                    print(f"  WARNING: image {img_id} {reason}")
                else:
                    results[img_id] = result

    # Print in original submission order so the table stays deterministic
    # across runs even though workers complete in any order.
    for idx, (img_id, _img) in enumerate(images, 1):
        if img_id not in results:
            continue
        cos_fp32, cos_fq, am_w, am_fp32, am_fq = results[img_id]
        cos_fp32_vals.append(cos_fp32)
        cos_fq_vals.append(cos_fq)
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
    print(f"  ─── W8A16 (compile + simulate) results over {n} images ───")
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
    print(f"    cos_fp32 avg ≥ 0.997 : "
          f"{'PASS' if arr_fp32.mean() >= 0.997 else 'FAIL'}  "
          f"(measured {float(arr_fp32.mean()):.6f})")
    print(f"    cos_fq   avg ≥ 0.998 : "
          f"{'PASS' if arr_fq.mean() >= 0.998 else 'FAIL'}  "
          f"(measured {float(arr_fq.mean()):.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
