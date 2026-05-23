#!/usr/bin/env python3
"""W8A32 — per-channel INT8 weights, FP32 activations, FP32 reference.

Measures the *ceiling* of weight-only quantization on DeiT-tiny:
how much logit signal is lost purely to weight rounding when
activations remain at full precision.

Compares against the user's 0.83 / 90% / 72% W8A8 baseline.
- If W8A32 cosine ≈ 0.99+, the W8A8 0.83 deficit is dominated by
  activation quant.
- If W8A32 cosine ≪ 1, weight quant itself is leaking signal —
  per-channel scales help but aren't enough; AdaRound / clipping needed.

Self-contained: bypasses AutoImageProcessor (no torchvision dep) and
downloads a handful of COCO val2017 images directly.
Not committed; diagnostic only.
"""
from __future__ import annotations

import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

sys.path.insert(0, "/home/user/ViT_accelerator/software")
from taccel.quantizer.fake_quant import apply_weight_quantization

MODEL_NAME = "facebook/deit-tiny-patch16-224"
N_IMAGES = 20  # match the user's baseline run

# ImageNet normalization (same as AutoImageProcessor would apply).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_BASE = "http://images.cocodataset.org/val2017/{:012d}.jpg"
COCO_VAL_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1425, 1503, 1532, 1584, 1761, 1818, 1993,
    2006, 2052, 2153, 2261, 2473, 2478, 2532, 2685, 3014, 3501,
    3717, 3845, 4024, 4519, 5037, 5190, 5586, 5802, 6040, 6444,
]


def fetch_image(img_id: int):
    try:
        r = requests.get(COCO_BASE.format(img_id), timeout=15)
        if r.status_code == 200:
            return img_id, Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return img_id, None


def collect_images(n: int):
    print(f"  Downloading up to {n} COCO val2017 images...")
    collected = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_image, i): i for i in COCO_VAL_IDS}
        for fut in as_completed(futures):
            img_id, img = fut.result()
            if img is not None:
                collected.append((img_id, img))
                print(f"    [{len(collected):>2}/{n}] id={img_id:6d} ✓", flush=True)
                if len(collected) >= n:
                    for f in futures:
                        f.cancel()
                    break
    collected.sort(key=lambda x: x[0])
    return collected[:n]


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Resize-shorter-side-to-256, center-crop 224, normalize. Returns [1,3,224,224]."""
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


def fp32_inference(model, pixel_values) -> np.ndarray:
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    return out.logits.squeeze(0).cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    ta = set(np.argsort(a)[-k:].tolist())
    tb = set(np.argsort(b)[-k:].tolist())
    return len(ta & tb) / k


def main() -> int:
    print(f"  Loading FP32 reference model {MODEL_NAME}...")
    model_fp32 = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
    model_fp32.eval()

    print("  Applying per-channel INT8 fake weight quantization...")
    model_wq, n_quantized = apply_weight_quantization(model_fp32)
    model_wq.eval()
    print(f"    Quantized {n_quantized} Linear/Conv2d weight tensors")

    images = collect_images(N_IMAGES)
    if not images:
        print("  ERROR: no images downloaded (no network?).")
        return 1

    cosines, top1_match, top5_overlap = [], 0, []
    for idx, (img_id, img) in enumerate(images, 1):
        px = preprocess_image(img)
        a = fp32_inference(model_fp32, px)
        b = fp32_inference(model_wq, px)
        cs = cosine_sim(a, b)
        cosines.append(cs)
        m = int(np.argmax(a) == np.argmax(b))
        top1_match += m
        ov = topk_overlap(a, b, k=5)
        top5_overlap.append(ov)
        print(f"    {idx:>2}  id={img_id:<6}  cos={cs:.6f}  top1={'✓' if m else '✗'}"
              f"  top5_overlap={ov:.2f}  argmax_fp32={int(np.argmax(a)):>4}"
              f"  argmax_w8={int(np.argmax(b)):>4}")

    arr = np.array(cosines)
    ov_arr = np.array(top5_overlap)
    n = len(images)
    print()
    print(f"  ─── W8A32 results over {n} images ───")
    print(f"    cosine_sim avg : {float(arr.mean()):.4f}")
    print(f"    cosine_sim p10 : {float(np.percentile(arr, 10)):.4f}")
    print(f"    cosine_sim min : {float(arr.min()):.4f}")
    print(f"    top-1 agreement: {top1_match}/{n} ({100.0 * top1_match / n:.0f}%)")
    print(f"    top-5 overlap  : {float(ov_arr.mean()) * 100:.0f}%")
    print()
    print(f"  Baseline W8A8 (per user): 0.83 / 90% top-1 / 72% top-5")
    print(f"  Gap attributable to ACTIVATION quant = W8A32 minus W8A8 numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
