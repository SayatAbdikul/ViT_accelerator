#!/usr/bin/env python3
"""W32A32 sanity check — proves fp32 inference is deterministic.

If this returns cosine != 1.0 (within float epsilon), the comparison rig
itself is broken and any W8A32 / W8A8 number we collect later is suspect.

Not committed; standalone tool.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForImageClassification

MODEL_NAME = "facebook/deit-tiny-patch16-224"
N_IMAGES = 8
RNG_SEED = 0

# ImageNet normalization (same as AutoImageProcessor would apply for DeiT-tiny).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_pixel_values(n: int, seed: int = 0) -> torch.Tensor:
    """Build [n, 3, 224, 224] float32 tensors with ImageNet normalization."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i in range(n):
        # Random uint8 image in HWC, scale to [0,1], normalize, transpose to CHW.
        u8 = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        x = u8.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = x.transpose(2, 0, 1)
    return torch.from_numpy(out)


def fp32_inference(model, pixel_values_1x3x224x224: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values_1x3x224x224)
    return outputs.logits.squeeze(0).cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    torch.manual_seed(RNG_SEED)
    print(f"  Loading {MODEL_NAME} from HF cache...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
    model.eval()

    pixel_values = make_pixel_values(N_IMAGES, seed=RNG_SEED)

    cosines = []
    max_abs_diff = 0.0
    for idx in range(N_IMAGES):
        px = pixel_values[idx:idx + 1]
        a = fp32_inference(model, px)
        b = fp32_inference(model, px)
        cs = cosine_sim(a, b)
        cosines.append(cs)
        diff = float(np.max(np.abs(a - b)))
        max_abs_diff = max(max_abs_diff, diff)
        print(f"    img {idx}: cosine={cs:.10f}  max|a-b|={diff:.3e}  argmax_a={int(np.argmax(a))} argmax_b={int(np.argmax(b))}")

    arr = np.array(cosines)
    print()
    print(f"  W32A32 sanity over {N_IMAGES} images:")
    print(f"    cosine_sim avg : {float(arr.mean()):.10f}")
    print(f"    cosine_sim min : {float(arr.min()):.10f}")
    print(f"    max|a-b|       : {max_abs_diff:.3e}")

    # Pass if every cosine is within 1e-6 of 1.0 (float epsilon tolerance).
    ok = bool(arr.min() >= 1.0 - 1e-6)
    print()
    print(f"  → SANITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
