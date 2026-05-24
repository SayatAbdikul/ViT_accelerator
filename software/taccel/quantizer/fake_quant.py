"""Fake quantization utilities: simulate the weight-only quantization
scheme used by the W8A16 / W8A32 paths inside a PyTorch model so we can
measure quantization error without running the accelerator simulator."""
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from .quantize import quantize_tensor, dequantize_tensor


def _quantize_dequantize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Apply per-channel INT8 quantization then dequantize back to FP32.

    Models the precision loss that would occur on hardware: the weight is
    rounded to the nearest INT8 value and then scaled back, leaving only
    the rounding error.
    """
    w_np = weight.detach().cpu().numpy().astype(np.float32)
    orig_shape = w_np.shape
    if w_np.ndim > 2:
        w_np = w_np.reshape(orig_shape[0], -1)

    q, scales = quantize_tensor(w_np, per_channel=True)
    w_rec = dequantize_tensor(q, scales).astype(np.float32)
    w_rec = w_rec.reshape(orig_shape)

    return torch.from_numpy(w_rec).to(weight.device).to(weight.dtype)


def apply_weight_quantization(model: nn.Module):
    """Return a copy of ``model`` with every Linear/Conv2d weight fake-quantized.

    Captures all weight rounding error with zero code change to the model
    forward pass itself. Returns ``(quantized_model, num_modules_quantized)``.
    """
    model_q = copy.deepcopy(model)
    quantized_count = 0
    for _, module in model_q.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = _quantize_dequantize_weight(module.weight.data)
            quantized_count += 1
    return model_q, quantized_count


def compute_metrics(
    logits_fp32: np.ndarray,
    logits_q: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy and similarity metrics between FP32 and quantized logits.

    Returns dict with: top1_match, top5_match, cosine_sim, logit_mse,
                       logit_mae, softmax_kl_div, logit_snr_db.
    """
    from scipy.spatial.distance import cosine

    results = {}
    top1_fp32 = int(np.argmax(logits_fp32))
    top1_q = int(np.argmax(logits_q))
    results["top1_match"] = (top1_fp32 == top1_q)
    results["top1_fp32"] = top1_fp32
    results["top1_q"] = top1_q

    top5_fp32 = set(np.argsort(logits_fp32)[-5:])
    top5_q = set(np.argsort(logits_q)[-5:])
    results["top5_match"] = len(top5_fp32 & top5_q) >= 1

    cos_dist = cosine(logits_fp32.astype(np.float64), logits_q.astype(np.float64))
    results["cosine_sim"] = float(1.0 - cos_dist)

    diff = logits_fp32.astype(np.float64) - logits_q.astype(np.float64)
    results["logit_mse"] = float(np.mean(diff ** 2))
    results["logit_mae"] = float(np.mean(np.abs(diff)))

    signal_power = float(np.mean(logits_fp32.astype(np.float64) ** 2))
    noise_power = float(np.mean(diff ** 2))
    if noise_power > 0 and signal_power > 0:
        results["logit_snr_db"] = float(10.0 * np.log10(signal_power / noise_power))
    else:
        results["logit_snr_db"] = float("inf")

    def safe_softmax(x):
        x = x.astype(np.float64)
        x = x - x.max()
        e = np.exp(np.clip(x, -500, 0))
        return e / e.sum()

    p_fp32 = safe_softmax(logits_fp32)
    p_q = safe_softmax(logits_q)
    eps = 1e-12
    kl = float(np.sum(p_fp32 * np.log((p_fp32 + eps) / (p_q + eps))))
    results["softmax_kl_div"] = kl

    return results
