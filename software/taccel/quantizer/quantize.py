"""Per-channel symmetric INT8 weight quantization."""
import numpy as np
from typing import Tuple


def quantize_tensor(tensor: np.ndarray, per_channel: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a 2D tensor to INT8 with per-channel symmetric quantization.

    Args:
        tensor: FP32 tensor of shape [out_channels, in_features]
        per_channel: if True, compute scale per output channel

    Returns:
        (int8_tensor, scales): quantized tensor and per-channel FP16 scales
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)

    if per_channel:
        max_vals = np.max(np.abs(tensor), axis=1)
        max_vals = np.maximum(max_vals, 1e-8)
        scales = max_vals / 127.0
    else:
        max_val = max(np.max(np.abs(tensor)), 1e-8)
        scales = np.full(tensor.shape[0], max_val / 127.0)

    scales_expanded = scales.reshape(-1, 1)
    q = np.clip(np.round(tensor / scales_expanded), -128, 127).astype(np.int8)

    return q, scales.astype(np.float16)


def dequantize_tensor(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize INT8 tensor back to FP32."""
    if q.ndim == 1:
        return q.astype(np.float32) * float(scales[0])
    scales_expanded = scales.astype(np.float32).reshape(-1, 1)
    return q.astype(np.float32) * scales_expanded
