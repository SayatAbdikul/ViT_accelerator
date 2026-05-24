"""Tests for the W8A16 / W8A32 weight-quantization entry points and the
mode-agnostic quantize_tensor / ScalePropagator utilities they depend on."""
import numpy as np
import torch
import torch.nn as nn

from taccel.quantizer.quantize import dequantize_tensor, quantize_tensor
from taccel.quantizer.scales import ScalePropagator


class TestQuantize:
    def test_basic_quantization(self):
        """Quantize and dequantize, error ≤ 1 LSB."""
        np.random.seed(42)
        W = np.random.randn(64, 128).astype(np.float32) * 2.0
        q, scales = quantize_tensor(W)

        assert q.dtype == np.int8
        assert scales.dtype == np.float16
        assert q.shape == W.shape
        assert len(scales) == 64  # per-channel

        W_rec = dequantize_tensor(q, scales)
        lsb = scales.astype(np.float32).reshape(-1, 1)
        err = np.abs(W - W_rec)
        assert np.all(err <= lsb), f"Max error {err.max():.6f} > 1 LSB {lsb.max():.6f}"

    def test_quantize_range(self):
        """All quantized values in [-128, 127]."""
        W = np.random.randn(32, 64).astype(np.float32) * 10.0
        q, _ = quantize_tensor(W)
        assert q.min() >= -128 and q.max() <= 127

    def test_per_channel_scales(self):
        """Each channel has independent scale."""
        W = np.zeros((4, 8), dtype=np.float32)
        W[0, 0] = 1.0
        W[1, 0] = 10.0
        W[2, 0] = 100.0
        W[3, 0] = 0.01
        _, scales = quantize_tensor(W)
        assert abs(float(scales[0]) - 1.0 / 127) < 1e-4
        assert abs(float(scales[1]) - 10.0 / 127) < 0.1
        assert abs(float(scales[2]) - 100.0 / 127) < 1.0

    def test_zero_tensor(self):
        """Zero tensor quantizes to zeros."""
        W = np.zeros((8, 16), dtype=np.float32)
        q, _ = quantize_tensor(W)
        assert np.all(q == 0)

    def test_conv_reshape(self):
        """Conv2d weights reshaped to 2D before quantization."""
        W = np.random.randn(192, 3, 16, 16).astype(np.float32)
        W_2d = W.reshape(192, -1)
        q, scales = quantize_tensor(W_2d)
        assert q.shape == (192, 768)
        assert len(scales) == 192


class TestScalePropagator:
    def test_prescale_bias(self):
        """Bias pre-scaling: bias_int32[ch] = round(bias_fp32[ch] / (act_scale * w_scale[ch]))."""
        sp = ScalePropagator()
        bias_fp32 = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        act_scale = np.array([0.1])
        w_scales = np.array([0.05, 0.1, 0.02], dtype=np.float32)
        bias_int32 = sp.prescale_bias(bias_fp32, act_scale, w_scales)
        assert bias_int32.dtype == np.int32
        for ch in range(3):
            recovered = bias_int32[ch] * float(act_scale[0]) * float(w_scales[ch])
            assert abs(recovered - bias_fp32[ch]) < 0.01

    def test_matmul_output_scale(self):
        sp = ScalePropagator()
        act_scale = np.array([0.05])
        w_scales = np.array([0.1, 0.2], dtype=np.float32)
        out_scale = sp.compute_matmul_output_scale(act_scale, w_scales)
        expected = np.array([0.005, 0.01])
        np.testing.assert_allclose(out_scale, expected, rtol=1e-5)


class TestW8A32QuantizeEntryPoint:
    """The W8A32 fork's weight-quant contract.

    ``W8A32_QUANTIZE`` is exported from ``taccel.quantizer`` as the single
    canonical entry point for the W8A32 path. It must be bit-identical to
    ``fake_quant.apply_weight_quantization`` (which is the reference the
    plan's accuracy ceiling was achieved against) so the compiler's
    in-program dequant weights match the fake-quant reference exactly.
    """

    def test_w8a32_quantize_export_is_importable(self):
        from taccel.quantizer import W8A32_QUANTIZE
        assert callable(W8A32_QUANTIZE)

    def test_w8a32_quantize_matches_per_channel_call(self):
        from taccel.quantizer import W8A32_QUANTIZE
        np.random.seed(123)
        w = np.random.randn(48, 96).astype(np.float32) * 2.5
        q_export, s_export = W8A32_QUANTIZE(w)
        q_direct, s_direct = quantize_tensor(w, per_channel=True)
        np.testing.assert_array_equal(q_export, q_direct)
        np.testing.assert_array_equal(s_export, s_direct)

    def test_w8a32_matches_fake_quant_bitwise(self):
        """``W8A32_QUANTIZE`` and ``apply_weight_quantization`` must produce
        bit-identical INT8 tensors and bit-identical dequantized FP32 weights
        for every Linear/Conv2d weight in a small synthetic model."""
        from taccel.quantizer import W8A32_QUANTIZE
        from taccel.quantizer.fake_quant import apply_weight_quantization

        torch.manual_seed(7)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 96)
                self.fc2 = nn.Linear(96, 32)
                self.conv = nn.Conv2d(3, 16, kernel_size=4, stride=2)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = Tiny()
        fq_model, count = apply_weight_quantization(model)
        assert count == 3, "All three weight modules must be quantized"

        for (name_orig, m_orig), (name_fq, m_fq) in zip(
            model.named_modules(), fq_model.named_modules()
        ):
            assert name_orig == name_fq
            if not isinstance(m_orig, (nn.Linear, nn.Conv2d)):
                continue
            w = m_orig.weight.detach().cpu().numpy().astype(np.float32)
            orig_shape = w.shape
            w2 = w.reshape(orig_shape[0], -1) if w.ndim > 2 else w
            q, scales = W8A32_QUANTIZE(w2)
            w_dq = dequantize_tensor(q, scales).astype(np.float32).reshape(orig_shape)
            w_fq = m_fq.weight.detach().cpu().numpy().astype(np.float32)
            np.testing.assert_array_equal(w_dq, w_fq), (
                f"{name_orig}: dequant mismatch between W8A32_QUANTIZE and "
                f"apply_weight_quantization — the bit-equivalence contract "
                f"that the plan's accuracy ceiling depends on is broken."
            )


class TestW8A16QuantizeEntryPoint:
    """W8A16 uses the same per-channel INT8 weight scheme as W8A32.

    ``W8A16_QUANTIZE`` is an alias for ``W8A32_QUANTIZE`` — exported under
    its mode-named identity so the parallel-modules pattern stays clean
    and so any future W8A16-specific weight-quant policy (e.g. clip-search
    tuned for FP16-narrowed dequant) has an obvious place to land.
    """

    def test_w8a16_quantize_is_exported_and_aliases_w8a32(self):
        from taccel.quantizer import W8A16_QUANTIZE, W8A32_QUANTIZE
        assert callable(W8A16_QUANTIZE)
        assert W8A16_QUANTIZE is W8A32_QUANTIZE

    def test_w8a16_quantize_matches_fake_quant_then_fp16_narrow(self):
        """W8A16_QUANTIZE → dequantize → narrow to FP16 must match what
        ``compile_w8a16`` stores in DRAM, which is exactly what the W8A16
        accuracy ceiling depends on."""
        from taccel.quantizer import W8A16_QUANTIZE
        from taccel.quantizer.fake_quant import apply_weight_quantization

        torch.manual_seed(11)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(48, 64)
                self.conv = nn.Conv2d(3, 8, kernel_size=4, stride=2)

            def forward(self, x):
                return self.fc(x)

        model = Tiny()
        fq_model, count = apply_weight_quantization(model)
        assert count == 2

        for (name_orig, m_orig), (name_fq, m_fq) in zip(
            model.named_modules(), fq_model.named_modules()
        ):
            assert name_orig == name_fq
            if not isinstance(m_orig, (nn.Linear, nn.Conv2d)):
                continue
            w = m_orig.weight.detach().cpu().numpy().astype(np.float32)
            orig_shape = w.shape
            w2 = w.reshape(orig_shape[0], -1) if w.ndim > 2 else w
            q, scales = W8A16_QUANTIZE(w2)
            w_dq = dequantize_tensor(q, scales).astype(np.float16).astype(np.float32).reshape(orig_shape)
            w_fq = m_fq.weight.detach().cpu().numpy().astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(w_dq, w_fq), (
                f"{name_orig}: dequant→fp16 mismatch between W8A16_QUANTIZE and "
                f"apply_weight_quantization → fp16"
            )
