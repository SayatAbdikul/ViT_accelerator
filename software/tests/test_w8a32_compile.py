"""Integration tests for the W8A32 compile + simulate path.

These tests live alongside the W8A8 baseline suite and run only when the
DeiT-tiny weights are reachable (the conftest gate). They prove:

1. ``Compiler(mode='w8a32').compile_w8a32(state_dict)`` emits a ProgramBinary
   whose instruction stream contains no INT8 quant-bracket opcodes
   (REQUANT / REQUANT_PC / DEQUANT_ADD / SOFTMAX_ATTNV).
2. ``SimulatorW8A32`` can execute the full program to HALT on synthetic
   pixel inputs without raising.
3. The classifier logits land in ABUF at the offset recorded in
   ``compiler_manifest['classifier_output']`` and produce non-trivial output.
4. The load-bearing accuracy gate: the W8A32 toolchain reproduces the
   ``fake_quant`` ceiling within tight numerical tolerance
   (cosine ≥ 0.999 vs ``apply_weight_quantization(model)`` forward),
   and the cosine vs the FP32 reference clears the ≥ 0.998 plan threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from taccel.compiler.compiler import Compiler
from taccel.model_config import ModelConfig
from taccel.golden_model.simulator_w8a32 import SimulatorW8A32
from taccel.isa.opcodes import Opcode
from taccel.isa.encoding import decode
from taccel.quantizer.quantize import quantize_tensor, dequantize_tensor
from taccel.quantizer.fake_quant import apply_weight_quantization


def _load_deit_tiny():
    """Load DeiT-tiny weights for compile tests. Skip if unreachable."""
    try:
        from transformers import ViTForImageClassification
    except ImportError:  # pragma: no cover
        pytest.skip("transformers not installed")
    try:
        model = ViTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"DeiT-tiny weights unreachable: {exc}")
    return model


def test_compile_w8a32_no_int8_quant_opcodes():
    """W8A32 program must not contain INT8-quant-bracket instructions."""
    model = _load_deit_tiny()
    c = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a32")
    program = c.compile_w8a32(model.state_dict())

    raw = program.instructions
    forbidden = {Opcode.REQUANT, Opcode.REQUANT_PC, Opcode.DEQUANT_ADD, Opcode.SOFTMAX_ATTNV}
    counts = {op: 0 for op in forbidden}
    for i in range(0, len(raw), 8):
        insn = decode(bytes(raw[i:i + 8]))
        if insn.opcode in forbidden:
            counts[insn.opcode] += 1
    bad = {op.name: cnt for op, cnt in counts.items() if cnt > 0}
    assert not bad, (
        f"W8A32 program contains INT8 quant-bracket opcodes: {bad}"
    )


def test_compile_w8a32_classifier_output_recorded():
    """compiler_manifest['classifier_output'] must point at the ABUF logits."""
    model = _load_deit_tiny()
    c = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a32")
    program = c.compile_w8a32(model.state_dict())
    co = program.compiler_manifest["classifier_output"]
    assert co["buf_id"] == 0  # BUF_ABUF
    assert co["logical_cols"] == 1000  # DeiT-tiny → ImageNet-1k
    assert co["N_pad"] == 1008  # pad_dim(1000)


def test_compile_w8a32_end_to_end_runs_to_halt():
    """Run a synthetic image through compile + simulate and reach HALT."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    model = _load_deit_tiny()
    sd = model.state_dict()
    c = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a32")
    program = c.compile_w8a32(sd)

    # FP32 reference and fake_quant ceiling on the same input
    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 224, 224)
    fq_model, _ = apply_weight_quantization(model)
    with torch.no_grad():
        ref_fp32 = model.eval()(pixel_values=pixel_values).logits.numpy()[0]
        ref_fq = fq_model.eval()(pixel_values=pixel_values).logits.numpy()[0]

    # Host-side patch embedding with fake-quant patch weights to match the
    # in-program dequant FP32 weights for the rest of the network.
    patch_w = sd['vit.embeddings.patch_embeddings.projection.weight'].numpy().astype(np.float32)
    patch_b = sd['vit.embeddings.patch_embeddings.projection.bias'].numpy().astype(np.float32)
    qw, scw = quantize_tensor(patch_w.reshape(patch_w.shape[0], -1), per_channel=True)
    patch_w_dq = dequantize_tensor(qw, scw).astype(np.float32).reshape(patch_w.shape)
    with torch.no_grad():
        patches = F.conv2d(
            pixel_values, torch.from_numpy(patch_w_dq),
            bias=torch.from_numpy(patch_b), stride=16,
        ).flatten(2).transpose(1, 2)[0].numpy().astype(np.float32)

    sim = SimulatorW8A32()
    sim.load_program(program)
    patch_bytes = patches.tobytes()
    sim.state.dram[program.input_offset:program.input_offset + len(patch_bytes)] = patch_bytes
    sim.run(max_instructions=program.insn_count + 10)
    assert sim.state.halted, "W8A32 program did not reach HALT"

    co = program.compiler_manifest["classifier_output"]
    logits = np.frombuffer(
        sim.state.abuf, dtype=np.float32,
        count=co["N_pad"], offset=co["offset_bytes"],
    )[:co["logical_cols"]]
    assert np.isfinite(logits).all(), "W8A32 produced NaN/Inf logits"
    assert np.any(logits != 0.0), "W8A32 produced all-zero logits"

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_fp32 = _cosine(logits, ref_fp32)
    cos_fq = _cosine(logits, ref_fq)

    # Load-bearing accuracy gate: the W8A32 toolchain must reproduce the
    # ``apply_weight_quantization`` ceiling within tight FP32 numerical
    # tolerance, AND match the FP32 reference within the same envelope the
    # fake_quant ceiling clears. The 0.999 vs-fake_quant threshold is the
    # stronger of the two — it asserts the compiler + simulator add no
    # measurable error on top of weight quantization (any drift > 1e-3
    # indicates a real bug like the seq-padding attention leak fixed in
    # codegen_w8a32._emit_qkt).
    assert cos_fq >= 0.999, (
        f"W8A32 cosine vs fake_quant = {cos_fq:.6f} below the 0.999 gate; "
        f"a regression has reintroduced numerical drift between the W8A32 "
        f"toolchain and the fake_quant reference."
    )
    assert cos_fp32 >= 0.998, (
        f"W8A32 cosine vs FP32 = {cos_fp32:.6f} below the 0.998 plan gate"
    )
