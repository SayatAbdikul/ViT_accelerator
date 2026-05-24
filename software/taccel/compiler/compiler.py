"""Top-level compiler: PyTorch model → ProgramBinary.

Modes
-----
- ``"w8a16"`` (default, shipping) — INT8 weights (per-channel) dequantized
  into FP16 DRAM, FP16 activations, FP32 accumulator. Software-only:
  matches the ``fake_quant`` reference within FP16 narrowing tolerance.
  ``software/taccel/compiler/codegen_w8a16.py`` and
  ``software/taccel/golden_model/simulator_w8a16.py``.

- ``"w8a32"`` — INT8 weights, FP32 activations, FP32 accumulator. Doubles
  the dequant-weight DRAM footprint vs w8a16 in exchange for an extra
  decimal digit of numerical headroom; used as the weight-quant accuracy
  ceiling reference.
"""
import struct
import numpy as np
from typing import Any, Dict, Literal, Optional
from ..assembler.assembler import ProgramBinary
from ..isa.encoding import encode
from ..isa.opcodes import Opcode, OPCODE_SHIFT, OPCODE_MASK, A_IMM28_SHIFT, MASK_28BIT
from ..quantizer.quantize import quantize_tensor, dequantize_tensor
from ..quantizer.scales import ScalePropagator
from ..model_config import ModelConfig
from .graph_extract import extract_vit_graph
from .codegen_w8a32 import CodeGeneratorW8A32
from .codegen_w8a16 import CodeGeneratorW8A16
from .passes import run_passes, default_pipeline_w8a32, default_pipeline_w8a16


class Compiler:
    """Compile a ViT-family model to ProgramBinary.

    A :class:`ModelConfig` controls the model dimensions; defaults to
    DeiT-tiny for backward compatibility with existing call sites.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None,
                 mode: Literal["w8a16", "w8a32"] = "w8a16"):
        if mode not in ("w8a16", "w8a32"):
            raise ValueError(
                f"mode must be 'w8a16' or 'w8a32', got {mode!r}"
            )
        self.cfg = cfg if cfg is not None else ModelConfig.deit_tiny()
        self.mode = mode
        self.scale_prop = ScalePropagator()

    # ── W8A32 path ────────────────────────────────────────────────────────

    def compile_w8a32(self, state_dict: dict) -> ProgramBinary:
        """Compile in W8A32 mode (INT8 weights, FP32 activations).

        Per-channel INT8 weight quantization is applied to every linear /
        embedding tensor; the result is dequantized back to FP32 and
        stored in DRAM. The systolic array then runs FP32 × FP32 → FP32,
        mirroring the ``fake_quant`` reference and bypassing all
        activation-side calibration / requant plumbing.
        """
        if self.mode != "w8a32":
            raise RuntimeError(
                "compile_w8a32 requires Compiler(mode='w8a32'); "
                f"current mode is {self.mode!r}"
            )

        fp32_weight_data: Dict[str, np.ndarray] = {}
        fp32_biases: Dict[str, np.ndarray] = {}
        prefix_root = self.cfg.module_prefix
        head_dim = self.cfg.head_dim

        for name, tensor in state_dict.items():
            if not hasattr(tensor, 'numpy'):
                continue
            arr = tensor.numpy().astype(np.float32)

            if 'cls_token' in name or 'position_embeddings' in name:
                while arr.ndim > 2:
                    arr = arr.squeeze(0)
                pad_row = (16 - arr.shape[0] % 16) % 16
                pad_col = (16 - arr.shape[1] % 16) % 16
                if pad_row or pad_col:
                    arr = np.pad(arr, ((0, pad_row), (0, pad_col)), mode='constant')
                fp32_weight_data[name] = arr.astype(np.float32)
                continue

            if 'weight' in name and arr.ndim >= 2:
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                q, scales = quantize_tensor(arr, per_channel=True)
                w_dq = dequantize_tensor(q, scales).astype(np.float32)
                w_dq = np.pad(
                    w_dq,
                    ((0, (16 - w_dq.shape[0] % 16) % 16),
                     (0, (16 - w_dq.shape[1] % 16) % 16)),
                    mode='constant',
                )
                w_dq = np.ascontiguousarray(w_dq.T)
                fp32_weight_data[name] = w_dq

            elif 'bias' in name and arr.ndim == 1:
                weight_name = name.replace('.bias', '.weight')
                if weight_name in state_dict:
                    w = state_dict[weight_name]
                    if hasattr(w, 'numpy') and w.numpy().ndim >= 2:
                        pad_len = (16 - len(arr) % 16) % 16
                        if pad_len:
                            arr = np.pad(arr, (0, pad_len), constant_values=0.0)
                        fp32_biases[name] = arr.astype(np.float32)
                    else:
                        # LayerNorm beta — FP16 to match the gamma convention.
                        fp32_weight_data[name] = arr.astype(np.float16)
                else:
                    fp32_biases[name] = arr.astype(np.float32)

            elif arr.ndim <= 2:
                # LayerNorm gamma and any other small 1D/2D tensor → FP16.
                fp32_weight_data[name] = arr.astype(np.float16)

        for layer_idx in range(self.cfg.depth):
            prefix = f"{prefix_root}.encoder.layer.{layer_idx}"
            for proj in ("query", "key", "value"):
                wname = f"{prefix}.attention.attention.{proj}.weight"
                bname = f"{prefix}.attention.attention.{proj}.bias"
                if wname not in state_dict:
                    continue
                w_full = state_dict[wname].numpy().astype(np.float32)
                b_full = (
                    state_dict[bname].numpy().astype(np.float32)
                    if bname in state_dict else None
                )
                for h in range(self.cfg.num_heads):
                    head_w = w_full[h * head_dim:(h + 1) * head_dim, :]
                    q, scales = quantize_tensor(head_w, per_channel=True)
                    head_w_dq = dequantize_tensor(q, scales).astype(np.float32)
                    head_w_dq = np.pad(
                        head_w_dq,
                        ((0, (16 - head_w_dq.shape[0] % 16) % 16),
                         (0, (16 - head_w_dq.shape[1] % 16) % 16)),
                        mode='constant',
                    )
                    head_w_dq = np.ascontiguousarray(head_w_dq.T)
                    fp32_weight_data[f"{wname}_h{h}"] = head_w_dq

                    if b_full is not None:
                        b_h = b_full[h * head_dim:(h + 1) * head_dim].astype(np.float32)
                        pad_len = (16 - len(b_h) % 16) % 16
                        if pad_len:
                            b_h = np.pad(b_h, (0, pad_len), constant_values=0.0)
                        fp32_biases[f"{bname}_h{h}"] = b_h

        graph = extract_vit_graph(self.cfg)
        ctx: Dict[str, Any] = {}
        graph = run_passes(graph, self.cfg, ctx, pipeline=default_pipeline_w8a32())

        codegen = CodeGeneratorW8A32(
            fp32_weight_data=fp32_weight_data,
            fp32_biases=fp32_biases,
            cfg=self.cfg,
        )
        instructions, dram_data = codegen.generate(graph)

        dram_temp_size = codegen.mem.dram_temp_total
        if dram_temp_size > 0:
            dram_data = dram_data + bytes(dram_temp_size)

        return self._finalize_program(
            instructions, dram_data, codegen, ctx,
            mode_label="w8a32", pos_embed_elem_bytes=4,
        )

    # ── W8A16 path (default) ──────────────────────────────────────────────

    def compile_w8a16(self, state_dict: dict) -> ProgramBinary:
        """Compile in W8A16 mode (INT8 weights, FP16 activations, FP32 accumulator).

        Mirrors :meth:`compile_w8a32` but narrows every dequantized weight
        and bias to FP16 before placing it in DRAM. The systolic array
        widens both FP16 operands back to FP32 for the multiply-accumulate
        (the standard mixed-precision convention) so the accumulator and
        SFU internal math stay in FP32.
        """
        if self.mode != "w8a16":
            raise RuntimeError(
                "compile_w8a16 requires Compiler(mode='w8a16'); "
                f"current mode is {self.mode!r}"
            )

        fp16_weight_data: Dict[str, np.ndarray] = {}
        fp16_biases: Dict[str, np.ndarray] = {}
        prefix_root = self.cfg.module_prefix
        head_dim = self.cfg.head_dim

        for name, tensor in state_dict.items():
            if not hasattr(tensor, 'numpy'):
                continue
            arr = tensor.numpy().astype(np.float32)

            if 'cls_token' in name or 'position_embeddings' in name:
                while arr.ndim > 2:
                    arr = arr.squeeze(0)
                pad_row = (16 - arr.shape[0] % 16) % 16
                pad_col = (16 - arr.shape[1] % 16) % 16
                if pad_row or pad_col:
                    arr = np.pad(arr, ((0, pad_row), (0, pad_col)), mode='constant')
                fp16_weight_data[name] = arr.astype(np.float16)
                continue

            if 'weight' in name and arr.ndim >= 2:
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                q, scales = quantize_tensor(arr, per_channel=True)
                w_dq = dequantize_tensor(q, scales).astype(np.float32)
                w_dq = np.pad(
                    w_dq,
                    ((0, (16 - w_dq.shape[0] % 16) % 16),
                     (0, (16 - w_dq.shape[1] % 16) % 16)),
                    mode='constant',
                )
                w_dq = np.ascontiguousarray(w_dq.T)
                # Narrow to FP16 only at the very end so the per-channel
                # scale × INT8 quantize step stays in FP32 (matches the
                # fake_quant reference; the FP16 narrow rounding is the
                # only added error vs the W8A32 path).
                fp16_weight_data[name] = w_dq.astype(np.float16)

            elif 'bias' in name and arr.ndim == 1:
                weight_name = name.replace('.bias', '.weight')
                if weight_name in state_dict:
                    w = state_dict[weight_name]
                    if hasattr(w, 'numpy') and w.numpy().ndim >= 2:
                        pad_len = (16 - len(arr) % 16) % 16
                        if pad_len:
                            arr = np.pad(arr, (0, pad_len), constant_values=0.0)
                        fp16_biases[name] = arr.astype(np.float16)
                    else:
                        fp16_weight_data[name] = arr.astype(np.float16)
                else:
                    fp16_biases[name] = arr.astype(np.float16)

            elif arr.ndim <= 2:
                fp16_weight_data[name] = arr.astype(np.float16)

        for layer_idx in range(self.cfg.depth):
            prefix = f"{prefix_root}.encoder.layer.{layer_idx}"
            for proj in ("query", "key", "value"):
                wname = f"{prefix}.attention.attention.{proj}.weight"
                bname = f"{prefix}.attention.attention.{proj}.bias"
                if wname not in state_dict:
                    continue
                w_full = state_dict[wname].numpy().astype(np.float32)
                b_full = (
                    state_dict[bname].numpy().astype(np.float32)
                    if bname in state_dict else None
                )
                for h in range(self.cfg.num_heads):
                    head_w = w_full[h * head_dim:(h + 1) * head_dim, :]
                    q, scales = quantize_tensor(head_w, per_channel=True)
                    head_w_dq = dequantize_tensor(q, scales).astype(np.float32)
                    head_w_dq = np.pad(
                        head_w_dq,
                        ((0, (16 - head_w_dq.shape[0] % 16) % 16),
                         (0, (16 - head_w_dq.shape[1] % 16) % 16)),
                        mode='constant',
                    )
                    head_w_dq = np.ascontiguousarray(head_w_dq.T)
                    fp16_weight_data[f"{wname}_h{h}"] = head_w_dq.astype(np.float16)

                    if b_full is not None:
                        b_h = b_full[h * head_dim:(h + 1) * head_dim].astype(np.float32)
                        pad_len = (16 - len(b_h) % 16) % 16
                        if pad_len:
                            b_h = np.pad(b_h, (0, pad_len), constant_values=0.0)
                        fp16_biases[f"{bname}_h{h}"] = b_h.astype(np.float16)

        graph = extract_vit_graph(self.cfg)
        ctx: Dict[str, Any] = {}
        graph = run_passes(graph, self.cfg, ctx, pipeline=default_pipeline_w8a16())

        codegen = CodeGeneratorW8A16(
            fp32_weight_data=fp16_weight_data,
            fp32_biases=fp16_biases,
            cfg=self.cfg,
        )
        instructions, dram_data = codegen.generate(graph)

        dram_temp_size = codegen.mem.dram_temp_total
        if dram_temp_size > 0:
            dram_data = dram_data + bytes(dram_temp_size)

        return self._finalize_program(
            instructions, dram_data, codegen, ctx,
            mode_label="w8a16", pos_embed_elem_bytes=2,
        )

    # ── shared finalization ───────────────────────────────────────────────

    def _finalize_program(self, instructions, dram_data, codegen, ctx,
                          *, mode_label: str, pos_embed_elem_bytes: int) -> ProgramBinary:
        """Encode instructions, patch SET_ADDR_LO immediates to absolute DRAM
        offsets, compute the framing-metadata fields, and assemble the
        :class:`ProgramBinary`. Shared between :meth:`compile_w8a32` and
        :meth:`compile_w8a16`; the only mode-dependent input is the element
        width used to compute the per-patch position-embedding offset."""
        prefix_root = self.cfg.module_prefix

        insn_bytes = bytearray()
        for insn in instructions:
            insn_bytes.extend(encode(insn))

        data_base = (len(insn_bytes) + 15) & ~15
        patched = bytearray(insn_bytes)
        for i in range(0, len(patched), 8):
            word = struct.unpack(">Q", patched[i:i + 8])[0]
            opcode_val = (word >> OPCODE_SHIFT) & OPCODE_MASK
            if opcode_val == Opcode.SET_ADDR_LO:
                old_imm28 = (word >> A_IMM28_SHIFT) & MASK_28BIT
                new_imm28 = (old_imm28 + data_base) & MASK_28BIT
                word = (word & ~(MASK_28BIT << A_IMM28_SHIFT)) | (new_imm28 << A_IMM28_SHIFT)
                patched[i:i + 8] = struct.pack(">Q", word)

        input_patches_dram_off = codegen.dram_layout.get("__input_patches__", 0)
        input_offset = data_base + input_patches_dram_off

        pos_emb_key = f"{prefix_root}.embeddings.position_embeddings"
        pos_emb_dram_start = codegen.dram_layout.get(pos_emb_key)
        if pos_emb_dram_start is not None:
            pos_embed_cls_dram_offset = data_base + pos_emb_dram_start
            pos_embed_patch_dram_offset = (
                data_base + pos_emb_dram_start + self.cfg.embed_dim * pos_embed_elem_bytes
            )
        else:
            pos_embed_cls_dram_offset = 0
            pos_embed_patch_dram_offset = 0
        cls_token_dram_start = codegen.dram_layout.get(
            f"{prefix_root}.embeddings.cls_token"
        )
        cls_token_dram_offset = (
            data_base + cls_token_dram_start if cls_token_dram_start is not None else 0
        )

        return ProgramBinary(
            instructions=bytes(patched),
            data=dram_data,
            entry_point=0,
            insn_count=len(instructions),
            data_base=data_base,
            input_offset=input_offset,
            pos_embed_patch_dram_offset=pos_embed_patch_dram_offset,
            pos_embed_cls_dram_offset=pos_embed_cls_dram_offset,
            cls_token_dram_offset=cls_token_dram_offset,
            trace_manifest=codegen.trace_manifest,
            compiler_manifest={
                "manifest_version": 1,
                "compiler": {"class": self.__class__.__name__, "mode": mode_label},
                "program_layout": {
                    "data_base": int(data_base),
                    "input_offset": int(input_offset),
                    "pos_embed_patch_dram_offset": int(pos_embed_patch_dram_offset),
                    "pos_embed_cls_dram_offset": int(pos_embed_cls_dram_offset),
                    "cls_token_dram_offset": int(cls_token_dram_offset),
                    "dram_temp_total": int(codegen.mem.dram_temp_total),
                },
                "seq_tiling": (
                    {
                        "needs_tiling": bool(ctx["seq_tiling_decision"].needs_tiling),
                        "tile_rows": int(ctx["seq_tiling_decision"].tile_rows),
                        "num_tiles": int(ctx["seq_tiling_decision"].num_tiles),
                    }
                    if "seq_tiling_decision" in ctx else {}
                ),
                "classifier_output": codegen.classifier_output or {},
            },
        )
