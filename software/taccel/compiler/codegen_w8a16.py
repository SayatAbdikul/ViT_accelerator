"""W8A16 code generator: parallel implementation of :mod:`codegen` for
the FP16-activation, INT8-weight-fake-quant path.

Cloned from :mod:`codegen_w8a32`; the per-site delta is element width.
Key differences from :class:`codegen.CodeGenerator`:

1. **Dequantized FP16 weights.** Weights are stored in DRAM as FP16 tensors
   that carry the INT8 rounding error (``w_int8 * w_scale``) and a small
   additional FP16-narrowing error. The systolic array widens both
   FP16 operands to FP32 for the multiply-accumulate, matching the
   standard mixed-precision convention. ACCUM stays FP32.

2. **No REQUANT / REQUANT_PC / DEQUANT_ADD / SCALE_MUL clip ops.** All
   activations flow as FP16 end-to-end (with FP32 internal reductions
   in the SFU); the scale-management plumbing that bridges INT8
   intermediates in W8A8 has no role.

3. **ABUF tiles are 2 bytes/element.** Every allocation that holds
   activations multiplies the per-tile element count by 2.

4. **Sequence tiling still triggers** even though the DeiT-tiny FP16
   residual (79.9 KB) fits in ABUF — the FC1 / classifier output
   ``[tile, mlp_dim] × 2`` exceeds the per-tile ABUF/2 cap. The W8A32
   tiled IR ops (``init_residual_tile``, ``tile_load``, ``tile_save``,
   ``concat_heads_tile`` and the per-tile versions of matmul/SFU/vadd)
   carry over unchanged; only the policy in
   ``memory_estimate_w8a16.decide_seq_tiling_w8a16`` flips.

5. **No trace manifest, no fused-path optimizations.** Same as W8A32.

6. **Bias added via VADD broadcast in ACCUM.** Biases are stored as FP16
   row vectors in DRAM; after MATMUL they are broadcast-added to the
   FP32 ACCUM (FP16 widen on read), then moved to ABUF via a flat
   BUF_COPY which preserves the FP32 bit pattern through the int32-
   typed ACCUM bytes and then narrows to FP16 on the downstream consumer.

7. **Attention mask uses -65504.0 (FP16 minimum).** Sufficient to
   underflow ``exp()`` to zero in the downstream softmax — same masking
   semantics as W8A32's -1e9 FP32, just clamped to FP16's representable
   range.

8. **Deferred-V load machinery is preserved structurally but never
   triggers.** With FP16 activations the per-head Q+K+V coexists in
   ABUF (3 × ~25 KB), so eager loading is correct. We keep the dispatch
   path so the codegen surface matches W8A32 line-for-line; the marking
   pass is gated off in :meth:`_mark_deferred_loads`.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..isa.opcodes import BUF_ABUF, BUF_WBUF, BUF_ACCUM
from ..isa.instructions import (
    MatmulInsn, VaddInsn, ScaleMulInsn, SoftmaxInsn, LayernormInsn, GeluInsn,
    LoadInsn, StoreInsn, BufCopyInsn,
    ConfigTileInsn, SyncInsn, NopInsn, HaltInsn, Instruction,
)
from ..model_config import ModelConfig
from .ir import IRNode, IRGraph
from .tiler import pad_dim, TILE
from .memory_alloc import MemoryAllocator, Allocation
from .sreg_allocator import SRegAllocator
from .scale_emitter import emit_scale
from .dma_emitter import AddrPlanner
from .sync_coalesce import coalesce_dma_syncs as _coalesce_dma_syncs

UNIT = 16
ELEM_BYTES = 2  # FP16 activations: 2 bytes per element


class CodeGeneratorW8A16:
    """Generate W8A16 ISA instructions from a seq-tiled IR graph.

    Inputs:
      * ``fp32_weight_data``: ``{name: fp16_ndarray}`` of dequantized FP16
        weights pre-padded and pre-transposed to the systolic layout
        ([K_pad, N_pad] for 2D weights). LayerNorm gamma/beta are stored
        as FP16 (matching the SFU convention). The field is named with
        the ``fp32`` prefix to match the W8A32 codegen API surface; in
        W8A16 the ndarrays are FP16.
      * ``fp32_biases``: ``{name: fp16_ndarray}`` of FP16 bias row vectors
        already padded to ``N_pad`` elements. Same naming convention.
      * ``cfg``: model dimensions.
    """

    def __init__(self,
                 fp32_weight_data: Dict[str, np.ndarray],
                 fp32_biases: Dict[str, np.ndarray],
                 cfg: ModelConfig):
        self.cfg = cfg
        self.fp32_weight_data = fp32_weight_data
        self.fp32_biases = fp32_biases
        self.mem = MemoryAllocator()
        self.instructions: List[Instruction] = []
        self.dram_layout: Dict[str, int] = {}
        self.dram_blob = bytearray()
        self.sregs = SRegAllocator()
        self.staging_dram_offsets: Dict[str, int] = {}
        # Tracks current value of each addr_reg so SET_ADDR_LO/HI writes
        # that wouldn't change the register are elided, and successive
        # DMA accesses within a 1 MB window of a cached base reuse it via
        # the M-type ``dram_off`` field.
        self.addr_planner = AddrPlanner()
        # W8A16 has no trace gate; manifest stays empty so downstream consumers
        # (ProgramBinary) see a structurally-valid but empty trace surface.
        self.trace_manifest: Dict[int, List[Dict[str, Any]]] = {}
        # Deferred-V load machinery is preserved structurally but never
        # populated in W8A16 — FP16 Q+K+V coexist comfortably in ABUF, so
        # eager V loading is fine. See :meth:`_mark_deferred_loads`.
        self._deferred_loads: Dict[str, IRNode] = {}
        # Sreg 15 is outside the SRegAllocator pools (ODD=1,3,5,7,9,11,13;
        # PAIR/QUAD use even slots ≤ 12) so we reserve it as the constant
        # 1.0 used by FP32→FP16 narrowing SCALE_MULs (the W8A16 replacement
        # for W8A32's flat BUF_COPY out of ACCUM). The SET_SCALE for this
        # register is emitted once at the top of ``generate``.
        self._scale_one_sreg: int = 15
        # Populated by ``generate`` so callers can fetch the FP16 logit row.
        self.classifier_output: Optional[Dict[str, int]] = None

    # ── public entry point ────────────────────────────────────────────

    def generate(self, graph: IRGraph) -> Tuple[List[Instruction], bytes]:
        self._layout_weights(graph)
        self._mark_deferred_loads(graph)
        # Initialize the narrowing-constant sreg (1.0) once up front. Every
        # FP32→FP16 narrowing SCALE_MUL emitted by ``_accum_to_abuf`` and the
        # strip-mined matmul path references this sreg.
        self._emit(emit_scale(self._scale_one_sreg, 1.0))
        last_uses = graph.compute_last_uses()
        for idx, node in enumerate(graph.nodes):
            self._emit_node(node)
            for inp_name, last_idx in last_uses.items():
                if last_idx == idx:
                    if self.mem.abuf.get(inp_name) is not None:
                        self.mem.abuf.free(inp_name)
        # Record the classifier output location (final logits) for host access.
        clf_alloc = self.mem.abuf.get("classifier")
        if clf_alloc is not None:
            self.classifier_output = {
                "buf_id": int(BUF_ABUF),
                "offset_units": int(clf_alloc.offset_units),
                "offset_bytes": int(clf_alloc.offset_units * UNIT),
                "M_pad": int(pad_dim(1)),
                "N_pad": int(pad_dim(self.cfg.num_classes)),
                "logical_cols": int(self.cfg.num_classes),
            }
        self.instructions.append(HaltInsn())
        self.instructions = _coalesce_dma_syncs(self.instructions)
        return self.instructions, bytes(self.dram_blob)

    def _mark_deferred_loads(self, graph: IRGraph):
        """No-op on the W8A16 path.

        The W8A32 codegen defers per-head V loads to avoid Q+K+V FP32 = 156 KB
        coexisting in 128 KB ABUF. In W8A16 each per-head V tile is ~13 KB
        FP16 and the full Q+K+V triple is ~75 KB — comfortably in ABUF —
        so the deferral is unnecessary. The method is preserved structurally
        (same call site in ``generate``) for parity with W8A32, but skips
        the marking pass.
        """
        return

    # ── DRAM weight layout ────────────────────────────────────────────

    def _layout_weights(self, graph: IRGraph):
        offset = 0
        for name, data in self.fp32_weight_data.items():
            blob = data.tobytes()
            self.dram_layout[name] = offset
            self.dram_blob.extend(blob)
            offset += len(blob)
        for name, bias in self.fp32_biases.items():
            self.dram_layout[name] = offset
            blob = bias.astype(np.float16).tobytes()
            self.dram_blob.extend(blob)
            offset += len(blob)
        # Zero-pad for K/V attention masking (FP16 zeros).
        zero_pad_size = self.cfg.pad_rows * self.cfg.head_dim * ELEM_BYTES
        self.dram_layout["__zero_pad__"] = offset
        self.dram_blob.extend(bytes(zero_pad_size))
        offset += zero_pad_size
        # Attention key mask: broadcast row added to QK^T before softmax so the
        # padded key columns (seq_len..seq_len_pad) contribute ≈ 0 mass instead
        # of ≈ 1/Z. In W8A16 the mask value is -65504.0 (FP16 minimum) — large
        # enough that the downstream softmax's exp() underflows to zero in
        # the masked columns. Same effect as W8A32's -1e9 FP32; the FP16
        # narrow is the only difference.
        attn_mask = np.zeros(self.cfg.seq_len_pad, dtype=np.float16)
        attn_mask[self.cfg.seq_len:] = np.float16(-65504.0)
        self.dram_layout["__attention_mask__"] = offset
        mask_blob = attn_mask.tobytes()
        self.dram_blob.extend(mask_blob)
        offset += len(mask_blob)
        # Input patches placeholder (FP16 patch embeddings written by host).
        input_patches_size = self.cfg.num_patches * self.cfg.embed_dim * ELEM_BYTES
        self.dram_layout["__input_patches__"] = offset
        self.dram_blob.extend(bytes(input_patches_size))
        offset += input_patches_size

        self.dram_temp_start = offset
        while len(self.dram_blob) % UNIT != 0:
            self.dram_blob.append(0)

    # ── node dispatch ─────────────────────────────────────────────────

    def _emit_node(self, node: IRNode):
        op = node.op
        if op == "init_residual_tile":
            self._emit_init_residual_tile(node)
        elif op == "tile_load":
            self._emit_tile_load(node)
        elif op == "tile_save":
            self._emit_tile_save(node)
        elif op == "concat_heads_tile":
            self._emit_concat_heads_tile(node)
        elif op == "layernorm":
            self._emit_layernorm(node)
        elif op == "matmul":
            self._emit_matmul(node)
        elif op == "matmul_qkt":
            self._emit_qkt(node)
        elif op == "matmul_attn_v":
            self._emit_attn_v(node)
        elif op == "softmax":
            self._emit_softmax(node)
        elif op == "gelu":
            self._emit_gelu(node)
        elif op == "scale_mul":
            self._emit_scale_mul(node)
        elif op == "vadd":
            self._emit_vadd(node)
        elif op == "cls_extract":
            self._emit_cls_extract(node)
        elif op == "reshape_heads":
            pass  # handled inside matmul_qkt
        else:
            raise NotImplementedError(
                f"W8A32 codegen does not support IR op '{op}' "
                f"(node='{node.name}')"
            )

    # ── helpers ───────────────────────────────────────────────────────

    def _emit(self, insn: Instruction):
        self.instructions.append(insn)

    def _emit_dma_load(self, buf_id: int, sram_off_units: int, size_bytes: int,
                       addr_reg: int, dram_byte_offset: int):
        setup, dram_off = self.addr_planner.plan_access(addr_reg, dram_byte_offset)
        self.instructions.extend(setup)
        xfer_units = (size_bytes + UNIT - 1) // UNIT
        self._emit(LoadInsn(
            buf_id=buf_id,
            sram_off=sram_off_units,
            xfer_len=min(xfer_units, 0xFFFF),
            addr_reg=addr_reg,
            dram_off=dram_off,
        ))

    def _emit_dma_store(self, buf_id: int, sram_off_units: int, size_bytes: int,
                        addr_reg: int, dram_byte_offset: int):
        setup, dram_off = self.addr_planner.plan_access(addr_reg, dram_byte_offset)
        self.instructions.extend(setup)
        xfer_units = (size_bytes + UNIT - 1) // UNIT
        self._emit(StoreInsn(
            buf_id=buf_id,
            sram_off=sram_off_units,
            xfer_len=min(xfer_units, 0xFFFF),
            addr_reg=addr_reg,
            dram_off=dram_off,
        ))

    def _dram_offset(self, name: str, context: str = "") -> int:
        if name not in self.dram_layout:
            raise KeyError(f"Missing DRAM symbol '{name}'"
                           + (f" while {context}" if context else ""))
        return self.dram_layout[name]

    def _resolve_staging_dram(self, name: str, total_bytes: int) -> int:
        cached = self.staging_dram_offsets.get(name)
        if cached is not None:
            return cached
        off = self.dram_temp_start + self.mem.alloc_dram_temp(name, total_bytes)
        self.staging_dram_offsets[name] = off
        return off

    def _accum_to_abuf(self, M_pad: int, N_pad: int, dst_off_units: int):
        """Narrow an FP32 [M_pad, N_pad] tile from ACCUM to FP16 ABUF.

        In W8A32 this is a flat BUF_COPY because ABUF holds FP32 too and the
        byte counts match. In W8A16 the source is FP32 (4 bytes/element) and
        the destination is FP16 (2 bytes/element), so a flat copy would
        scramble the data. Instead, we emit a SCALE_MUL with scale=1.0,
        which the simulator routes through the same FP32-internal-math /
        FP16-narrow-on-write path as every other ABUF write.
        """
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))
        self._emit(ScaleMulInsn(
            src1_buf=BUF_ACCUM, src1_off=0,
            src2_buf=BUF_WBUF, src2_off=0,  # unused by SCALE_MUL
            dst_buf=BUF_ABUF, dst_off=dst_off_units,
            sreg=self._scale_one_sreg,
        ))
        self._emit(SyncInsn(resource_mask=0b001))

    def _load_fp32_bias_to_wbuf(self, bias_name: str, N_pad: int) -> int:
        """DMA an FP32 bias row into WBUF and return its offset_units.

        Caller owns the WBUF allocation key ``bias_<bias_name>`` and must
        free it after use.
        """
        bias_dram = self._dram_offset(bias_name, f"loading FP32 bias '{bias_name}'")
        bias_bytes = N_pad * ELEM_BYTES
        bias_alloc = self.mem.wbuf.alloc(f"bias_{bias_name}", bias_bytes)
        self._emit_dma_load(BUF_WBUF, bias_alloc.offset_units, bias_bytes, 1, bias_dram)
        self._emit(SyncInsn(resource_mask=0b001))
        return bias_alloc.offset_units

    # ── init_residual_tile ────────────────────────────────────────────

    def _emit_init_residual_tile(self, node: IRNode):
        """Stream one tile of CLS+patches+pos_embed FP32 → DRAM staging."""
        tile_idx = int(node.attrs["tile_idx"])
        tile_rows = int(node.attrs["tile_rows"])
        logical_rows = int(node.attrs["logical_rows"])
        cls_name = node.inputs[0]
        pos_name = node.inputs[1]

        embed = self.cfg.embed_dim
        num_patches = self.cfg.num_patches

        cls_dram = self._dram_offset(cls_name, "init_residual_tile cls")
        pos_dram = self._dram_offset(pos_name, "init_residual_tile pos_embed")
        patches_dram = self.dram_layout["__input_patches__"]

        seq_row_start = tile_idx * tile_rows

        # ABUF tile sized to tile_rows × embed in FP32 bytes.
        tile_bytes = tile_rows * embed * ELEM_BYTES
        scratch_name = f"_init_tile_scratch_{tile_idx}"
        scratch = self.mem.abuf.alloc(scratch_name, tile_bytes)

        if seq_row_start == 0:
            self._emit_dma_load(BUF_ABUF, scratch.offset_units, embed * ELEM_BYTES, 0, cls_dram)
            self._emit(SyncInsn(resource_mask=0b001))
            patches_in_tile = min(logical_rows - 1, num_patches)
            if patches_in_tile > 0:
                patch_dst_units = scratch.offset_units + (embed * ELEM_BYTES) // UNIT
                self._emit_dma_load(
                    BUF_ABUF, patch_dst_units,
                    patches_in_tile * embed * ELEM_BYTES, 1, patches_dram,
                )
                self._emit(SyncInsn(resource_mask=0b001))
        else:
            patch_start = seq_row_start - 1
            patches_in_tile = max(0, min(logical_rows, num_patches - patch_start))
            if patches_in_tile > 0:
                patch_src = patches_dram + patch_start * embed * ELEM_BYTES
                self._emit_dma_load(
                    BUF_ABUF, scratch.offset_units,
                    patches_in_tile * embed * ELEM_BYTES, 0, patch_src,
                )
                self._emit(SyncInsn(resource_mask=0b001))

        # Pos_embed tile → WBUF (FP32 rows).
        pos_bytes = tile_rows * embed * ELEM_BYTES
        pos_alloc = self.mem.wbuf.alloc(f"_init_pos_tile{tile_idx}", pos_bytes)
        pos_src = pos_dram + seq_row_start * embed * ELEM_BYTES
        self._emit_dma_load(BUF_WBUF, pos_alloc.offset_units, pos_bytes, 1, pos_src)
        self._emit(SyncInsn(resource_mask=0b001))

        # VADD ABUF[scratch] + WBUF[pos] → ABUF[scratch]. ABUF path = M×N + M×N.
        m_tiles = tile_rows // TILE
        n_tiles = pad_dim(embed) // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))
        self._emit(VaddInsn(
            src1_buf=BUF_ABUF, src1_off=scratch.offset_units,
            src2_buf=BUF_WBUF, src2_off=pos_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=scratch.offset_units,
        ))
        self.mem.wbuf.free(f"_init_pos_tile{tile_idx}")

        # Store the (logical_rows × embed) FP32 result to the residual stage.
        dst_base = self._resolve_staging_dram(
            node.attrs["dst_dram"], int(node.attrs["total_dram_bytes"]) * ELEM_BYTES,
        )
        dst_off = dst_base + int(node.attrs["dst_offset_bytes"]) * ELEM_BYTES
        self._emit_dma_store(
            BUF_ABUF, scratch.offset_units,
            logical_rows * embed * ELEM_BYTES, 2, dst_off,
        )
        self._emit(SyncInsn(resource_mask=0b001))
        self.mem.abuf.free(scratch_name)

    # ── tile_load / tile_save ─────────────────────────────────────────

    def _emit_tile_load(self, node: IRNode):
        if node.name in self._deferred_loads:
            # DMA emitted later inside the matmul_attn_v handler so the
            # tile doesn't compete with Q and K for ABUF.
            return
        rows, cols = node.output_shape
        M_pad = pad_dim(rows)
        N_pad = pad_dim(cols)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        src_base = self._resolve_staging_dram(
            node.attrs["src_dram"], int(node.attrs["total_dram_bytes"]) * ELEM_BYTES,
        )
        src_off = src_base + int(node.attrs.get("src_offset_bytes", 0)) * ELEM_BYTES
        load_bytes = rows * cols * ELEM_BYTES
        self._emit_dma_load(BUF_ABUF, out_alloc.offset_units, load_bytes, 3, src_off)
        self._emit(SyncInsn(resource_mask=0b001))

    def _emit_deferred_tile_load(self, name: str):
        """Materialise a deferred tile_load DMA (used by matmul_attn_v)."""
        node = self._deferred_loads[name]
        rows, cols = node.output_shape
        M_pad = pad_dim(rows)
        N_pad = pad_dim(cols)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        src_base = self._resolve_staging_dram(
            node.attrs["src_dram"], int(node.attrs["total_dram_bytes"]) * ELEM_BYTES,
        )
        src_off = src_base + int(node.attrs.get("src_offset_bytes", 0)) * ELEM_BYTES
        load_bytes = rows * cols * ELEM_BYTES
        self._emit_dma_load(BUF_ABUF, out_alloc.offset_units, load_bytes, 3, src_off)
        self._emit(SyncInsn(resource_mask=0b001))

    def _emit_tile_save(self, node: IRNode):
        rows, cols = node.output_shape
        in_name = node.inputs[0]
        in_alloc = self.mem.abuf.get(in_name)
        src_buf = BUF_ABUF
        if in_alloc is None:
            in_alloc = self.mem.wbuf.get(in_name)
            src_buf = BUF_WBUF
        if in_alloc is None:
            raise KeyError(
                f"tile_save '{node.name}' missing source allocation '{in_name}' "
                f"in ABUF or WBUF"
            )
        dst_base = self._resolve_staging_dram(
            node.attrs["dst_dram"], int(node.attrs["total_dram_bytes"]) * ELEM_BYTES,
        )
        dst_off = dst_base + int(node.attrs.get("dst_offset_bytes", 0)) * ELEM_BYTES
        save_bytes = rows * cols * ELEM_BYTES
        self._emit_dma_store(src_buf, in_alloc.offset_units, save_bytes, 2, dst_off)
        self._emit(SyncInsn(resource_mask=0b001))
        if src_buf == BUF_WBUF:
            self.mem.wbuf.free(in_name)

    # ── concat_heads_tile ─────────────────────────────────────────────

    def _emit_concat_heads_tile(self, node: IRNode):
        """Concatenate per-head attn_v outputs into ABUF for the out_proj tile.

        Each output row interleaves all heads' data:
            ABUF[r, h*head_dim:(h+1)*head_dim] ← AV_stage_h[seq_row=tile_idx*tile_rows+r, :]
        """
        tile_idx = int(node.attrs["tile_idx"])
        tile_rows = int(node.attrs["tile_rows"])
        logical_rows = int(node.attrs["logical_rows"])
        head_dim = int(node.attrs["head_dim"])
        num_heads = int(node.attrs["num_heads"])
        av_total_elems = int(node.attrs["av_stage_total_bytes"])  # element count in W8A8 IR
        embed = num_heads * head_dim

        M_pad = pad_dim(logical_rows)
        N_pad = pad_dim(embed)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)

        head_byte_dim = head_dim * ELEM_BYTES
        embed_byte_dim = embed * ELEM_BYTES

        for h, stage_name in enumerate(node.inputs):
            head_base = self._resolve_staging_dram(stage_name, av_total_elems * ELEM_BYTES)
            for r in range(logical_rows):
                seq_row = tile_idx * tile_rows + r
                src_off = head_base + seq_row * head_byte_dim
                dst_off_units = (
                    out_alloc.offset_units
                    + (r * embed_byte_dim + h * head_byte_dim) // UNIT
                )
                self._emit_dma_load(BUF_ABUF, dst_off_units, head_byte_dim, 3, src_off)
                self._emit(SyncInsn(resource_mask=0b001))

    # ── layernorm ─────────────────────────────────────────────────────

    def _emit_layernorm(self, node: IRNode):
        """LAYERNORM in FP32: in/out via ABUF, gamma+beta packed in WBUF."""
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        gamma_name = node.inputs[1]
        beta_name = node.inputs[2]
        gamma_dram = self._dram_offset(gamma_name, f"loading LN gamma for '{node.name}'")
        beta_dram = self._dram_offset(beta_name, f"loading LN beta for '{node.name}'")
        # Gamma and beta are FP16 ndim=1 tensors of length N (the SFU expects FP16).
        gb_bytes = N_pad * 4  # FP16 gamma[N] + FP16 beta[N] = 4 bytes per channel
        gb_alloc = self.mem.wbuf.alloc(f"gb_{node.name}", gb_bytes)
        self._emit_dma_load(BUF_WBUF, gb_alloc.offset_units, N_pad * 2, 1, gamma_dram)
        self._emit(SyncInsn(resource_mask=0b001))
        beta_off_units = gb_alloc.offset_units + (N_pad * 2) // UNIT
        self._emit_dma_load(BUF_WBUF, beta_off_units, N_pad * 2, 1, beta_dram)
        self._emit(SyncInsn(resource_mask=0b001))

        in_alloc = self.mem.abuf.get(node.inputs[0]) or \
                   self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad * ELEM_BYTES)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        # sreg is ignored by the W8A32 SFU LN — pass any valid index.
        self._emit(LayernormInsn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_WBUF, src2_off=gb_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            sreg=0,
        ))
        self._emit(SyncInsn(resource_mask=0b100))
        self.mem.wbuf.free(f"gb_{node.name}")

    # ── matmul ────────────────────────────────────────────────────────

    # FP32 weights for big linears (FC1/FC2/classifier/ViT-B out_proj) exceed
    # the 256 KB WBUF. We N-strip the weight so each chunk fits with room for
    # the bias and intermediate strip outputs. 192 KB leaves 64 KB headroom.
    _WBUF_WEIGHT_BUDGET = 192 * 1024

    def _pick_n_strip(self, K_pad: int, N_pad: int) -> int:
        """Pick an N-strip size for N-strip-mined matmul.

        Returns ``N_pad`` (one strip) when the full weight fits, otherwise the
        largest 16-aligned ``N_strip`` such that ``K_pad * N_strip * 4`` stays
        within ``_WBUF_WEIGHT_BUDGET`` and ``N_pad`` is a clean multiple of it.
        """
        full_bytes = K_pad * N_pad * ELEM_BYTES
        if full_bytes <= self._WBUF_WEIGHT_BUDGET:
            return N_pad
        max_strip = self._WBUF_WEIGHT_BUDGET // (K_pad * ELEM_BYTES)
        max_strip = (max_strip // 16) * 16
        if max_strip < 16:
            max_strip = 16
        # Prefer a clean divisor of N_pad to avoid a short trailing strip.
        for cand in range(max_strip, 0, -16):
            if N_pad % cand == 0:
                return cand
        return max_strip

    def _emit_matmul(self, node: IRNode):
        """Generic matmul with optional bias.

        Path: ABUF activations × WBUF FP32 weights → ACCUM (FP32)
              → VADD ACCUM + bias_row (1×N) → ACCUM
              → BUF_COPY ACCUM → ABUF (FP32 bytes preserved).

        Weights that overflow WBUF are N-strip-mined: each chunk loads one
        ``N_strip``-wide weight slab, runs MATMUL → bias → BUF_COPY to the
        proper N-window of the full ABUF output, then frees the slab.
        """
        M, N = node.output_shape
        weight_name = node.inputs[1]
        weight = self.fp32_weight_data.get(weight_name)
        if weight is None:
            raise KeyError(f"Missing weight '{weight_name}' for matmul '{node.name}'")

        K = weight.shape[0]
        M_pad = pad_dim(M)
        N_pad = pad_dim(N)
        K_pad = pad_dim(K)
        N_strip = self._pick_n_strip(K_pad, N_pad)
        num_strips = N_pad // N_strip
        bias_name = node.attrs.get("bias")
        has_bias = bool(bias_name and bias_name in self.fp32_biases)

        # Activations: source already in ABUF from a previous tile_load / op.
        act_alloc = self.mem.abuf.get(node.inputs[0]) or \
                    self.mem.abuf.alloc(node.inputs[0], M_pad * K_pad * ELEM_BYTES)

        # Output ABUF [M_pad, N_pad] FP32; strips fill it column-window by window.
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        w_dram_base = self._dram_offset(weight_name, f"loading weight '{weight_name}'")

        for k in range(num_strips):
            n_start = k * N_strip
            # Weight slab: [K_pad, N_strip] starting at column n_start.
            slab_bytes_per_row = N_strip * ELEM_BYTES
            slab_bytes = K_pad * slab_bytes_per_row
            slab_dram = w_dram_base + n_start * K_pad * ELEM_BYTES
            slab_alloc = self.mem.wbuf.alloc(f"_w_{weight_name}_s{k}", slab_bytes)
            if num_strips == 1:
                # Whole-weight load (the fp32_weight_data tensor is exactly
                # [K_pad, N_pad] FP32, so a flat DMA is correct).
                self._emit_dma_load(BUF_WBUF, slab_alloc.offset_units, slab_bytes, 0, w_dram_base)
                self._emit(SyncInsn(resource_mask=0b001))
            else:
                # The DRAM weight is stored as [K_pad, N_pad] row-major; we need
                # [K_pad, N_strip] starting at column n_start. Stream one
                # K-row of N_strip cols per DMA (K_pad DMAs per strip).
                for k_row in range(K_pad):
                    row_dram = w_dram_base + (k_row * N_pad + n_start) * ELEM_BYTES
                    row_dst_units = slab_alloc.offset_units + (k_row * slab_bytes_per_row) // UNIT
                    self._emit_dma_load(BUF_WBUF, row_dst_units, slab_bytes_per_row, 0, row_dram)
                    self._emit(SyncInsn(resource_mask=0b001))

            # MATMUL ABUF × WBUF[slab] → ACCUM (FP32 [M_pad, N_strip]).
            m_tiles = M_pad // TILE
            ns_tiles = N_strip // TILE
            k_tiles = K_pad // TILE
            self._emit(ConfigTileInsn(M=m_tiles - 1, N=ns_tiles - 1, K=k_tiles - 1))
            self._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=act_alloc.offset_units,
                src2_buf=BUF_WBUF, src2_off=slab_alloc.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            ))
            self._emit(SyncInsn(resource_mask=0b010))
            self.mem.wbuf.free(f"_w_{weight_name}_s{k}")

            if has_bias:
                # Load only the N_strip slice of the bias row.
                bias_dram = self._dram_offset(bias_name) + n_start * ELEM_BYTES
                bias_slice_bytes = N_strip * ELEM_BYTES
                bias_alloc = self.mem.wbuf.alloc(f"bias_{bias_name}_s{k}", bias_slice_bytes)
                self._emit_dma_load(BUF_WBUF, bias_alloc.offset_units, bias_slice_bytes, 1, bias_dram)
                self._emit(SyncInsn(resource_mask=0b001))
                self._emit(VaddInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_WBUF, src2_off=bias_alloc.offset_units,
                    dst_buf=BUF_ACCUM, dst_off=0,
                ))
                self.mem.wbuf.free(f"bias_{bias_name}_s{k}")

            if num_strips == 1:
                # ACCUM [M_pad, N_pad] → ABUF as one narrowing SCALE_MUL.
                self._accum_to_abuf(M_pad, N_pad, out_alloc.offset_units)
            else:
                # Strip-mined path. The W8A32 codegen scatters per-row via
                # flat BUF_COPY from ACCUM (FP32, 4 B/elem) into the ABUF
                # column window (also FP32, 4 B/elem). For W8A16 the source
                # is still FP32 but the destination is FP16 (2 B/elem), so
                # we cannot copy bytes directly. Two steps:
                #   1. SCALE_MUL the whole [M_pad, N_strip] ACCUM tile to a
                #      packed FP16 scratchpad in ABUF (no scatter).
                #   2. Per-row flat BUF_COPY from the scratchpad into the
                #      out_alloc column window (FP16 → FP16, byte counts
                #      match so the flat copy is correct).
                scratch_bytes = M_pad * N_strip * ELEM_BYTES
                scratch_alloc = self.mem.abuf.alloc(
                    f"_narrow_{node.name}_s{k}", scratch_bytes,
                )
                ms_tiles = M_pad // TILE
                ns_tiles = N_strip // TILE
                self._emit(ConfigTileInsn(M=ms_tiles - 1, N=ns_tiles - 1, K=0))
                self._emit(ScaleMulInsn(
                    src1_buf=BUF_ACCUM, src1_off=0,
                    src2_buf=BUF_WBUF, src2_off=0,
                    dst_buf=BUF_ABUF, dst_off=scratch_alloc.offset_units,
                    sreg=self._scale_one_sreg,
                ))
                self._emit(SyncInsn(resource_mask=0b001))

                row_units_strip = (N_strip * ELEM_BYTES) // UNIT
                row_units_full = (N_pad * ELEM_BYTES) // UNIT
                col_offset_units = (n_start * ELEM_BYTES) // UNIT
                for r in range(M_pad):
                    src_off = (
                        scratch_alloc.offset_units
                        + (r * N_strip * ELEM_BYTES) // UNIT
                    )
                    dst_off = out_alloc.offset_units + r * row_units_full + col_offset_units
                    self._emit(BufCopyInsn(
                        src_buf=BUF_ABUF, src_off=src_off,
                        dst_buf=BUF_ABUF, dst_off=dst_off,
                        length=row_units_strip,
                    ))
                self._emit(SyncInsn(resource_mask=0b001))
                self.mem.abuf.free(f"_narrow_{node.name}_s{k}")

    # ── matmul_qkt ────────────────────────────────────────────────────

    def _emit_qkt(self, node: IRNode):
        """Q × K^T strip-mined over Q's M dimension.

        Strategy:
          1. BUF_COPY transpose K (ABUF [seq, head_dim]) → WBUF [head_dim, seq]
             (uses the FP32-aware transpose path in SimulatorW8A32).
          2. For each 16-row strip of Q: MATMUL Q_strip × K^T → ACCUM
             then SCALE_MUL by 1/sqrt(d_head) → ACCUM in place
             then SOFTMAX ACCUM → WBUF row strip.
          3. After all strips: WBUF holds [seq_pad, seq_pad] FP32 softmax.

        Output: WBUF allocation under ``node.name`` with the full softmax.
        """
        head_idx = node.attrs["head_idx"]
        seq_len = node.output_shape[0]
        head_dim = self.cfg.head_dim
        M_pad = pad_dim(seq_len)
        K_pad = pad_dim(head_dim)
        scale = float(node.attrs.get("scale", head_dim ** -0.5))

        # Zero out K padding rows (LN(zero) = beta would leak).
        k_alloc = self.mem.abuf.get(node.inputs[1])
        if k_alloc is None:
            k_alloc = self.mem.abuf.alloc(node.inputs[1], M_pad * K_pad * ELEM_BYTES)
        if M_pad > seq_len:
            pad_rows = M_pad - seq_len
            k_pad_units = k_alloc.offset_units + (seq_len * K_pad * ELEM_BYTES) // UNIT
            zero_pad_dram = self._dram_offset("__zero_pad__", "loading K padding mask")
            self._emit_dma_load(BUF_ABUF, k_pad_units, pad_rows * K_pad * ELEM_BYTES, 3,
                                zero_pad_dram)
            self._emit(SyncInsn(resource_mask=0b001))

        # BUF_COPY transpose K → WBUF (FP32 element transpose).
        length_units = (M_pad * K_pad * ELEM_BYTES) // UNIT
        kt_wbuf = self.mem.wbuf.alloc(f"kt_head{head_idx}_{node.name}", K_pad * M_pad * ELEM_BYTES)
        self._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=k_alloc.offset_units,
            dst_buf=BUF_WBUF, dst_off=kt_wbuf.offset_units,
            length=length_units,
            src_rows=M_pad // TILE,
            transpose=1,
        ))
        self._emit(SyncInsn(resource_mask=0b001))

        q_alloc = self.mem.abuf.get(node.inputs[0])
        if q_alloc is None:
            q_alloc = self.mem.abuf.alloc(node.inputs[0], M_pad * K_pad * ELEM_BYTES)

        n_tiles = M_pad // TILE
        k_tiles = K_pad // TILE
        num_strips = M_pad // TILE

        # Full softmax output lives in WBUF [M_pad, M_pad] FP32.
        softmax_wbuf = self.mem.wbuf.alloc(node.name, M_pad * M_pad * ELEM_BYTES)

        # Attention mask in WBUF: [seq_len_pad] FP32 with -1e9 in the padded
        # columns. Loaded once, broadcast-added to each Q-strip's QK^T row
        # before softmax to suppress probability mass on padded keys.
        mask_dram = self._dram_offset("__attention_mask__", "loading attention mask")
        mask_bytes = self.cfg.seq_len_pad * ELEM_BYTES
        mask_wbuf = self.mem.wbuf.alloc(f"attn_mask_{node.name}", mask_bytes)
        self._emit_dma_load(BUF_WBUF, mask_wbuf.offset_units, mask_bytes, 0, mask_dram)
        self._emit(SyncInsn(resource_mask=0b001))

        # Pre-load 1/sqrt(d) into an sreg so SCALE_MUL can apply it per strip.
        scale_sreg = self.sregs.alloc_single()
        self._emit(emit_scale(scale_sreg, scale))

        strip_byte_stride = TILE * M_pad * ELEM_BYTES

        for s in range(num_strips):
            self._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=k_tiles - 1))
            q_strip_units = q_alloc.offset_units + (s * TILE * K_pad * ELEM_BYTES) // UNIT
            # Q[strip] × K^T → ACCUM (FP32).
            self._emit(MatmulInsn(
                src1_buf=BUF_ABUF, src1_off=q_strip_units,
                src2_buf=BUF_WBUF, src2_off=kt_wbuf.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
                flags=0,
            ))
            self._emit(SyncInsn(resource_mask=0b010))

            # SCALE_MUL by 1/sqrt(d_head) on ACCUM, in place.
            self._emit(ConfigTileInsn(M=0, N=n_tiles - 1, K=0))
            self._emit(ScaleMulInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_WBUF, src2_off=0,
                dst_buf=BUF_ACCUM, dst_off=0,
                sreg=scale_sreg,
            ))
            self._emit(SyncInsn(resource_mask=0b100))

            # Mask padded keys: VADD ACCUM + WBUF[mask] → ACCUM. The W8A32
            # VADD broadcasts a single src2 row across all M ACCUM rows, so
            # one [seq_len_pad] row of mask handles every Q-row in the strip.
            self._emit(VaddInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_WBUF, src2_off=mask_wbuf.offset_units,
                dst_buf=BUF_ACCUM, dst_off=0,
            ))

            # SOFTMAX ACCUM → WBUF strip (FP32 strip-mined).
            strip_off_units = softmax_wbuf.offset_units + (s * strip_byte_stride) // UNIT
            self._emit(SoftmaxInsn(
                src1_buf=BUF_ACCUM, src1_off=0,
                src2_buf=BUF_WBUF, src2_off=0,
                dst_buf=BUF_WBUF, dst_off=strip_off_units,
                sreg=0,
            ))
            self._emit(SyncInsn(resource_mask=0b100))

        self.mem.wbuf.free(f"attn_mask_{node.name}")
        self.mem.wbuf.free(f"kt_head{head_idx}_{node.name}")

    # ── matmul_attn_v ─────────────────────────────────────────────────

    def _emit_attn_v(self, node: IRNode):
        """softmax × V → ACCUM → ABUF (FP32 everywhere)."""
        seq_len = node.output_shape[0]
        head_dim = node.output_shape[1]
        M_pad = pad_dim(seq_len)
        N_pad = pad_dim(head_dim)

        # Softmax output is in WBUF under the softmax node name (rename in _emit_softmax).
        attn_alloc = self.mem.wbuf.get(node.inputs[0])
        if attn_alloc is None:
            raise KeyError(
                f"matmul_attn_v '{node.name}' missing softmax allocation "
                f"'{node.inputs[0]}' in WBUF"
            )

        # Materialise the deferred V load (if any) now that Q and K have been freed.
        if node.inputs[1] in self._deferred_loads and \
                self.mem.abuf.get(node.inputs[1]) is None:
            self._emit_deferred_tile_load(node.inputs[1])
        v_alloc = self.mem.abuf.get(node.inputs[1])
        if v_alloc is None:
            v_alloc = self.mem.abuf.alloc(node.inputs[1], M_pad * N_pad * ELEM_BYTES)

        # Zero pad V's trailing rows.
        if M_pad > seq_len:
            pad_rows = M_pad - seq_len
            v_pad_units = v_alloc.offset_units + (seq_len * N_pad * ELEM_BYTES) // UNIT
            zero_pad_dram = self._dram_offset("__zero_pad__", "loading V padding mask")
            self._emit_dma_load(BUF_ABUF, v_pad_units, pad_rows * N_pad * ELEM_BYTES, 3,
                                zero_pad_dram)
            self._emit(SyncInsn(resource_mask=0b001))

        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        k_tiles = M_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=k_tiles - 1))
        # src1 = softmax in WBUF, src2 = V in ABUF (both FP32).
        self._emit(MatmulInsn(
            src1_buf=BUF_WBUF, src1_off=attn_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=v_alloc.offset_units,
            dst_buf=BUF_ACCUM, dst_off=0,
            flags=0,
        ))
        self._emit(SyncInsn(resource_mask=0b010))

        self.mem.wbuf.free(node.inputs[0])

        # ACCUM → ABUF (the AttnV result for downstream concat / tile_save).
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        self._accum_to_abuf(M_pad, N_pad, out_alloc.offset_units)

    # ── softmax / scale_mul (rename-only in tiled path) ───────────────

    def _emit_scale_mul(self, node: IRNode):
        """In W8A32 the QKT scale is folded into the QKT emitter; this IR
        node only needs to propagate the WBUF allocation under the new name."""
        in_alloc = self.mem.wbuf.get(node.inputs[0])
        if in_alloc is not None:
            alloc = self.mem.wbuf.allocations.pop(node.inputs[0])
            alloc.name = node.name
            self.mem.wbuf.allocations[node.name] = alloc

    def _emit_softmax(self, node: IRNode):
        """Softmax was emitted per-strip inside ``_emit_qkt``; this IR node
        is a rename so downstream attn_v can find the WBUF allocation."""
        in_alloc = self.mem.wbuf.get(node.inputs[0])
        if in_alloc is not None and node.inputs[0] in self.mem.wbuf.allocations:
            alloc = self.mem.wbuf.allocations.pop(node.inputs[0])
            alloc.name = node.name
            self.mem.wbuf.allocations[node.name] = alloc

    # ── gelu ──────────────────────────────────────────────────────────

    def _emit_gelu(self, node: IRNode):
        """GELU in FP32 on ABUF tiles."""
        # The seq_tiling pass sets ``inline_with`` for the GELU node that
        # used to be fused with FC1. In W8A32 there is no fusion — we always
        # emit GELU as a separate SFU op reading from ABUF.
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        in_alloc = self.mem.abuf.get(node.inputs[0]) or \
                   self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad * ELEM_BYTES)
        out_alloc = self.mem.abuf.alloc(node.name, M_pad * N_pad * ELEM_BYTES)
        self._emit(GeluInsn(
            src1_buf=BUF_ABUF, src1_off=in_alloc.offset_units,
            src2_buf=BUF_WBUF, src2_off=0,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            sreg=0,
        ))
        self._emit(SyncInsn(resource_mask=0b100))

    # ── vadd ──────────────────────────────────────────────────────────

    def _emit_vadd(self, node: IRNode):
        """FP32 elementwise add on ABUF tiles (residual connection)."""
        M_pad = pad_dim(node.output_shape[0])
        N_pad = pad_dim(node.output_shape[1])
        m_tiles = M_pad // TILE
        n_tiles = N_pad // TILE
        self._emit(ConfigTileInsn(M=m_tiles - 1, N=n_tiles - 1, K=0))

        src1_alloc = self.mem.abuf.get(node.inputs[0]) or \
                     self.mem.abuf.alloc(node.inputs[0], M_pad * N_pad * ELEM_BYTES)
        src2_alloc = self.mem.abuf.get(node.inputs[1]) or \
                     self.mem.abuf.alloc(node.inputs[1], M_pad * N_pad * ELEM_BYTES)

        # Write in-place into src2's slot (residual1 / residual2 last-use semantics).
        self._emit(VaddInsn(
            src1_buf=BUF_ABUF, src1_off=src1_alloc.offset_units,
            src2_buf=BUF_ABUF, src2_off=src2_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=src2_alloc.offset_units,
        ))
        # Rename src2's allocation to the output node.
        alloc = self.mem.abuf.allocations.pop(node.inputs[1], None)
        if alloc is not None:
            alloc.name = node.name
            self.mem.abuf.allocations[node.name] = alloc

    # ── cls_extract ───────────────────────────────────────────────────

    def _emit_cls_extract(self, node: IRNode):
        """Copy ABUF row 0 (the CLS token) into a fresh allocation."""
        N = self.cfg.embed_dim
        in_alloc = self.mem.abuf.get(node.inputs[0])
        if in_alloc is None:
            in_alloc = self.mem.abuf.alloc(
                node.inputs[0],
                self.cfg.seq_len_pad * pad_dim(N) * ELEM_BYTES,
            )
        out_alloc = self.mem.abuf.alloc(node.name, pad_dim(N) * ELEM_BYTES)
        length_units = (N * ELEM_BYTES) // UNIT
        self._emit(BufCopyInsn(
            src_buf=BUF_ABUF, src_off=in_alloc.offset_units,
            dst_buf=BUF_ABUF, dst_off=out_alloc.offset_units,
            length=length_units,
        ))


__all__ = ["CodeGeneratorW8A16"]
