# Precision Modes

The TACCEL software toolchain supports two precision modes. W8A16 is the
default shipping path; W8A32 is the FP32 weight-quant ceiling reference.

| Mode | Weights | Activations | Accumulator | Entry point | Simulator |
|---|---|---|---|---|---|
| **`w8a16`** (default) | INT8 per-channel → FP16 dequant in DRAM | FP16 | FP32 (bit-aliased) | `Compiler.compile_w8a16()` | `SimulatorW8A16` |
| **`w8a32`** | INT8 per-channel → FP32 dequant in DRAM | FP32 | FP32 (bit-aliased) | `Compiler.compile_w8a32()` | `SimulatorW8A32` |

## Why two modes?

Both modes are software-only and target the same accelerator ISA. They
differ only in the element width carried through the ABUF (activation
buffer) and the DRAM-dequant weight footprint.

- **W8A32** is the *measurement* mode: it preserves FP32 precision end
  to end and matches `fake_quant.apply_weight_quantization` bit-for-bit
  on the dequantized weights. Its cosine vs FP32 reference is the
  ceiling that any weight-only quantization scheme can achieve on this
  toolchain.

- **W8A16** is the *shipping* mode: it halves the dequant-weight DRAM
  footprint vs W8A32 (e.g. 344 MB → 172 MB for ViT-B) and halves the
  per-element ABUF cost. The FP16 narrowing introduces ~3-decimal-digit
  per-tensor rounding noise; over 12 transformer blocks this compounds
  to ~one-nine looser cosine vs the W8A32 ceiling. Both gates stay
  tight: `cos vs FP32 ≥ 0.997`, `cos vs fake_quant ≥ 0.998`.

## Why no W8A8

INT8 *weights* are easy on ViT (the per-channel scheme above achieves
~0.999 cosine vs FP32 on its own). INT8 *activations* are not: ViT
activation distributions (especially post-LayerNorm and post-GELU) have
heavy tails that per-tensor INT8 scaling can't capture, attention
softmax probabilities concentrate near 0/1 and lose resolution under
INT8, and 12 blocks of compounding INT8 rounding can cost 3–10% top-1
without SmoothQuant / Hessian-guided / twin-uniform calibration.

The legacy W8A8 path was removed because its calibration plumbing was
tightly coupled to the unmasked attention semantics and to per-block
activation scales tuned on a small image set. The remaining accuracy
gap to FP32 was an INT8-activation-floor problem, not a toolchain bug
— solving it would have required full PTQ infrastructure that the
W8A16 / W8A32 paths sidestep entirely.

## Load-bearing accuracy gates

Both paths must clear these gates on the 20-image frozen COCO eval set:

| Gate | W8A16 | W8A32 |
|---|---|---|
| `cos(logits, FP32 reference)` ≥ | 0.997 | 0.998 |
| `cos(logits, fake_quant ceiling)` ≥ | 0.998 | 0.999 |
| top-1 vs FP32 reference | 20/20 | 20/20 |

The `fake_quant` reference is
`taccel.quantizer.fake_quant.apply_weight_quantization`: a PyTorch model
with the exact per-channel INT8 weight-quant scheme applied (dequantized
back to FP32 in-place), so the accelerator's weight-quant ceiling is
reproducible without running the simulator.

## Codegen and golden-model differences (W8A16 vs W8A32)

### Attention mask

W8A32 stores the attention key mask as an FP32 row with `-1e9` in
padded columns; W8A16 uses `-65504.0` (the FP16 clamp). In both modes
the mask is loaded once per `_emit_qkt` call and broadcast-VADD'd into
the FP32 ACCUM between the QK^T scale and the SOFTMAX, so softmax
probability mass on padded keys underflows to zero.

### ACCUM endpoint conversion

W8A32 moves ACCUM → ABUF with a flat BUF_COPY (both buffers are 4 B
per element). W8A16 uses SCALE_MUL with `scale = 1.0` (reserved
`sreg=15` in the codegen) so the simulator's narrow-on-write path
runs: ACCUM[4 B FP32] → ABUF[2 B FP16].

### Deferred V loads

W8A32's per-head Q+K+V FP32 footprint (~150 KB) exceeds ABUF, so the
W8A32 codegen defers loading V until after the QK^T matmul completes.
W8A16's FP16 Q+K+V (~38 KB) fits comfortably; the W8A16 codegen loads
all three eagerly and skips the deferred-V machinery.

## How to compile and run

```bash
# Compile a DeiT-tiny program.bin in W8A16 (default).
python -m tools.compile_model --weights pytorch_model.bin -o program.bin

# Compile in W8A32.
python -m tools.compile_model --weights pytorch_model.bin -o program.bin --mode w8a32

# Simulate (mode must match the compile mode).
python -m tools.run_golden program.bin --input patches.npy --mode w8a16

# Accuracy benchmarks on the 20 frozen images. benchmark_w8a16 fans the
# per-image torch forwards + SimulatorW8A16 dispatch across processes by
# default (--workers=$(nproc)); pass --workers 1 to force sequential.
python -m tools.benchmark_w8a16 --max-images 20
python -m tools.benchmark_w8a32 --max-images 20

# Memory budget report for either mode.
python -m tools.profile_memory --mode w8a16
python -m tools.profile_memory --mode w8a32

# W8A16 RTL-vs-golden bit-exact parity gate (load-bearing acceptance
# contract; ~1–2 hours of Verilator time for the full 20 images).
python -m tools.batch_compare_rtl_golden --max-images 20
```

## Codegen passes that affect both modes

Two thin compiler passes ride on top of the W8A16/W8A32 codegen and
cut the emitted program by ~49% without changing semantics. They are
silent (no logging, no manifest changes) and the bit-exact parity
gate is the verification.

* **`compiler/dma_emitter.py::AddrPlanner`.** Caches the value the
  RTL/golden will hold in each of the four 56-bit `addr_regs` and
  emits SET_ADDR_LO/HI only on actual change. Also uses the M-type
  `dram_off` field (16-bit, ×16-byte) to walk inside a 1 MB window of
  the cached base without re-emitting SET_ADDR at all. RTL
  `dma_engine.sv` already computes `addr_regs[reg] + dram_off*16`, so
  the change is compiler-side only.
* **`compiler/sync_coalesce.py::coalesce_dma_syncs`.** Drops
  SYNC(`0b001`) bits whose hazard the RTL already enforces at issue:
  helper consumers (`BUF_COPY` line 287, `SCALE_MUL` / `VADD`
  line 321) and SFU consumers (`SOFTMAX` / `LAYERNORM` / `GELU`
  line 366) auto-stall on `dma_busy` in `control_unit.sv`. The pass
  **keeps** SYNCs that fence `OP_MATMUL` (line 340 does not check
  `dma_busy`), and crucially keeps SYNCs between adjacent
  `OP_LOAD` / `OP_STORE` — the DMA engine has no command queue
  (`dma_engine.sv` line 187), so a second dispatch pulse arriving
  while DMA is mid-flight is silently dropped. That class of SYNC is
  architectural, not defensive, and removing it produces a silent
  miscompile (verified to fail the bit-exact gate during this work).

DeiT-tiny W8A16 instruction count drops from 1,288,764 to 657,846
(−49.0%); program bytes 10.3 MB → 5.3 MB. SET_ADDR_LO falls from
314,050 to 822 and SET_ADDR_HI from 314,050 to 4; the SYNC reduction
is modest (~3,644 instructions) because most SYNCs in DeiT-tiny
fence DMA→DMA and DMA→MATMUL, neither of which can be relaxed.

## RTL parity contract

The W8A16 RTL must produce **bit-exact** FP16 classifier logits
against `SimulatorW8A16` on every image of the 20-image frozen
benchmark. `software/tools/batch_compare_rtl_golden.py` is the sole
acceptance gate: it asserts
`np.array_equal(rtl_logits.view(np.uint16), golden_logits.view(np.uint16))`
with zero ULPs of slack. The gate is load-bearing — a divergence must
be root-caused at the responsible rounding step (systolic PE, SFU
narrowing, helper VADD/SCALE_MUL, or per-tile FP16 commit) and fixed
at that site. **Reviewers must refuse weakening the assertion to a
tolerance** without a written rationale that names the legitimately
divergent rounding step.

## File layout

The two modes are implemented as **parallel modules** that sit beside
each other. Editing one path never touches the other.

| Purpose | W8A16 file | W8A32 file |
|---|---|---|
| Compiler entry point | `Compiler.compile_w8a16` | `Compiler.compile_w8a32` |
| Codegen | `taccel/compiler/codegen_w8a16.py` | `taccel/compiler/codegen_w8a32.py` |
| Sequence tiling policy | `passes/memory_estimate_w8a16.py` | `passes/memory_estimate_w8a32.py` |
| Pass pipeline factory | `passes.default_pipeline_w8a16()` | `passes.default_pipeline_w8a32()` |
| Machine state | `golden_model/state_w8a16.py` | `golden_model/state_w8a32.py` |
| Simulator | `golden_model/simulator_w8a16.py` | `golden_model/simulator_w8a32.py` |
| Systolic dispatch | `golden_model/systolic_w8a16.py` | `golden_model/systolic_w8a32.py` |
| SFU | `golden_model/sfu_w8a16.py` | `golden_model/sfu_w8a32.py` |
| Benchmark tool | `tools/benchmark_w8a16.py` | `tools/benchmark_w8a32.py` |
| Quantize entry point | `quantizer.W8A16_QUANTIZE` | `quantizer.W8A32_QUANTIZE` (alias) |
| End-to-end test | `tests/test_w8a16_compile.py` | `tests/test_w8a32_compile.py` |
| Sim unit tests | `tests/test_w8a16_simulator.py` | `tests/test_w8a32_simulator.py` |
| Foundation tests | `tests/test_w8a16_foundation.py` | `tests/test_w8a32_foundation.py` |

The base classes in `taccel/golden_model/{simulator,state,sfu,systolic}.py`
remain as internal infrastructure: `SimulatorW8A16` and `SimulatorW8A32`
inherit from the base `Simulator` for byte-mover (LOAD/STORE/BUF_COPY)
and control-flow (CONFIG_TILE/SYNC/HALT) ops, all of which are
mode-agnostic. The mode-specific dispatchers override the math ops.
