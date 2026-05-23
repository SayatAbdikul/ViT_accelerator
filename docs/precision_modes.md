# Precision modes

This document explains the two precision modes supported by the TACCEL
software toolchain, why both exist, and which one to use for which job.

## Summary

| Mode | Weights | Activations | Accumulators | Compiler | Golden sim | RTL parity |
|---|---|---|---|---|---|---|
| **`w8a8`** (default) | INT8 per-channel | INT8 per-tensor | INT32 | `Compiler.compile()` | `Simulator` | Bit-exact |
| **`w8a32`** | INT8 per-channel → FP32 dequant in DRAM | FP32 | FP32 (reinterpret ACCUM) | `Compiler(mode='w8a32').compile_w8a32()` | `SimulatorW8A32` | **Suspended** |

`w8a8` is the production path. The RTL is hard-wired INT8 across all 5
compute module groups (`systolic_array`, `systolic_pe`,
`systolic_controller`, `blocking_helper_engine`, `sfu_engine`), and
`batch_compare_rtl_golden.py` signs off on exact integer logit equality
between the RTL and the W8A8 golden model.

`w8a32` is a software-only path that bypasses every activation-side
calibration knob and emits no `REQUANT*` / `DEQUANT_ADD` opcodes. It
exists to answer one question: **what accuracy ceiling does
weight-quantization-only inference reach, end-to-end through the real
compiler + golden simulator?**

## Why both exist — the bisection that motivated W8A32

A precision-sweep diagnostic split the accuracy story of W8A8 inference
into two contributions: weight quantization and activation quantization.

| Configuration | Cosine vs FP32 | Top-1 (20 COCO val) | Top-5 overlap |
|---|---:|---:|---:|
| FP32 reference (HF DeiT-tiny) | 1.0000 | 20/20 | 100% |
| Fake-quant ceiling (weights only, PyTorch hooks) | 0.9995 | 19/20 (95%) | 96% |
| W8A8 production (full toolchain) | 0.8324 | 18/20 (90%) | 72% |

The fake-quant ceiling was a PyTorch shortcut — it applied our exact
per-channel INT8 quantization to every Linear / Conv2d weight, then ran
the full FP32 forward pass. It never exercised the compiler, IR,
codegen, or golden simulator. Reaching the same ≈ 0.9995 cosine through
the real toolchain was the load-bearing invariant for the W8A32 fork.

The 0.17 cosine deficit between W8A8 and the fake-quant ceiling is
attributable entirely to *activation* quantization compounding across
12 transformer blocks. Knowing this lets future calibration / mixed-
precision investigations target the right surface.

## Architecture differences

### Activations and accumulators

In W8A32, activations stay FP32 in ABUF (128 KB of FP32 = 32 K elements,
4× tighter than W8A8's INT8 element budget). ACCUM is bit-aliased as
FP32: the byte layout is identical to W8A8's INT32, but `accum.view(np.float32)`
is used in `simulator_w8a32`. WBUF and DRAM are dtype-agnostic — DRAM
holds the FP32 dequantized weights directly.

### Sequence tiling

Because the FP32 element width is 4× larger, the M2 sequence-tiling
pass kicks in for DeiT-tiny (the residual stream is 156 KB > 128 KB
ABUF). The W8A32 codegen lands on `tile_rows = 16` for DeiT-tiny, and
the same policy with a tighter per-tile FC1 cap. ViT-B compile in
W8A32 hits the M3 WBUF wide-weight boundary in the same place W8A8
does — wide weights still need N-strip-to-DRAM independent of
activation precision.

### Attention mask

The W8A32 attention path needs an explicit mask on padded key columns:
because the simulator's softmax normalizes over all 208 padded columns,
the 11 zero-padded keys would otherwise leak ≈ 5% probability mass per
query row, compounding to a ≈ 0.02 cosine deficit over 12 blocks. The
codegen inserts an `__attention_mask__` FP32 row (`-1e9` in padded
columns) and VADD-broadcasts it into ACCUM between SCALE_MUL and
SOFTMAX inside each Q-strip iteration. See
`software/taccel/compiler/codegen_w8a32.py:_emit_qkt`.

### Quantizer pipeline

`W8A32_QUANTIZE` (exported from `taccel.quantizer`) is the canonical
weight-quant entry point — `quantize_tensor(per_channel=True)`, the
same scheme `fake_quant.apply_weight_quantization` uses. The
calibration / SmoothQuant / Hessian-guided / twin-uniform / bias-
correction modules are all activation-quant-only and explicitly
dormant on the W8A32 path (their module docstrings note this).

## RTL parity is suspended in W8A32

The RTL implements INT8 weights × INT8 activations → INT32 accumulators
across 5 module groups (`systolic_array`, `systolic_pe`,
`systolic_controller`, `blocking_helper_engine`, `sfu_engine`). All of
those would silently compute garbage if fed FP32 bit patterns on the
INT8 datapath. A full W8A32 RTL rewrite (FP32 datapath in systolic, FP32
in SFU/helper, REQUANT/REQUANT_PC removal, ABUF widening) is multi-month
work and is **not** in scope for the W8A32 fork.

Tools enforce this by erroring out instead of silently skipping:

```bash
$ python software/tools/batch_compare_rtl_golden.py --mode w8a32 --weights ...
RTL parity is suspended in W8A32 mode.
The 5 RTL module groups ... are hardwired INT8 and would silently
compute garbage on FP32 bit patterns. See docs/precision_modes.md ...
```

## How to use each mode

### W8A8 (production, default)

```bash
# Compile a model
python software/tools/compile_model.py --weights software/pytorch_model.bin -o program.bin

# Run the golden simulator
python software/tools/run_golden.py program.bin --input input.bin

# RTL-vs-golden sign-off
python software/tools/batch_compare_rtl_golden.py --weights software/pytorch_model.bin --image-dir software/images/frozen_benchmark

# Production accuracy benchmark (uses full W8A8 calibration plumbing)
python software/tools/benchmark_fp32_vs_int8.py --max-images 20
```

### W8A32 (accuracy investigation, software only)

```bash
# Compile a model in W8A32 mode (no calibration needed)
python software/tools/compile_model.py --mode w8a32 --weights software/pytorch_model.bin -o program_w8a32.bin

# Run the W8A32 golden simulator
python software/tools/run_golden.py --mode w8a32 program_w8a32.bin --input patches_fp32.npy

# End-to-end W8A32 accuracy benchmark
python software/tools/benchmark_w8a32.py --max-images 20

# Memory profile in W8A32 mode (4× tighter element budgets)
python software/tools/profile_memory.py --mode w8a32 --model vit-base

# Equivalent of the production benchmark, but routed to benchmark_w8a32.py
python software/tools/benchmark_fp32_vs_int8.py --mode w8a32 --max-images 20
```

## Load-bearing accuracy gate

The W8A32 invariant is enforced by
`software/tests/test_w8a32_compile.py::test_compile_w8a32_end_to_end_runs_to_halt`:

```python
assert cos_fq >= 0.999       # bit-equivalence with fake_quant
assert cos_fp32 >= 0.998     # within 1e-3 of the fake-quant ceiling
```

`cos_fq` is the strictest gate. The W8A32 toolchain should add **no**
measurable error on top of weight quantization, so any drift > 1e-3
indicates a regression (e.g. the seq-padding attention leak that the
mask in `_emit_qkt` fixes). Reviewers must refuse threshold weakening
without a written rationale.

## Path forward

Once the W8A32 ceiling is locked, the natural follow-ons are:

- **Mixed-precision exploration**: W8A8 with selective FP32 sites
  (e.g. only FC1 activations in FP32) to chip away at the 0.17 cosine
  deficit without paying the full FP32 cost.
- **W4A8 weight quantization (AWQ)**: orthogonal to this fork; slots
  in via a new `quantize_tensor(bits=4, per_channel=True)` path.
- **W8A32 RTL rewrite**: if a precision target locks W8A32 as the
  shipping mode, the RTL needs a FP32 datapath rewrite (systolic INT8 ×
  FP32, FP32 SFU/helper, REQUANT removal, ABUF widening). Multi-month;
  out of scope until W8A32 accuracy is proven and a precision target
  is committed.

## File-layout cheatsheet

| Purpose | W8A8 file | W8A32 file |
|---|---|---|
| Compiler entry | `compiler.py::Compiler.compile` | `compiler.py::Compiler.compile_w8a32` |
| Codegen | `compiler/codegen.py` | `compiler/codegen_w8a32.py` |
| Seq-tiling policy | `compiler/passes/memory_estimate.py` | `compiler/passes/memory_estimate_w8a32.py` |
| Simulator | `golden_model/simulator.py` | `golden_model/simulator_w8a32.py` |
| SFU | `golden_model/sfu.py` | `golden_model/sfu_w8a32.py` |
| Systolic | `golden_model/systolic.py` | `golden_model/systolic_w8a32.py` |
| Machine state | `golden_model/state.py::MachineState` | `golden_model/state_w8a32.py::MachineStateW8A32` |
| Memory helpers | `golden_model/memory.py` (`read_int8_tile`, `read_int32_tile`) | `golden_model/memory.py` (`read_fp32_tile`, `write_fp32_tile`) |
| Quantizer entry | `quantizer/quantize.py::quantize_tensor` (per-tensor or per-channel) | `quantizer/__init__.py::W8A32_QUANTIZE` (always per-channel) |
| Benchmark | `tools/benchmark_fp32_vs_int8.py` | `tools/benchmark_w8a32.py` |
| RTL parity | `tools/batch_compare_rtl_golden.py` | suspended |
