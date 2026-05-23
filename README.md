# TACCEL Transformer Accelerator

TACCEL is an experimental INT8 transformer accelerator and compiler stack for
running a quantized DeiT-tiny vision transformer workload in RTL. The project
contains both the hardware model and the software toolchain needed to assemble,
compile, simulate, and compare accelerator programs against a Python golden
model.

The current target workload is `facebook/deit-tiny-patch16-224`.

## What This Repository Contains

This repo is organized around three cooperating layers:

- `software/`: Python ISA, assembler, compiler, quantizer, golden model, and
  RTL-vs-golden comparison tools.
- `rtl/`: SystemVerilog RTL for the accelerator, plus Verilator and cocotb
  testbenches.
- `docs/`: design plans, debug plans, synthesis notes, and historical
  investigation notes.

The most important design principle is that the RTL is verified against the
same compiler and golden model used by the software stack. The goal is not just
to unit-test isolated modules, but to run compiler-generated programs through
RTL and compare their architectural outputs against the golden model.

## Target Model And Dataflow

The baseline compiler flow targets DeiT-tiny:

| Property | Value |
|---|---:|
| Sequence length | 197 tokens, padded to 208 |
| Embedding dimension | 192 |
| Attention heads | 3 |
| Head dimension | 64 |
| MLP hidden dimension | 768 |
| Transformer blocks | 12 |
| Classifier outputs | 1000 |

The current host/runtime contract assumes:

- Patch embedding is performed on the CPU.
- The accelerator starts from pre-embedded INT8 patch tokens in DRAM.
- DRAM contains weights, biases, FP16 scale tables, CLS token, and position
  embeddings.
- Accelerator SRAM offsets are expressed in 16-byte units.

## Accelerator Architecture

The RTL implements a small fixed-function transformer accelerator:

| Unit | Purpose |
|---|---|
| Fetch/decode/control | Fetch 64-bit instructions from DRAM and dispatch work |
| DMA engine | Move data between DRAM and SRAM buffers |
| Systolic array | 16x16 INT8 x INT8 matrix multiply with INT32 accumulation |
| Blocking helper engine | Local copies, requantization, scale multiply, VADD, DEQUANT_ADD |
| SFU | LayerNorm, Softmax, GELU, attention@V helper paths |
| SRAM subsystem | ABUF, WBUF, and ACCUM dual-port SRAM models |

The architectural SRAM buffers are:

| Buffer | Size | Data view | Addressing unit |
|---|---:|---|---|
| ABUF | 128 KB | INT8 activations | 16 bytes |
| WBUF | 256 KB | INT8 weights / FP16 params / INT32 bias | 16 bytes |
| ACCUM | 64 KB | INT32 accumulators | 16 bytes |

## Instruction Set

The custom ISA is defined in `software/taccel/isa` and implemented in RTL by
the fetch, decode, control, DMA, helper, SFU, and systolic units.

Implemented instruction groups:

- System/setup: `NOP`, `HALT`, `SYNC`, `CONFIG_TILE`, `SET_SCALE`,
  `SET_ADDR_LO`, `SET_ADDR_HI`
- Data movement: `LOAD`, `STORE`, `BUF_COPY`
- Matrix compute: `MATMUL`
- Quantization and helper compute: `REQUANT`, `REQUANT_PC`, `SCALE_MUL`,
  `VADD`, `DEQUANT_ADD`
- SFU compute: `SOFTMAX`, `LAYERNORM`, `GELU`, `SOFTMAX_ATTNV`

See `software/docs/isa_spec.md` and `docs/rtl_plan.md` for the detailed ISA
contract.

## Software Stack

The Python toolchain provides:

- ISA definitions and binary encoding/decoding.
- Text assembler and disassembler.
- INT8 quantization and calibration utilities.
- DeiT-tiny graph extraction and tile-level code generation.
- A sequential golden-model simulator.
- Debug and sign-off tools for comparing RTL runs against golden traces.

Useful entry points:

- `software/tools/asm.py`: assemble text assembly to `program.bin`.
- `software/tools/disasm.py`: disassemble a binary program.
- `software/tools/compile_model.py`: compile a model into an accelerator
  program.
- `software/tools/run_golden.py`: run a program in the Python golden model.
- `software/tools/compare_rtl_golden.py`: compile/run/compare RTL against the
  golden model.
- `software/tools/batch_compare_rtl_golden.py`: run one or more images through
  the RTL-vs-golden compare flow.

## Precision Modes

The software toolchain supports two precision modes, selectable via
`--mode` on the user-facing tools (`compile_model.py`, `run_golden.py`,
`benchmark_fp32_vs_int8.py`, `batch_compare_rtl_golden.py`,
`profile_memory.py`):

| Mode | Weights | Activations | Accumulators | RTL parity |
|---|---|---|---|---|
| `w8a8` (default) | INT8 per-channel | INT8 per-tensor | INT32 | Yes — bit-exact |
| `w8a32` | INT8 per-channel → FP32 dequant in DRAM | FP32 | FP32 (reinterpret ACCUM) | **Suspended** |

W8A8 is the production path: it matches the RTL exactly and is what
`batch_compare_rtl_golden.py` signs off against. W8A32 is a
software-only path used to measure the accuracy ceiling of
weight-quantization-only inference end-to-end through the real compiler
+ golden simulator. It bypasses every activation-side calibration knob
and emits no `REQUANT*` / `DEQUANT_ADD` opcodes. The 5 INT8-hardwired
RTL module groups (`systolic_array`, `systolic_pe`,
`systolic_controller`, `blocking_helper_engine`, `sfu_engine`) would
compute garbage on FP32 bit patterns, so RTL parity is explicitly
disabled in `w8a32` mode; running `batch_compare_rtl_golden.py
--mode w8a32` errors out with a pointer to `docs/precision_modes.md`.

For the W8A32 end-to-end accuracy benchmark:

```bash
./.venv/bin/python3 software/tools/benchmark_w8a32.py --max-images 20
```

See `docs/precision_modes.md` for the motivation behind the W8A32 fork,
the load-bearing accuracy gate (cos vs FP32 ≥ 0.998, cos vs fake_quant
≥ 0.999), and the path forward (mixed precision / W4A8 / etc.).

## RTL Stack

Important RTL files:

- `rtl/src/taccel_top.sv`: top-level accelerator integration.
- `rtl/src/control_unit.sv`: instruction retirement, dispatch, barriers, and
  fault handling.
- `rtl/src/fetch_unit.sv`: instruction fetch path.
- `rtl/src/decode_unit.sv`: instruction decoder.
- `rtl/src/dma_engine.sv`: DRAM/SRAM load and store engine.
- `rtl/src/blocking_helper_engine.sv`: blocking local helper operations.
- `rtl/src/sfu_engine.sv`: SFU operations.
- `rtl/src/systolic/`: systolic PE, array, and controller.
- `rtl/src/memory/`: SRAM and register-file models.

Verilator is the primary sign-off simulator. cocotb tests exist for additional
ISA-visible coverage, but the current debug and sign-off flow is centered on
native Verilator benches and the program-level runner.

## Current Status

The project can build the RTL testbench suite and run compiler-generated
programs through the Verilator runner. The debug infrastructure can emit
snapshots, SRAM write logs, systolic traces, and replay payloads to isolate
first divergences against the golden model.

Current numerical and synthesis status:

- All FP32 SFU helpers are now synthesizable SystemVerilog. The full set —
  `round`, `add`, `sub`, `mul`, `div`, `sqrt`, `exp`, `erf`, `gelu`,
  `quantize_i8`, and `from_fp16` — lives in
  `rtl/src/include/fp32_prim_pkg.sv` as bit-exact `fp32_*_bits` functions.
  The normative spec is `rtl/src/include/ARITH_CONTRACT.md`.
- The grep gate at `rtl/verilator/Makefile`'s `synth_gate` target enforces
  that `rtl/src/` (excluding `rtl/src/tb/`) contains no `DPI-C` imports,
  no `real` types, and none of `$realtobits`/`$bitstoreal`/`$rtoi`/`$itor`.
- RTL is bit-exact-equivalent to the golden model by construction:
  `software/taccel/golden_model/sfu.py` calls `software/taccel/utils/fp32_prim_ref.py`,
  which implements the same algorithms as the RTL `fp32_*_bits` package using
  sequential FP32 left-folds. The sign-off invariant is exact integer logit
  equality at `software/tools/batch_compare_rtl_golden.py:141`
  (`raw_logits_exact = golden_logits == rtl_logits`, zero tolerance).

Remaining work for FPGA bring-up (documented as future work, not in `main`):

- **SFU FSM serialization** — the `F_ROW_COMPUTE`, `F_ATTN_PREP`, and
  `F_ATTN_V_LATCH` states in `rtl/src/sfu_engine.sv` still execute 208-element
  computations combinationally. They must be broken into element-serial
  sub-states sharing one FP datapath instance before timing can close on FPGA.
- **Vivado synthesis dry-run** and target-clock closure (≈300–400 MHz
  UltraScale+ realistic for FP32 div/sqrt). Portable RTL only — no vendor FP
  IP, since vendor rounding would break the exact-logit contract.

## Setup

Python dependencies are listed in `software/requirements.txt`. A typical local
setup is:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r software/requirements.txt
```

RTL simulation requires Verilator and a C++17 compiler. On macOS with
Homebrew, for example:

```bash
brew install verilator
```

The repository currently expects model weights and sample images to live under
`software/`, for example:

- `software/pytorch_model.bin`
- `software/images/frozen_benchmark/000000002006.jpg`

## Common Commands

Run the Python test suite:

```bash
./.venv/bin/python3 -m pytest software/tests -q
```

Build and run the full native Verilator suite:

```bash
make -C rtl/verilator all
```

Run selected RTL tests:

```bash
make -C rtl/verilator test_decode
make -C rtl/verilator test_control
make -C rtl/verilator test_dma
make -C rtl/verilator test_helpers
make -C rtl/verilator test_fp32_prims
make -C rtl/verilator test_sfu
make -C rtl/verilator test_systolic
make -C rtl/verilator test_systolic_qkt
```

Build the full-program RTL runner:

```bash
make -C rtl/verilator run_program
```

Run one-image RTL-vs-golden batch compare:

```bash
./.venv/bin/python3 software/tools/batch_compare_rtl_golden.py \
  --weights software/pytorch_model.bin \
  --image software/images/frozen_benchmark/000000002006.jpg \
  --max-images 1 \
  --summary-out /tmp/batch_compare_one_image/summary.json \
  --work-dir /tmp/batch_compare_one_image \
  --keep-work
```

Run a lower-level compile-and-compare flow:

```bash
./.venv/bin/python3 software/tools/compare_rtl_golden.py \
  --summary-out /tmp/compare_rtl_golden/summary.json \
  --work-dir /tmp/compare_rtl_golden \
  compile \
  --scenario baseline_default \
  --weights software/pytorch_model.bin \
  --image software/images/frozen_benchmark/000000002006.jpg
```

Enable replay-backed QK/SFU regressions when a replay payload bundle exists:

```bash
RTL_QKT_REPLAY_DIR=/tmp/rtl_debug_stepaa2/replay_payloads \
  make -C rtl/verilator test_sfu test_systolic_qkt
```

## Debugging And Provenance Flow

A major part of this project is the RTL/golden debug workflow. When a full
compare fails, the tools can emit:

- `rtl_summary.json`: execution status, cycles, logits, faults, violations.
- `rtl_snapshot_manifest.json` and `rtl_snapshot_data.bin`: checkpoint tensor
  snapshots.
- `first_divergence.json`: first raw checkpoint mismatch.
- `effective_first_divergence.json`: first mismatch after approved rebasing.
- SRAM write logs for ABUF, WBUF, and ACCUM.
- Systolic window traces and hidden snapshots.
- Replay payload bundles for small native Verilator reducers.

The intended workflow is:

1. Run full RTL-vs-golden compare.
2. Inspect the first raw divergence.
3. Use emitted replay payloads to build or run a focused reducer.
4. Diff SRAM provenance and checkpoint artifacts.
5. Classify the mismatch as a real RTL bug, a replay/source issue, a capture
   issue, or an approved nonblocking/rounding artifact.

This workflow is documented in more detail in:

- `docs/rtl_debugging_plan.md`
- `docs/rtl_debug_plan.md`
- `rtl/TESTBENCHES.md`

## Development Notes

- Prefer adding focused Verilator tests for RTL bugs before broad program-level
  tests.
- Keep golden-model semantics and RTL-visible semantics aligned; do not rely on
  Python file boundaries to infer hardware dispatch.
- Be careful with FP32 behavior. NumPy, C++ libm, SV `real`, and custom RTL
  arithmetic can differ by one LSB unless the operation order and rounding
  points are intentionally frozen.
- Do not run multiple Verilator builds that target the same build directory in
  parallel; they can trample generated archives.

## Toward An LLM Accelerator

This repository is a useful foundation for an LLM accelerator, especially the
ISA/toolchain/test infrastructure and systolic/DMA/control blocks. However, an
LLM accelerator would require substantial new architecture:

- autoregressive decode scheduling
- KV-cache layout and update paths
- RoPE/RMSNorm/SwiGLU operator support
- prefill vs. decode tiling strategies
- long-context memory planning
- new golden-model and checkpoint coverage

Treat this codebase as a strong transformer-accelerator prototype rather than a
drop-in finished LLM accelerator.

## Further Reading

- `software/CODEBASE.md`: detailed software stack walkthrough.
- `rtl/TESTBENCHES.md`: RTL testbench ownership and commands.
- `docs/rtl_plan.md`: hardware/software contract and RTL roadmap.
- `docs/archive/rtl_synthesis_plan_phase0-4_COMPLETED.md`: synthesis-oriented planning notes from the Phase 0–4 DPI/`real` migration (now complete).
- `software/docs/isa_spec.md`: ISA details.

