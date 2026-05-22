# Plan: Make the TACCEL RTL Synthesis-Ready

## Context

The RTL today is **functionally correct but not synthesizable**. It was built simulation-first to establish golden-model parity. Now that parity is achieved (292/292 Python tests, `effective_pass: True`), the next step is to turn the design into something a synthesis tool (Vivado / Quartus / Yosys / DC) will accept and close timing on.

Two classes of construct currently block synthesis:

1. **DPI-C calls** — 6 imports across `sfu_engine.sv` and `blocking_helper_engine.sv` (`sfu_fp32_div`, `sfu_fp32_exp`, `sfu_fp32_sqrt`, `sfu_fp32_gelu`, `sfu_fp32_quantize_i8`, plus one in blocking helper). These are simulation-only handoffs to C++.
2. **SystemVerilog `real` (IEEE-754 FP64) arithmetic** — 82 occurrences across the two engines and `fp32_prim_pkg.sv`. `real` is explicitly non-synthesizable; it's treated as `double` only by the simulator.

A partial migration is already in flight:
- `rtl/src/include/fp32_prim_pkg.sv` (**untracked, 435 lines**) implements bit-level FP32 round/add/sub/mul purely in SV, plus `real`-interface shims
- `rtl/src/tb/tb_fp32_prim.sv` + `rtl/verilator/test_fp32_prims.cpp` (**both untracked**) bit-exact-check the package against native C++ `float`
- Staged edits in `sfu_engine.sv` and `blocking_helper_engine.sv` have swapped 4 of the 9 DPI calls for package calls (`add/sub/mul/round`) but still use `real` as the data type and still call DPI for `div/exp/sqrt/gelu/quantize_i8`

Transcendentals are the hard part: IEEE-754 does **not** mandate correctly-rounded results for `exp`/`gelu`, so bit-matching C++ `expf` in hardware is not a stable target (libm varies across platforms). Synthesis-readiness therefore requires **changing the golden model's reference to match a hardware-realizable algorithm**, not forcing the hardware to match libm.

Outcome of this plan: all RTL passes lint-for-synthesis (no DPI, no `real`, no `$realtobits`/`$bitstoreal`), a reproducible arithmetic contract between golden model and RTL, and a documented path to pipelined timing closure.

---

## Scope decisions (decide before Phase 1)

| Question | Recommended default | Why |
|---|---|---|
| Keep FP32 inside the SFU, or go fixed-point? | **Keep FP32**, synthesized from `fp32_prim_pkg` | Minimizes blast radius on the golden model; matches what real ML accelerators do for LN/softmax/GELU |
| `exp` / `GELU` reference algorithm | **Minimax polynomial with 2^x range reduction** (exp) and **piecewise polynomial** (GELU), each with a documented Python reference | Fully synthesizable, well-understood, modest area. Rules out matching libm. |
| `div` / `sqrt` algorithm | **Newton-Raphson with IEEE-754 correct rounding** OR **SRT radix-4** | Standard and synth-friendly; pick once |
| Target | **FPGA first (Xilinx/AMD or Intel)**, ASIC later | FPGA confirms synthesizability without committing to a PDK |
| Clock target | **Pick one concrete number** (e.g. 500 MHz FPGA) | Drives pipelining depth in Phase 5 |

These are the main fork points. Change them if you disagree and the rest of the plan still works.

---

## Phased Plan

Each phase is independently valuable — if you stop after Phase N, the codebase is still in a better state than it started.

### Phase 0 — Ground rules and guardrails (0.5 day)

**Goal:** lock in what "synthesizable" means and put a gate in the build to enforce it.

- Commit the untracked files (`fp32_prim_pkg.sv`, `tb_fp32_prim.sv`, `test_fp32_prims.cpp`, Makefile delta) so the Phase-1 work has a stable base
- Add a `SYNTH_BUILD` flag in `rtl/verilator/Makefile` and `rtl/cocotb/Makefile` that, when set, compiles with a linter rule banning DPI imports and `real` types
  - `verilator --lint-only -Wwarn-REALCVT -Wwarn-DPIBAD` plus a custom grep gate on `/real\b|import "DPI-C"/` in `rtl/src/` (excluding `rtl/src/tb/`)
- Document the FP32 contract in a new `rtl/src/include/ARITH_CONTRACT.md` (one file, ~50 lines): what each primitive does, which are correctly-rounded, which are approximations with error bounds

**Verification:** `make SYNTH_BUILD=1 lint` fails today (because of DPI and `real`); that failure becomes the target we drive to zero.

---

### Phase 1 — Complete the pure-SV FP32 primitive package (~4 days)

**Goal:** extend `fp32_prim_pkg.sv` so every FP32 op used by the engines has a synthesizable bit-level implementation, each with a bit-exact reference in Python.

**Work:**

1. Verify existing `fp32_add_bits` / `fp32_sub_bits` / `fp32_mul_bits` / `fp32_round_bits` pass `test_fp32_prims` today (256 random + edge + real-shim cases already written at `rtl/verilator/test_fp32_prims.cpp:115-217`)
2. Add:
   - `fp32_div_bits` — IEEE-754 correctly-rounded; implement via Newton-Raphson reciprocal + corrective rounding, or SRT radix-4
   - `fp32_sqrt_bits` — IEEE-754 correctly-rounded; Newton-Raphson with one corrective step
   - `fp32_exp_bits` — minimax polynomial in range-reduced argument; not correctly-rounded, ≤2 ULP target. Publish the polynomial coefficients + range reduction formula in `ARITH_CONTRACT.md`
   - `fp32_gelu_bits` — either `x/2 * (1 + erf(x/√2))` built from `fp32_exp_bits`, or a piecewise polynomial (simpler, bigger tables). Document the choice.
   - `fp32_quantize_i8_bits` — `round_half_even(x/scale)` + saturate; trivial
   - `fp32_from_fp16_bits` — pure integer rewiring of 16-bit IEEE-754 half → 32-bit IEEE-754 single; replaces `fp16_to_real` today at `sfu_engine.sv:211-235` and `blocking_helper_engine.sv:317-341`
3. For each new primitive:
   - Add a Python reference function (bit-exact) in a new `software/taccel/utils/fp32_prim_ref.py`
   - Add a directed + random stress check in `test_fp32_prims.cpp` that compares against the Python reference's exported bit-patterns

**Do not** remove the `_real` shims yet — keep them as a bridge for the in-progress engines. They get deleted in Phase 4.

**Verification:** `make test_fp32_prims` all cases pass; `fp32_prim_ref.py` and SV package agree on every tested point.

---

### Phase 2 — Align the golden model to the new arithmetic (~3 days)

**Goal:** the Python `simulator.py` no longer calls NumPy `exp`/`sqrt`/`/` for SFU paths. It calls the same algorithms the RTL will use. After this phase, the golden model *is* the spec; the RTL just has to be bit-equivalent to it.

**Work:**

1. In `software/taccel/golden_model/sfu.py`, replace the raw NumPy FP32 ops in SOFTMAX / LAYERNORM / GELU / SOFTMAX_ATTNV with calls to `fp32_prim_ref.py`
2. Re-run `compare_rtl_golden.py compile` and `batch_compare_rtl_golden.py`. The known 1-LSB LAYERNORM artifact at `block3_ln2[95,158]` should disappear (both sides now use the same accumulation order and the same rounding). If new divergences appear, they indicate edge cases in the primitive package to fix.
3. Regenerate any pinned test vectors in `software/tests/` that depend on the old NumPy path. The vast majority of the 292 tests don't care — they test structural behavior, not transcendental bit-patterns. Walk through failures and classify: (a) tolerable numeric drift → update expected value, (b) real regression → investigate.
4. Add a `tools/lib/fp32_calibrate_golden.py` helper that replays calibration against the new golden so calibration scales are derived against the same numeric surface the RTL will produce

**Critical file locations:**
- `software/taccel/golden_model/sfu.py:1-299` — SFU ops
- `software/taccel/utils/` — new `fp32_prim_ref.py` lives here
- `software/tests/test_golden_model.py`, `test_compiler.py` — expect some expected-value churn

**Verification:** `pytest software/tests/ -x` all pass. `batch_compare_rtl_golden.py` shows no divergences beyond the scenarios we know about (and ideally zero, since RTL and golden now share their arithmetic reference).

---

### Phase 3 — Strip `real` and DPI from the two engines (~5 days)

**Goal:** `sfu_engine.sv` and `blocking_helper_engine.sv` contain only `logic` / `logic [31:0]` (FP32 bits) / `logic [7:0]` data; no `real`, no DPI-C.

**Critical file locations:**

- `rtl/src/sfu_engine.sv:68-72` — 5 DPI imports to delete
- `rtl/src/sfu_engine.sv:129-141` — storage declared as `real`; change to `logic [31:0]` arrays
- `rtl/src/sfu_engine.sv:195-289` — `pow2_int`, `fp16_to_real`, `quantize_to_i8`, `gelu_real` — rewrite as bit-level
- `rtl/src/sfu_engine.sv:437-453, 627, 654, 680-722, 849, 873, 897-898` — all the FSM arithmetic sites that use `real` variables
- `rtl/src/blocking_helper_engine.sv:70` — one remaining DPI import
- `rtl/src/blocking_helper_engine.sv:301-349` — `pow2_int`, `fp16_to_real`, `round_half_even` helpers
- `rtl/src/blocking_helper_engine.sv:420-441` — VADD FSM arithmetic using `real`

**Work:**

1. Rewrite `fp16_to_real` as `fp32_from_fp16_bits` (pure bit rewiring: sign stays, exp rebiased from 15→127, mantissa left-padded from 10→23 bits; handle subnormals/inf/NaN).
2. Change all `real` storage (`row_data_q`, `attn_accum_q`, `gamma_q`, `beta_q`, scale regs, debug regs) to `logic [31:0]` (FP32 bits).
3. Replace every `fp32_*_real` and DPI call with `fp32_*_bits`.
4. Preserve the `/* verilator public_flat_rd */` observability pragmas; widen the C++ testbench helpers in `rtl/verilator/include/testbench.h` that unpack FP32 state (they already exist for the debug outputs — just need to read 32-bit registers instead of 64-bit `double`).
5. Run the full Verilator suite, then `batch_compare_rtl_golden.py`. Any new divergence vs. Phase 2 golden is a porting bug in the engine rewrite — fix before moving on.

**Verification:** grep `rtl/src/sfu_engine.sv rtl/src/blocking_helper_engine.sv` for `\breal\b` and `DPI-C` returns zero. All existing tests still pass.

---

### Phase 4 — Strip the `_real` shims from `fp32_prim_pkg.sv` (~0.5 day)

**Goal:** the package itself is synthesizable. After Phase 3 nothing calls the shims.

**Work:**

1. Delete from `rtl/src/include/fp32_prim_pkg.sv`:
   - `fp32_pow2` (line 30, FP64 helper)
   - `fp32_from_fp64_bits` (line 79)
   - `fp32_real_to_bits` (line 134)
   - `fp32_bits_to_real` (line 138)
   - `fp32_round_real`, `fp32_add_real`, `fp32_sub_real`, `fp32_mul_real` (lines 413-430)
2. Delete the `real`-interface block from `tb_fp32_prim.sv` and the `expect_real_shims` harness in `test_fp32_prims.cpp` (only the bit-level checks remain)
3. Delete the leftover `extern "C"` shims in `rtl/verilator/include/testbench.h:36-100` that aren't already gone

**Verification:** `grep -rn "real\b\|\$realtobits\|\$bitstoreal" rtl/src/` returns only the testbench tree; `make SYNTH_BUILD=1 lint` passes.

---

### Phase 5 — Pipeline the FP32 primitives and the SFU FSM (~2–4 weeks)

**Goal:** Phases 1–4 produce *combinational* FP32 primitives that will synthesize but won't close timing — a single-cycle `fp32_div_bits` has multi-nanosecond critical paths. Real synthesis needs pipelined primitives and an FSM that tolerates multi-cycle latency.

**Work:**

1. Pick the clock target (default: 500 MHz FPGA). Estimate cycle budgets:
   - `fp32_add/sub`: 2–3 cycles
   - `fp32_mul`: 3–4 cycles
   - `fp32_div`: 8–16 cycles
   - `fp32_sqrt`: 8–16 cycles
   - `fp32_exp`: 4–8 cycles (table + polynomial)
   - `fp32_gelu`: 6–12 cycles
2. Refactor each primitive in `fp32_prim_pkg.sv` from function to **pipelined module** with `valid_in` / `valid_out` / `ready` handshake
3. Rewrite `sfu_engine.sv`'s inner compute loops to use these pipelined modules — introduce a shallow scoreboard so the FSM can issue one element per cycle and drain results over N cycles
4. Rewrite the GELU and SOFTMAX_ATTNV paths similarly; these are the deepest chains
5. Verify against the Phase-2 golden at every step

**This is the bulk of the effort.** Everything before is mechanical; this is real microarchitecture work. Budget accordingly.

**Verification:** `batch_compare_rtl_golden.py` still passes; timing reports from Vivado/Quartus close at the target frequency.

---

### Phase 6 — Synthesis dry-run and lint cleanup (~3 days)

**Goal:** actually run synthesis, see what the tool complains about, fix it.

**Work:**

1. Pick an FPGA target (e.g. Xilinx UltraScale+, ~200 KLUT class)
2. Run `vivado -mode batch` synthesis on the full hierarchy rooted at `rtl/src/taccel_top.sv`
3. Triage warnings by severity:
   - **Critical:** inferred latches (missing `default:` in case / missing `else`), combinational loops, X-propagation, wrong BRAM inference
   - **Important:** timing violations at target freq (→ iterate pipelining from Phase 5)
   - **Cosmetic:** width-mismatch warnings, unused signals
4. Fix in priority order. Common fixes: add `default:` to every case, fully initialize all FFs in reset, replace any inferred latch with explicit FF, ensure SRAMs are hit by only one writer per port (already true in the design).
5. Also run Verilator lint with `-Wall -Wwarn-style` as a secondary quality gate (stricter than synthesis lint and catches style issues early)

**Critical files to review for latches / case completeness:**
- `rtl/src/control_unit.sv` (large FSM)
- `rtl/src/blocking_helper_engine.sv` (1511 lines, nested FSMs — known refactor candidate)
- Post-Phase-5 `rtl/src/sfu_engine.sv`

**Verification:** clean synthesis report (zero inferred latches, zero combinational loops); timing met or within 10% of target with clear path to close.

---

### Phase 7 — Optional: refactor `blocking_helper_engine.sv` (~1 week)

Not strictly required for synthesis but flagged as a maintenance issue. 1511 lines, 6 operations in one FSM. If synthesis reveals timing problems concentrated here, split into per-op submodules (`buf_copy_engine.sv`, `vadd_engine.sv`, `requant_engine.sv`, `dequant_add_engine.sv`) with a top-level mux. Otherwise defer.

---

## What stays untouched

These parts of the RTL are already synthesis-clean:

- `rtl/src/include/taccel_pkg.sv` — pure parameters and enums
- `rtl/src/memory/sram_dp.sv` — already has `(* ram_style = "block" *)`, sync read/write
- `rtl/src/memory/sram_subsystem.sv`, `register_file.sv`
- `rtl/src/systolic/*` — INT8 MACs, no FP anywhere
- `rtl/src/decode_unit.sv`, `fetch_unit.sv`, `dma_engine.sv`, `control_unit.sv`, `taccel_top.sv` — pure integer / handshake logic

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase-2 changes to golden break calibration, dropping accuracy | Re-run calibration on 20-image frozen benchmark after Phase 2; if top-1 drops >1%, investigate before Phase 3. Keep old path available behind a flag during transition. |
| `fp32_exp_bits` polynomial has >1 ULP error that compounds in softmax | Run numerical study on one layer's worth of actual softmax inputs; choose polynomial degree to bound worst-case relative error. Document in `ARITH_CONTRACT.md`. |
| Phase-5 pipelining changes timing of async handshakes with control unit | Keep the `sfu_dispatch / sfu_busy` contract; internal pipelining is transparent to `taccel_top` |
| Synthesis reveals fundamental FSM issues only after much work | Phase 6 is before committing to PDK/ASIC; FPGA-first strategy catches this cheaply |

---

## Final verification checklist

1. `grep -rn "real\b\|import \"DPI-C\"\|\$realtobits\|\$bitstoreal" rtl/src/` → zero hits outside `rtl/src/tb/`
2. `make SYNTH_BUILD=1 lint` passes
3. `pytest software/tests/ -x` → all 292 tests pass
4. `python software/tools/batch_compare_rtl_golden.py --image-dir software/images/frozen_benchmark/ --summary-out /tmp/batch_synth.json` → zero divergences (or only the known classified ones)
5. Vivado / Quartus synthesis report on `taccel_top`: zero inferred latches, zero combinational loops, timing met at chosen clock

---

## Effort summary

| Phase | Effort | Cumulative |
|---|---|---|
| 0: Guardrails | 0.5 d | 0.5 d |
| 1: Complete SV primitives | 4 d | 4.5 d |
| 2: Align golden model | 3 d | 7.5 d |
| 3: Strip `real`/DPI from engines | 5 d | 12.5 d |
| 4: Strip shims from package | 0.5 d | 13 d |
| 5: Pipelining | 2–4 weeks | 4–5 weeks |
| 6: Synthesis dry-run | 3 d | 4.5–5.5 weeks |
| 7: Optional helper refactor | 1 week | 5.5–6.5 weeks |

Phases 0–4 alone get you a fully simulation-equivalent RTL with no DPI and no `real` — a significant milestone and ~13 days of work. Phases 5–6 get you to actual timing closure.
