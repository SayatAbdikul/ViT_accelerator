#!/usr/bin/env python3
"""W8A16 RTL-vs-golden bit-exact parity comparator.

The load-bearing W8A16 invariant: the Verilator-built RTL simulator and
``SimulatorW8A16`` must produce **bit-exact** FP16 classifier logits when
driven with the same program and the same FP16 patch input. The
foundations for this are:

* Phase 1: ``software/taccel/golden_model/systolic_w8a16.py`` uses a
  sequential FP32 K-loop accumulator that matches the RTL PE order.
* Phase 2: ``rtl/src/systolic/`` widens to FP16 inputs and FP32 MAC
  using ``fp32_prim_pkg::fp32_mul_bits`` + ``fp32_add_bits``.
* Phase 3: ``rtl/src/sfu_engine.sv`` keeps FP32-internal math; ABUF
  endpoints widen/narrow via ``fp32_from_fp16_bits`` /
  ``fp32_to_fp16_bits``.
* Phase 4: ``rtl/src/blocking_helper_engine.sv`` rewires VADD/SCALE_MUL
  to the same widen/narrow pattern.
* Phase 5: decode_unit/control_unit reject INT8-bridging opcodes with
  FAULT_UNSUPPORTED_OP so dead paths cannot drift.

This script:

  1. Loads (or compiles) a W8A16 ``ProgramBinary``.
  2. Writes the FP16 patch bytes to a temp file and spawns the Verilator
     runner ``rtl/verilator/build/run_program/Vtaccel_top`` with
     ``--abuf-dump-out`` so the final ABUF is materialised on disk.
  3. Runs ``SimulatorW8A16`` against the same program with the same
     patches in-process.
  4. Slices the classifier logits from both ABUF images at
     ``compiler_manifest['classifier_output']['offset_bytes']`` and
     compares them as ``np.uint16`` views (FP16 bit patterns).
  5. Returns 0 on bit-exact match, 1 on first-divergence detection.

**The bit-exact gate may not be weakened.** If logits diverge, the
divergence must be root-caused in one of the phases listed above. Do
not paper over a divergence with ``np.allclose`` — that defeats the
contract this script enforces.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.assembler.assembler import ProgramBinary  # noqa: E402
from taccel.golden_model.simulator_w8a16 import SimulatorW8A16  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO_ROOT / "rtl" / "verilator" / "build" / "run_program" / "Vtaccel_top"
ABUF_BYTES = 131072  # matches taccel_pkg::ABUF_BYTES


@dataclass
class ParityResult:
    """Outcome of a single RTL-vs-golden comparison."""
    image_id: Optional[int]
    passed: bool
    rtl_status: str
    golden_halted: bool
    rtl_cycles: int
    first_divergence_index: Optional[int]
    rtl_logit_bits: Optional[int]
    golden_logit_bits: Optional[int]
    rtl_argmax: Optional[int]
    golden_argmax: Optional[int]
    rtl_summary_json: dict


def _ensure_runner(runner_path: Path) -> None:
    if not runner_path.exists():
        raise SystemExit(
            f"RTL runner binary missing: {runner_path}\n"
            f"Build it first:  make -C rtl/verilator run_program"
        )


def _slice_logits_bits(abuf_bytes: bytes, classifier_output: dict) -> np.ndarray:
    """Return FP16 logits as uint16 bit patterns sliced at the manifest offset."""
    off = classifier_output["offset_bytes"]
    n_pad = classifier_output["N_pad"]
    logical = classifier_output["logical_cols"]
    fp16 = np.frombuffer(abuf_bytes, dtype=np.float16, count=n_pad, offset=off)[:logical]
    return fp16.view(np.uint16).copy()


def _run_rtl(
    runner: Path,
    program_path: Path,
    patches_fp16: np.ndarray,
    *,
    max_cycles: int,
    work_dir: Path,
) -> tuple[Optional[bytes], dict, int]:
    """Invoke the Verilator runner.

    Returns (abuf_bytes_or_None, summary_dict, exit_code). The runner uses
    documented exit codes:
      0 — program halted cleanly
      2 — parse error or CLI/usage error
      3 — summary.timeout (program did not halt within --max-cycles)
      4 — summary.violations non-empty (e.g. forbidden async-engine overlap)
    Non-zero exits are returned as data — they're legitimate parity-failure
    signals, not script errors. Callers should consult summary["status"] /
    summary["fault_*"] for context.
    """
    rows, cols = patches_fp16.shape
    raw_path = work_dir / "patches_fp16.bin"
    raw_path.write_bytes(patches_fp16.tobytes())

    json_out = work_dir / "summary.json"
    abuf_out = work_dir / "abuf_dump.bin"

    # patch-cols is the FP16 byte count for one patch row (cols * 2). The
    # runner pad-aligns to 16 bytes per row, and the codegen lays out one
    # FP16 patch (192 elements = 384 bytes) per 16-byte-aligned row, which
    # is already 16-aligned, so no padding needed.
    patch_cols_bytes = cols * 2

    cmd = [
        str(runner),
        "--program", str(program_path),
        "--json-out", str(json_out),
        "--abuf-dump-out", str(abuf_out),
        "--patches-raw", str(raw_path),
        "--patch-rows", str(rows),
        "--patch-cols", str(patch_cols_bytes),
        "--max-cycles", str(max_cycles),
        "--num-classes", "0",  # we do not consume the ACCUM-INT32 logits dump
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Parse errors (exit 2) leave the summary as a stub with parse_error set;
    # the runner does not produce an ABUF dump. Surface them as a script
    # error — they indicate a malformed program / CLI, not a parity result.
    if proc.returncode == 2:
        try:
            stub = json.loads(json_out.read_text())
            parse_err = stub.get("parse_error", "")
        except Exception:
            parse_err = proc.stderr.strip()
        raise RuntimeError(
            f"RTL runner refused the program (exit 2): {parse_err}\n"
            f"  cmd: {' '.join(cmd)}"
        )

    summary: dict
    try:
        summary = json.loads(json_out.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"RTL runner did not produce a readable summary.json: {exc}\n"
            f"  exit={proc.returncode} stderr={proc.stderr.strip()}\n"
            f"  cmd: {' '.join(cmd)}"
        )

    abuf_bytes: Optional[bytes] = None
    if abuf_out.exists():
        raw = abuf_out.read_bytes()
        if len(raw) != ABUF_BYTES:
            raise RuntimeError(
                f"Unexpected ABUF dump size: {len(raw)} (want {ABUF_BYTES})"
            )
        abuf_bytes = raw
    elif proc.returncode == 0:
        # A clean exit must always produce the dump.
        raise RuntimeError(
            f"RTL runner exited cleanly but did not write {abuf_out}"
        )
    # exit 3 / 4 with no abuf is fine — the runner aborted before the dump.
    return abuf_bytes, summary, proc.returncode


def _run_golden(program: ProgramBinary, patches_fp16: np.ndarray) -> bytes:
    """Run SimulatorW8A16 in-process, return full ABUF bytes at sim end."""
    sim = SimulatorW8A16()
    sim.load_program(program)
    patch_bytes = patches_fp16.tobytes()
    sim.state.dram[program.input_offset:program.input_offset + len(patch_bytes)] = patch_bytes
    sim.run(max_instructions=program.insn_count + 10)
    if not sim.state.halted:
        raise RuntimeError("SimulatorW8A16 did not reach HALT")
    return bytes(sim.state.abuf)


def compare_program(
    program: ProgramBinary,
    patches_fp16: np.ndarray,
    *,
    runner: Path = DEFAULT_RUNNER,
    # DeiT-tiny W8A16 with the Phase-2 doubled K-strip cadence takes
    # ~50–100M cycles. 500M gives plenty of headroom; the runner returns
    # exit code 3 (summary.timeout=true) if the program does not HALT.
    max_cycles: int = 500_000_000,
    image_id: Optional[int] = None,
) -> ParityResult:
    """Compare RTL vs golden for one (program, patches) pair.

    Returns a :class:`ParityResult` describing the outcome. Bit-exactness
    is the contract; callers should treat ``passed=False`` as a failure
    unless they have a written rationale for the divergence.
    """
    _ensure_runner(runner)

    co = program.compiler_manifest["classifier_output"]

    with tempfile.TemporaryDirectory(prefix="rtl_vs_golden_") as tmp:
        tmp_path = Path(tmp)
        # Stage the program.bin in the temp dir to keep paths short.
        prog_path = tmp_path / "program.bin"
        prog_path.write_bytes(program.to_bytes())

        rtl_abuf, rtl_summary, rtl_exit = _run_rtl(
            runner, prog_path, patches_fp16,
            max_cycles=max_cycles, work_dir=tmp_path,
        )
        golden_abuf = _run_golden(program, patches_fp16)

    rtl_status = rtl_summary.get("status", "unknown")
    rtl_cycles = int(rtl_summary.get("cycles", 0))
    gold_bits = _slice_logits_bits(golden_abuf, co)
    gold_argmax = int(np.argmax(gold_bits.view(np.float16).astype(np.float32)))

    # Runner did not produce a valid ABUF (timeout / violation / fault). The
    # RTL did not run the program to completion, so there's no bit-exact
    # comparison to make — record the runner status and fail the parity gate.
    if rtl_abuf is None or rtl_status != "halted":
        return ParityResult(
            image_id=image_id,
            passed=False,
            rtl_status=rtl_status,
            golden_halted=True,
            rtl_cycles=rtl_cycles,
            first_divergence_index=None,
            rtl_logit_bits=None,
            golden_logit_bits=None,
            rtl_argmax=None,
            golden_argmax=gold_argmax,
            rtl_summary_json=rtl_summary,
        )

    rtl_bits = _slice_logits_bits(rtl_abuf, co)
    rtl_argmax = int(np.argmax(rtl_bits.view(np.float16).astype(np.float32)))

    if np.array_equal(rtl_bits, gold_bits):
        return ParityResult(
            image_id=image_id,
            passed=True,
            rtl_status=rtl_status,
            golden_halted=True,
            rtl_cycles=rtl_cycles,
            first_divergence_index=None,
            rtl_logit_bits=None,
            golden_logit_bits=None,
            rtl_argmax=rtl_argmax,
            golden_argmax=gold_argmax,
            rtl_summary_json=rtl_summary,
        )

    diff_mask = rtl_bits != gold_bits
    first = int(np.argmax(diff_mask))
    return ParityResult(
        image_id=image_id,
        passed=False,
        rtl_status=rtl_status,
        golden_halted=True,
        rtl_cycles=rtl_cycles,
        first_divergence_index=first,
        rtl_logit_bits=int(rtl_bits[first]),
        golden_logit_bits=int(gold_bits[first]),
        rtl_argmax=rtl_argmax,
        golden_argmax=gold_argmax,
        rtl_summary_json=rtl_summary,
    )


def format_result(result: ParityResult) -> str:
    tag = f"image_id={result.image_id}" if result.image_id is not None else "single"
    if result.passed:
        return (
            f"[PASS] {tag}\n"
            f"  rtl_status={result.rtl_status} cycles={result.rtl_cycles}\n"
            f"  argmax: rtl={result.rtl_argmax} golden={result.golden_argmax}"
        )

    # The RTL didn't reach HALT — surface why instead of an arbitrary "logit[0]
    # diverged" message (we don't have RTL logits to diverge with).
    if result.first_divergence_index is None:
        fault_ctx = result.rtl_summary_json.get("fault_context", {})
        body = (
            f"  rtl_status={result.rtl_status} cycles={result.rtl_cycles} "
            f"(no bit-exact comparison — runner did not halt)\n"
            f"  golden argmax={result.golden_argmax}"
        )
        if fault_ctx.get("valid"):
            body += (
                f"\n  fault: code={fault_ctx.get('fault_code')} "
                f"pc={fault_ctx.get('pc')} "
                f"opcode={fault_ctx.get('opcode')} "
                f"source={fault_ctx.get('source_name')}"
            )
        return f"[FAIL] {tag}\n{body}"

    return (
        f"[FAIL] {tag}\n"
        f"  rtl_status={result.rtl_status} cycles={result.rtl_cycles}\n"
        f"  first divergence at logit[{result.first_divergence_index}]: "
        f"rtl=0x{result.rtl_logit_bits:04x} golden=0x{result.golden_logit_bits:04x}\n"
        f"  argmax: rtl={result.rtl_argmax} golden={result.golden_argmax}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_program(path: Path) -> ProgramBinary:
    return ProgramBinary.from_bytes(path.read_bytes())


def _load_patches(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.frombuffer(path.read_bytes(), dtype=np.float16)
    arr = arr.astype(np.float16)
    if arr.ndim == 1:
        # Assume packed (rows*cols,) — caller must reshape themselves before
        # using this script in a pipeline; for the bit-exact gate we accept
        # the 2-D form.
        raise SystemExit("--patches must be a 2-D (rows, cols) FP16 array")
    return arr


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--program", required=True, type=Path,
                        help="Compiled W8A16 program.bin")
    parser.add_argument("--patches", required=True, type=Path,
                        help="2-D FP16 patch tensor (.npy or raw .bin)")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER,
                        help="Path to the Verilator-built run_program binary")
    parser.add_argument("--max-cycles", type=int, default=500_000_000,
                        help="RTL cycle budget (default 500M; DeiT-tiny W8A16 ~50-100M)")
    parser.add_argument("--image-id", type=int, default=None,
                        help="Optional image identifier to embed in the report")
    args = parser.parse_args(argv)

    program = _load_program(args.program)
    patches = _load_patches(args.patches)

    result = compare_program(
        program, patches,
        runner=args.runner,
        max_cycles=args.max_cycles,
        image_id=args.image_id,
    )
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
