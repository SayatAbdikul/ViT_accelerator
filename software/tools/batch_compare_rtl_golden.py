#!/usr/bin/env python3
"""W8A16 RTL-vs-golden bit-exact parity gate over the 20 frozen images.

This is the load-bearing acceptance gate for the W8A16 RTL fork
(Phase 6 of ``.claude/plans/now-create-a-comprehensive-cheerful-lampson.md``).

For each image in the frozen benchmark set:

  1. Preprocess pixels (resize 256, center-crop 224, ImageNet normalise).
  2. Run the host-side patch projection (same fake-quant patch weights
     the in-program dequant pipeline uses) and narrow to FP16.
  3. Drive the Verilator runner + ``SimulatorW8A16`` through the same
     compiled program with the same FP16 patch input.
  4. Slice FP16 logits from both ABUF images at the manifest offset
     and assert bit-exact equality on the uint16 view.

A PASS is **bit-exact** equality of every FP16 logit bit. The gate
mirrors the deleted W8A8 harness and remains the single load-bearing
contract for RTL correctness. **Do not weaken** to a tolerance unless
you have a written rationale that explains the specific rounding step
that legitimately diverges.

Runtime warning: Verilator simulation of a full DeiT-tiny program takes
several minutes per image; this gate is a once-per-CI run, not a quick
loop. The default 20-image evaluation can take ~1–2 hours.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taccel.compiler.cache import load_or_compile  # noqa: E402
from taccel.model_config import ModelConfig  # noqa: E402
from taccel.quantizer.fake_quant import apply_weight_quantization  # noqa: E402
from taccel.quantizer.quantize import quantize_tensor, dequantize_tensor  # noqa: E402

# Re-use the per-image comparator and the frozen-image helpers from
# their respective tools so this driver stays a thin loop.
from tools.compare_rtl_golden import (  # noqa: E402
    DEFAULT_RUNNER, ParityResult, compare_program, format_result,
)
from tools.benchmark_w8a16 import (  # noqa: E402
    DEFAULT_IMAGE_IDS, LOCAL_FROZEN_IMAGE_DIR, MODEL_NAME,
    _host_patch_embed, _load_local_images, _preprocess_image,
)


def _compile_program(model_name: str):
    """Load DeiT-tiny and W8A16-compile once for the whole batch."""
    from transformers import ViTForImageClassification

    print(f"Loading {model_name}...")
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()

    print("Compiling DeiT-tiny in W8A16 mode (cached on disk; skip on hit)...")
    program = load_or_compile(
        ModelConfig.deit_tiny(), state_dict, mode="w8a16", verbose=True,
    )
    print(f"  {program.insn_count} instructions, "
          f"{len(program.data):,} bytes data")
    return state_dict, program


# Per-worker globals populated via fork inheritance from the parent.
_W_STATE_DICT = None
_W_PROGRAM = None
_W_RUNNER = None
_W_MAX_CYCLES = None


def _init_worker(state_dict, program, runner, max_cycles):
    """ProcessPoolExecutor initializer — caches large objects in module
    globals so per-task pickling is just (img_id, img). With fork, the
    inheritance is COW and essentially free."""
    global _W_STATE_DICT, _W_PROGRAM, _W_RUNNER, _W_MAX_CYCLES
    _W_STATE_DICT = state_dict
    _W_PROGRAM = program
    _W_RUNNER = runner
    _W_MAX_CYCLES = max_cycles


def _run_one_image(args) -> ParityResult:
    """Worker entry point: patch-embed one image and run compare_program."""
    img_id, img = args
    pixel_values = _preprocess_image(img)
    patches = _host_patch_embed(pixel_values, _W_STATE_DICT)
    return compare_program(
        _W_PROGRAM, patches,
        runner=_W_RUNNER,
        max_cycles=_W_MAX_CYCLES,
        image_id=img_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W8A16 RTL-vs-golden bit-exact parity gate (20-image)."
    )
    parser.add_argument("--max-images", type=int, default=len(DEFAULT_IMAGE_IDS),
                        help="Number of frozen benchmark images (default 20).")
    parser.add_argument("--image-dir", default=LOCAL_FROZEN_IMAGE_DIR,
                        help="Directory holding the local frozen benchmark cache.")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER,
                        help="Path to the Verilator-built run_program binary.")
    parser.add_argument("--max-cycles", type=int, default=500_000_000,
                        help="RTL cycle budget per image (default 500M; DeiT-tiny W8A16 ~50-100M).")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop iterating on the first bit-exact divergence.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel Vtaccel_top runners (default 2; combined "
                             "with the default SIM_THREADS=4 build this saturates "
                             "the 8 physical cores on a 16-thread box). Set to 1 "
                             "for sequential execution; raise if Vtaccel_top was "
                             "built with SIM_THREADS=2 or fewer.")
    args = parser.parse_args(argv)

    state_dict, program = _compile_program(MODEL_NAME)
    image_ids = DEFAULT_IMAGE_IDS[: args.max_images]
    images = _load_local_images(image_ids, args.image_dir)

    workers = max(1, min(args.workers, len(images)))

    def _print_result(idx: int, r: ParityResult) -> None:
        diff = (
            f"{r.first_divergence_index}"
            if r.first_divergence_index is not None else "—"
        )
        print(f"  {idx:>3}  {r.image_id:>6}  "
              f"{'PASS' if r.passed else 'FAIL':>6}  "
              f"{r.rtl_cycles:>10}  {diff:>12}  "
              f"({r.rtl_argmax}, {r.golden_argmax})", flush=True)

    # Keyed by img_id so we can print in submission order at the end
    # regardless of which worker finished first.
    by_id: dict[int, ParityResult] = {}

    print()
    print(f"  Running RTL-vs-golden bit-exact gate over {len(images)} images "
          f"({workers} worker{'s' if workers != 1 else ''})...")
    print(f"  Note: each worker spawns a Vtaccel_top subprocess (~30-60 min "
          f"per image at SIM_THREADS=4).")
    print(f"  {'#':>3}  {'id':>6}  {'status':>6}  {'cycles':>10}  "
          f"{'first_diff':>12}  argmax(rtl, golden)")

    if workers == 1:
        # Sequential path: same code as the parallel worker but in-process,
        # so --stop-on-fail can short-circuit cleanly without the pool's
        # in-flight-cancel quirks.
        _init_worker(state_dict, program, args.runner, args.max_cycles)
        for idx, (img_id, img) in enumerate(images, 1):
            r = _run_one_image((img_id, img))
            by_id[img_id] = r
            _print_result(idx, r)
            if not r.passed and args.stop_on_fail:
                print(format_result(r))
                break
    else:
        # Parallel path: fork-based pool inherits the program + state_dict
        # via COW. With workers=2 and SIM_THREADS=4 the box runs two
        # Vtaccel_top processes across the 8 physical cores.
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(state_dict, program, args.runner, args.max_cycles),
        ) as ex:
            futures = {
                ex.submit(_run_one_image, (img_id, img)): (idx, img_id)
                for idx, (img_id, img) in enumerate(images, 1)
            }
            for fut in as_completed(futures):
                idx, img_id = futures[fut]
                r = fut.result()
                by_id[img_id] = r
                _print_result(idx, r)
                if not r.passed and args.stop_on_fail:
                    print(format_result(r))
                    # Cancel any not-yet-started tasks. Already-running
                    # Vtaccel_top subprocesses will finish their current
                    # work — there's no clean cross-process abort — but
                    # the gate has already failed and we'll exit nonzero.
                    for pending in futures:
                        pending.cancel()
                    break

    # Replay in original submission order for a deterministic summary.
    results = [by_id[img_id] for img_id, _ in images if img_id in by_id]

    n = len(results)
    n_pass = sum(1 for r in results if r.passed)
    print()
    print(f"  ─── W8A16 RTL-vs-golden bit-exact gate: {n_pass}/{n} PASS ───")
    if n_pass != n:
        print()
        print("  Failures per image:")
        for r in results:
            if r.passed:
                continue
            if r.first_divergence_index is None:
                # Runner did not halt — no logits to diverge with.
                print(f"   image {r.image_id}: rtl_status={r.rtl_status} "
                      f"(cycles={r.rtl_cycles}); no bit-exact comparison")
            else:
                print(f"   image {r.image_id}: "
                      f"logit[{r.first_divergence_index}] "
                      f"rtl=0x{r.rtl_logit_bits:04x} "
                      f"golden=0x{r.golden_logit_bits:04x}")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
