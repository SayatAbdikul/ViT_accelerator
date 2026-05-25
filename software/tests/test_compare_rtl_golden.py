"""W8A16 RTL-vs-golden bit-exact parity tests.

This is the load-bearing acceptance gate established by Phase 6 of the
W8A16 RTL fork plan. It asserts that the Verilator-built RTL simulator
and ``SimulatorW8A16`` produce **bit-exact** FP16 classifier logits on
the frozen benchmark images.

The single-image smoke (``test_compare_rtl_golden_smoke``) is what the
day-to-day pytest suite runs once Verilator and the DeiT-tiny weights
are present. The 20-image batch
(``test_compare_rtl_golden_batch_20_images``) is the full acceptance
gate; it is marked ``slow`` because Verilator simulation of a full
DeiT-tiny program takes several minutes per image, so the whole batch
is ~1–2 hours of wall time. Run it explicitly with
``pytest -m slow`` (or directly via the command below) when shipping a
phase-changing RTL diff.

Skip behavior:

* When ``verilator`` is not on PATH, ``conftest.py``'s ``collect_ignore``
  drops the whole module — fresh clones / CI jobs without Verilator
  installed get a clean run.
* When the DeiT-tiny weights or the local frozen image cache are
  missing, the ``assets_available`` fixture skips per-test.
* Bit-exact parity has no per-environment knob: a PASS means
  ``np.array_equal`` over the FP16 uint16 view. **Do not weaken** this
  gate to a tolerance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "software" / "tools"
RTL_VERILATOR_DIR = REPO_ROOT / "rtl" / "verilator"
RUN_PROGRAM_BINARY = RTL_VERILATOR_DIR / "build" / "run_program" / "Vtaccel_top"


# Module-scope autouse fixture: require a pre-built Verilator runner.
# We intentionally do NOT trigger the ~10-minute Verilator build from
# pytest: a CI-style invocation that ran `make -C rtl/verilator
# run_program` separately gets the test; a fresh clone without the
# pre-build cleanly skips. conftest.py already drops this file when
# `verilator` is missing on PATH; this fixture adds the "binary exists"
# precondition on top.
@pytest.fixture(scope="module", autouse=True)
def _ensure_run_program_built():
    if not RUN_PROGRAM_BINARY.exists():
        pytest.skip(
            f"RTL runner binary missing: {RUN_PROGRAM_BINARY}\n"
            f"  Build with:  make -C rtl/verilator run_program"
        )


def _import_compare_tools():
    """Lazy import so module-collection doesn't hit transformers/torch."""
    sys.path.insert(0, str(REPO_ROOT / "software"))
    from tools.compare_rtl_golden import compare_program, format_result  # noqa: E402
    from tools.benchmark_w8a16 import (  # noqa: E402
        DEFAULT_IMAGE_IDS, LOCAL_FROZEN_IMAGE_DIR, MODEL_NAME,
        _host_patch_embed, _load_local_images, _preprocess_image,
    )
    return {
        "compare_program": compare_program,
        "format_result": format_result,
        "DEFAULT_IMAGE_IDS": DEFAULT_IMAGE_IDS,
        "LOCAL_FROZEN_IMAGE_DIR": LOCAL_FROZEN_IMAGE_DIR,
        "MODEL_NAME": MODEL_NAME,
        "_host_patch_embed": _host_patch_embed,
        "_load_local_images": _load_local_images,
        "_preprocess_image": _preprocess_image,
    }


def _compile_deit_tiny_w8a16():
    """Compile DeiT-tiny in W8A16 mode; skip if weights unreachable."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import ViTForImageClassification
    from taccel.compiler.compiler import Compiler
    from taccel.model_config import ModelConfig

    try:
        model = ViTForImageClassification.from_pretrained(
            "facebook/deit-tiny-patch16-224"
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"DeiT-tiny weights unreachable: {exc}")
    model.eval()
    state_dict = model.state_dict()
    compiler = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a16")
    program = compiler.compile_w8a16(state_dict)
    return state_dict, program


def _patches_for_image(image_id: int, helpers: dict, state_dict) -> np.ndarray:
    """Load + preprocess + host-side patch-embed a single frozen image."""
    images = helpers["_load_local_images"]([image_id], helpers["LOCAL_FROZEN_IMAGE_DIR"])
    _, img = images[0]
    pixel_values = helpers["_preprocess_image"](img)
    return helpers["_host_patch_embed"](pixel_values, state_dict)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_compare_rtl_golden_smoke(assets_available):
    """One-image smoke: compile, run RTL, run golden, bit-exact logits."""
    helpers = _import_compare_tools()
    state_dict, program = _compile_deit_tiny_w8a16()

    image_id = helpers["DEFAULT_IMAGE_IDS"][0]
    patches = _patches_for_image(image_id, helpers, state_dict)

    result = helpers["compare_program"](
        program, patches,
        runner=RUN_PROGRAM_BINARY,
        max_cycles=500_000_000,
        image_id=image_id,
    )
    assert result.passed, (
        "W8A16 RTL-vs-golden parity gate failed (bit-exact contract).\n"
        + helpers["format_result"](result)
    )


@pytest.mark.integration
@pytest.mark.slow
def test_compare_rtl_golden_batch_20_images(assets_available):
    """Full 20-image bit-exact gate — the load-bearing W8A16 invariant."""
    helpers = _import_compare_tools()
    state_dict, program = _compile_deit_tiny_w8a16()

    image_ids = helpers["DEFAULT_IMAGE_IDS"][:20]
    failures = []
    for image_id in image_ids:
        patches = _patches_for_image(image_id, helpers, state_dict)
        result = helpers["compare_program"](
            program, patches,
            runner=RUN_PROGRAM_BINARY,
            max_cycles=500_000_000,
            image_id=image_id,
        )
        if not result.passed:
            failures.append(helpers["format_result"](result))

    assert not failures, (
        "W8A16 RTL-vs-golden 20-image bit-exact gate failed.\n\n"
        + "\n\n".join(failures)
    )
