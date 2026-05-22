"""Shared pytest fixtures and asset-availability gates.

Two responsibilities:

1. **Session fixtures** for repeated setup (repo_root, assembler) so individual
   test files stop rolling their own.

2. **Asset-availability gate**: ``software/pytorch_model.bin`` (5.7 MB DeiT-tiny
   weights) and ``software/images/`` (gitignored COCO subset) are not in the
   repo. Three tests assert preset/dataset sizes against the local image set,
   and one test module imports from ``images.download_imagenet_class``. When
   the assets are absent we cleanly skip those tests instead of failing them,
   so a fresh clone runs the suite green.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_DIR = REPO_ROOT / "software"
IMAGES_DIR = SOFTWARE_DIR / "images"
WEIGHTS_PATH = SOFTWARE_DIR / "pytorch_model.bin"
IMAGES_PACKAGE_INIT = IMAGES_DIR / "__init__.py"


# Tests whose assertions only make sense when the local image dataset is
# present (they check len(preset["eval_image_ids"]) against a fixed sample
# count, or call discover_cats_dogs_samples() and expect 200 entries).
_ASSET_DEPENDENT_TEST_NAMES = {
    "test_discover_cats_dogs_samples_is_stable_and_labeled",
    "test_diagnostic_preset_cats_dogs_uses_all_local_samples",
    "test_diagnostic_preset_imagenet_class0_uses_all_local_samples",
}


def _images_available() -> bool:
    return IMAGES_DIR.exists()


def _weights_available() -> bool:
    return WEIGHTS_PATH.exists()


# ─── Collection-time gate ─────────────────────────────────────────────────────
# test_download_imagenet_class.py does `from images.download_imagenet_class
# import save_class_images` at module scope, which ImportErrors when the
# gitignored images/ package is missing. Skipping the module at collection
# time keeps `pytest software/tests` clean on a fresh clone.
collect_ignore = []
if not IMAGES_PACKAGE_INIT.exists():
    collect_ignore.append("test_download_imagenet_class.py")


def pytest_collection_modifyitems(config, items):
    """Auto-mark and auto-skip tests that need the gitignored image dataset.

    Two ways a test is treated as asset-dependent:
      - Its name is in _ASSET_DEPENDENT_TEST_NAMES above (manual allowlist).
      - It carries the ``requires_assets`` marker (new tests should prefer
        this; the allowlist is here only because the existing failing tests
        predate the marker).
    """
    if _images_available() and _weights_available():
        return
    skip_marker = pytest.mark.skip(
        reason="Asset-dependent test: needs software/images/ "
               "and/or software/pytorch_model.bin (both gitignored)"
    )
    for item in items:
        if item.name in _ASSET_DEPENDENT_TEST_NAMES or "requires_assets" in item.keywords:
            item.add_marker(skip_marker)


# ─── Session fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repo root (parent of ``software/``)."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def software_dir() -> Path:
    """Absolute path to ``software/``."""
    return SOFTWARE_DIR


@pytest.fixture(scope="session")
def assembler():
    """Cached ``Assembler()`` instance — stateless, safe to share across tests."""
    from taccel.assembler.assembler import Assembler
    return Assembler()


@pytest.fixture
def assets_available():
    """Per-test gate: skip if model weights or image dataset are missing.

    New tests that need real data should depend on this fixture rather than
    being added to the allowlist above. Equivalent to marking the test with
    ``@pytest.mark.requires_assets`` but available as an explicit dependency
    for fixtures that need to load assets in their setup.
    """
    if not (_images_available() and _weights_available()):
        pytest.skip(
            "Requires software/pytorch_model.bin and software/images/ "
            "(both gitignored)"
        )
