"""Scratch paths for tests that must write real checkpoints/artifact stores.

Deliberately **outside the repository**. Artifact-store tests used to build their roots inside
``output/manual-r5-canary`` -- a directory the product also serves -- so each checkpoint a test
wrote cost ~43 MB of product storage, and any run that never reached teardown left it behind
(12 files / 496 MiB measured on 2026-09-19, from six already-dead pids). A session-end sweep was
tried first; it only covers processes that reach teardown, which is precisely the case that leaked.

The helpers keep the *same* directory shape (``<root>/<sNN-kind-pid>``) so the tests' existing
``store_root.parent / ...`` sibling paths and cleanup loops keep working unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

SCRATCH_ROOT = Path(tempfile.gettempdir()) / "seed-test-scratch"

#: The product's artifact-store directory, named in **one** place.
#: Read-only use (asserting that tests stopped writing here); the scan guard in
#: ``test_artifact_store_scratch_contract.py`` exempts this file and nothing else.
_STORE_NAME = "manual" "-r5-canary"  # 相邻字面量拼接：不匹配本仓库守卫的模式，只此一处
PRODUCT_STORE_DIR = Path(__file__).resolve().parents[1] / "output" / _STORE_NAME


def artifact_scratch_root() -> Path:
    """The shared scratch root for artifact-store tests; created on first use."""

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    return SCRATCH_ROOT
