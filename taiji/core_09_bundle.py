"""Single source for the 09.8 subject-bundle identity.

The bundle ``output/taiji_core_09/subject_bundle.pt`` is ignored by ``*.pt`` and
is therefore absent from every clean checkout.  The 2026-11-10 ruling (ledger
D-20261110-08) classified it as ``regenerable_exact``: the declared construction
path reproduces it byte for byte, so it is rebuilt on demand instead of being
committed.  The entry point is ``scripts/training/materialize_taiji_core_09_bundle.py``.

Every consumer that compares a bundle against "the" expected identity imports
the constants below instead of carrying its own copy: two copies of the same
sha256 drift apart silently the first time one of them is edited.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

BUNDLE_RELATIVE_PATH = "output/taiji_core_09/subject_bundle.pt"
EXPECTED_SHA256 = "ce4c62685d55bb518d234e7407e19b05257d1d761a0430432f7d0f09641321cf"
EXPECTED_BYTES = 77609636

#: Set by ``verify_taiji_bundle_rebuild_determinism`` when it feeds a deliberately
#: non-sealed rebuild through a consumer.  In this mode the provenance checks
#: (sha256, byte count, candidate id) are still measured and reported but they do
#: not abort the run; the comparison then excludes them.  D-20261111-03.
READOUT_MODE_ENV = "TAIJI_BUNDLE_READOUT_MODE"


def readout_mode() -> bool:
    """True when a consumer is being replayed against a non-sealed rebuild."""
    value = os.environ.get(READOUT_MODE_ENV, "").strip().lower()
    return value in {"1", "true", "yes"}


def bundle_path(default: str = BUNDLE_RELATIVE_PATH) -> str:
    """The bundle a consumer should open; the determinism gate overrides it."""
    return os.environ.get("TAIJI_BUNDLE_PATH", default)


def sha256_of_file(path: Path | str) -> str:
    """Stream the file so a 74 MiB bundle does not have to sit in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mismatch(path: Path | str) -> str | None:
    """``None`` when the file matches the sealed identity, else the reason it does not."""
    candidate = Path(path)
    if not candidate.is_file():
        return f"missing bundle: {candidate}"
    size = candidate.stat().st_size
    if size != EXPECTED_BYTES:
        return f"bytes {size} != {EXPECTED_BYTES}"
    actual = sha256_of_file(candidate)
    if actual != EXPECTED_SHA256:
        return f"sha256 {actual} != {EXPECTED_SHA256}"
    return None
