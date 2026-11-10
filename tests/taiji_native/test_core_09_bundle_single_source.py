"""The 09.8 bundle identity must have exactly one written source.

``taiji/core_09_bundle.py`` owns the sealed sha256; until 2026-11-10 six scripts
each carried their own copy.  Two copies drift apart the first time one of them
is edited, and the drift is silent: the readings stay sealed against a value no
runner can reproduce.  The identity therefore lives in one module and this test
fails as soon as a second copy reappears in shipped code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from taiji.core_09_bundle import EXPECTED_SHA256

ROOT = Path(__file__).resolve().parents[2]
OWNER = "taiji/core_09_bundle.py"
SCAN_ROOTS = ("taiji", "scripts")


def _shipped_python_files() -> list[Path]:
    """Git-tracked files only: an untracked local draft is not shipped code."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *SCAN_ROOTS],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [ROOT / name for name in listed.split("\0") if name.endswith(".py")]


def test_the_sealed_sha256_is_spelled_out_in_exactly_one_shipped_file():
    offenders = []
    for path in _shipped_python_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative == OWNER:
            continue
        if EXPECTED_SHA256 in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(relative)
    assert offenders == []
