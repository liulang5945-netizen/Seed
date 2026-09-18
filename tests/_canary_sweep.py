"""Sweep of test-made intermediates under ``output/manual-r5-canary/`` (DEBT-I7, second instance).

Artifact-store tests build their roots as ``<repo>/output/manual-r5-canary/sNN-<kind>-<pid>`` and
never remove them, so the directory the *product* also uses keeps growing: measured across four
full-suite runs on 2026-09-18 it went from 10 entries / 335 MB to 14 / 500 MB.

Because every residue name embeds the pid of the process that wrote it, a session can delete exactly
its own files and nothing else.  That is why ``sweep`` takes the pid and the directory as arguments
instead of hard-coding them: the contract test runs it against a scratch directory.

Nothing here touches ``output/_archived_manual-r5-canary_20260917/`` (the 1052 entries archived on
09-17) or any entry this process did not create.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

#: ``sNN-<kind>-<pid>`` with an optional ``.pt``; the pid is the only reliable ownership marker.
_RESIDUE = re.compile(r"-(\d+)(\.pt)?$")


def sweep(directory: Path, pid: int) -> list[str]:
    """Remove residue written by ``pid`` and return the removed names, sorted.

    A name that does not end in ``-<pid>`` (or ``-<pid>.pt``) is left alone, so pre-existing files
    -- ``README.md``, ``native-canary.pt`` -- and another session's in-flight files survive.
    """

    if not directory.is_dir():
        return []
    removed: list[str] = []
    for entry in sorted(directory.iterdir()):
        match = _RESIDUE.search(entry.name)
        if match is None or int(match.group(1)) != pid:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed.append(entry.name)
    return removed
