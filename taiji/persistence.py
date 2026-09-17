"""Crash-safe local file writes for Taiji checkpoints.

Taiji is a self-contained cognitive substrate: it must not import the Seed
runtime, the seed_platform governance layer, Legacy NeuroPlex, or HuggingFace
transformers (enforced by ``tests/taiji_native/test_architecture_contract.py``
and ``test_naming_boundary_contract.py``).  Checkpoint saving therefore keeps
its own atomic-write helper here instead of reaching into ``seed.persistence``.

``atomic_save`` is the ``torch.save`` equivalent of the write-then-``os.replace``
pattern already used by ``taiji/structural_artifact_store.py``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def atomic_save(payload: Mapping[str, Any], path: Path | str) -> Path:
    """``torch.save`` with crash-safe replace semantics.

    The temporary file is created in the destination directory so the final
    ``os.replace`` is atomic on the same filesystem.  A successful save sweeps
    stale half-written temporals from a previously killed process, which could
    not run its own cleanup branch.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(target.name + ".*.tmp"):
        try:
            stale.unlink()
        except OSError as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("atomic_save stale cleanup skipped (non-fatal): %s", exc)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target
