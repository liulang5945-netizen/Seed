"""Tests for the N34 long-run durability harness (zero training)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import pytest
import torch

from taiji.long_run_harness import (
    CrashInjected,
    LongRunHarness,
    SealedSplitGuard,
    SealViolation,
)


def _step(record, i: int) -> None:
    record.add_pending({"id": f"task-{i}", "kind": "unfinished"})
    record.append_experience("session", {"i": i})
    record.capsule["h"] = torch.tensor([float(i), 1.0])


def test_multi_session_continuity(tmp_path: Path) -> None:
    h = LongRunHarness(tmp_path, "subj")
    for _i in range(5):
        h.run_session(_step)
    record, nxt = h.resume()
    assert nxt == 5
    assert len(record.pending) == 5
    assert len(record.experience) == 5
    assert h.verify_chain()
    assert len(h.chain) == 5
    assert h.chain[-1] == h._iter_audit()[-1]["chain"]


def test_crash_and_resume(tmp_path: Path) -> None:
    h = LongRunHarness(tmp_path, "subj")
    for _ in range(3):
        h.run_session(_step)
    before, idx = h.resume()
    with pytest.raises(CrashInjected):
        h.run_session(_step, crash_before_commit=True)
    after, idx2 = h.resume()
    assert idx == idx2 == 3
    assert before.digest() == after.digest()
    assert len(after.pending) == 3
    h.run_session(_step)  # session 3 now commits
    record, nxt = h.resume()
    assert nxt == 4 and len(record.pending) == 4


def test_audit_tamper_detected(tmp_path: Path) -> None:
    h = LongRunHarness(tmp_path, "subj")
    h.run_session(_step)
    h.run_session(_step)
    lines = h.audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    import json

    entry = json.loads(lines[1])
    entry["record_digest"] = "0" * 64
    lines[1] = json.dumps(entry)
    h.audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit chain broken"):
        h.verify_chain()


def test_sealed_split_guard() -> None:
    g = SealedSplitGuard()
    g.require("val")
    with pytest.raises(SealViolation):
        g.require("check")
    with pytest.raises(SealViolation):
        g.require("final")
    assert g.attempts_count() == 2


def test_harness_sealed_read_attempt(tmp_path: Path) -> None:
    h = LongRunHarness(tmp_path, "subj")
    with pytest.raises(SealViolation):
        h.read_split("check")
    assert h.guard.attempts_count() == 1


def test_deterministic_chain(tmp_path: Path) -> None:
    a = LongRunHarness(tmp_path / "a", "subj")
    b = LongRunHarness(tmp_path / "b", "subj")
    for _ in range(4):
        a.run_session(_step)
        b.run_session(_step)
    assert a.chain == b.chain
