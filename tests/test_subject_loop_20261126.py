"""Tests for the integrated subject-loop harness (zero training)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import pytest

from taiji.subject_loop import SubjectLoop


def _run(loop: SubjectLoop, n: int, start: int = 5000) -> None:
    for i in range(n):
        loop.run_session(start + i)


def test_multi_session_loop(tmp_path: Path) -> None:
    loop = SubjectLoop(tmp_path, "subject-0")
    _run(loop, 3)
    record = loop.store.load("subject-0")
    assert loop.session_index == 3
    assert len(loop.chain) == 3
    assert loop.chain == [
        e["chain"]
        for e in [
            __import__("json").loads(l)
            for l in loop.audit_path.read_text(encoding="utf-8").splitlines()
        ]
    ]
    assert record.pending == []
    evidences = [e for e in record.experience if e["kind"] == "evidence"]
    assert len(evidences) >= 3
    out = loop.summary(
        loop._confidence_value(record, "turn_action_correct") and [0.5], [1], ["不确定"]
    )
    assert 0.0 <= out.ece <= 1.0
    assert out.ledger_posterior >= 0.5


def test_crash_and_resume(tmp_path: Path) -> None:
    loop = SubjectLoop(tmp_path, "subject-0")
    _run(loop, 2)
    before = loop.store.load("subject-0").digest()
    with pytest.raises(RuntimeError):
        loop.run_session(5099, crash_before_commit=True)
    after = loop.store.load("subject-0").digest()
    assert before == after
    assert loop.session_index == 2
    loop.run_session(5099)
    assert loop.session_index == 3


def test_ledger_rebuilt_across_sessions(tmp_path: Path) -> None:
    loop = SubjectLoop(tmp_path, "subject-0")
    _run(loop, 2)
    record = loop.store.load("subject-0")
    ledger = loop._ledger(record)
    evidences = [e for e in record.experience if e["kind"] == "evidence"]
    assert len(ledger.evidence) == len(evidences)
    assert ledger.claims["turn_action_correct"].status == "accepted"


def test_deterministic_loops(tmp_path: Path) -> None:
    a = SubjectLoop(tmp_path / "a", "subject-0")
    b = SubjectLoop(tmp_path / "b", "subject-0")
    _run(a, 3)
    _run(b, 3)
    assert a.chain == b.chain
    assert a.store.load("subject-0").digest() == b.store.load("subject-0").digest()


def test_sealed_read_blocked(tmp_path: Path) -> None:
    loop = SubjectLoop(tmp_path, "subject-0")
    for split in ("check", "final"):
        with pytest.raises(PermissionError):
            loop.sealed_read(split)


def test_confidence_verbalisation(tmp_path: Path) -> None:
    loop = SubjectLoop(tmp_path, "subject-0")
    _run(loop, 1)
    record = loop.store.load("subject-0")
    conf = loop._confidence(record, "turn_action_correct")
    assert conf in {"我很确定", "比较确定", "不太确定", "不确定"}
