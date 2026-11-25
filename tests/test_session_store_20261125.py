"""Unit tests for the N33.2 cross-session store (`taiji/session_store.py`).

Zero-training: round-trip bit-exact, cross-process round-trip, pending/experience persistence,
integrity failure on corruption, and no cross-subject contamination.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import pytest
import torch

from taiji.session_store import SCHEMA, SessionRecord, SessionStore


def _record(subject: str = "subj-a") -> SessionRecord:
    r = SessionRecord(subject_id=subject)
    r.capsule = {"h": torch.tensor([1.5, -2.0, 0.25]), "w": torch.zeros(2, 3)}
    r.add_pending({"id": "task-1", "kind": "unfinished", "state": "waiting"})
    r.append_experience("task_started", {"task": "task-1"}, source="owner")
    return r


def test_roundtrip_bit_exact(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    r = _record()
    digest = r.digest()
    store.save(r)
    back = store.load("subj-a")
    assert back.digest() == digest
    assert torch.equal(back.capsule["h"], r.capsule["h"])
    assert back.pending == r.pending
    assert back.experience == r.experience
    assert back.clock == r.clock


def test_pending_resolution_persists(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    r = _record()
    r.resolve_pending("task-1", "done")
    store.save(r)
    back = store.load("subj-a")
    assert back.pending == []
    assert any(e["kind"] == "resolved_pending" for e in back.experience)


def test_corrupted_store_raises(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.save(_record())
    path = tmp_path / "subj-a.session"
    data = torch.load(path, weights_only=False)
    data["metadata"]["clock"] = 999  # tamper
    torch.save(data, path)
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load("subj-a")


def test_no_cross_subject_contamination(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.save(_record("subj-a"))
    r2 = SessionRecord(subject_id="subj-b")
    r2.capsule = {"h": torch.tensor([9.0])}
    store.save(r2)
    a = store.load("subj-a")
    b = store.load("subj-b")
    assert torch.equal(a.capsule["h"], torch.tensor([1.5, -2.0, 0.25]))
    assert torch.equal(b.capsule["h"], torch.tensor([9.0]))
    assert store.subjects() == ["subj-a", "subj-b"]


def test_cross_process_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    digest = _record().digest()
    store.save(_record())
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"E:\\Seed")
        from taiji.session_store import SessionStore
        store = SessionStore(r"{tmp_path}")
        rec = store.load("subj-a")
        print(rec.digest())
        """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == digest


def test_invalid_ids_rejected(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(SessionRecord(subject_id="../escape"))
    r = SessionRecord(subject_id="ok")
    with pytest.raises(ValueError):
        r.add_pending({"id": "", "kind": "x"})
    with pytest.raises(KeyError):
        r.resolve_pending("missing", "done")


def test_schema_constant() -> None:
    assert SCHEMA == "taiji-session-store-v1"
