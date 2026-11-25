"""Long-run durability harness (N34 preparation; zero training).

N34 is the long-run endurance stage. Before any training card, the run loop must be able to
survive many sessions, report a verifiable audit chain, resume after a simulated crash, and
refuse to touch the sealed ``check``/``final`` splits.

This module is a small, deterministic harness over :class:`taiji.session_store.SessionStore`:

* every session loads the subject record, mutates it through a caller-supplied function, saves
  it atomically, and appends one audit entry whose hash chains to the previous entry;
* ``crash_before_commit`` simulates a process death between mutation and save, so the next
  ``resume()`` continues from the last committed session;
* ``SealedSplitGuard`` blocks and logs any attempt to read ``check``/``final``;
* no training, no confirmation seeds, no wall-clock dependence (deterministic given the calls).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from taiji.session_store import SessionRecord, SessionStore

SCHEMA = "taiji-long-run-v1"
SEALED_SPLITS = ("check", "final")


class CrashInjected(RuntimeError):
    """Raised to simulate a process death before the session commit."""


class SealViolation(RuntimeError):
    """Raised when a caller tries to read a sealed split."""


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class SealedSplitGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def require(self, split: str) -> None:
        if split in SEALED_SPLITS:
            self.attempts.append(split)
            raise SealViolation(f"split {split!r} is sealed for the long-run harness")

    def attempts_count(self) -> int:
        return len(self.attempts)


class LongRunHarness:
    """One subject, many sessions, chained audit, crash/resume, sealed-split guard."""

    def __init__(self, root: str | Path, subject_id: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.subject_id = subject_id
        self.store = SessionStore(self.root)
        self.audit_path = self.root / f"{subject_id}.audit.jsonl"
        self.guard = SealedSplitGuard()
        self.chain: list[str] = []
        self.session_index = 0
        self._replay_chain()

    # ------------------------------------------------------------------ audit
    def _iter_audit(self) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        out = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def _replay_chain(self) -> None:
        prev = "0" * 64
        for e in self._iter_audit():
            digest = e["record_digest"]
            expected = hashlib.sha256((prev + digest).encode("utf-8")).hexdigest()
            if e.get("chain") != expected or e.get("prev") != prev:
                raise ValueError(f"audit chain broken at session {e.get('session')}")
            prev = expected
        self.chain = [e["chain"] for e in self._iter_audit()]
        self.session_index = len(self.chain)

    def verify_chain(self) -> bool:
        self._replay_chain()
        return True

    # ---------------------------------------------------------------- sessions
    def _load(self) -> SessionRecord:
        if self.session_index == 0:
            return SessionRecord(subject_id=self.subject_id)
        return self.store.load(self.subject_id)

    def run_session(
        self,
        session_fn: Callable[[SessionRecord, int], None],
        *,
        crash_before_commit: bool = False,
    ) -> dict[str, Any]:
        record = self._load()
        session_fn(record, self.session_index)
        if crash_before_commit:
            raise CrashInjected(f"crash before commit at session {self.session_index}")
        self.store.save(record)
        prev = self.chain[-1] if self.chain else "0" * 64
        digest = record.digest()
        chain = hashlib.sha256((prev + digest).encode("utf-8")).hexdigest()
        entry = {
            "schema": SCHEMA,
            "session": self.session_index,
            "subject_id": self.subject_id,
            "record_digest": digest,
            "prev": prev,
            "chain": chain,
        }
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(_canonical(entry) + "\n")
            fh.flush()
        self.chain.append(chain)
        self.session_index += 1
        return entry

    def resume(self) -> tuple[SessionRecord, int]:
        self._replay_chain()
        record = (
            self.store.load(self.subject_id)
            if self.session_index > 0
            else SessionRecord(subject_id=self.subject_id)
        )
        return record, self.session_index

    def read_split(self, split: str) -> None:
        """Sealed splits must never be read by long-run code; val/train are allowed."""
        if split not in SEALED_SPLITS:
            raise NotImplementedError(
                "long-run harness does not expose data splits; guard test only"
            )
        self.guard.require(split)
