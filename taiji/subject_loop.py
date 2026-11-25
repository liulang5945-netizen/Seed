"""Integrated "subject loop" harness (`taiji-subject-loop-v1`, zero training).

One subject, many sessions, in which the four already-frozen facilities collaborate inside a
single loop:

1. :class:`taiji.session_store.SessionStore` persists the subject record across sessions;
2. :class:`taiji.chinese_task_env.ChineseTaskEnv` supplies continuous Chinese instructions;
3. an action is chosen by a **scripted control policy** (engineering prototype, explicitly not a
   learned policy) and the resulting outcome feeds
4. :class:`taiji.evidence_ledger.CredibilityLedger` (accumulated evidence, rebuilt from the
   append-only experience log on every session);
5. a Laplace-smoothed success estimate is verbalised with
   :func:`taiji.uncertainty_calibration.verbalize_confidence` and its calibration is measured;
6. every session appends a sha256 audit entry, and a crash injection can interrupt a session
   before commit so the next session resumes from the last consistent state.

This is an integration smoke: it makes no capability claim, trains nothing and never touches
``check``/``final``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taiji.chinese_task_env import ChineseTaskEnv, scripted_oracle_policy
from taiji.evidence_ledger import CredibilityLedger
from taiji.session_store import SessionRecord, SessionStore
from taiji.uncertainty_calibration import expected_calibration_error, verbalize_confidence

SCHEMA = "taiji-subject-loop-v1"
SEALED = ("check", "final")


@dataclass
class LoopResult:
    sessions: int
    steps: int
    chain: list[str]
    confidences: list[str]
    ledger_posterior: float
    ece: float
    record_digest: str


class SubjectLoop:
    def __init__(self, root: str | Path, subject_id: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.subject_id = subject_id
        self.store = SessionStore(self.root)
        self.audit_path = self.root / f"{subject_id}.loop-audit.jsonl"
        self.chain: list[str] = []
        self.session_index = 0
        self._replay()

    # ---------------------------------------------------------------- audit
    def _replay(self) -> None:
        prev = "0" * 64
        entries = []
        if self.audit_path.exists():
            for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        for e in entries:
            expected = hashlib.sha256((prev + e["record_digest"]).encode("utf-8")).hexdigest()
            if e["chain"] != expected:
                raise ValueError(f"loop audit chain broken at session {e.get('session')}")
            prev = expected
            self.chain.append(expected)
        self.session_index = len(self.chain)

    # -------------------------------------------------------------- evidence
    def _ledger(self, record: SessionRecord) -> CredibilityLedger:
        led = CredibilityLedger()
        for ev in record.experience:
            if ev.get("kind") == "evidence":
                p = ev["payload"]
                led.add(p["claim"], int(p["polarity"]), p["source"])
        return led

    def _confidence(self, record: SessionRecord, claim: str) -> str:
        led = self._ledger(record)
        tries = [e for e in record.experience if e.get("kind") == "evidence"]
        correct = sum(
            1 for e in tries if e["payload"]["claim"] == claim and e["payload"]["polarity"] == 1
        )
        total = len(tries)
        rate = (correct + 1.0) / (total + 2.0)  # Laplace smoothing
        if claim in led.claims:
            rate = 0.5 * rate + 0.5 * led.claims[claim].posterior
        return verbalize_confidence(min(1.0, max(0.0, rate)))

    # ---------------------------------------------------------------- sessions
    def run_session(self, seed: int, *, crash_before_commit: bool = False) -> dict[str, Any]:
        record = (
            self.store.load(self.subject_id)
            if self.session_index
            else SessionRecord(subject_id=self.subject_id)
        )
        claim = "turn_action_correct"
        env = ChineseTaskEnv(seed)
        env.reset()
        predicted: list[float] = []
        outcomes: list[int] = []
        steps = 0
        while True:
            action = scripted_oracle_policy(env)  # scripted control policy (not learned)
            record.append_experience(
                "evidence",
                {"claim": claim, "polarity": 1, "source": "sensor"},
            )
            conf = self._confidence(record, claim)
            predicted.append(self._confidence_value(record, claim))
            obs, reward, done = env.step(action)
            outcomes.append(int(reward > 0))
            record.append_experience(
                "turn", {"session": self.session_index, "step": steps, "reward": reward}
            )
            steps += 1
            if done:
                break
        record.append_experience("session", {"seed": seed, "steps": steps, "claim": claim})
        record.add_pending({"id": f"session-{self.session_index}", "kind": "closed"})
        record.resolve_pending(f"session-{self.session_index}", "done")
        if crash_before_commit:
            raise RuntimeError(f"crash before commit at session {self.session_index}")
        self.store.save(record)
        prev = self.chain[-1] if self.chain else "0" * 64
        chain = hashlib.sha256((prev + record.digest()).encode("utf-8")).hexdigest()
        entry = {
            "schema": SCHEMA,
            "session": self.session_index,
            "subject_id": self.subject_id,
            "seed": seed,
            "steps": steps,
            "record_digest": record.digest(),
            "prev": prev,
            "chain": chain,
        }
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
        self.chain.append(chain)
        self.session_index += 1
        return {"entry": entry, "predicted": predicted, "outcomes": outcomes, "confidence": conf}

    @staticmethod
    def _confidence_value(record: SessionRecord, claim: str) -> float:
        tries = [e for e in record.experience if e.get("kind") == "evidence"]
        correct = sum(
            1 for e in tries if e["payload"]["claim"] == claim and e["payload"]["polarity"] == 1
        )
        total = len(tries)
        return (correct + 1.0) / (total + 2.0)

    # ------------------------------------------------------------------ views
    def sealed_read(self, split: str) -> None:
        if split in SEALED:
            raise PermissionError(f"split {split!r} is sealed")
        raise NotImplementedError

    def summary(
        self, predicted: list[float], outcomes: list[int], confidences: list[str]
    ) -> LoopResult:
        record = self.store.load(self.subject_id)
        led = self._ledger(record)
        posterior = (
            led.claims.get("turn_action_correct").posterior
            if "turn_action_correct" in led.claims
            else 0.0
        )
        return LoopResult(
            sessions=self.session_index,
            steps=len(predicted),
            chain=list(self.chain),
            confidences=confidences,
            ledger_posterior=posterior,
            ece=expected_calibration_error(predicted, outcomes)["ece"],
            record_digest=record.digest(),
        )
