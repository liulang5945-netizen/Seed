"""External Action Lifecycle & 4-Boundary Recovery Ledger (`taiji-action-lifecycle-v1`).

Implements `TAIJI_NATIVE_ARCHITECTURE_V1 §4.7` & `GOV-ACT`:
  * Fail-closed 9-state external action lifecycle distinguishing internal preparation
    from external dispatch, unknown timeout, observed receipt, exactly-once settlement,
    and compensation failure;
  * Idempotency keys prevent duplicate dispatches and duplicate outcome settlement;
  * `record_model_rollback` preserves all dispatched/observed/settled external facts
    (zero pseudo-rollback of world history);
  * 4-boundary crash recovery across `prepared`, `dispatched`, `observed`, and `settled`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "taiji-action-lifecycle-v1"

ACTION_STATES: tuple[str, ...] = (
    "proposed",
    "rejected",
    "prepared",
    "dispatched",
    "unknown",
    "observed",
    "settled",
    "compensation_failed",
    "unresolved",
)

ALLOWED_ACTION_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"prepared", "rejected"}),
    "prepared": frozenset({"dispatched", "rejected"}),
    "dispatched": frozenset({"observed", "unknown"}),
    "unknown": frozenset({"observed", "unresolved"}),
    "observed": frozenset({"settled"}),
    "settled": frozenset({"compensation_failed"}),
    "rejected": frozenset(),
    "unresolved": frozenset(),
    "compensation_failed": frozenset(),
}


class ActionLifecycleError(RuntimeError):
    """Raised on illegal external action state transitions."""


class DuplicateSettlementError(RuntimeError):
    """Raised when attempting to settle an already-settled action twice."""


@dataclass
class ActionRecord:
    action_id: str
    idempotency_key: str
    intent_digest: str
    goal_revision: int
    belief_revision: int
    state: str = "proposed"
    dispatch_attempts: int = 0
    receipt_id: str | None = None
    outcome_payload: dict[str, Any] | None = None
    settled_once: bool = False
    compensation_of: str | None = None
    reject_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRecord:
        return cls(**data)


class ActionLifecycleLedger:
    """Fail-closed durable ledger for §4.7 external action execution & recovery."""

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.records: dict[str, ActionRecord] = {}
        self.by_idempotency: dict[str, str] = {}
        self.rollback_notes: list[dict[str, Any]] = []

    def _persist(self) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_name(f".{self.ledger_path.name}.{os.getpid()}.tmp")
        payload = {
            "schema": SCHEMA,
            "records": {k: v.to_dict() for k, v in self.records.items()},
            "by_idempotency": dict(self.by_idempotency),
            "rollback_notes": list(self.rollback_notes),
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    @classmethod
    def open(cls, ledger_path: str | Path) -> ActionLifecycleLedger:
        p = Path(ledger_path)
        ledger = cls(p)
        if not p.exists():
            return ledger
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ActionLifecycleError(f"Invalid ledger schema: {data.get('schema')}")
        for aid, rdict in data.get("records", {}).items():
            ledger.records[aid] = ActionRecord.from_dict(rdict)
        ledger.by_idempotency = dict(data.get("by_idempotency", {}))
        ledger.rollback_notes = list(data.get("rollback_notes", []))
        return ledger

    def _transition(self, rec: ActionRecord, target: str, reason: str) -> None:
        if target not in ACTION_STATES:
            raise ActionLifecycleError(f"Unknown state {target!r}")
        if target not in ALLOWED_ACTION_TRANSITIONS[rec.state]:
            raise ActionLifecycleError(
                f"Illegal action transition {rec.state!r} -> {target!r} for {rec.action_id}"
            )
        rec.history.append(
            {
                "ts_utc": datetime.now(UTC).isoformat(),
                "from": rec.state,
                "to": target,
                "reason": reason,
            }
        )
        rec.state = target

    def propose(
        self,
        action_id: str,
        *,
        idempotency_key: str,
        intent_digest: str,
        goal_revision: int,
        belief_revision: int,
        compensation_of: str | None = None,
    ) -> ActionRecord:
        existing_id = self.by_idempotency.get(idempotency_key)
        if existing_id is not None:
            return self.records[existing_id]
        rec = ActionRecord(
            action_id=action_id,
            idempotency_key=idempotency_key,
            intent_digest=intent_digest,
            goal_revision=int(goal_revision),
            belief_revision=int(belief_revision),
            compensation_of=compensation_of,
            history=[
                {
                    "ts_utc": datetime.now(UTC).isoformat(),
                    "from": None,
                    "to": "proposed",
                    "reason": "proposed",
                }
            ],
        )
        self.records[action_id] = rec
        self.by_idempotency[idempotency_key] = action_id
        self._persist()
        return rec

    def prepare(self, action_id: str, reason: str = "authorized") -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "prepared", reason)
        self._persist()
        return rec

    def reject(self, action_id: str, reason: str) -> ActionRecord:
        rec = self.records[action_id]
        rec.reject_reason = reason
        self._transition(rec, "rejected", reason)
        self._persist()
        return rec

    def dispatch(self, action_id: str, reason: str = "dispatched") -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "dispatched", reason)
        rec.dispatch_attempts += 1
        self._persist()
        return rec

    def mark_unknown(self, action_id: str, reason: str = "timeout_or_disconnect") -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "unknown", reason)
        self._persist()
        return rec

    def mark_unresolved(self, action_id: str, reason: str = "reconcile_failed") -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "unresolved", reason)
        self._persist()
        return rec

    def observe(
        self, action_id: str, *, receipt_id: str, outcome_payload: dict[str, Any]
    ) -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "observed", f"receipt:{receipt_id}")
        rec.receipt_id = receipt_id
        rec.outcome_payload = dict(outcome_payload)
        self._persist()
        return rec

    def settle(self, action_id: str, reason: str = "settled") -> ActionRecord:
        rec = self.records[action_id]
        if rec.settled_once:
            raise DuplicateSettlementError(f"Action {action_id} already settled")
        self._transition(rec, "settled", reason)
        rec.settled_once = True
        self._persist()
        return rec

    def mark_compensation_failed(self, action_id: str, reason: str) -> ActionRecord:
        rec = self.records[action_id]
        self._transition(rec, "compensation_failed", reason)
        self._persist()
        return rec

    def recover_in_flight(self) -> dict[str, list[str]]:
        """4-boundary recovery scan after restart/crash."""
        summary: dict[str, list[str]] = {
            "prepared_ready_or_cancel": [],
            "dispatched_moved_to_unknown": [],
            "observed_ready_to_settle": [],
            "already_settled": [],
        }
        for aid, rec in self.records.items():
            if rec.state == "prepared":
                summary["prepared_ready_or_cancel"].append(aid)
            elif rec.state == "dispatched":
                self._transition(rec, "unknown", "crash_recovery_unknown_outcome")
                summary["dispatched_moved_to_unknown"].append(aid)
            elif rec.state == "observed":
                summary["observed_ready_to_settle"].append(aid)
            elif rec.state == "settled":
                summary["already_settled"].append(aid)
        self._persist()
        return summary

    def record_model_rollback(self, reason: str) -> dict[str, Any]:
        """Log internal rollback without erasing dispatched/observed/settled external facts."""
        note = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "reason": reason,
            "preserved_external_actions": [
                aid
                for aid, rec in self.records.items()
                if rec.state in ("dispatched", "unknown", "observed", "settled")
            ],
        }
        self.rollback_notes.append(note)
        self._persist()
        return note
