"""Deterministic, checkpointable arbitration for structural candidate batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STRUCTURAL_CANDIDATE_BATCH_CHECKPOINT_FORMAT = "taiji-structural-candidate-batch-v1"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuralCandidateBatch:
    """A deterministic reservation and arbitration decision for candidates."""

    batch_id: str
    candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    deferred_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    candidate_states: tuple[tuple[str, str], ...]
    reasons: tuple[tuple[str, str], ...]
    reserved_resource_cost: int
    reservation_remaining: int
    structural_budget_before: int
    available_budget_before: int
    topology_digest: str
    arbitration_digest: str
    revision: int = 0
    status: str = "planned"

    def __post_init__(self) -> None:
        if not str(self.batch_id):
            raise ValueError("structural candidate batch_id must not be empty")
        if self.status not in {"planned", "running", "completed", "failed_closed"}:
            raise ValueError("unsupported structural candidate batch status")
        all_ids = tuple(str(item) for item in self.candidate_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("structural candidate batch candidate_ids must be unique")
        if any(not item for item in all_ids):
            raise ValueError("structural candidate batch candidate_ids must be non-empty")
        selected = tuple(str(item) for item in self.selected_candidate_ids)
        deferred = tuple(str(item) for item in self.deferred_candidate_ids)
        rejected = tuple(str(item) for item in self.rejected_candidate_ids)
        groups = (selected, deferred, rejected)
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("structural candidate batch decision ids must be unique")
        if set(selected) & set(deferred) or set(selected) & set(rejected) or set(deferred) & set(rejected):
            raise ValueError("structural candidate batch decision groups must be disjoint")
        if set(selected) | set(deferred) | set(rejected) != set(all_ids):
            raise ValueError("structural candidate batch decisions must cover all candidates")
        states = tuple((str(key), str(value)) for key, value in self.candidate_states)
        if tuple(key for key, _ in states) != all_ids:
            raise ValueError("structural candidate batch states must follow candidate_ids")
        allowed_states = {
            "reserved",
            "deferred",
            "rejected",
            "admitted",
            "rolled_back",
            "failed_closed",
            "policy_rejected",
        }
        if any(not key or value not in allowed_states for key, value in states):
            raise ValueError("unsupported structural candidate batch candidate state")
        reasons = tuple((str(key), str(value)) for key, value in self.reasons)
        if len({key for key, _ in reasons}) != len(reasons) or any(
            not key or not value for key, value in reasons
        ):
            raise ValueError("structural candidate batch reasons must be unique and non-empty")
        if int(self.reserved_resource_cost) < 0 or int(self.reservation_remaining) < 0:
            raise ValueError("structural candidate batch reservation cannot be negative")
        if int(self.reservation_remaining) > int(self.reserved_resource_cost):
            raise ValueError("structural candidate batch remaining reservation exceeds total")
        if int(self.structural_budget_before) < 0 or int(self.available_budget_before) < 0:
            raise ValueError("structural candidate batch budget cannot be negative")
        if int(self.available_budget_before) > int(self.structural_budget_before):
            raise ValueError("structural candidate batch available budget exceeds budget")
        for name in ("topology_digest", "arbitration_digest"):
            if not str(getattr(self, name)):
                raise ValueError(f"structural candidate batch {name} must not be empty")
        if int(self.revision) < 0:
            raise ValueError("structural candidate batch revision cannot be negative")
        object.__setattr__(self, "batch_id", str(self.batch_id))
        object.__setattr__(self, "candidate_ids", all_ids)
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "deferred_candidate_ids", deferred)
        object.__setattr__(self, "rejected_candidate_ids", rejected)
        object.__setattr__(self, "candidate_states", states)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "reserved_resource_cost", int(self.reserved_resource_cost))
        object.__setattr__(self, "reservation_remaining", int(self.reservation_remaining))
        object.__setattr__(self, "structural_budget_before", int(self.structural_budget_before))
        object.__setattr__(self, "available_budget_before", int(self.available_budget_before))
        object.__setattr__(self, "topology_digest", str(self.topology_digest))
        object.__setattr__(self, "arbitration_digest", str(self.arbitration_digest))
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def state_by_candidate(self) -> dict[str, str]:
        return dict(self.candidate_states)

    @property
    def reason_by_candidate(self) -> dict[str, str]:
        return dict(self.reasons)

    @property
    def active_reservation(self) -> bool:
        return self.reservation_remaining > 0

    def identity_payload(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "deferred_candidate_ids": list(self.deferred_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "reserved_resource_cost": self.reserved_resource_cost,
            "structural_budget_before": self.structural_budget_before,
            "available_budget_before": self.available_budget_before,
            "topology_digest": self.topology_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_CANDIDATE_BATCH_CHECKPOINT_FORMAT,
            "batch_id": self.batch_id,
            "candidate_ids": list(self.candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "deferred_candidate_ids": list(self.deferred_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "candidate_states": {key: value for key, value in self.candidate_states},
            "reasons": {key: value for key, value in self.reasons},
            "reserved_resource_cost": self.reserved_resource_cost,
            "reservation_remaining": self.reservation_remaining,
            "structural_budget_before": self.structural_budget_before,
            "available_budget_before": self.available_budget_before,
            "topology_digest": self.topology_digest,
            "arbitration_digest": self.arbitration_digest,
            "revision": self.revision,
            "status": self.status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralCandidateBatch:
        if payload.get("format") != STRUCTURAL_CANDIDATE_BATCH_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural candidate batch format")
        raw_states = payload.get("candidate_states", {})
        raw_reasons = payload.get("reasons", {})
        if not isinstance(raw_states, Mapping) or not isinstance(raw_reasons, Mapping):
            raise ValueError("structural candidate batch states and reasons must be mappings")
        candidate_ids = tuple(str(item) for item in payload.get("candidate_ids", ()))
        return cls(
            batch_id=str(payload["batch_id"]),
            candidate_ids=candidate_ids,
            selected_candidate_ids=tuple(
                str(item) for item in payload.get("selected_candidate_ids", ())
            ),
            deferred_candidate_ids=tuple(
                str(item) for item in payload.get("deferred_candidate_ids", ())
            ),
            rejected_candidate_ids=tuple(
                str(item) for item in payload.get("rejected_candidate_ids", ())
            ),
            candidate_states=tuple((item, str(raw_states[item])) for item in candidate_ids),
            reasons=tuple((str(key), str(value)) for key, value in raw_reasons.items()),
            reserved_resource_cost=int(payload.get("reserved_resource_cost", 0)),
            reservation_remaining=int(payload.get("reservation_remaining", 0)),
            structural_budget_before=int(payload.get("structural_budget_before", 0)),
            available_budget_before=int(payload.get("available_budget_before", 0)),
            topology_digest=str(payload["topology_digest"]),
            arbitration_digest=str(payload["arbitration_digest"]),
            revision=int(payload.get("revision", 0)),
            status=str(payload.get("status", "planned")),
        )


def structural_candidate_batch_digest(
    *,
    candidate_payloads: tuple[Mapping[str, Any], ...],
    topology_digest: str,
    structural_budget: int,
    available_budget: int,
) -> str:
    """Return the deterministic identity for one arbitration attempt."""

    return _canonical_digest(
        {
            "candidate_payloads": list(candidate_payloads),
            "topology_digest": str(topology_digest),
            "structural_budget": int(structural_budget),
            "available_budget": int(available_budget),
        }
    )
