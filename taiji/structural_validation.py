"""Independent validation policy for candidate structural changes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STRUCTURAL_VALIDATION_GATE_FORMAT = "taiji-structural-validation-gate-v1"
STRUCTURAL_ADMISSION_RESULT_FORMAT = "taiji-structural-admission-result-v1"


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StructuralValidationGateDecision:
    """Content-addressed decision over independently measured candidate metrics."""

    candidate_id: str
    passed: bool
    holdout_gain: float
    retention_regression: float
    lesion_effect: float
    resource_state: float
    resource_cost: int
    structural_budget: int
    minimum_holdout_gain: float
    maximum_retention_regression: float
    minimum_lesion_effect: float
    minimum_resource_state: float
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    decision_digest: str

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("structural validation candidate_id must not be empty")
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
            "minimum_holdout_gain",
            "maximum_retention_regression",
            "minimum_lesion_effect",
            "minimum_resource_state",
        ):
            _unit(getattr(self, name), f"structural validation {name}")
        if int(self.resource_cost) <= 0:
            raise ValueError("structural validation resource_cost must be positive")
        if int(self.structural_budget) < 0:
            raise ValueError("structural validation structural_budget cannot be negative")
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if not evidence_ids or any(not item for item in evidence_ids):
            raise ValueError("structural validation evidence_ids must not be empty")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("structural validation evidence_ids must be unique")
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons):
            raise ValueError("structural validation reasons must not be empty")
        if bool(self.passed) != (not reasons):
            raise ValueError("structural validation passed must match reasons")
        if not str(self.decision_digest):
            raise ValueError("structural validation decision_digest must not be empty")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "passed", bool(self.passed))
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
            "minimum_holdout_gain",
            "maximum_retention_regression",
            "minimum_lesion_effect",
            "minimum_resource_state",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        object.__setattr__(self, "structural_budget", int(self.structural_budget))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "decision_digest", str(self.decision_digest))
        expected_digest = _digest(self._payload_without_digest())
        if self.decision_digest != expected_digest:
            raise ValueError("structural validation decision digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_VALIDATION_GATE_FORMAT,
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "holdout_gain": self.holdout_gain,
            "retention_regression": self.retention_regression,
            "lesion_effect": self.lesion_effect,
            "resource_state": self.resource_state,
            "resource_cost": self.resource_cost,
            "structural_budget": self.structural_budget,
            "minimum_holdout_gain": self.minimum_holdout_gain,
            "maximum_retention_regression": self.maximum_retention_regression,
            "minimum_lesion_effect": self.minimum_lesion_effect,
            "minimum_resource_state": self.minimum_resource_state,
            "evidence_ids": list(self.evidence_ids),
            "reasons": list(self.reasons),
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_without_digest()
        payload["decision_digest"] = self.decision_digest
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StructuralValidationGateDecision:
        if payload.get("format") != STRUCTURAL_VALIDATION_GATE_FORMAT:
            raise ValueError("unsupported structural validation gate format")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            passed=bool(payload["passed"]),
            holdout_gain=float(payload["holdout_gain"]),
            retention_regression=float(payload["retention_regression"]),
            lesion_effect=float(payload["lesion_effect"]),
            resource_state=float(payload["resource_state"]),
            resource_cost=int(payload["resource_cost"]),
            structural_budget=int(payload["structural_budget"]),
            minimum_holdout_gain=float(payload["minimum_holdout_gain"]),
            maximum_retention_regression=float(payload["maximum_retention_regression"]),
            minimum_lesion_effect=float(payload["minimum_lesion_effect"]),
            minimum_resource_state=float(payload["minimum_resource_state"]),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
            decision_digest=str(payload["decision_digest"]),
        )


@dataclass(frozen=True)
class StructuralAdmissionResult:
    """Auditable result for one atomic topology admission attempt."""

    candidate_id: str
    proposal_id: str
    status: str
    decision_digest: str
    parent_checkpoint_digest: str
    child_checkpoint_digest: str
    topology_before_digest: str
    topology_after_digest: str
    structural_budget_before: int
    structural_budget_after: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id) or not str(self.proposal_id):
            raise ValueError("structural admission identifiers must not be empty")
        if self.status not in {"admitted", "rejected", "rolled_back", "failed_closed"}:
            raise ValueError("unsupported structural admission status")
        for name in (
            "decision_digest",
            "parent_checkpoint_digest",
            "child_checkpoint_digest",
            "topology_before_digest",
            "topology_after_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"structural admission {name} must not be empty")
        if min(int(self.structural_budget_before), int(self.structural_budget_after)) < 0:
            raise ValueError("structural admission budget cannot be negative")
        if self.error is not None and not str(self.error):
            raise ValueError("structural admission error must not be empty")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "proposal_id", str(self.proposal_id))
        object.__setattr__(self, "decision_digest", str(self.decision_digest))
        object.__setattr__(self, "parent_checkpoint_digest", str(self.parent_checkpoint_digest))
        object.__setattr__(self, "child_checkpoint_digest", str(self.child_checkpoint_digest))
        object.__setattr__(self, "topology_before_digest", str(self.topology_before_digest))
        object.__setattr__(self, "topology_after_digest", str(self.topology_after_digest))
        object.__setattr__(self, "structural_budget_before", int(self.structural_budget_before))
        object.__setattr__(self, "structural_budget_after", int(self.structural_budget_after))
        object.__setattr__(self, "error", None if self.error is None else str(self.error))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_ADMISSION_RESULT_FORMAT,
            "candidate_id": self.candidate_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "decision_digest": self.decision_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "child_checkpoint_digest": self.child_checkpoint_digest,
            "topology_before_digest": self.topology_before_digest,
            "topology_after_digest": self.topology_after_digest,
            "structural_budget_before": self.structural_budget_before,
            "structural_budget_after": self.structural_budget_after,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StructuralAdmissionResult:
        if payload.get("format") != STRUCTURAL_ADMISSION_RESULT_FORMAT:
            raise ValueError("unsupported structural admission result format")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            proposal_id=str(payload["proposal_id"]),
            status=str(payload["status"]),
            decision_digest=str(payload["decision_digest"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            child_checkpoint_digest=str(payload["child_checkpoint_digest"]),
            topology_before_digest=str(payload["topology_before_digest"]),
            topology_after_digest=str(payload["topology_after_digest"]),
            structural_budget_before=int(payload["structural_budget_before"]),
            structural_budget_after=int(payload["structural_budget_after"]),
            error=None if payload.get("error") is None else str(payload["error"]),
        )


def evaluate_structural_candidate_validation(
    candidate_id: str,
    *,
    holdout_gain: float,
    retention_regression: float,
    lesion_effect: float,
    resource_state: float,
    resource_cost: int,
    structural_budget: int,
    evidence_ids: Sequence[str],
    minimum_holdout_gain: float = 0.05,
    maximum_retention_regression: float = 0.05,
    minimum_lesion_effect: float = 0.05,
    minimum_resource_state: float = 0.40,
) -> StructuralValidationGateDecision:
    """Evaluate candidate metrics without changing any model or ledger state."""

    holdout_gain = _unit(holdout_gain, "structural validation holdout_gain")
    retention_regression = _unit(
        retention_regression,
        "structural validation retention_regression",
    )
    lesion_effect = _unit(lesion_effect, "structural validation lesion_effect")
    resource_state = _unit(resource_state, "structural validation resource_state")
    minimum_holdout_gain = _unit(
        minimum_holdout_gain,
        "structural validation minimum_holdout_gain",
    )
    maximum_retention_regression = _unit(
        maximum_retention_regression,
        "structural validation maximum_retention_regression",
    )
    minimum_lesion_effect = _unit(
        minimum_lesion_effect,
        "structural validation minimum_lesion_effect",
    )
    minimum_resource_state = _unit(
        minimum_resource_state,
        "structural validation minimum_resource_state",
    )
    resource_cost = int(resource_cost)
    structural_budget = int(structural_budget)
    if resource_cost <= 0:
        raise ValueError("structural validation resource_cost must be positive")
    if structural_budget < 0:
        raise ValueError("structural validation structural_budget cannot be negative")
    normalized_evidence_ids = tuple(str(item) for item in evidence_ids)
    if not normalized_evidence_ids or any(not item for item in normalized_evidence_ids):
        raise ValueError("structural validation evidence_ids must not be empty")
    if len(set(normalized_evidence_ids)) != len(normalized_evidence_ids):
        raise ValueError("structural validation evidence_ids must be unique")

    reasons: list[str] = []
    if holdout_gain < minimum_holdout_gain:
        reasons.append("holdout_gain_below_threshold")
    if retention_regression > maximum_retention_regression:
        reasons.append("retention_regression_above_threshold")
    if lesion_effect < minimum_lesion_effect:
        reasons.append("lesion_effect_below_threshold")
    if resource_state < minimum_resource_state:
        reasons.append("resource_state_below_threshold")
    if resource_cost > structural_budget:
        reasons.append("structural_budget_insufficient")
    payload = {
        "format": STRUCTURAL_VALIDATION_GATE_FORMAT,
        "candidate_id": str(candidate_id),
        "passed": not reasons,
        "holdout_gain": holdout_gain,
        "retention_regression": retention_regression,
        "lesion_effect": lesion_effect,
        "resource_state": resource_state,
        "resource_cost": resource_cost,
        "structural_budget": structural_budget,
        "minimum_holdout_gain": minimum_holdout_gain,
        "maximum_retention_regression": maximum_retention_regression,
        "minimum_lesion_effect": minimum_lesion_effect,
        "minimum_resource_state": minimum_resource_state,
        "evidence_ids": list(normalized_evidence_ids),
        "reasons": reasons,
    }
    return StructuralValidationGateDecision(
        candidate_id=str(candidate_id),
        passed=not reasons,
        holdout_gain=holdout_gain,
        retention_regression=retention_regression,
        lesion_effect=lesion_effect,
        resource_state=resource_state,
        resource_cost=resource_cost,
        structural_budget=structural_budget,
        minimum_holdout_gain=minimum_holdout_gain,
        maximum_retention_regression=maximum_retention_regression,
        minimum_lesion_effect=minimum_lesion_effect,
        minimum_resource_state=minimum_resource_state,
        evidence_ids=normalized_evidence_ids,
        reasons=tuple(reasons),
        decision_digest=_digest(payload),
    )
