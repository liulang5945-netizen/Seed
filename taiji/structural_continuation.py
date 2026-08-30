"""Checkpointable capacity pressure and rollback records for structural growth."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STRUCTURAL_CAPACITY_PRESSURE_FORMAT = "taiji-structural-capacity-pressure-v1"
STRUCTURAL_CANDIDATE_ROLLBACK_FORMAT = "taiji-structural-candidate-rollback-v1"


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuralRegionCapacityPressure:
    """A measured, explicit capacity pressure snapshot for one region."""

    region_id: str
    unit_count: int
    capacity_limit: int
    pending_candidate_count: int
    reserved_resource_cost: int
    structural_budget: int
    occupancy: float
    queue_pressure: float
    reservation_pressure: float
    pressure: float
    pressure_digest: str

    def __post_init__(self) -> None:
        if not str(self.region_id):
            raise ValueError("structural capacity pressure region_id must not be empty")
        if min(
            int(self.unit_count),
            int(self.capacity_limit),
            int(self.pending_candidate_count),
            int(self.reserved_resource_cost),
            int(self.structural_budget),
        ) < 0:
            raise ValueError("structural capacity pressure counts cannot be negative")
        if int(self.capacity_limit) <= 0:
            raise ValueError("structural capacity pressure capacity_limit must be positive")
        for name in (
            "occupancy",
            "queue_pressure",
            "reservation_pressure",
            "pressure",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"structural capacity pressure {name} must be in [0, 1]")
        if not str(self.pressure_digest):
            raise ValueError("structural capacity pressure digest must not be empty")
        object.__setattr__(self, "region_id", str(self.region_id))
        for name in (
            "unit_count",
            "capacity_limit",
            "pending_candidate_count",
            "reserved_resource_cost",
            "structural_budget",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "occupancy",
            "queue_pressure",
            "reservation_pressure",
            "pressure",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "pressure_digest", str(self.pressure_digest))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_CAPACITY_PRESSURE_FORMAT,
            "region_id": self.region_id,
            "unit_count": self.unit_count,
            "capacity_limit": self.capacity_limit,
            "pending_candidate_count": self.pending_candidate_count,
            "reserved_resource_cost": self.reserved_resource_cost,
            "structural_budget": self.structural_budget,
            "occupancy": self.occupancy,
            "queue_pressure": self.queue_pressure,
            "reservation_pressure": self.reservation_pressure,
            "pressure": self.pressure,
            "pressure_digest": self.pressure_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralRegionCapacityPressure:
        if payload.get("format") != STRUCTURAL_CAPACITY_PRESSURE_FORMAT:
            raise ValueError("unsupported structural capacity pressure format")
        return cls(
            region_id=str(payload["region_id"]),
            unit_count=int(payload["unit_count"]),
            capacity_limit=int(payload["capacity_limit"]),
            pending_candidate_count=int(payload.get("pending_candidate_count", 0)),
            reserved_resource_cost=int(payload.get("reserved_resource_cost", 0)),
            structural_budget=int(payload.get("structural_budget", 0)),
            occupancy=float(payload["occupancy"]),
            queue_pressure=float(payload.get("queue_pressure", 0.0)),
            reservation_pressure=float(payload.get("reservation_pressure", 0.0)),
            pressure=float(payload["pressure"]),
            pressure_digest=str(payload["pressure_digest"]),
        )


def measure_structural_region_capacity_pressure(
    *,
    region_id: str,
    unit_count: int,
    capacity_limit: int,
    pending_candidate_count: int,
    reserved_resource_cost: int,
    structural_budget: int,
) -> StructuralRegionCapacityPressure:
    """Measure capacity without mutating topology, budget, or candidate state."""

    limit = int(capacity_limit)
    if limit <= 0:
        raise ValueError("structural capacity pressure capacity_limit must be positive")
    occupancy = min(1.0, max(0.0, float(unit_count) / limit))
    queue_pressure = min(1.0, max(0.0, float(pending_candidate_count) / limit))
    reservation_pressure = min(1.0, max(0.0, float(reserved_resource_cost) / limit))
    pressure = max(occupancy, queue_pressure, reservation_pressure)
    identity = {
        "region_id": str(region_id),
        "unit_count": int(unit_count),
        "capacity_limit": limit,
        "pending_candidate_count": int(pending_candidate_count),
        "reserved_resource_cost": int(reserved_resource_cost),
        "structural_budget": int(structural_budget),
        "occupancy": occupancy,
        "queue_pressure": queue_pressure,
        "reservation_pressure": reservation_pressure,
        "pressure": pressure,
    }
    return StructuralRegionCapacityPressure(
        **identity,
        pressure_digest=_digest(identity),
    )


@dataclass(frozen=True)
class StructuralCandidateRollback:
    """Auditable reversal of one admitted candidate."""

    batch_id: str
    candidate_id: str
    proposal_id: str
    status: str
    admission_parent_checkpoint_digest: str
    admission_child_checkpoint_digest: str
    rollback_checkpoint_digest: str
    topology_before_digest: str
    topology_after_digest: str
    structural_budget_before: int
    structural_budget_after: int
    resource_cost: int
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"rolled_back", "failed_closed", "not_rollbackable"}:
            raise ValueError("unsupported structural candidate rollback status")
        for name in (
            "batch_id",
            "candidate_id",
            "proposal_id",
            "admission_parent_checkpoint_digest",
            "admission_child_checkpoint_digest",
            "rollback_checkpoint_digest",
            "topology_before_digest",
            "topology_after_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"structural candidate rollback {name} must not be empty")
        if min(int(self.structural_budget_before), int(self.structural_budget_after)) < 0:
            raise ValueError("structural candidate rollback budget cannot be negative")
        if int(self.resource_cost) <= 0:
            raise ValueError("structural candidate rollback resource_cost must be positive")
        if self.status != "rolled_back" and not str(self.reason):
            raise ValueError("failed structural candidate rollback requires a reason")
        object.__setattr__(self, "batch_id", str(self.batch_id))
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "proposal_id", str(self.proposal_id))
        object.__setattr__(self, "admission_parent_checkpoint_digest", str(self.admission_parent_checkpoint_digest))
        object.__setattr__(self, "admission_child_checkpoint_digest", str(self.admission_child_checkpoint_digest))
        object.__setattr__(self, "rollback_checkpoint_digest", str(self.rollback_checkpoint_digest))
        object.__setattr__(self, "topology_before_digest", str(self.topology_before_digest))
        object.__setattr__(self, "topology_after_digest", str(self.topology_after_digest))
        object.__setattr__(self, "structural_budget_before", int(self.structural_budget_before))
        object.__setattr__(self, "structural_budget_after", int(self.structural_budget_after))
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        object.__setattr__(self, "reason", str(self.reason))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_CANDIDATE_ROLLBACK_FORMAT,
            "batch_id": self.batch_id,
            "candidate_id": self.candidate_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "admission_parent_checkpoint_digest": self.admission_parent_checkpoint_digest,
            "admission_child_checkpoint_digest": self.admission_child_checkpoint_digest,
            "rollback_checkpoint_digest": self.rollback_checkpoint_digest,
            "topology_before_digest": self.topology_before_digest,
            "topology_after_digest": self.topology_after_digest,
            "structural_budget_before": self.structural_budget_before,
            "structural_budget_after": self.structural_budget_after,
            "resource_cost": self.resource_cost,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralCandidateRollback:
        if payload.get("format") != STRUCTURAL_CANDIDATE_ROLLBACK_FORMAT:
            raise ValueError("unsupported structural candidate rollback format")
        return cls(
            batch_id=str(payload["batch_id"]),
            candidate_id=str(payload["candidate_id"]),
            proposal_id=str(payload["proposal_id"]),
            status=str(payload["status"]),
            admission_parent_checkpoint_digest=str(payload["admission_parent_checkpoint_digest"]),
            admission_child_checkpoint_digest=str(payload["admission_child_checkpoint_digest"]),
            rollback_checkpoint_digest=str(payload["rollback_checkpoint_digest"]),
            topology_before_digest=str(payload["topology_before_digest"]),
            topology_after_digest=str(payload["topology_after_digest"]),
            structural_budget_before=int(payload["structural_budget_before"]),
            structural_budget_after=int(payload["structural_budget_after"]),
            resource_cost=int(payload["resource_cost"]),
            reason=str(payload.get("reason", "")),
        )

