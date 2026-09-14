"""Zero-impact candidate artifacts for Taiji R4 residual growth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import StructuralTopologyProposal
from .internalization import content_digest

ADAPTIVE_RESIDUAL_CANDIDATE_FORMAT = "taiji-adaptive-residual-candidate-v1"
ADAPTIVE_RESIDUAL_CANDIDATE_VERSION = 1


def _text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class AdaptiveResidualGrowthCandidate:
    """An auditable, non-mutating proposal to grow one residual unit.

    The embedded topology proposal is intentionally still ``pending``.  This
    artifact records what shadow materialization would do, but it contains no
    newly initialized weights and does not alter the live bridge.
    """

    bridge_id: str
    candidate_id: str
    unit_id: str
    parent_checkpoint_digest: str
    source_checkpoint_digest: str
    decision_digest: str
    evidence_ids: tuple[str, ...]
    parent_unit_count: int
    proposed_unit_count: int
    parent_edge_count: int
    proposed_edge_count: int
    topology_diff: tuple[tuple[str, int], ...]
    resource_cost: int
    structural_budget: int
    proposal: StructuralTopologyProposal
    candidate_digest: str

    def __post_init__(self) -> None:
        _text(self.bridge_id, "adaptive residual candidate bridge_id")
        _text(self.candidate_id, "adaptive residual candidate candidate_id")
        _text(self.unit_id, "adaptive residual candidate unit_id")
        _text(self.parent_checkpoint_digest, "adaptive residual candidate parent digest")
        _text(self.source_checkpoint_digest, "adaptive residual candidate source digest")
        _text(self.decision_digest, "adaptive residual candidate decision digest")
        _text(self.candidate_digest, "adaptive residual candidate digest")
        evidence_ids = tuple(str(item).strip() for item in self.evidence_ids)
        if (
            not evidence_ids
            or len(set(evidence_ids)) != len(evidence_ids)
            or any(not item for item in evidence_ids)
        ):
            raise ValueError(
                "adaptive residual candidate evidence_ids must be unique and non-empty"
            )
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if (
            min(
                int(self.parent_unit_count),
                int(self.proposed_unit_count),
                int(self.parent_edge_count),
                int(self.proposed_edge_count),
                int(self.resource_cost),
                int(self.structural_budget),
            )
            < 0
        ):
            raise ValueError("adaptive residual candidate counts cannot be negative")
        if int(self.proposed_unit_count) <= int(self.parent_unit_count):
            raise ValueError("adaptive residual candidate must increase unit capacity")
        if int(self.proposed_edge_count) < int(self.parent_edge_count):
            raise ValueError("adaptive residual candidate cannot reduce edge capacity")
        if int(self.resource_cost) <= 0:
            raise ValueError("adaptive residual candidate resource cost must be positive")
        if int(self.resource_cost) > int(self.structural_budget):
            raise ValueError("adaptive residual candidate exceeds structural budget")
        if not isinstance(self.proposal, StructuralTopologyProposal):
            raise TypeError("adaptive residual candidate proposal is invalid")
        if self.proposal.status != "pending":
            raise ValueError("adaptive residual candidate proposal must remain pending")
        if self.proposal.substrate_id != self.bridge_id:
            raise ValueError("adaptive residual candidate proposal targets another bridge")
        if self.proposal.target_kind != "neuron" or self.proposal.operation != "add":
            raise ValueError("adaptive residual candidate proposal must add one neuron")
        if self.proposal.requested_units != 1:
            raise ValueError("adaptive residual candidate supports one neuron at a time")
        if self.proposal.resource_cost != int(self.resource_cost):
            raise ValueError("adaptive residual candidate resource cost does not match proposal")
        if self.proposal.evidence_ids != evidence_ids:
            raise ValueError("adaptive residual candidate evidence does not match proposal")
        if dict(self.proposal.specification).get("unit_id") != self.unit_id:
            raise ValueError("adaptive residual candidate unit identity does not match proposal")
        diff = tuple((str(key), int(value)) for key, value in self.topology_diff)
        if len({key for key, _value in diff}) != len(diff) or any(not key for key, _value in diff):
            raise ValueError("adaptive residual candidate topology diff keys must be unique")
        object.__setattr__(self, "topology_diff", diff)
        expected = content_digest(self._payload_without_digest())
        if self.candidate_digest != expected:
            raise ValueError("adaptive residual candidate digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_RESIDUAL_CANDIDATE_FORMAT,
            "version": ADAPTIVE_RESIDUAL_CANDIDATE_VERSION,
            "kind": "candidate",
            "bridge_id": self.bridge_id,
            "candidate_id": self.candidate_id,
            "unit_id": self.unit_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "source_checkpoint_digest": self.source_checkpoint_digest,
            "decision_digest": self.decision_digest,
            "evidence_ids": list(self.evidence_ids),
            "parent_unit_count": int(self.parent_unit_count),
            "proposed_unit_count": int(self.proposed_unit_count),
            "parent_edge_count": int(self.parent_edge_count),
            "proposed_edge_count": int(self.proposed_edge_count),
            "topology_diff": {key: int(value) for key, value in self.topology_diff},
            "resource_cost": int(self.resource_cost),
            "structural_budget": int(self.structural_budget),
            "proposal": self.proposal.to_payload(),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "candidate_digest": self.candidate_digest}

    @classmethod
    def create(
        cls,
        *,
        bridge_id: str,
        unit_id: str,
        parent_checkpoint_digest: str,
        source_checkpoint_digest: str,
        decision_digest: str,
        evidence_ids: tuple[str, ...],
        parent_unit_count: int,
        proposed_unit_count: int,
        parent_edge_count: int,
        proposed_edge_count: int,
        topology_diff: tuple[tuple[str, int], ...],
        resource_cost: int,
        structural_budget: int,
        proposal: StructuralTopologyProposal,
    ) -> AdaptiveResidualGrowthCandidate:
        normalized_evidence = tuple(str(item).strip() for item in evidence_ids)
        identity: dict[str, Any] = {
            "format": ADAPTIVE_RESIDUAL_CANDIDATE_FORMAT,
            "version": ADAPTIVE_RESIDUAL_CANDIDATE_VERSION,
            "kind": "candidate",
            "bridge_id": str(bridge_id).strip(),
            "unit_id": str(unit_id).strip(),
            "parent_checkpoint_digest": str(parent_checkpoint_digest).strip(),
            "source_checkpoint_digest": str(source_checkpoint_digest).strip(),
            "decision_digest": str(decision_digest).strip(),
            "evidence_ids": list(normalized_evidence),
            "parent_unit_count": int(parent_unit_count),
            "proposed_unit_count": int(proposed_unit_count),
            "parent_edge_count": int(parent_edge_count),
            "proposed_edge_count": int(proposed_edge_count),
            "topology_diff": {str(key): int(value) for key, value in topology_diff},
            "resource_cost": int(resource_cost),
            "structural_budget": int(structural_budget),
            "proposal": proposal.to_payload(),
        }
        candidate_id = f"r4-candidate:{content_digest(identity)}"
        without_digest: dict[str, Any] = {**identity, "candidate_id": candidate_id}
        return cls(
            bridge_id=without_digest["bridge_id"],
            candidate_id=candidate_id,
            unit_id=without_digest["unit_id"],
            parent_checkpoint_digest=without_digest["parent_checkpoint_digest"],
            source_checkpoint_digest=without_digest["source_checkpoint_digest"],
            decision_digest=without_digest["decision_digest"],
            evidence_ids=normalized_evidence,
            parent_unit_count=without_digest["parent_unit_count"],
            proposed_unit_count=without_digest["proposed_unit_count"],
            parent_edge_count=without_digest["parent_edge_count"],
            proposed_edge_count=without_digest["proposed_edge_count"],
            topology_diff=topology_diff,
            resource_cost=without_digest["resource_cost"],
            structural_budget=without_digest["structural_budget"],
            proposal=proposal,
            candidate_digest=content_digest(without_digest),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdaptiveResidualGrowthCandidate:
        if payload.get("format") != ADAPTIVE_RESIDUAL_CANDIDATE_FORMAT:
            raise ValueError("unsupported adaptive residual candidate format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_CANDIDATE_VERSION:
            raise ValueError("unsupported adaptive residual candidate version")
        if payload.get("kind") != "candidate":
            raise ValueError("adaptive residual payload is not a candidate")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "candidate_digest"}
        )
        if str(payload.get("candidate_digest", "")) != expected:
            raise ValueError("adaptive residual candidate digest mismatch")
        proposal_payload = payload.get("proposal")
        if not isinstance(proposal_payload, Mapping):
            raise ValueError("adaptive residual candidate proposal is invalid")
        diff_payload = payload.get("topology_diff")
        if not isinstance(diff_payload, Mapping):
            raise ValueError("adaptive residual candidate topology diff is invalid")
        return cls(
            bridge_id=str(payload["bridge_id"]),
            candidate_id=str(payload["candidate_id"]),
            unit_id=str(payload["unit_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            source_checkpoint_digest=str(payload["source_checkpoint_digest"]),
            decision_digest=str(payload["decision_digest"]),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
            parent_unit_count=int(payload["parent_unit_count"]),
            proposed_unit_count=int(payload["proposed_unit_count"]),
            parent_edge_count=int(payload["parent_edge_count"]),
            proposed_edge_count=int(payload["proposed_edge_count"]),
            topology_diff=tuple((str(key), int(value)) for key, value in diff_payload.items()),
            resource_cost=int(payload["resource_cost"]),
            structural_budget=int(payload["structural_budget"]),
            proposal=StructuralTopologyProposal.from_payload(proposal_payload),
            candidate_digest=str(payload["candidate_digest"]),
        )


__all__ = [
    "ADAPTIVE_RESIDUAL_CANDIDATE_FORMAT",
    "ADAPTIVE_RESIDUAL_CANDIDATE_VERSION",
    "AdaptiveResidualGrowthCandidate",
]
