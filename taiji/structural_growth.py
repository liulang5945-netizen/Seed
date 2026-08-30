"""Substrate-driven structural growth signals for the native Taiji runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

STRUCTURAL_GROWTH_CHECKPOINT_FORMAT = "taiji-structural-growth-v1"
STRUCTURAL_PRUNING_CHECKPOINT_FORMAT = "taiji-structural-pruning-v1"
STRUCTURAL_RUNTIME_OBSERVATION_CHECKPOINT_FORMAT = "taiji-structural-runtime-observation-v1"
STRUCTURAL_PROPOSAL_CANDIDATE_FORMAT = "taiji-structural-proposal-candidate-v1"
STRUCTURAL_MAINTENANCE_RESULT_FORMAT = "taiji-structural-maintenance-result-v1"
STRUCTURAL_CANDIDATE_VALIDATION_FORMAT = "taiji-structural-candidate-validation-v1"
STRUCTURAL_EVIDENCE_PARTITIONS = frozenset({"runtime", "train", "holdout", "retention"})


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


@dataclass(frozen=True)
class StructuralRuntimeObservation:
    """One checkpointable structural observation produced by a real runtime tick.

    ``prediction_error`` is optional because a tick without an expected
    activity target can still provide usage and resource evidence for
    pruning, but it must not be mistaken for supervised growth evidence.
    """

    network_id: str
    region_id: str
    tick: int
    usage: float
    resource_pressure: float
    prediction_error: float | None
    learning_gain: float
    holdout_transfer: float
    evidence_id: str
    task_slice_id: str = ""
    partition: str = "runtime"

    def __post_init__(self) -> None:
        if not str(self.network_id):
            raise ValueError("structural runtime network_id must not be empty")
        if not str(self.region_id):
            raise ValueError("structural runtime region_id must not be empty")
        if int(self.tick) <= 0:
            raise ValueError("structural runtime tick must be positive")
        _unit(self.usage, "structural runtime usage")
        _unit(self.resource_pressure, "structural runtime resource_pressure")
        if self.prediction_error is not None:
            _unit(self.prediction_error, "structural runtime prediction_error")
        _unit(self.learning_gain, "structural runtime learning_gain")
        _unit(self.holdout_transfer, "structural runtime holdout_transfer")
        if not str(self.evidence_id):
            raise ValueError("structural runtime evidence_id must not be empty")
        if self.partition not in STRUCTURAL_EVIDENCE_PARTITIONS:
            raise ValueError("unsupported structural runtime evidence partition")
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "tick", int(self.tick))
        object.__setattr__(self, "usage", float(self.usage))
        object.__setattr__(self, "resource_pressure", float(self.resource_pressure))
        object.__setattr__(
            self,
            "prediction_error",
            None if self.prediction_error is None else float(self.prediction_error),
        )
        object.__setattr__(self, "learning_gain", float(self.learning_gain))
        object.__setattr__(self, "holdout_transfer", float(self.holdout_transfer))
        object.__setattr__(self, "evidence_id", str(self.evidence_id))
        object.__setattr__(self, "task_slice_id", str(self.task_slice_id))
        object.__setattr__(self, "partition", str(self.partition))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_RUNTIME_OBSERVATION_CHECKPOINT_FORMAT,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "tick": self.tick,
            "usage": self.usage,
            "resource_pressure": self.resource_pressure,
            "prediction_error": self.prediction_error,
            "learning_gain": self.learning_gain,
            "holdout_transfer": self.holdout_transfer,
            "evidence_id": self.evidence_id,
            "task_slice_id": self.task_slice_id,
            "partition": self.partition,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralRuntimeObservation:
        if payload.get("format") != STRUCTURAL_RUNTIME_OBSERVATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural runtime observation format")
        error = payload.get("prediction_error")
        return cls(
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            tick=int(payload["tick"]),
            usage=float(payload["usage"]),
            resource_pressure=float(payload["resource_pressure"]),
            prediction_error=None if error is None else float(error),
            learning_gain=float(payload["learning_gain"]),
            holdout_transfer=float(payload.get("holdout_transfer", 0.0)),
            evidence_id=str(payload["evidence_id"]),
            task_slice_id=str(payload.get("task_slice_id", "")),
            partition=str(payload.get("partition", "runtime")),
        )


@dataclass(frozen=True)
class StructuralProposalCandidate:
    """A runtime-evidence candidate awaiting the topology ledger."""

    candidate_id: str
    network_id: str
    target_kind: str
    operation: str
    substrate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_tick: int
    priority: float
    specification: tuple[tuple[str, Any], ...] = ()
    resource_cost: int = 1
    depends_on_candidate_ids: tuple[str, ...] = ()
    conflict_keys: tuple[str, ...] = ()
    parent_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("structural candidate_id must not be empty")
        if not str(self.network_id):
            raise ValueError("structural candidate network_id must not be empty")
        if self.target_kind not in {"neuron", "region", "connection"}:
            raise ValueError("unsupported structural candidate target_kind")
        if self.operation not in {"add", "prune", "split", "merge"}:
            raise ValueError("unsupported structural candidate operation")
        substrates = tuple(str(item) for item in self.substrate_ids)
        evidence = tuple(str(item) for item in self.evidence_ids)
        if not substrates or any(not item for item in substrates):
            raise ValueError("structural candidate substrate_ids must not be empty")
        if not evidence or any(not item for item in evidence):
            raise ValueError("structural candidate evidence_ids must not be empty")
        if len(set(evidence)) != len(evidence):
            raise ValueError("structural candidate evidence_ids cannot contain duplicates")
        if int(self.source_tick) <= 0:
            raise ValueError("structural candidate source_tick must be positive")
        _unit(self.priority, "structural candidate priority")
        if int(self.resource_cost) <= 0:
            raise ValueError("structural candidate resource_cost must be positive")
        specification = tuple((str(key), value) for key, value in self.specification)
        if len({key for key, _ in specification}) != len(specification):
            raise ValueError("structural candidate specification keys must be unique")
        dependencies = tuple(str(item) for item in self.depends_on_candidate_ids)
        if any(not item for item in dependencies):
            raise ValueError("structural candidate dependencies must not be empty")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("structural candidate dependencies cannot contain duplicates")
        if str(self.candidate_id) in dependencies:
            raise ValueError("structural candidate cannot depend on itself")
        conflicts = tuple(str(item) for item in self.conflict_keys)
        if any(not item for item in conflicts):
            raise ValueError("structural candidate conflict_keys must not be empty")
        if len(set(conflicts)) != len(conflicts):
            raise ValueError("structural candidate conflict_keys cannot contain duplicates")
        if self.parent_checkpoint_id is not None and not str(self.parent_checkpoint_id):
            raise ValueError("structural candidate parent_checkpoint_id must not be empty")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "substrate_ids", substrates)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "source_tick", int(self.source_tick))
        object.__setattr__(self, "priority", float(self.priority))
        object.__setattr__(self, "specification", specification)
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        object.__setattr__(self, "depends_on_candidate_ids", dependencies)
        object.__setattr__(self, "conflict_keys", conflicts)
        object.__setattr__(
            self,
            "parent_checkpoint_id",
            None if self.parent_checkpoint_id is None else str(self.parent_checkpoint_id),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_PROPOSAL_CANDIDATE_FORMAT,
            "candidate_id": self.candidate_id,
            "network_id": self.network_id,
            "target_kind": self.target_kind,
            "operation": self.operation,
            "substrate_ids": list(self.substrate_ids),
            "evidence_ids": list(self.evidence_ids),
            "source_tick": self.source_tick,
            "priority": self.priority,
            "specification": {key: value for key, value in self.specification},
            "resource_cost": self.resource_cost,
            "depends_on_candidate_ids": list(self.depends_on_candidate_ids),
            "conflict_keys": list(self.conflict_keys),
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralProposalCandidate:
        if payload.get("format") != STRUCTURAL_PROPOSAL_CANDIDATE_FORMAT:
            raise ValueError("unsupported structural proposal candidate format")
        specification = payload.get("specification", {})
        if not isinstance(specification, Mapping):
            raise ValueError("structural candidate specification must be a mapping")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            network_id=str(payload["network_id"]),
            target_kind=str(payload["target_kind"]),
            operation=str(payload["operation"]),
            substrate_ids=tuple(str(item) for item in payload.get("substrate_ids", ())),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            source_tick=int(payload["source_tick"]),
            priority=float(payload["priority"]),
            specification=tuple((str(key), value) for key, value in specification.items()),
            resource_cost=int(payload.get("resource_cost", 1)),
            depends_on_candidate_ids=tuple(
                str(item) for item in payload.get("depends_on_candidate_ids", ())
            ),
            conflict_keys=tuple(str(item) for item in payload.get("conflict_keys", ())),
            parent_checkpoint_id=(
                None
                if payload.get("parent_checkpoint_id") is None
                else str(payload["parent_checkpoint_id"])
            ),
        )


@dataclass(frozen=True)
class StructuralMaintenanceResult:
    """Auditable result for one item in a structural maintenance cycle."""

    candidate_id: str
    proposal_id: str | None
    status: str
    validation_score: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("structural maintenance candidate_id must not be empty")
        if self.proposal_id is not None and not str(self.proposal_id):
            raise ValueError("structural maintenance proposal_id must not be empty")
        if self.status not in {
            "committed",
            "rejected",
            "missing_holdout",
            "failed_closed",
            "already_applied",
        }:
            raise ValueError("unsupported structural maintenance status")
        _unit(self.validation_score, "structural maintenance validation_score")
        if self.error is not None and not str(self.error):
            raise ValueError("structural maintenance error must not be empty")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(
            self,
            "proposal_id",
            None if self.proposal_id is None else str(self.proposal_id),
        )
        object.__setattr__(self, "validation_score", float(self.validation_score))
        object.__setattr__(self, "error", None if self.error is None else str(self.error))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_MAINTENANCE_RESULT_FORMAT,
            "candidate_id": self.candidate_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "validation_score": self.validation_score,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralMaintenanceResult:
        if payload.get("format") != STRUCTURAL_MAINTENANCE_RESULT_FORMAT:
            raise ValueError("unsupported structural maintenance result format")
        proposal_id = payload.get("proposal_id")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            proposal_id=None if proposal_id is None else str(proposal_id),
            status=str(payload["status"]),
            validation_score=float(payload.get("validation_score", 0.0)),
            error=None if payload.get("error") is None else str(payload["error"]),
        )


@dataclass(frozen=True)
class StructuralCandidateValidation:
    """Candidate-only validation evidence before topology admission."""

    candidate_id: str
    proposal_id: str | None
    status: str
    validation_score: float
    parent_checkpoint_digest: str
    validation_checkpoint_digest: str
    topology_before_digest: str
    topology_after_digest: str
    structural_budget_before: int
    structural_budget_after: int
    evidence_ids: tuple[str, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("structural candidate validation candidate_id must not be empty")
        if self.proposal_id is not None and not str(self.proposal_id):
            raise ValueError("structural candidate validation proposal_id must not be empty")
        if self.status not in {"validated", "rejected", "failed_closed"}:
            raise ValueError("unsupported structural candidate validation status")
        _unit(self.validation_score, "structural candidate validation_score")
        for name in (
            "parent_checkpoint_digest",
            "validation_checkpoint_digest",
            "topology_before_digest",
            "topology_after_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"structural candidate validation {name} must not be empty")
        if min(int(self.structural_budget_before), int(self.structural_budget_after)) < 0:
            raise ValueError("structural candidate validation budget cannot be negative")
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("structural candidate validation evidence_ids must be unique")
        if self.error is not None and not str(self.error):
            raise ValueError("structural candidate validation error must not be empty")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(
            self,
            "proposal_id",
            None if self.proposal_id is None else str(self.proposal_id),
        )
        object.__setattr__(self, "validation_score", float(self.validation_score))
        object.__setattr__(self, "parent_checkpoint_digest", str(self.parent_checkpoint_digest))
        object.__setattr__(
            self,
            "validation_checkpoint_digest",
            str(self.validation_checkpoint_digest),
        )
        object.__setattr__(self, "topology_before_digest", str(self.topology_before_digest))
        object.__setattr__(self, "topology_after_digest", str(self.topology_after_digest))
        object.__setattr__(self, "structural_budget_before", int(self.structural_budget_before))
        object.__setattr__(self, "structural_budget_after", int(self.structural_budget_after))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "error", None if self.error is None else str(self.error))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_CANDIDATE_VALIDATION_FORMAT,
            "candidate_id": self.candidate_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "validation_score": self.validation_score,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "validation_checkpoint_digest": self.validation_checkpoint_digest,
            "topology_before_digest": self.topology_before_digest,
            "topology_after_digest": self.topology_after_digest,
            "structural_budget_before": self.structural_budget_before,
            "structural_budget_after": self.structural_budget_after,
            "evidence_ids": list(self.evidence_ids),
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralCandidateValidation:
        if payload.get("format") != STRUCTURAL_CANDIDATE_VALIDATION_FORMAT:
            raise ValueError("unsupported structural candidate validation format")
        proposal_id = payload.get("proposal_id")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            proposal_id=None if proposal_id is None else str(proposal_id),
            status=str(payload["status"]),
            validation_score=float(payload.get("validation_score", 0.0)),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            validation_checkpoint_digest=str(payload["validation_checkpoint_digest"]),
            topology_before_digest=str(payload["topology_before_digest"]),
            topology_after_digest=str(payload["topology_after_digest"]),
            structural_budget_before=int(payload["structural_budget_before"]),
            structural_budget_after=int(payload["structural_budget_after"]),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            error=None if payload.get("error") is None else str(payload["error"]),
        )


@dataclass(frozen=True)
class StructuralGrowthDynamics:
    """Configurable persistence and evidence policy for structural birth."""

    ema_rate: float = 0.25
    error_threshold: float = 0.65
    holdout_transfer_threshold: float = 0.60
    minimum_resource_state: float = 0.40
    minimum_holdout_gain: float = 0.05
    maximum_restructure_holdout_regression: float = 0.05
    required_error_steps: int = 3
    growth_resource_cost: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_rate) <= 1.0:
            raise ValueError("structural growth ema_rate must be in (0, 1]")
        _unit(self.error_threshold, "structural growth error_threshold")
        _unit(self.holdout_transfer_threshold, "structural growth holdout_transfer_threshold")
        _unit(self.minimum_resource_state, "structural growth minimum_resource_state")
        _unit(self.minimum_holdout_gain, "structural growth minimum_holdout_gain")
        _unit(
            self.maximum_restructure_holdout_regression,
            "structural growth maximum_restructure_holdout_regression",
        )
        if int(self.required_error_steps) <= 0:
            raise ValueError("structural growth required_error_steps must be positive")
        if int(self.growth_resource_cost) <= 0:
            raise ValueError("structural growth resource cost must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralGrowthDynamics:
        return cls(**dict(payload))


@dataclass
class StructuralGrowthRegionState:
    """Online evidence state for one substrate region."""

    region_id: str
    error_ema: float = 0.0
    holdout_transfer_ema: float = 0.0
    resource_state_ema: float = 1.0
    consecutive_error_steps: int = 0
    proposal_count: int = 0
    observation_count: int = 0

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("structural growth region_id must not be empty")
        self.error_ema = _unit(self.error_ema, "structural growth error_ema")
        self.holdout_transfer_ema = _unit(
            self.holdout_transfer_ema,
            "structural growth holdout_transfer_ema",
        )
        self.resource_state_ema = _unit(
            self.resource_state_ema,
            "structural growth resource_state_ema",
        )
        if (
            min(
                int(self.consecutive_error_steps),
                int(self.proposal_count),
                int(self.observation_count),
            )
            < 0
        ):
            raise ValueError("structural growth counters cannot be negative")
        self.consecutive_error_steps = int(self.consecutive_error_steps)
        self.proposal_count = int(self.proposal_count)
        self.observation_count = int(self.observation_count)

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralGrowthRegionState:
        return cls(
            region_id=str(payload["region_id"]),
            error_ema=float(payload.get("error_ema", 0.0)),
            holdout_transfer_ema=float(payload.get("holdout_transfer_ema", 0.0)),
            resource_state_ema=float(payload.get("resource_state_ema", 1.0)),
            consecutive_error_steps=int(payload.get("consecutive_error_steps", 0)),
            proposal_count=int(payload.get("proposal_count", 0)),
            observation_count=int(payload.get("observation_count", 0)),
        )


@dataclass(frozen=True)
class StructuralGrowthDecision:
    """An auditable decision to emit a substrate growth proposal."""

    region_id: str
    should_grow: bool
    proposal_ordinal: int
    evidence_ids: tuple[str, ...]
    error_ema: float
    holdout_transfer_ema: float
    resource_state_ema: float
    consecutive_error_steps: int


class AdaptiveStructuralGrowthController:
    """Turn persistent substrate error into proposals without semantic tables.

    The controller never names actions, intents, tokens or tasks.  It only
    tracks regional predictive pressure, available resources and transfer
    evidence.  A proposal is emitted after a configurable persistence window;
    the caller still has to pass it through Taiji's budget/checkpoint/lesion/
    rollback ledger before the new unit becomes live.
    """

    def __init__(
        self,
        *,
        dynamics: StructuralGrowthDynamics | None = None,
    ) -> None:
        self.dynamics = dynamics or StructuralGrowthDynamics()
        self._regions: dict[str, StructuralGrowthRegionState] = {}
        self.total_observations = 0

    @property
    def regions(self) -> tuple[StructuralGrowthRegionState, ...]:
        return tuple(self._regions.values())

    def _region(self, region_id: str) -> StructuralGrowthRegionState:
        key = str(region_id)
        if not key:
            raise ValueError("structural growth region_id must not be empty")
        return self._regions.setdefault(key, StructuralGrowthRegionState(key))

    def observe(
        self,
        region_id: str,
        *,
        prediction_error: float,
        resource_state: float,
        holdout_transfer: float,
        evidence_ids: Sequence[str],
    ) -> StructuralGrowthDecision:
        """Update regional evidence and optionally emit one growth signal."""

        ids = tuple(str(item) for item in evidence_ids)
        if not ids or any(not item for item in ids):
            raise ValueError("structural growth evidence_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("structural growth evidence_ids cannot contain duplicates")
        error = _unit(prediction_error, "structural growth prediction_error")
        resource = _unit(resource_state, "structural growth resource_state")
        transfer = _unit(holdout_transfer, "structural growth holdout_transfer")
        region = self._region(region_id)
        rate = float(self.dynamics.ema_rate)
        region.error_ema = (1.0 - rate) * region.error_ema + rate * error
        region.resource_state_ema = (1.0 - rate) * region.resource_state_ema + rate * resource
        region.holdout_transfer_ema = (1.0 - rate) * region.holdout_transfer_ema + rate * transfer
        region.consecutive_error_steps = (
            region.consecutive_error_steps + 1
            if region.error_ema >= float(self.dynamics.error_threshold)
            else 0
        )
        region.observation_count += 1
        self.total_observations += 1
        should_grow = bool(
            region.consecutive_error_steps >= int(self.dynamics.required_error_steps)
            and region.holdout_transfer_ema >= float(self.dynamics.holdout_transfer_threshold)
            and region.resource_state_ema >= float(self.dynamics.minimum_resource_state)
        )
        if should_grow:
            region.proposal_count += 1
            region.consecutive_error_steps = 0
        return StructuralGrowthDecision(
            region_id=region.region_id,
            should_grow=should_grow,
            proposal_ordinal=region.proposal_count,
            evidence_ids=ids,
            error_ema=region.error_ema,
            holdout_transfer_ema=region.holdout_transfer_ema,
            resource_state_ema=region.resource_state_ema,
            consecutive_error_steps=region.consecutive_error_steps,
        )

    def next_unit_id(self, region_id: str, existing_unit_ids: Sequence[str]) -> str:
        """Allocate a collision-free structural identity, not a semantic label."""

        region = self._region(region_id)
        existing = {str(item) for item in existing_unit_ids}
        ordinal = max(1, region.proposal_count)
        while True:
            candidate = f"{region.region_id}.grown.{ordinal}"
            if candidate not in existing:
                return candidate
            ordinal += 1

    def next_region_id(self, parent_region_id: str, existing_region_ids: Sequence[str]) -> str:
        """Allocate a collision-free child region identity from substrate lineage."""

        parent = str(parent_region_id)
        if not parent:
            raise ValueError("structural growth parent_region_id must not be empty")
        existing = {str(item) for item in existing_region_ids}
        region = self._region(parent)
        ordinal = max(1, region.proposal_count)
        while True:
            candidate = f"{parent}.region.{ordinal}"
            if candidate not in existing:
                return candidate
            ordinal += 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_GROWTH_CHECKPOINT_FORMAT,
            "dynamics": self.dynamics.to_payload(),
            "total_observations": self.total_observations,
            "regions": {
                region_id: region.to_payload() for region_id, region in self._regions.items()
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> AdaptiveStructuralGrowthController:
        if payload.get("format") != STRUCTURAL_GROWTH_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural growth checkpoint format")
        dynamics_payload = payload.get("dynamics", {})
        regions_payload = payload.get("regions", {})
        if not isinstance(dynamics_payload, Mapping) or not isinstance(regions_payload, Mapping):
            raise ValueError("structural growth checkpoint fields must be mappings")
        controller = cls(dynamics=StructuralGrowthDynamics.from_payload(dynamics_payload))
        controller._regions = {
            str(region_id): StructuralGrowthRegionState.from_payload(region_payload)
            for region_id, region_payload in regions_payload.items()
        }
        if set(controller._regions) != {str(key) for key in regions_payload}:
            raise ValueError("structural growth region identities do not match")
        controller.total_observations = int(payload.get("total_observations", 0))
        if controller.total_observations < 0:
            raise ValueError("structural growth total_observations cannot be negative")
        return controller


@dataclass(frozen=True)
class StructuralPruningDynamics:
    """Configurable persistence and evidence policy for structural removal."""

    ema_rate: float = 0.25
    maximum_usage: float = 0.15
    minimum_resource_pressure: float = 0.65
    maximum_learning_gain: float = 0.10
    required_underuse_steps: int = 3
    maximum_holdout_regression: float = 0.05
    pruning_resource_cost: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_rate) <= 1.0:
            raise ValueError("structural pruning ema_rate must be in (0, 1]")
        _unit(self.maximum_usage, "structural pruning maximum_usage")
        _unit(
            self.minimum_resource_pressure,
            "structural pruning minimum_resource_pressure",
        )
        _unit(self.maximum_learning_gain, "structural pruning maximum_learning_gain")
        if int(self.required_underuse_steps) <= 0:
            raise ValueError("structural pruning required_underuse_steps must be positive")
        _unit(
            self.maximum_holdout_regression,
            "structural pruning maximum_holdout_regression",
        )
        if int(self.pruning_resource_cost) <= 0:
            raise ValueError("structural pruning resource cost must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralPruningDynamics:
        return cls(**dict(payload))


@dataclass
class StructuralPruningRegionState:
    """Online evidence state for one candidate structural substrate."""

    region_id: str
    usage_ema: float = 1.0
    resource_pressure_ema: float = 0.0
    learning_gain_ema: float = 1.0
    consecutive_underuse_steps: int = 0
    proposal_count: int = 0
    observation_count: int = 0

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("structural pruning region_id must not be empty")
        self.usage_ema = _unit(self.usage_ema, "structural pruning usage_ema")
        self.resource_pressure_ema = _unit(
            self.resource_pressure_ema,
            "structural pruning resource_pressure_ema",
        )
        self.learning_gain_ema = _unit(
            self.learning_gain_ema,
            "structural pruning learning_gain_ema",
        )
        if (
            min(
                int(self.consecutive_underuse_steps),
                int(self.proposal_count),
                int(self.observation_count),
            )
            < 0
        ):
            raise ValueError("structural pruning counters cannot be negative")
        self.consecutive_underuse_steps = int(self.consecutive_underuse_steps)
        self.proposal_count = int(self.proposal_count)
        self.observation_count = int(self.observation_count)

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralPruningRegionState:
        return cls(
            region_id=str(payload["region_id"]),
            usage_ema=float(payload.get("usage_ema", 1.0)),
            resource_pressure_ema=float(payload.get("resource_pressure_ema", 0.0)),
            learning_gain_ema=float(payload.get("learning_gain_ema", 1.0)),
            consecutive_underuse_steps=int(payload.get("consecutive_underuse_steps", 0)),
            proposal_count=int(payload.get("proposal_count", 0)),
            observation_count=int(payload.get("observation_count", 0)),
        )


@dataclass(frozen=True)
class StructuralPruningDecision:
    """An auditable decision to emit a substrate pruning proposal."""

    region_id: str
    should_prune: bool
    proposal_ordinal: int
    evidence_ids: tuple[str, ...]
    usage_ema: float
    resource_pressure_ema: float
    learning_gain_ema: float
    consecutive_underuse_steps: int


class AdaptiveStructuralPruningController:
    """Turn persistent substrate underuse, pressure and stagnation into signals."""

    def __init__(
        self,
        *,
        dynamics: StructuralPruningDynamics | None = None,
    ) -> None:
        self.dynamics = dynamics or StructuralPruningDynamics()
        self._regions: dict[str, StructuralPruningRegionState] = {}
        self.total_observations = 0

    @property
    def regions(self) -> tuple[StructuralPruningRegionState, ...]:
        return tuple(self._regions.values())

    def _region(self, region_id: str) -> StructuralPruningRegionState:
        key = str(region_id)
        if not key:
            raise ValueError("structural pruning region_id must not be empty")
        return self._regions.setdefault(key, StructuralPruningRegionState(key))

    def observe(
        self,
        region_id: str,
        *,
        usage: float,
        resource_pressure: float,
        learning_gain: float,
        evidence_ids: Sequence[str],
    ) -> StructuralPruningDecision:
        """Update substrate evidence and optionally emit one prune signal."""

        return self.observe_substrate(
            region_id,
            usage=usage,
            resource_pressure=resource_pressure,
            learning_gain=learning_gain,
            evidence_ids=evidence_ids,
        )

    def observe_substrate(
        self,
        substrate_id: str,
        *,
        usage: float,
        resource_pressure: float,
        learning_gain: float,
        evidence_ids: Sequence[str],
    ) -> StructuralPruningDecision:
        """Update any substrate identity, including a region or a connection."""

        ids = tuple(str(item) for item in evidence_ids)
        if not ids or any(not item for item in ids):
            raise ValueError("structural pruning evidence_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("structural pruning evidence_ids cannot contain duplicates")
        usage_value = _unit(usage, "structural pruning usage")
        pressure_value = _unit(resource_pressure, "structural pruning resource_pressure")
        gain_value = _unit(learning_gain, "structural pruning learning_gain")
        region = self._region(substrate_id)
        rate = float(self.dynamics.ema_rate)
        region.usage_ema = (1.0 - rate) * region.usage_ema + rate * usage_value
        region.resource_pressure_ema = (
            1.0 - rate
        ) * region.resource_pressure_ema + rate * pressure_value
        region.learning_gain_ema = (1.0 - rate) * region.learning_gain_ema + rate * gain_value
        region.consecutive_underuse_steps = (
            region.consecutive_underuse_steps + 1
            if (
                region.usage_ema <= float(self.dynamics.maximum_usage)
                and region.resource_pressure_ema >= float(self.dynamics.minimum_resource_pressure)
                and region.learning_gain_ema <= float(self.dynamics.maximum_learning_gain)
            )
            else 0
        )
        region.observation_count += 1
        self.total_observations += 1
        should_prune = region.consecutive_underuse_steps >= int(
            self.dynamics.required_underuse_steps
        )
        if should_prune:
            region.proposal_count += 1
            region.consecutive_underuse_steps = 0
        return StructuralPruningDecision(
            region_id=region.region_id,
            should_prune=should_prune,
            proposal_ordinal=region.proposal_count,
            evidence_ids=ids,
            usage_ema=region.usage_ema,
            resource_pressure_ema=region.resource_pressure_ema,
            learning_gain_ema=region.learning_gain_ema,
            consecutive_underuse_steps=region.consecutive_underuse_steps,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_PRUNING_CHECKPOINT_FORMAT,
            "dynamics": self.dynamics.to_payload(),
            "total_observations": self.total_observations,
            "regions": {
                region_id: region.to_payload() for region_id, region in self._regions.items()
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> AdaptiveStructuralPruningController:
        if payload.get("format") != STRUCTURAL_PRUNING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural pruning checkpoint format")
        dynamics_payload = payload.get("dynamics", {})
        regions_payload = payload.get("regions", {})
        if not isinstance(dynamics_payload, Mapping) or not isinstance(regions_payload, Mapping):
            raise ValueError("structural pruning checkpoint fields must be mappings")
        controller = cls(dynamics=StructuralPruningDynamics.from_payload(dynamics_payload))
        controller._regions = {
            str(region_id): StructuralPruningRegionState.from_payload(region_payload)
            for region_id, region_payload in regions_payload.items()
        }
        if set(controller._regions) != {str(key) for key in regions_payload}:
            raise ValueError("structural pruning region identities do not match")
        controller.total_observations = int(payload.get("total_observations", 0))
        if controller.total_observations < 0:
            raise ValueError("structural pruning total_observations cannot be negative")
        return controller
