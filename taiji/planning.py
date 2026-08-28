"""Goal-directed candidate scoring and outcome-driven progress updates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .config import (
    DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE,
    DEFAULT_RECOVERY_INTERACTION_RESIDUAL_TOLERANCE,
)
from .contracts import Goal, GoalState, Outcome, PlanCandidate, PlanState, WorldAction

PLANNING_CHECKPOINT_FORMAT = "taiji-planning-v1"
RECOVERY_ARCHIVE_CHECKPOINT_FORMAT = "taiji-recovery-archive-v1"
RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT = "taiji-recovery-strategy-ledger-v1"
RECOVERY_READER_DEPENDENCY_CHECKPOINT_FORMAT = "taiji-recovery-reader-dependency-v1"
RECOVERY_READER_ATTRIBUTION_CHECKPOINT_FORMAT = "taiji-recovery-reader-attribution-v1"
RECOVERY_READER_INTERACTION_CHECKPOINT_FORMAT = "taiji-recovery-reader-interaction-v1"
RECOVERY_READER_INTERACTION_GROUP_CHECKPOINT_FORMAT = "taiji-recovery-reader-interaction-group-v1"
RECOVERY_READER_CREDIT_CONSISTENCY_CHECKPOINT_FORMAT = "taiji-recovery-reader-credit-consistency-v1"
RECOVERY_READER_ORDER_INVARIANCE_TOLERANCE = DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE
RECOVERY_READER_CREDIT_CONSERVATION_TOLERANCE = 1e-7
RECOVERY_READER_CREDIT_CONSISTENCY_READER_KINDS = (
    "semantic",
    "procedural",
    "sequence",
    "concept",
)


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _connected_components(
    active_ids: set[str], connections: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic connected components for audited strategy relations."""

    parent = {rollout_id: rollout_id for rollout_id in active_ids}

    def find(rollout_id: str) -> str:
        root = rollout_id
        while parent[root] != root:
            root = parent[root]
        while parent[rollout_id] != rollout_id:
            previous = rollout_id
            rollout_id = parent[rollout_id]
            parent[previous] = root
        return root

    for connection in connections:
        members = tuple(dict.fromkeys(str(item) for item in connection))
        if len(members) < 2 or not set(members) <= active_ids:
            continue
        first_root = find(members[0])
        for member in members[1:]:
            member_root = find(member)
            if first_root != member_root:
                parent[member_root] = first_root
    grouped: dict[str, list[str]] = {}
    for rollout_id in sorted(active_ids):
        grouped.setdefault(find(rollout_id), []).append(rollout_id)
    return tuple(sorted((tuple(group) for group in grouped.values()), key=lambda item: item))


@dataclass(frozen=True)
class PlanningConfig:
    reward_weight: float = 0.60
    progress_weight: float = 1.00
    success_weight: float = 0.80
    uncertainty_weight: float = 1.20
    resource_weight: float = 0.40
    conflict_weight: float = 0.80
    unseen_uncertainty_multiplier: float = 1.50
    stochastic_uncertainty_multiplier: float = 1.00
    conflicted_uncertainty_multiplier: float = 2.00
    concept_weight: float = 0.40
    concept_sequence_weight: float = 0.60
    outcome_progress_gain: float = 0.40
    discount: float = 0.90
    replan_error_threshold: float = 0.25
    recovery_error_threshold: float = 1.00
    world_calibration_quantile: float = 0.95
    world_calibration_std_multiplier: float = 2.0
    world_calibration_margin: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if float(value) < 0.0:
                raise ValueError(f"planning {name} cannot be negative")
        _unit(self.outcome_progress_gain, "outcome_progress_gain")
        _unit(self.discount, "discount")
        _nonnegative_finite(self.replan_error_threshold, "replan_error_threshold")
        _nonnegative_finite(self.recovery_error_threshold, "recovery_error_threshold")
        _unit(self.world_calibration_quantile, "world_calibration_quantile")
        _nonnegative_finite(
            self.world_calibration_std_multiplier,
            "world_calibration_std_multiplier",
        )
        _nonnegative_finite(self.world_calibration_margin, "world_calibration_margin")

    def to_payload(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PlanningConfig:
        return cls(**{name: float(value) for name, value in payload.items()})


@dataclass(frozen=True)
class PlanningCandidate:
    """One executable action plus its current world-model estimates."""

    candidate_id: str
    action: WorldAction
    predicted_reward: float
    success_probability: float
    expected_progress: float
    uncertainty: float = 0.0
    uncertainty_mode: str = "unknown"
    resource_cost: float = 0.0
    conflict: float = 0.0
    concept_affinity: float = 0.0
    prediction_provenance: str = "planner"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("planning candidate_id cannot be empty")
        if not math.isfinite(float(self.predicted_reward)):
            raise ValueError("predicted_reward must be finite")
        _unit(self.success_probability, "success_probability")
        _unit(self.expected_progress, "expected_progress")
        _unit(self.uncertainty, "uncertainty")
        _unit(self.resource_cost, "resource_cost")
        _unit(self.conflict, "conflict")
        _unit(self.concept_affinity, "concept_affinity")
        if not self.uncertainty_mode:
            raise ValueError("planning uncertainty_mode cannot be empty")
        if not self.prediction_provenance:
            raise ValueError("prediction provenance cannot be empty")


@dataclass(frozen=True)
class RecoveryRolloutLineage:
    """The runtime boundary under which a recovery rollout was synthesized."""

    capability_tick: int
    capability_actions: tuple[int, ...]
    capability_action_kinds: tuple[str, ...]
    affordance_id: str
    affordance_content_identity: str
    action_semantic_key: tuple[str, ...]
    schema_revision: int

    def __post_init__(self) -> None:
        if int(self.capability_tick) < 0:
            raise ValueError("recovery capability_tick cannot be negative")
        if not self.capability_actions or len(self.capability_actions) != len(
            self.capability_action_kinds
        ):
            raise ValueError("recovery capability actions and kinds must be aligned")
        if len(set(self.capability_actions)) != len(self.capability_actions):
            raise ValueError("recovery capability actions must be unique")
        if len(set(self.capability_action_kinds)) != len(self.capability_action_kinds):
            raise ValueError("recovery capability action_kinds must be unique")
        if any(int(action) < 0 for action in self.capability_actions):
            raise ValueError("recovery capability actions cannot be negative")
        if any(not kind for kind in self.capability_action_kinds):
            raise ValueError("recovery capability action_kinds cannot be empty")
        if not self.affordance_id:
            raise ValueError("recovery affordance_id cannot be empty")
        if not self.affordance_content_identity:
            raise ValueError("recovery affordance_content_identity cannot be empty")
        if not self.action_semantic_key:
            raise ValueError("recovery action_semantic_key cannot be empty")
        if int(self.schema_revision) < 0:
            raise ValueError("recovery schema_revision cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "capability_tick": self.capability_tick,
            "capability_actions": list(self.capability_actions),
            "capability_action_kinds": list(self.capability_action_kinds),
            "affordance_id": self.affordance_id,
            "affordance_content_identity": self.affordance_content_identity,
            "action_semantic_key": list(self.action_semantic_key),
            "schema_revision": self.schema_revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryRolloutLineage:
        return cls(
            capability_tick=int(payload["capability_tick"]),
            capability_actions=tuple(int(item) for item in payload["capability_actions"]),
            capability_action_kinds=tuple(str(item) for item in payload["capability_action_kinds"]),
            affordance_id=str(payload["affordance_id"]),
            affordance_content_identity=str(payload["affordance_content_identity"]),
            action_semantic_key=tuple(str(item) for item in payload["action_semantic_key"]),
            schema_revision=int(payload["schema_revision"]),
        )


@dataclass(frozen=True)
class PlanningDecision:
    """The scored executable plans and the currently selected action."""

    goal_id: str
    plan: PlanState
    selected: PlanningCandidate


@dataclass(frozen=True)
class ImaginedRollout:
    """A multi-step world-model hypothesis with explicit provenance."""

    rollout_id: str
    goal_id: str
    steps: tuple[PlanningCandidate, ...]
    confidence: float
    provenance: str = "imagined"
    concept_sequence_affinity: float = 0.0
    recovery_lineage: RecoveryRolloutLineage | None = None

    def __post_init__(self) -> None:
        if not self.rollout_id or not self.goal_id:
            raise ValueError("imagined rollout ids cannot be empty")
        if not self.steps:
            raise ValueError("imagined rollout requires at least one step")
        _unit(self.confidence, "rollout confidence")
        _unit(self.concept_sequence_affinity, "concept_sequence_affinity")
        if self.provenance != "imagined":
            raise ValueError("rollout provenance must be imagined")
        if self.recovery_lineage is not None and not isinstance(
            self.recovery_lineage, RecoveryRolloutLineage
        ):
            raise TypeError("recovery_lineage must be a RecoveryRolloutLineage or None")

    def to_payload(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "goal_id": self.goal_id,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "concept_sequence_affinity": self.concept_sequence_affinity,
            "recovery_lineage": (
                None if self.recovery_lineage is None else self.recovery_lineage.to_payload()
            ),
            "steps": [
                {
                    "candidate_id": step.candidate_id,
                    "action": step.action.to_payload(),
                    "predicted_reward": step.predicted_reward,
                    "success_probability": step.success_probability,
                    "expected_progress": step.expected_progress,
                    "uncertainty": step.uncertainty,
                    "uncertainty_mode": step.uncertainty_mode,
                    "resource_cost": step.resource_cost,
                    "conflict": step.conflict,
                    "concept_affinity": step.concept_affinity,
                    "prediction_provenance": step.prediction_provenance,
                }
                for step in self.steps
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ImaginedRollout:
        return cls(
            rollout_id=str(payload["rollout_id"]),
            goal_id=str(payload["goal_id"]),
            confidence=float(payload["confidence"]),
            provenance=str(payload.get("provenance", "imagined")),
            concept_sequence_affinity=float(payload.get("concept_sequence_affinity", 0.0)),
            recovery_lineage=(
                None
                if payload.get("recovery_lineage") is None
                else RecoveryRolloutLineage.from_payload(payload["recovery_lineage"])
            ),
            steps=tuple(
                PlanningCandidate(
                    candidate_id=str(item["candidate_id"]),
                    action=WorldAction.from_payload(item["action"]),
                    predicted_reward=float(item["predicted_reward"]),
                    success_probability=float(item["success_probability"]),
                    expected_progress=float(item["expected_progress"]),
                    uncertainty=float(item.get("uncertainty", 0.0)),
                    uncertainty_mode=str(item.get("uncertainty_mode", "unknown")),
                    resource_cost=float(item.get("resource_cost", 0.0)),
                    conflict=float(item.get("conflict", 0.0)),
                    concept_affinity=float(item.get("concept_affinity", 0.0)),
                    prediction_provenance=str(item.get("prediction_provenance", "planner")),
                )
                for item in payload.get("steps", ())
            ),
        )


@dataclass(frozen=True)
class RecoveryPortfolio:
    """Checkpointable portfolio of parallel recovery rollout branches."""

    portfolio_id: str
    goal_id: str
    candidates: tuple[ImaginedRollout, ...]
    statuses: tuple[tuple[str, str], ...] = ()
    retired_rollout_ids: tuple[str, ...] = ()
    selected_rollout_id: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.portfolio_id or not self.goal_id:
            raise ValueError("recovery portfolio ids cannot be empty")
        if not self.candidates:
            raise ValueError("recovery portfolio requires candidates")
        candidate_ids = tuple(candidate.rollout_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("recovery portfolio candidate rollout ids must be unique")
        if not self.statuses:
            object.__setattr__(
                self,
                "statuses",
                tuple((rollout_id, "active") for rollout_id in candidate_ids),
            )
        status_ids = tuple(rollout_id for rollout_id, _ in self.statuses)
        if status_ids != candidate_ids:
            raise ValueError("recovery portfolio statuses must align with candidates")
        allowed = {"active", "selected", "pruned", "expired"}
        if any(status not in allowed for _, status in self.statuses):
            raise ValueError("unsupported recovery portfolio branch status")
        if self.selected_rollout_id is not None and self.selected_rollout_id not in candidate_ids:
            raise ValueError("recovery portfolio selected rollout is missing")
        if any(not rollout_id for rollout_id in self.retired_rollout_ids):
            raise ValueError("recovery portfolio retired rollout ids cannot be empty")
        if len(set(self.retired_rollout_ids)) != len(self.retired_rollout_ids):
            raise ValueError("recovery portfolio retired rollout ids must be unique")
        if int(self.revision) < 0:
            raise ValueError("recovery portfolio revision cannot be negative")

    def status_for(self, rollout_id: str) -> str | None:
        return dict(self.statuses).get(rollout_id)

    def active_candidates(self) -> tuple[ImaginedRollout, ...]:
        active = {
            rollout_id for rollout_id, status in self.statuses if status in {"active", "selected"}
        }
        return tuple(candidate for candidate in self.candidates if candidate.rollout_id in active)

    def mark_selected(self, rollout_id: str) -> RecoveryPortfolio:
        if self.status_for(rollout_id) not in {"active", "selected"}:
            raise ValueError("recovery portfolio cannot select an unknown rollout")
        statuses = tuple(
            (
                candidate_id,
                (
                    "selected"
                    if candidate_id == rollout_id
                    else "active" if status == "selected" else status
                ),
            )
            for candidate_id, status in self.statuses
        )
        return replace(
            self,
            statuses=statuses,
            selected_rollout_id=rollout_id,
            revision=self.revision + 1,
        )

    def mark_expired(self, *rollout_ids: str) -> RecoveryPortfolio:
        expired = set(rollout_ids)
        statuses = tuple(
            (candidate_id, "expired" if candidate_id in expired else status)
            for candidate_id, status in self.statuses
        )
        return replace(
            self,
            statuses=statuses,
            selected_rollout_id=(
                None if self.selected_rollout_id in expired else self.selected_rollout_id
            ),
            revision=self.revision + 1,
        )

    def mark_pruned(self, *rollout_ids: str) -> RecoveryPortfolio:
        pruned = set(rollout_ids)
        statuses = tuple(
            (candidate_id, "pruned" if candidate_id in pruned else status)
            for candidate_id, status in self.statuses
        )
        return replace(
            self,
            statuses=statuses,
            selected_rollout_id=(
                None if self.selected_rollout_id in pruned else self.selected_rollout_id
            ),
            revision=self.revision + 1,
        )

    def replace_candidate(self, rollout: ImaginedRollout) -> RecoveryPortfolio:
        if self.status_for(rollout.rollout_id) is None:
            return self
        candidates = tuple(
            rollout if candidate.rollout_id == rollout.rollout_id else candidate
            for candidate in self.candidates
        )
        return replace(self, candidates=candidates, revision=self.revision + 1)

    def to_payload(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "goal_id": self.goal_id,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "statuses": [[rollout_id, status] for rollout_id, status in self.statuses],
            "retired_rollout_ids": list(self.retired_rollout_ids),
            "selected_rollout_id": self.selected_rollout_id,
            "revision": self.revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryPortfolio:
        return cls(
            portfolio_id=str(payload["portfolio_id"]),
            goal_id=str(payload["goal_id"]),
            candidates=tuple(
                ImaginedRollout.from_payload(dict(item)) for item in payload.get("candidates", ())
            ),
            statuses=tuple((str(item[0]), str(item[1])) for item in payload.get("statuses", ())),
            retired_rollout_ids=tuple(str(item) for item in payload.get("retired_rollout_ids", ())),
            selected_rollout_id=(
                None
                if payload.get("selected_rollout_id") is None
                else str(payload["selected_rollout_id"])
            ),
            revision=int(payload.get("revision", 0)),
        )

    def archive_entries(
        self,
        source_episode_id: str,
        *,
        completed_rollout_id: str | None = None,
        outcome_reward: float | None = None,
        outcome_success: bool | None = None,
        terminal: bool = False,
        evidence_count: int = 0,
        outcome_consistency: float = 1.0,
    ) -> tuple[RecoveryArchiveEntry, ...]:
        """Summarize branches without retaining executable candidates."""

        if not source_episode_id:
            raise ValueError("recovery archive source episode cannot be empty")
        entries: list[RecoveryArchiveEntry] = []
        for candidate in self.candidates:
            status = self.status_for(candidate.rollout_id)
            if status in {"pruned", "expired"}:
                lifecycle = "expired"
            elif (
                status == "selected"
                and candidate.rollout_id == completed_rollout_id
                and terminal
                and outcome_success is True
            ):
                lifecycle = "completed"
            else:
                lifecycle = "abandoned"
            lineage = candidate.recovery_lineage
            entries.append(
                RecoveryArchiveEntry(
                    portfolio_id=self.portfolio_id,
                    rollout_id=candidate.rollout_id,
                    source_episode_id=source_episode_id,
                    goal_id=self.goal_id,
                    lifecycle=lifecycle,
                    action_kinds=tuple(step.action.kind for step in candidate.steps),
                    capability_tick=(None if lineage is None else lineage.capability_tick),
                    affordance_id="" if lineage is None else lineage.affordance_id,
                    affordance_content_identity=(
                        "" if lineage is None else lineage.affordance_content_identity
                    ),
                    resource_cost=sum(step.resource_cost for step in candidate.steps),
                    outcome_reward=(
                        outcome_reward if candidate.rollout_id == completed_rollout_id else None
                    ),
                    outcome_success=(
                        outcome_success if candidate.rollout_id == completed_rollout_id else None
                    ),
                    terminal=bool(terminal and candidate.rollout_id == completed_rollout_id),
                    evidence_count=(
                        int(evidence_count) if candidate.rollout_id == completed_rollout_id else 0
                    ),
                    outcome_consistency=(
                        float(outcome_consistency)
                        if candidate.rollout_id == completed_rollout_id
                        else 0.0
                    ),
                    portfolio_revision=self.revision,
                )
            )
        return tuple(entries)


@dataclass(frozen=True)
class RecoveryArchiveEntry:
    """Non-executable cross-episode summary of one recovery branch."""

    portfolio_id: str
    rollout_id: str
    source_episode_id: str
    goal_id: str
    lifecycle: str
    action_kinds: tuple[str, ...]
    capability_tick: int | None = None
    affordance_id: str = ""
    affordance_content_identity: str = ""
    resource_cost: float = 0.0
    outcome_reward: float | None = None
    outcome_success: bool | None = None
    terminal: bool = False
    evidence_count: int = 0
    outcome_consistency: float = 0.0
    portfolio_revision: int = 0

    def __post_init__(self) -> None:
        if not self.portfolio_id or not self.rollout_id:
            raise ValueError("recovery archive ids cannot be empty")
        if not self.source_episode_id or not self.goal_id:
            raise ValueError("recovery archive lineage ids cannot be empty")
        if self.lifecycle not in {"completed", "abandoned", "expired"}:
            raise ValueError("unsupported recovery archive lifecycle")
        if not self.action_kinds or any(not kind for kind in self.action_kinds):
            raise ValueError("recovery archive action_kinds cannot be empty")
        if self.capability_tick is not None and int(self.capability_tick) < 0:
            raise ValueError("recovery archive capability_tick cannot be negative")
        _nonnegative_finite(self.resource_cost, "recovery archive resource_cost")
        if self.outcome_reward is not None and not math.isfinite(float(self.outcome_reward)):
            raise ValueError("recovery archive outcome_reward must be finite")
        if int(self.evidence_count) < 0:
            raise ValueError("recovery archive evidence_count cannot be negative")
        _unit(self.outcome_consistency, "recovery archive outcome_consistency")
        if int(self.portfolio_revision) < 0:
            raise ValueError("recovery archive portfolio_revision cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "portfolio_id": self.portfolio_id,
            "rollout_id": self.rollout_id,
            "source_episode_id": self.source_episode_id,
            "goal_id": self.goal_id,
            "lifecycle": self.lifecycle,
            "action_kinds": list(self.action_kinds),
            "capability_tick": self.capability_tick,
            "affordance_id": self.affordance_id,
            "affordance_content_identity": self.affordance_content_identity,
            "resource_cost": self.resource_cost,
            "outcome_reward": self.outcome_reward,
            "outcome_success": self.outcome_success,
            "terminal": self.terminal,
            "evidence_count": self.evidence_count,
            "outcome_consistency": self.outcome_consistency,
            "portfolio_revision": self.portfolio_revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryArchiveEntry:
        return cls(
            portfolio_id=str(payload["portfolio_id"]),
            rollout_id=str(payload["rollout_id"]),
            source_episode_id=str(payload["source_episode_id"]),
            goal_id=str(payload["goal_id"]),
            lifecycle=str(payload["lifecycle"]),
            action_kinds=tuple(str(item) for item in payload["action_kinds"]),
            capability_tick=(
                None if payload.get("capability_tick") is None else int(payload["capability_tick"])
            ),
            affordance_id=str(payload.get("affordance_id", "")),
            affordance_content_identity=str(payload.get("affordance_content_identity", "")),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            outcome_reward=(
                None if payload.get("outcome_reward") is None else float(payload["outcome_reward"])
            ),
            outcome_success=(
                None if payload.get("outcome_success") is None else bool(payload["outcome_success"])
            ),
            terminal=bool(payload.get("terminal", False)),
            evidence_count=int(payload.get("evidence_count", 0)),
            outcome_consistency=float(payload.get("outcome_consistency", 0.0)),
            portfolio_revision=int(payload.get("portfolio_revision", 0)),
        )


@dataclass(frozen=True)
class RecoveryPortfolioArchive:
    """Bounded cross-episode recovery memory; entries are never executable."""

    entries: tuple[RecoveryArchiveEntry, ...] = ()
    capacity: int = 256
    revision: int = 0

    def __post_init__(self) -> None:
        if int(self.capacity) <= 0:
            raise ValueError("recovery archive capacity must be positive")
        if int(self.revision) < 0:
            raise ValueError("recovery archive revision cannot be negative")
        keys = tuple((entry.source_episode_id, entry.rollout_id) for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("recovery archive entries must be unique")
        if len(self.entries) > int(self.capacity):
            raise ValueError("recovery archive entries exceed capacity")

    def append(self, new_entries: Sequence[RecoveryArchiveEntry]) -> RecoveryPortfolioArchive:
        additions = tuple(new_entries)
        if not additions:
            return self
        merged = list(self.entries)
        positions = {
            (entry.source_episode_id, entry.rollout_id): index for index, entry in enumerate(merged)
        }
        for entry in additions:
            key = (entry.source_episode_id, entry.rollout_id)
            if key in positions:
                merged[positions[key]] = entry
            else:
                positions[key] = len(merged)
                merged.append(entry)
        retained = tuple(merged[-int(self.capacity) :])
        return replace(self, entries=retained, revision=self.revision + 1)

    @property
    def archived_rollout_ids(self) -> tuple[str, ...]:
        return tuple(entry.rollout_id for entry in self.entries)

    def lifecycle_for(self, rollout_id: str) -> str | None:
        for entry in reversed(self.entries):
            if entry.rollout_id == rollout_id:
                return entry.lifecycle
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_ARCHIVE_CHECKPOINT_FORMAT,
            "entries": [entry.to_payload() for entry in self.entries],
            "capacity": self.capacity,
            "revision": self.revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryPortfolioArchive:
        if payload.get("format", RECOVERY_ARCHIVE_CHECKPOINT_FORMAT) != (
            RECOVERY_ARCHIVE_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery archive checkpoint format")
        return cls(
            entries=tuple(
                RecoveryArchiveEntry.from_payload(dict(item)) for item in payload.get("entries", ())
            ),
            capacity=int(payload.get("capacity", 256)),
            revision=int(payload.get("revision", 0)),
        )


@dataclass(frozen=True)
class RecoveryStrategyApproval:
    """Evidence-gated admission of one completed branch into long-term memory."""

    approval_id: str
    rollout_id: str
    source_episode_id: str
    goal_id: str
    memory_id: str
    action_kinds: tuple[str, ...]
    evidence_count: int
    outcome_reward: float
    resource_cost: float = 0.0
    outcome_consistency: float = 1.0
    revision: int = 0

    def __post_init__(self) -> None:
        if not all(
            (
                self.approval_id,
                self.rollout_id,
                self.source_episode_id,
                self.goal_id,
                self.memory_id,
            )
        ):
            raise ValueError("recovery strategy approval ids cannot be empty")
        if not self.action_kinds or any(not kind for kind in self.action_kinds):
            raise ValueError("recovery strategy approval action_kinds cannot be empty")
        if int(self.evidence_count) <= 0:
            raise ValueError("recovery strategy approval evidence_count must be positive")
        if not math.isfinite(float(self.outcome_reward)):
            raise ValueError("recovery strategy approval outcome_reward must be finite")
        _nonnegative_finite(self.resource_cost, "recovery strategy approval resource_cost")
        _unit(self.outcome_consistency, "recovery strategy approval outcome_consistency")
        if int(self.revision) < 0:
            raise ValueError("recovery strategy approval revision cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "rollout_id": self.rollout_id,
            "source_episode_id": self.source_episode_id,
            "goal_id": self.goal_id,
            "memory_id": self.memory_id,
            "action_kinds": list(self.action_kinds),
            "evidence_count": self.evidence_count,
            "outcome_reward": self.outcome_reward,
            "resource_cost": self.resource_cost,
            "outcome_consistency": self.outcome_consistency,
            "revision": self.revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryStrategyApproval:
        return cls(
            approval_id=str(payload["approval_id"]),
            rollout_id=str(payload["rollout_id"]),
            source_episode_id=str(payload["source_episode_id"]),
            goal_id=str(payload["goal_id"]),
            memory_id=str(payload["memory_id"]),
            action_kinds=tuple(str(item) for item in payload["action_kinds"]),
            evidence_count=int(payload["evidence_count"]),
            outcome_reward=float(payload["outcome_reward"]),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            outcome_consistency=float(payload.get("outcome_consistency", 1.0)),
            revision=int(payload.get("revision", 0)),
        )


@dataclass(frozen=True)
class _RecoverySelectionUnit:
    approvals: tuple[RecoveryStrategyApproval, ...]
    score: float
    evidence_count: int
    resource_cost: float
    rollout_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryStrategyLedger:
    """Admission and revocation ledger for recovery-derived long-term memory."""

    evidence_threshold: int = 2
    memory_budget: float = 1.0
    evidence_weight: float = 0.5
    consistency_weight: float = 0.3
    resource_weight: float = 0.2
    approvals: tuple[RecoveryStrategyApproval, ...] = ()
    revoked_rollout_ids: tuple[str, ...] = ()
    revision: int = 0
    interaction_residual_tolerance: float = DEFAULT_RECOVERY_INTERACTION_RESIDUAL_TOLERANCE
    interaction_order_tolerance: float = DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE
    interaction_audit_available: bool = False
    interactions: tuple[RecoveryReaderInteraction, ...] = ()
    interaction_groups: tuple[RecoveryReaderInteractionGroup, ...] = ()

    def __post_init__(self) -> None:
        if int(self.evidence_threshold) <= 0:
            raise ValueError("recovery strategy evidence_threshold must be positive")
        _nonnegative_finite(self.memory_budget, "recovery strategy memory_budget")
        if float(self.memory_budget) <= 0.0:
            raise ValueError("recovery strategy memory_budget must be positive")
        weights = (
            float(self.evidence_weight),
            float(self.consistency_weight),
            float(self.resource_weight),
        )
        if any(weight <= 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("recovery strategy competition weights must be positive and sum to 1")
        approval_ids = tuple(approval.approval_id for approval in self.approvals)
        if len(set(approval_ids)) != len(approval_ids):
            raise ValueError("recovery strategy approvals must be unique")
        if any(not rollout_id for rollout_id in self.revoked_rollout_ids):
            raise ValueError("recovery strategy revoked rollout ids cannot be empty")
        if len(set(self.revoked_rollout_ids)) != len(self.revoked_rollout_ids):
            raise ValueError("recovery strategy revoked rollout ids must be unique")
        if int(self.revision) < 0:
            raise ValueError("recovery strategy ledger revision cannot be negative")
        _nonnegative_finite(
            self.interaction_residual_tolerance, "recovery strategy interaction residual tolerance"
        )
        _nonnegative_finite(
            self.interaction_order_tolerance, "recovery strategy interaction order tolerance"
        )
        if not isinstance(self.interaction_audit_available, bool):
            raise ValueError("recovery strategy interaction audit flag must be boolean")
        if any(not isinstance(item, RecoveryReaderInteraction) for item in self.interactions):
            raise ValueError("recovery strategy interactions must be typed records")
        if any(
            not isinstance(item, RecoveryReaderInteractionGroup) for item in self.interaction_groups
        ):
            raise ValueError("recovery strategy interaction groups must be typed records")

    def admit(
        self,
        entry: RecoveryArchiveEntry,
        *,
        memory_id: str,
    ) -> RecoveryStrategyLedger:
        """Admit only a completed, terminal, evidence-backed archive entry."""

        if entry.lifecycle != "completed" or not entry.terminal:
            raise ValueError("only completed terminal recovery can enter strategy memory")
        if int(entry.evidence_count) < int(self.evidence_threshold):
            return self
        if not memory_id:
            raise ValueError("recovery strategy approval memory_id cannot be empty")
        if entry.rollout_id in self.revoked_rollout_ids:
            return self
        approval = RecoveryStrategyApproval(
            approval_id=f"{entry.source_episode_id}:{entry.rollout_id}",
            rollout_id=entry.rollout_id,
            source_episode_id=entry.source_episode_id,
            goal_id=entry.goal_id,
            memory_id=memory_id,
            action_kinds=entry.action_kinds,
            evidence_count=entry.evidence_count,
            outcome_reward=0.0 if entry.outcome_reward is None else entry.outcome_reward,
            resource_cost=entry.resource_cost,
            outcome_consistency=entry.outcome_consistency,
            revision=self.revision + 1,
        )
        approvals = tuple(
            existing for existing in self.approvals if existing.approval_id != approval.approval_id
        )
        return replace(
            self,
            approvals=(*approvals, approval),
            revision=self.revision + 1,
        )

    def revoke(self, rollout_id: str) -> RecoveryStrategyLedger:
        if not rollout_id:
            raise ValueError("recovery strategy rollout_id cannot be empty")
        if rollout_id in self.revoked_rollout_ids:
            return self
        return replace(
            self,
            revoked_rollout_ids=(*self.revoked_rollout_ids, rollout_id),
            revision=self.revision + 1,
        )

    def active_approvals(self) -> tuple[RecoveryStrategyApproval, ...]:
        revoked = set(self.revoked_rollout_ids)
        return tuple(approval for approval in self.approvals if approval.rollout_id not in revoked)

    def is_active(self, rollout_id: str) -> bool:
        return any(approval.rollout_id == rollout_id for approval in self.active_approvals())

    def competition_score(self, approval: RecoveryStrategyApproval) -> float:
        """Score one active strategy using configured evidence and resource signals."""

        evidence_signal = min(
            1.0,
            float(approval.evidence_count) / float(self.evidence_threshold),
        )
        resource_signal = 1.0 - min(
            1.0,
            float(approval.resource_cost) / float(self.memory_budget),
        )
        return (
            float(self.evidence_weight) * evidence_signal
            + float(self.consistency_weight) * float(approval.outcome_consistency)
            + float(self.resource_weight) * resource_signal
        )

    @property
    def ranked_approvals(self) -> tuple[RecoveryStrategyApproval, ...]:
        """Return active strategies in deterministic competition order."""

        return tuple(
            sorted(
                self.active_approvals(),
                key=lambda approval: (
                    -self.competition_score(approval),
                    -int(approval.evidence_count),
                    float(approval.resource_cost),
                    approval.approval_id,
                ),
            )
        )

    def _select_ranked_approvals(self) -> tuple[RecoveryStrategyApproval, ...]:
        selected: list[RecoveryStrategyApproval] = []
        selected_memory_ids: set[str] = set()
        consumed = 0.0
        for approval in self.ranked_approvals:
            cost = float(approval.resource_cost)
            if approval.memory_id in selected_memory_ids:
                continue
            if consumed + cost > float(self.memory_budget) + 1e-8:
                continue
            selected.append(approval)
            selected_memory_ids.add(approval.memory_id)
            consumed += cost
        return tuple(selected)

    @property
    def selected_approvals(self) -> tuple[RecoveryStrategyApproval, ...]:
        """Greedily select ranked strategies without exceeding memory budget."""

        return self.select_with_interaction_audit(
            self.interactions,
            self.interaction_groups,
            audit_available=self.interaction_audit_available,
            residual_tolerance=self.interaction_residual_tolerance,
            order_tolerance=self.interaction_order_tolerance,
        )

    def select_with_interaction_audit(
        self,
        interactions: Sequence[RecoveryReaderInteraction] = (),
        groups: Sequence[RecoveryReaderInteractionGroup] = (),
        *,
        audit_available: bool = False,
        residual_tolerance: float,
        order_tolerance: float,
    ) -> tuple[RecoveryStrategyApproval, ...]:
        """Select strategies while treating unproven interactions as atomic.

        Before a reader has an interaction audit, the normal deterministic
        competition policy is used to bootstrap the first consolidation. Once
        an audit exists, a pair is independently selectable only when its
        residual and order delta are within the configured tolerances. Missing
        pair evidence is fail-closed: the pair is joined into an atomic unit
        instead of being assumed additive. Connected atomic pairs are selected
        together, or rejected together when their combined cost exceeds the
        memory budget.
        """

        residual_limit = _nonnegative_finite(
            residual_tolerance, "recovery interaction residual_tolerance"
        )
        order_limit = _nonnegative_finite(order_tolerance, "recovery interaction order_tolerance")
        active = self.active_approvals()
        group_items = tuple(groups)
        if any(not isinstance(item, RecoveryReaderInteractionGroup) for item in group_items):
            raise ValueError("recovery strategy interaction groups must be typed records")
        if len(active) < 2 or (not interactions and not group_items and not audit_available):
            return self._select_ranked_approvals()

        active_ids = {approval.rollout_id for approval in active}
        audited_pairs: set[frozenset[str]] = set()
        atomic_connections: list[tuple[str, ...]] = []
        for interaction in interactions:
            pair = frozenset(interaction.strategy_rollout_ids)
            if len(pair) != 2 or not pair <= active_ids:
                continue
            audited_pairs.add(pair)
            if (
                interaction.interaction_residual_l2 > residual_limit
                or interaction.order_delta_l2 > order_limit
                or not interaction.order_invariant
            ):
                atomic_connections.append(tuple(pair))

        for group in group_items:
            group_ids = frozenset(group.strategy_rollout_ids)
            if len(group_ids) < 3 or not group_ids <= active_ids:
                continue
            if (
                group.higher_order_residual_l2 > residual_limit
                or group.order_delta_l2 > order_limit
                or not group.order_invariant
                or not group.credit_decomposition_safe
            ):
                atomic_connections.append(tuple(group_ids))

        for index, first in enumerate(active):
            for second in active[index + 1 :]:
                pair = frozenset((first.rollout_id, second.rollout_id))
                if pair not in audited_pairs:
                    atomic_connections.append(tuple(pair))

        rank = {approval.rollout_id: index for index, approval in enumerate(self.ranked_approvals)}
        grouped: dict[str, list[RecoveryStrategyApproval]] = {}
        for approval in active:
            grouped.setdefault(approval.rollout_id, []).append(approval)
        for component in _connected_components(active_ids, atomic_connections):
            root = component[0]
            for rollout_id in component[1:]:
                grouped.setdefault(root, []).extend(grouped.pop(rollout_id, ()))

        units: list[_RecoverySelectionUnit] = []
        for grouped_approvals in grouped.values():
            ordered = tuple(
                sorted(grouped_approvals, key=lambda approval: rank[approval.rollout_id])
            )
            memory_ids = tuple(approval.memory_id for approval in ordered)
            if len(set(memory_ids)) != len(memory_ids):
                continue
            units.append(
                _RecoverySelectionUnit(
                    approvals=ordered,
                    score=min(self.competition_score(approval) for approval in ordered),
                    evidence_count=min(int(approval.evidence_count) for approval in ordered),
                    resource_cost=sum(float(approval.resource_cost) for approval in ordered),
                    rollout_ids=tuple(approval.rollout_id for approval in ordered),
                )
            )

        selected: list[RecoveryStrategyApproval] = []
        selected_memory_ids: set[str] = set()
        consumed = 0.0
        for unit in sorted(
            units,
            key=lambda item: (
                -item.score,
                -item.evidence_count,
                item.resource_cost,
                item.rollout_ids,
            ),
        ):
            if any(approval.memory_id in selected_memory_ids for approval in unit.approvals):
                continue
            if consumed + unit.resource_cost > float(self.memory_budget) + 1e-8:
                continue
            selected.extend(unit.approvals)
            selected_memory_ids.update(approval.memory_id for approval in unit.approvals)
            consumed += unit.resource_cost
        return tuple(selected)

    def record_interaction_audit(
        self,
        interactions: Sequence[RecoveryReaderInteraction],
        *,
        groups: Sequence[RecoveryReaderInteractionGroup] = (),
        audit_available: bool | None = None,
    ) -> RecoveryStrategyLedger:
        """Persist the latest reader audit used by the canonical selector."""

        items = tuple(interactions)
        if any(not isinstance(item, RecoveryReaderInteraction) for item in items):
            raise ValueError("recovery strategy interactions must be typed records")
        group_items = tuple(groups)
        if any(not isinstance(item, RecoveryReaderInteractionGroup) for item in group_items):
            raise ValueError("recovery strategy interaction groups must be typed records")
        if audit_available is None:
            audit = bool(items or group_items) or self.interaction_audit_available
        else:
            if not isinstance(audit_available, bool):
                raise ValueError("recovery strategy interaction audit flag must be boolean")
            audit = audit_available
        return replace(
            self,
            interaction_audit_available=audit,
            interactions=items,
            interaction_groups=group_items,
            revision=self.revision + 1,
        )

    @property
    def selected_rollout_ids(self) -> tuple[str, ...]:
        return tuple(approval.rollout_id for approval in self.selected_approvals)

    @property
    def approved_memory_ids(self) -> tuple[str, ...]:
        return tuple(approval.memory_id for approval in self.active_approvals())

    @property
    def active_memory_ids(self) -> tuple[str, ...]:
        """Return memory IDs selected for downstream consolidation/readout."""

        return tuple(approval.memory_id for approval in self.selected_approvals)

    @property
    def selected_memory_ids(self) -> tuple[str, ...]:
        return self.active_memory_ids

    @property
    def revoked_memory_ids(self) -> tuple[str, ...]:
        revoked = set(self.revoked_rollout_ids)
        return tuple(
            approval.memory_id for approval in self.approvals if approval.rollout_id in revoked
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT,
            "evidence_threshold": self.evidence_threshold,
            "memory_budget": self.memory_budget,
            "evidence_weight": self.evidence_weight,
            "consistency_weight": self.consistency_weight,
            "resource_weight": self.resource_weight,
            "approvals": [approval.to_payload() for approval in self.approvals],
            "revoked_rollout_ids": list(self.revoked_rollout_ids),
            "revision": self.revision,
            "interaction_residual_tolerance": self.interaction_residual_tolerance,
            "interaction_order_tolerance": self.interaction_order_tolerance,
            "interaction_audit_available": self.interaction_audit_available,
            "interactions": [item.to_payload() for item in self.interactions],
            "interaction_groups": [item.to_payload() for item in self.interaction_groups],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryStrategyLedger:
        if payload.get("format", RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT) != (
            RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery strategy ledger checkpoint format")
        return cls(
            evidence_threshold=int(payload.get("evidence_threshold", 2)),
            memory_budget=float(payload.get("memory_budget", 1.0)),
            evidence_weight=float(payload.get("evidence_weight", 0.5)),
            consistency_weight=float(payload.get("consistency_weight", 0.3)),
            resource_weight=float(payload.get("resource_weight", 0.2)),
            approvals=tuple(
                RecoveryStrategyApproval.from_payload(dict(item))
                for item in payload.get("approvals", ())
            ),
            revoked_rollout_ids=tuple(str(item) for item in payload.get("revoked_rollout_ids", ())),
            revision=int(payload.get("revision", 0)),
            interaction_residual_tolerance=float(
                payload.get(
                    "interaction_residual_tolerance",
                    DEFAULT_RECOVERY_INTERACTION_RESIDUAL_TOLERANCE,
                )
            ),
            interaction_order_tolerance=float(
                payload.get(
                    "interaction_order_tolerance", DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE
                )
            ),
            interaction_audit_available=bool(
                payload.get("interaction_audit_available", bool(payload.get("interactions", ())))
            ),
            interactions=tuple(
                RecoveryReaderInteraction.from_payload(dict(item))
                for item in payload.get("interactions", ())
            ),
            interaction_groups=tuple(
                RecoveryReaderInteractionGroup.from_payload(dict(item))
                for item in payload.get("interaction_groups", ())
            ),
        )


@dataclass(frozen=True)
class RecoveryReaderContribution:
    """Leave-one-out effect of one strategy on one downstream reader."""

    reader_kind: str
    strategy_rollout_id: str
    memory_id: str
    effect_delta_l2: float
    credit: float
    replay_epochs: int
    replay_learning_rate: float
    method: str = "leave-one-out"

    def __post_init__(self) -> None:
        if not self.reader_kind:
            raise ValueError("recovery reader contribution reader_kind cannot be empty")
        if not self.strategy_rollout_id:
            raise ValueError("recovery reader contribution rollout id cannot be empty")
        if not self.memory_id:
            raise ValueError("recovery reader contribution memory id cannot be empty")
        if not math.isfinite(float(self.effect_delta_l2)) or float(self.effect_delta_l2) < 0.0:
            raise ValueError("recovery reader contribution effect must be finite and non-negative")
        _unit(self.credit, "recovery reader contribution credit")
        if int(self.replay_epochs) <= 0:
            raise ValueError("recovery reader contribution replay_epochs must be positive")
        if (
            not math.isfinite(float(self.replay_learning_rate))
            or float(self.replay_learning_rate) <= 0.0
        ):
            raise ValueError(
                "recovery reader contribution replay_learning_rate must be positive and finite"
            )
        if not self.method:
            raise ValueError("recovery reader contribution method cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_READER_ATTRIBUTION_CHECKPOINT_FORMAT,
            "reader_kind": self.reader_kind,
            "strategy_rollout_id": self.strategy_rollout_id,
            "memory_id": self.memory_id,
            "effect_delta_l2": self.effect_delta_l2,
            "credit": self.credit,
            "replay_epochs": self.replay_epochs,
            "replay_learning_rate": self.replay_learning_rate,
            "method": self.method,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderContribution:
        if payload.get("format", RECOVERY_READER_ATTRIBUTION_CHECKPOINT_FORMAT) != (
            RECOVERY_READER_ATTRIBUTION_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery reader attribution checkpoint format")
        return cls(
            reader_kind=str(payload["reader_kind"]),
            strategy_rollout_id=str(payload["strategy_rollout_id"]),
            memory_id=str(payload["memory_id"]),
            effect_delta_l2=float(payload.get("effect_delta_l2", 0.0)),
            credit=float(payload.get("credit", 0.0)),
            replay_epochs=int(payload.get("replay_epochs", 1)),
            replay_learning_rate=float(payload.get("replay_learning_rate", 0.1)),
            method=str(payload.get("method", "leave-one-out")),
        )


@dataclass(frozen=True)
class RecoveryReaderInteraction:
    """Replayable pairwise effect between two strategies on one reader.

    The effect values are deterministic L2-like distances from the same
    baseline reader checkpoint. ``interaction_delta_l2`` is signed: positive
    means the pair has more effect than the two isolated effects added
    together, while negative means interference. ``order_delta_l2`` records
    whether replaying the pair in the opposite order changes the reader.
    """

    reader_kind: str
    strategy_rollout_ids: tuple[str, str]
    memory_ids: tuple[str, str]
    pair_effect_l2: float
    additive_effect_l2: float
    interaction_delta_l2: float
    interaction_residual_l2: float
    order_delta_l2: float
    order_invariant: bool
    replay_epochs: int
    replay_learning_rate: float
    method: str = "pairwise-replay"
    replay_action_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reader_kind:
            raise ValueError("recovery reader interaction reader_kind cannot be empty")
        if len(self.strategy_rollout_ids) != 2 or any(
            not rollout_id for rollout_id in self.strategy_rollout_ids
        ):
            raise ValueError("recovery reader interaction needs two rollout ids")
        if len(set(self.strategy_rollout_ids)) != 2:
            raise ValueError("recovery reader interaction rollout ids must be distinct")
        if len(self.memory_ids) != 2 or any(not memory_id for memory_id in self.memory_ids):
            raise ValueError("recovery reader interaction needs two memory ids")
        if len(set(self.memory_ids)) != 2:
            raise ValueError("recovery reader interaction memory ids must be distinct")
        for value, name in (
            (self.pair_effect_l2, "pair effect"),
            (self.additive_effect_l2, "additive effect"),
            (self.interaction_residual_l2, "interaction residual"),
            (self.order_delta_l2, "order delta"),
        ):
            _nonnegative_finite(value, f"recovery reader interaction {name}")
        if not math.isfinite(float(self.interaction_delta_l2)):
            raise ValueError("recovery reader interaction delta must be finite")
        if not math.isclose(
            float(self.interaction_residual_l2),
            abs(float(self.interaction_delta_l2)),
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("recovery reader interaction residual must match delta")
        if bool(self.order_invariant) != (
            float(self.order_delta_l2) <= RECOVERY_READER_ORDER_INVARIANCE_TOLERANCE
        ):
            raise ValueError("recovery reader interaction order invariance does not match delta")
        if int(self.replay_epochs) <= 0:
            raise ValueError("recovery reader interaction replay_epochs must be positive")
        if (
            not math.isfinite(float(self.replay_learning_rate))
            or float(self.replay_learning_rate) <= 0.0
        ):
            raise ValueError(
                "recovery reader interaction replay_learning_rate must be positive and finite"
            )
        if not self.method:
            raise ValueError("recovery reader interaction method cannot be empty")
        if any(not action_kind for action_kind in self.replay_action_kinds):
            raise ValueError("recovery reader interaction action kinds cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_READER_INTERACTION_CHECKPOINT_FORMAT,
            "reader_kind": self.reader_kind,
            "strategy_rollout_ids": list(self.strategy_rollout_ids),
            "memory_ids": list(self.memory_ids),
            "pair_effect_l2": self.pair_effect_l2,
            "additive_effect_l2": self.additive_effect_l2,
            "interaction_delta_l2": self.interaction_delta_l2,
            "interaction_residual_l2": self.interaction_residual_l2,
            "order_delta_l2": self.order_delta_l2,
            "order_invariant": self.order_invariant,
            "replay_epochs": self.replay_epochs,
            "replay_learning_rate": self.replay_learning_rate,
            "method": self.method,
            "replay_action_kinds": list(self.replay_action_kinds),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderInteraction:
        if payload.get("format", RECOVERY_READER_INTERACTION_CHECKPOINT_FORMAT) != (
            RECOVERY_READER_INTERACTION_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery reader interaction checkpoint format")
        strategy_rollout_ids = tuple(str(item) for item in payload.get("strategy_rollout_ids", ()))
        memory_ids = tuple(str(item) for item in payload.get("memory_ids", ()))
        if len(strategy_rollout_ids) != 2 or len(memory_ids) != 2:
            raise ValueError("recovery reader interaction payload must contain two ids per kind")
        return cls(
            reader_kind=str(payload["reader_kind"]),
            strategy_rollout_ids=(strategy_rollout_ids[0], strategy_rollout_ids[1]),
            memory_ids=(memory_ids[0], memory_ids[1]),
            pair_effect_l2=float(payload.get("pair_effect_l2", 0.0)),
            additive_effect_l2=float(payload.get("additive_effect_l2", 0.0)),
            interaction_delta_l2=float(payload.get("interaction_delta_l2", 0.0)),
            interaction_residual_l2=float(payload.get("interaction_residual_l2", 0.0)),
            order_delta_l2=float(payload.get("order_delta_l2", 0.0)),
            order_invariant=bool(payload.get("order_invariant", True)),
            replay_epochs=int(payload.get("replay_epochs", 1)),
            replay_learning_rate=float(payload.get("replay_learning_rate", 0.1)),
            method=str(payload.get("method", "pairwise-replay")),
            replay_action_kinds=tuple(str(item) for item in payload.get("replay_action_kinds", ())),
        )


@dataclass(frozen=True)
class RecoveryReaderInteractionGroup:
    """Replayable higher-order effect for one connected strategy group."""

    reader_kind: str
    strategy_rollout_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    group_effect_l2: float
    additive_effect_l2: float
    pairwise_interaction_delta_l2: float
    pairwise_predicted_effect_l2: float
    higher_order_delta_l2: float
    higher_order_residual_l2: float
    order_delta_l2: float
    order_invariant: bool
    replay_epochs: int
    replay_learning_rate: float
    method: str = "higher-order-group-replay"
    singleton_effect_l2: tuple[float, ...] = ()
    replay_digest: str = ""
    attribution_digest: str = ""
    replay_action_kinds: tuple[str, ...] = ()
    member_increment_l2: tuple[float, ...] = ()
    interaction_credit_l2: tuple[float, ...] = ()
    residual_credit_l2: float = 0.0
    credit_conservation_error_l2: float = 0.0
    credit_decomposition_complete: bool = False

    def __post_init__(self) -> None:
        if not self.reader_kind:
            raise ValueError("recovery reader interaction group reader_kind cannot be empty")
        if len(self.strategy_rollout_ids) < 3 or any(
            not rollout_id for rollout_id in self.strategy_rollout_ids
        ):
            raise ValueError("recovery reader interaction group needs at least three rollout ids")
        if len(set(self.strategy_rollout_ids)) != len(self.strategy_rollout_ids):
            raise ValueError("recovery reader interaction group rollout ids must be distinct")
        if len(self.memory_ids) != len(self.strategy_rollout_ids) or any(
            not memory_id for memory_id in self.memory_ids
        ):
            raise ValueError("recovery reader interaction group memory ids must match rollouts")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("recovery reader interaction group memory ids must be distinct")
        for value, name in (
            (self.group_effect_l2, "group effect"),
            (self.additive_effect_l2, "additive effect"),
            (self.higher_order_residual_l2, "higher-order residual"),
            (self.order_delta_l2, "order delta"),
        ):
            _nonnegative_finite(value, f"recovery reader interaction group {name}")
        for value, name in (
            (self.pairwise_interaction_delta_l2, "pairwise interaction delta"),
            (self.pairwise_predicted_effect_l2, "pairwise predicted effect"),
            (self.higher_order_delta_l2, "higher-order delta"),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"recovery reader interaction group {name} must be finite")
        if not math.isclose(
            float(self.higher_order_residual_l2),
            abs(float(self.higher_order_delta_l2)),
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError("recovery reader interaction group residual must match delta")
        if bool(self.order_invariant) != (
            float(self.order_delta_l2) <= RECOVERY_READER_ORDER_INVARIANCE_TOLERANCE
        ):
            raise ValueError(
                "recovery reader interaction group order invariance does not match delta"
            )
        if int(self.replay_epochs) <= 0:
            raise ValueError("recovery reader interaction group replay_epochs must be positive")
        if (
            not math.isfinite(float(self.replay_learning_rate))
            or float(self.replay_learning_rate) <= 0.0
        ):
            raise ValueError(
                "recovery reader interaction group replay_learning_rate must be positive and finite"
            )
        if not self.method:
            raise ValueError("recovery reader interaction group method cannot be empty")
        if self.singleton_effect_l2:
            if len(self.singleton_effect_l2) != len(self.strategy_rollout_ids):
                raise ValueError(
                    "recovery reader interaction group singleton effects must match rollouts"
                )
            for effect in self.singleton_effect_l2:
                _nonnegative_finite(effect, "recovery reader interaction group singleton effect")
        if any(not action_kind for action_kind in self.replay_action_kinds):
            raise ValueError("recovery reader interaction group action kinds cannot be empty")
        if self.member_increment_l2:
            if len(self.member_increment_l2) != len(self.strategy_rollout_ids):
                raise ValueError(
                    "recovery reader interaction group member increments must match rollouts"
                )
            for effect in self.member_increment_l2:
                _nonnegative_finite(effect, "recovery reader interaction group member increment")
        if self.interaction_credit_l2:
            expected_count = (
                len(self.strategy_rollout_ids) * (len(self.strategy_rollout_ids) - 1) // 2
            )
            if len(self.interaction_credit_l2) != expected_count:
                raise ValueError(
                    "recovery reader interaction group interaction credits must match pairs"
                )
            if any(not math.isfinite(float(effect)) for effect in self.interaction_credit_l2):
                raise ValueError(
                    "recovery reader interaction group interaction credits must be finite"
                )
        if not math.isfinite(float(self.residual_credit_l2)):
            raise ValueError("recovery reader interaction group residual credit must be finite")
        _nonnegative_finite(
            self.credit_conservation_error_l2,
            "recovery reader interaction group credit conservation error",
        )
        if not isinstance(self.credit_decomposition_complete, bool):
            raise ValueError("recovery reader interaction group credit flag must be boolean")
        if self.credit_decomposition_complete:
            if not self.member_increment_l2 or not self.interaction_credit_l2:
                raise ValueError(
                    "complete recovery reader interaction group credit needs subset effects"
                )
            if self.singleton_effect_l2 != self.member_increment_l2:
                raise ValueError(
                    "recovery reader interaction group member increments must match singleton effects"
                )
            decomposed_effect = (
                sum(self.member_increment_l2)
                + sum(self.interaction_credit_l2)
                + float(self.residual_credit_l2)
            )
            computed_error = abs(float(self.group_effect_l2) - decomposed_effect)
            if not math.isclose(
                float(self.credit_conservation_error_l2),
                computed_error,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "recovery reader interaction group credit conservation error is stale"
                )

    @property
    def credit_decomposition_safe(self) -> bool:
        """Return whether this group can be treated as separable evidence."""

        return bool(
            self.credit_decomposition_complete
            and self.order_invariant
            and self.credit_conservation_error_l2 <= RECOVERY_READER_CREDIT_CONSERVATION_TOLERANCE
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_READER_INTERACTION_GROUP_CHECKPOINT_FORMAT,
            "reader_kind": self.reader_kind,
            "strategy_rollout_ids": list(self.strategy_rollout_ids),
            "memory_ids": list(self.memory_ids),
            "group_effect_l2": self.group_effect_l2,
            "additive_effect_l2": self.additive_effect_l2,
            "pairwise_interaction_delta_l2": self.pairwise_interaction_delta_l2,
            "pairwise_predicted_effect_l2": self.pairwise_predicted_effect_l2,
            "higher_order_delta_l2": self.higher_order_delta_l2,
            "higher_order_residual_l2": self.higher_order_residual_l2,
            "order_delta_l2": self.order_delta_l2,
            "order_invariant": self.order_invariant,
            "replay_epochs": self.replay_epochs,
            "replay_learning_rate": self.replay_learning_rate,
            "method": self.method,
            "singleton_effect_l2": list(self.singleton_effect_l2),
            "replay_digest": self.replay_digest,
            "attribution_digest": self.attribution_digest,
            "replay_action_kinds": list(self.replay_action_kinds),
            "member_increment_l2": list(self.member_increment_l2),
            "interaction_credit_l2": list(self.interaction_credit_l2),
            "residual_credit_l2": self.residual_credit_l2,
            "credit_conservation_error_l2": self.credit_conservation_error_l2,
            "credit_decomposition_complete": self.credit_decomposition_complete,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderInteractionGroup:
        if payload.get("format", RECOVERY_READER_INTERACTION_GROUP_CHECKPOINT_FORMAT) != (
            RECOVERY_READER_INTERACTION_GROUP_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery reader interaction group checkpoint format")
        strategy_rollout_ids = tuple(str(item) for item in payload.get("strategy_rollout_ids", ()))
        memory_ids = tuple(str(item) for item in payload.get("memory_ids", ()))
        return cls(
            reader_kind=str(payload["reader_kind"]),
            strategy_rollout_ids=strategy_rollout_ids,
            memory_ids=memory_ids,
            group_effect_l2=float(payload.get("group_effect_l2", 0.0)),
            additive_effect_l2=float(payload.get("additive_effect_l2", 0.0)),
            pairwise_interaction_delta_l2=float(payload.get("pairwise_interaction_delta_l2", 0.0)),
            pairwise_predicted_effect_l2=float(payload.get("pairwise_predicted_effect_l2", 0.0)),
            higher_order_delta_l2=float(payload.get("higher_order_delta_l2", 0.0)),
            higher_order_residual_l2=float(payload.get("higher_order_residual_l2", 0.0)),
            order_delta_l2=float(payload.get("order_delta_l2", 0.0)),
            order_invariant=bool(payload.get("order_invariant", True)),
            replay_epochs=int(payload.get("replay_epochs", 1)),
            replay_learning_rate=float(payload.get("replay_learning_rate", 0.1)),
            method=str(payload.get("method", "higher-order-group-replay")),
            singleton_effect_l2=tuple(
                float(item) for item in payload.get("singleton_effect_l2", ())
            ),
            replay_digest=str(payload.get("replay_digest", "")),
            attribution_digest=str(payload.get("attribution_digest", "")),
            replay_action_kinds=tuple(str(item) for item in payload.get("replay_action_kinds", ())),
            member_increment_l2=tuple(
                float(item) for item in payload.get("member_increment_l2", ())
            ),
            interaction_credit_l2=tuple(
                float(item) for item in payload.get("interaction_credit_l2", ())
            ),
            residual_credit_l2=float(
                payload.get("residual_credit_l2", payload.get("higher_order_delta_l2", 0.0))
            ),
            credit_conservation_error_l2=float(payload.get("credit_conservation_error_l2", 0.0)),
            credit_decomposition_complete=bool(payload.get("credit_decomposition_complete", False)),
        )


def recovery_reader_credit_profile(
    group: RecoveryReaderInteractionGroup,
) -> tuple[float, ...]:
    """Return an order-independent normalized signed credit profile.

    Reader magnitudes are not expected to match: a semantic state distance and
    a concept-organ state distance live on different scales.  The profile only
    compares the decomposition shape, with member, pair, and explicit residual
    components normalized by their total absolute mass.
    """

    if not group.credit_decomposition_complete:
        return ()
    member_by_id = dict(zip(group.strategy_rollout_ids, group.member_increment_l2, strict=True))
    pair_by_ids = {
        frozenset(pair): float(credit)
        for pair, credit in zip(
            (
                (first, second)
                for index, first in enumerate(group.strategy_rollout_ids)
                for second in group.strategy_rollout_ids[index + 1 :]
            ),
            group.interaction_credit_l2,
            strict=True,
        )
    }
    ordered_ids = tuple(sorted(group.strategy_rollout_ids))
    components = [float(member_by_id[rollout_id]) for rollout_id in ordered_ids]
    components.extend(
        pair_by_ids[frozenset((first, second))]
        for index, first in enumerate(ordered_ids)
        for second in ordered_ids[index + 1 :]
    )
    components.append(float(group.residual_credit_l2))
    scale = sum(abs(value) for value in components)
    if scale <= 1e-12:
        return tuple(0.0 for _ in components)
    return tuple(value / scale for value in components)


def recovery_reader_credit_structure_digest(
    group: RecoveryReaderInteractionGroup,
) -> str:
    """Hash the reader-independent topology of one group's credit split."""

    profile = recovery_reader_credit_profile(group)
    if not profile:
        return ""
    signature = (
        tuple(sorted(group.strategy_rollout_ids)),
        len(group.strategy_rollout_ids),
        len(group.interaction_credit_l2),
        len(profile),
        bool(group.order_invariant),
        bool(group.credit_decomposition_complete),
    )
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecoveryReaderCreditConsistency:
    """Cross-reader audit for one strategy group's replay attribution.

    State and checkpoint digests are evidence of version/reader drift, not
    equality requirements.  The gate compares only the reader-independent
    decomposition topology and normalized signed credit profile.
    """

    strategy_rollout_ids: tuple[str, ...]
    reader_kinds: tuple[str, ...]
    credit_structure_digests: tuple[str, ...]
    credit_profiles: tuple[tuple[float, ...], ...]
    base_checkpoint_digests: tuple[str, ...]
    state_digests: tuple[str, ...]
    max_credit_drift_l1: float
    credit_drift_tolerance: float
    coverage_complete: bool
    structure_consistent: bool
    within_tolerance: bool
    checkpoint_complete: bool
    reader_attribution_safe: tuple[bool, ...] = ()
    changed_reader_kinds: tuple[str, ...] = ()
    method: str = "cross-reader-normalized-credit-v1"

    def __post_init__(self) -> None:
        if len(self.strategy_rollout_ids) < 3 or any(
            not rollout_id for rollout_id in self.strategy_rollout_ids
        ):
            raise ValueError("recovery reader credit consistency needs a strategy group")
        if len(set(self.strategy_rollout_ids)) != len(self.strategy_rollout_ids):
            raise ValueError("recovery reader credit consistency rollout ids must be unique")
        if not self.reader_kinds or any(not reader_kind for reader_kind in self.reader_kinds):
            raise ValueError("recovery reader credit consistency needs reader kinds")
        if len(set(self.reader_kinds)) != len(self.reader_kinds):
            raise ValueError("recovery reader credit consistency reader kinds must be unique")
        field_lengths = (
            len(self.credit_structure_digests),
            len(self.credit_profiles),
            len(self.base_checkpoint_digests),
            len(self.state_digests),
        )
        if any(length != len(self.reader_kinds) for length in field_lengths):
            raise ValueError("recovery reader credit consistency fields must align by reader")
        if any(
            not math.isfinite(float(value)) for profile in self.credit_profiles for value in profile
        ):
            raise ValueError("recovery reader credit consistency profiles must be finite")
        if not math.isfinite(float(self.max_credit_drift_l1)) or self.max_credit_drift_l1 < 0.0:
            raise ValueError(
                "recovery reader credit consistency drift must be finite and non-negative"
            )
        _nonnegative_finite(
            self.credit_drift_tolerance,
            "recovery reader credit consistency tolerance",
        )
        for value, name in (
            (self.coverage_complete, "coverage_complete"),
            (self.structure_consistent, "structure_consistent"),
            (self.within_tolerance, "within_tolerance"),
            (self.checkpoint_complete, "checkpoint_complete"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"recovery reader credit consistency {name} must be boolean")
        safe = self.reader_attribution_safe
        if not safe:
            safe = (False,) * len(self.reader_kinds)
            object.__setattr__(self, "reader_attribution_safe", safe)
        if len(safe) != len(self.reader_kinds) or any(
            not isinstance(value, bool) for value in safe
        ):
            raise ValueError("recovery reader credit consistency attribution flags must align")
        if any(reader_kind not in self.reader_kinds for reader_kind in self.changed_reader_kinds):
            raise ValueError("recovery reader credit consistency changed reader is unknown")
        if not self.method:
            raise ValueError("recovery reader credit consistency method cannot be empty")

    @property
    def complete(self) -> bool:
        """Return whether all four readers support a comparable safe audit."""

        return bool(
            self.coverage_complete
            and self.structure_consistent
            and self.within_tolerance
            and self.checkpoint_complete
        )

    @property
    def safe(self) -> bool:
        """Return whether the group may use every reader's attribution."""

        return bool(self.complete and all(self.reader_attribution_safe))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_READER_CREDIT_CONSISTENCY_CHECKPOINT_FORMAT,
            "strategy_rollout_ids": list(self.strategy_rollout_ids),
            "reader_kinds": list(self.reader_kinds),
            "credit_structure_digests": list(self.credit_structure_digests),
            "credit_profiles": [list(profile) for profile in self.credit_profiles],
            "base_checkpoint_digests": list(self.base_checkpoint_digests),
            "state_digests": list(self.state_digests),
            "max_credit_drift_l1": self.max_credit_drift_l1,
            "credit_drift_tolerance": self.credit_drift_tolerance,
            "coverage_complete": self.coverage_complete,
            "structure_consistent": self.structure_consistent,
            "within_tolerance": self.within_tolerance,
            "checkpoint_complete": self.checkpoint_complete,
            "reader_attribution_safe": list(self.reader_attribution_safe),
            "changed_reader_kinds": list(self.changed_reader_kinds),
            "method": self.method,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderCreditConsistency:
        if payload.get("format", RECOVERY_READER_CREDIT_CONSISTENCY_CHECKPOINT_FORMAT) != (
            RECOVERY_READER_CREDIT_CONSISTENCY_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery reader credit consistency checkpoint format")
        return cls(
            strategy_rollout_ids=tuple(
                str(item) for item in payload.get("strategy_rollout_ids", ())
            ),
            reader_kinds=tuple(str(item) for item in payload.get("reader_kinds", ())),
            credit_structure_digests=tuple(
                str(item) for item in payload.get("credit_structure_digests", ())
            ),
            credit_profiles=tuple(
                tuple(float(value) for value in profile)
                for profile in payload.get("credit_profiles", ())
            ),
            base_checkpoint_digests=tuple(
                str(item) for item in payload.get("base_checkpoint_digests", ())
            ),
            state_digests=tuple(str(item) for item in payload.get("state_digests", ())),
            max_credit_drift_l1=float(payload.get("max_credit_drift_l1", 0.0)),
            credit_drift_tolerance=float(payload.get("credit_drift_tolerance", 0.0)),
            coverage_complete=bool(payload.get("coverage_complete", False)),
            structure_consistent=bool(payload.get("structure_consistent", False)),
            within_tolerance=bool(payload.get("within_tolerance", False)),
            checkpoint_complete=bool(payload.get("checkpoint_complete", False)),
            reader_attribution_safe=tuple(
                bool(value) for value in payload.get("reader_attribution_safe", ())
            ),
            changed_reader_kinds=tuple(
                str(item) for item in payload.get("changed_reader_kinds", ())
            ),
            method=str(payload.get("method", "cross-reader-normalized-credit-v1")),
        )


@dataclass(frozen=True)
class RecoveryReaderDependency:
    """Provenance slice showing which strategies feed one downstream reader."""

    reader_kind: str
    memory_ids: tuple[str, ...] = ()
    strategy_rollout_ids: tuple[str, ...] = ()
    revision: int = 0
    contributions: tuple[RecoveryReaderContribution, ...] = ()
    interactions: tuple[RecoveryReaderInteraction, ...] = ()
    interaction_groups: tuple[RecoveryReaderInteractionGroup, ...] = ()
    interaction_audit_complete: bool = False
    base_checkpoint: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    base_checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if not self.reader_kind:
            raise ValueError("recovery reader dependency reader_kind cannot be empty")
        if any(not memory_id for memory_id in self.memory_ids):
            raise ValueError("recovery reader dependency memory ids cannot be empty")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("recovery reader dependency memory ids must be unique")
        if len(self.memory_ids) != len(self.strategy_rollout_ids):
            raise ValueError("recovery reader dependency ids must have equal lengths")
        if any(not rollout_id for rollout_id in self.strategy_rollout_ids):
            raise ValueError("recovery reader dependency rollout ids cannot be empty")
        if len(set(self.strategy_rollout_ids)) != len(self.strategy_rollout_ids):
            raise ValueError("recovery reader dependency rollout ids must be unique")
        if any(
            not isinstance(contribution, RecoveryReaderContribution)
            or contribution.reader_kind != self.reader_kind
            for contribution in self.contributions
        ):
            raise ValueError("recovery reader contributions must belong to their reader")
        if len({contribution.strategy_rollout_id for contribution in self.contributions}) != len(
            self.contributions
        ):
            raise ValueError("recovery reader contributions must be unique per rollout")
        if any(
            contribution.strategy_rollout_id not in self.strategy_rollout_ids
            or contribution.memory_id not in self.memory_ids
            for contribution in self.contributions
        ):
            raise ValueError("recovery reader contribution provenance must match dependency ids")
        if any(
            not isinstance(interaction, RecoveryReaderInteraction)
            or interaction.reader_kind != self.reader_kind
            for interaction in self.interactions
        ):
            raise ValueError("recovery reader interactions must belong to their reader")
        if any(
            not isinstance(group, RecoveryReaderInteractionGroup)
            or group.reader_kind != self.reader_kind
            for group in self.interaction_groups
        ):
            raise ValueError("recovery reader interaction groups must belong to their reader")
        if not isinstance(self.interaction_audit_complete, bool):
            raise ValueError("recovery reader interaction audit flag must be boolean")
        interaction_keys = {
            frozenset(interaction.strategy_rollout_ids) for interaction in self.interactions
        }
        if len(interaction_keys) != len(self.interactions):
            raise ValueError("recovery reader interactions must be unique per rollout pair")
        group_keys = {frozenset(group.strategy_rollout_ids) for group in self.interaction_groups}
        if len(group_keys) != len(self.interaction_groups):
            raise ValueError("recovery reader interaction groups must be unique per rollout group")
        memory_by_rollout = dict(zip(self.strategy_rollout_ids, self.memory_ids, strict=True))
        if any(
            any(
                memory_by_rollout.get(rollout_id) != memory_id
                for rollout_id, memory_id in zip(
                    interaction.strategy_rollout_ids, interaction.memory_ids, strict=True
                )
            )
            for interaction in self.interactions
        ):
            raise ValueError("recovery reader interaction provenance must match dependency ids")
        if any(
            any(
                memory_by_rollout.get(rollout_id) != memory_id
                for rollout_id, memory_id in zip(
                    group.strategy_rollout_ids, group.memory_ids, strict=True
                )
            )
            for group in self.interaction_groups
        ):
            raise ValueError(
                "recovery reader interaction group provenance must match dependency ids"
            )
        if self.base_checkpoint is not None and not self.base_checkpoint_digest:
            raise ValueError("recovery reader base checkpoint requires a content digest")
        if int(self.revision) < 0:
            raise ValueError("recovery reader dependency revision cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "reader_kind": self.reader_kind,
            "memory_ids": list(self.memory_ids),
            "strategy_rollout_ids": list(self.strategy_rollout_ids),
            "revision": self.revision,
            "contributions": [item.to_payload() for item in self.contributions],
            "interactions": [item.to_payload() for item in self.interactions],
            "interaction_groups": [item.to_payload() for item in self.interaction_groups],
            "interaction_audit_complete": self.interaction_audit_complete,
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_digest": self.base_checkpoint_digest,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderDependency:
        return cls(
            reader_kind=str(payload["reader_kind"]),
            memory_ids=tuple(str(item) for item in payload.get("memory_ids", ())),
            strategy_rollout_ids=tuple(
                str(item) for item in payload.get("strategy_rollout_ids", ())
            ),
            revision=int(payload.get("revision", 0)),
            contributions=tuple(
                RecoveryReaderContribution.from_payload(dict(item))
                for item in payload.get("contributions", ())
            ),
            interactions=tuple(
                RecoveryReaderInteraction.from_payload(dict(item))
                for item in payload.get("interactions", ())
            ),
            interaction_groups=tuple(
                RecoveryReaderInteractionGroup.from_payload(dict(item))
                for item in payload.get("interaction_groups", ())
            ),
            interaction_audit_complete=bool(payload.get("interaction_audit_complete", False)),
            base_checkpoint=(
                None if payload.get("base_checkpoint") is None else dict(payload["base_checkpoint"])
            ),
            base_checkpoint_digest=str(payload.get("base_checkpoint_digest", "")),
        )


@dataclass(frozen=True)
class RecoveryReaderDependencyGraph:
    """Checkpointable dependency graph for selective reader rebuilds."""

    dependencies: tuple[RecoveryReaderDependency, ...] = ()
    revision: int = 0
    credit_consistency: tuple[RecoveryReaderCreditConsistency, ...] = ()

    def __post_init__(self) -> None:
        reader_kinds = tuple(dependency.reader_kind for dependency in self.dependencies)
        if len(set(reader_kinds)) != len(reader_kinds):
            raise ValueError("recovery reader dependency reader kinds must be unique")
        if int(self.revision) < 0:
            raise ValueError("recovery reader dependency graph revision cannot be negative")
        consistency_keys = {
            frozenset(item.strategy_rollout_ids) for item in self.credit_consistency
        }
        if len(consistency_keys) != len(self.credit_consistency):
            raise ValueError("recovery reader credit consistency groups must be unique")
        if any(
            not isinstance(item, RecoveryReaderCreditConsistency)
            for item in self.credit_consistency
        ):
            raise ValueError("recovery reader credit consistency records must be typed")

    def bind(
        self,
        reader_kind: str,
        approvals: Sequence[RecoveryStrategyApproval],
        *,
        contributions: Sequence[RecoveryReaderContribution] = (),
        interactions: Sequence[RecoveryReaderInteraction] = (),
        interaction_groups: Sequence[RecoveryReaderInteractionGroup] = (),
        interaction_audit_complete: bool | None = None,
        base_checkpoint: dict[str, Any] | None = None,
        base_checkpoint_digest: str = "",
    ) -> RecoveryReaderDependencyGraph:
        if not reader_kind:
            raise ValueError("recovery reader dependency reader_kind cannot be empty")
        selected = tuple(approvals)
        memory_ids = tuple(dict.fromkeys(approval.memory_id for approval in selected))
        rollout_ids = tuple(dict.fromkeys(approval.rollout_id for approval in selected))
        contribution_items = tuple(contributions)
        interaction_items = tuple(interactions)
        interaction_group_items = tuple(interaction_groups)
        audit_complete = (
            bool(interaction_items or interaction_group_items)
            if interaction_audit_complete is None
            else interaction_audit_complete
        )
        if not isinstance(audit_complete, bool):
            raise ValueError("recovery reader interaction audit flag must be boolean")
        approval_by_rollout = {approval.rollout_id: approval for approval in selected}
        if any(
            contribution.reader_kind != reader_kind
            or contribution.strategy_rollout_id not in approval_by_rollout
            or contribution.memory_id
            != approval_by_rollout[contribution.strategy_rollout_id].memory_id
            for contribution in contribution_items
        ):
            raise ValueError("recovery reader contributions must match selected approvals")
        if any(
            interaction.reader_kind != reader_kind
            or any(
                rollout_id not in approval_by_rollout
                or approval_by_rollout[rollout_id].memory_id != memory_id
                for rollout_id, memory_id in zip(
                    interaction.strategy_rollout_ids, interaction.memory_ids, strict=True
                )
            )
            for interaction in interaction_items
        ):
            raise ValueError("recovery reader interactions must match selected approvals")
        if any(
            group.reader_kind != reader_kind
            or any(
                rollout_id not in approval_by_rollout
                or approval_by_rollout[rollout_id].memory_id != memory_id
                for rollout_id, memory_id in zip(
                    group.strategy_rollout_ids, group.memory_ids, strict=True
                )
            )
            for group in interaction_group_items
        ):
            raise ValueError("recovery reader interaction groups must match selected approvals")
        dependency = RecoveryReaderDependency(
            reader_kind=reader_kind,
            memory_ids=memory_ids,
            strategy_rollout_ids=rollout_ids,
            revision=self.revision + 1,
            contributions=contribution_items,
            interactions=interaction_items,
            interaction_groups=interaction_group_items,
            interaction_audit_complete=audit_complete,
            base_checkpoint=base_checkpoint,
            base_checkpoint_digest=str(base_checkpoint_digest),
        )
        dependencies = tuple(item for item in self.dependencies if item.reader_kind != reader_kind)
        return replace(self, dependencies=(*dependencies, dependency), revision=self.revision + 1)

    def dependency_for(self, reader_kind: str) -> RecoveryReaderDependency | None:
        return next(
            (
                dependency
                for dependency in self.dependencies
                if dependency.reader_kind == reader_kind
            ),
            None,
        )

    def reader_kinds_for_rollout(self, rollout_id: str) -> tuple[str, ...]:
        return tuple(
            dependency.reader_kind
            for dependency in self.dependencies
            if rollout_id in dependency.strategy_rollout_ids
        )

    def retain_selected(
        self, approvals: Sequence[RecoveryStrategyApproval]
    ) -> RecoveryReaderDependencyGraph:
        """Propagate competition/revocation to every already-bound reader."""

        selected_by_rollout = {approval.rollout_id: approval for approval in approvals}
        dependencies = tuple(
            RecoveryReaderDependency(
                reader_kind=dependency.reader_kind,
                memory_ids=tuple(
                    selected_by_rollout[rollout_id].memory_id
                    for rollout_id in dependency.strategy_rollout_ids
                    if rollout_id in selected_by_rollout
                ),
                strategy_rollout_ids=tuple(
                    rollout_id
                    for rollout_id in dependency.strategy_rollout_ids
                    if rollout_id in selected_by_rollout
                ),
                revision=self.revision + 1,
                contributions=tuple(
                    contribution
                    for contribution in dependency.contributions
                    if contribution.strategy_rollout_id in selected_by_rollout
                ),
                interactions=tuple(
                    interaction
                    for interaction in dependency.interactions
                    if all(
                        rollout_id in selected_by_rollout
                        for rollout_id in interaction.strategy_rollout_ids
                    )
                ),
                interaction_groups=tuple(
                    group
                    for group in dependency.interaction_groups
                    if all(
                        rollout_id in selected_by_rollout
                        for rollout_id in group.strategy_rollout_ids
                    )
                ),
                interaction_audit_complete=dependency.interaction_audit_complete,
                base_checkpoint=dependency.base_checkpoint,
                base_checkpoint_digest=dependency.base_checkpoint_digest,
            )
            for dependency in self.dependencies
        )
        consistency = tuple(
            item
            for item in self.credit_consistency
            if all(rollout_id in selected_by_rollout for rollout_id in item.strategy_rollout_ids)
        )
        return replace(
            self,
            dependencies=dependencies,
            credit_consistency=consistency,
            revision=self.revision + 1,
        )

    def record_credit_consistency(
        self, audits: Sequence[RecoveryReaderCreditConsistency]
    ) -> RecoveryReaderDependencyGraph:
        """Persist the latest cross-reader group audits."""

        items = tuple(audits)
        if any(not isinstance(item, RecoveryReaderCreditConsistency) for item in items):
            raise ValueError("recovery reader credit consistency records must be typed")
        return replace(self, credit_consistency=items, revision=self.revision + 1)

    @property
    def interaction_audit_available(self) -> bool:
        """Return whether any bound reader has completed interaction auditing."""

        return any(
            dependency.interaction_audit_complete or bool(dependency.interaction_groups)
            for dependency in self.dependencies
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_READER_DEPENDENCY_CHECKPOINT_FORMAT,
            "dependencies": [dependency.to_payload() for dependency in self.dependencies],
            "credit_consistency": [item.to_payload() for item in self.credit_consistency],
            "revision": self.revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryReaderDependencyGraph:
        if payload.get("format", RECOVERY_READER_DEPENDENCY_CHECKPOINT_FORMAT) != (
            RECOVERY_READER_DEPENDENCY_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery reader dependency checkpoint format")
        return cls(
            dependencies=tuple(
                RecoveryReaderDependency.from_payload(dict(item))
                for item in payload.get("dependencies", ())
            ),
            credit_consistency=tuple(
                RecoveryReaderCreditConsistency.from_payload(dict(item))
                for item in payload.get("credit_consistency", ())
            ),
            revision=int(payload.get("revision", 0)),
        )


def build_recovery_reader_credit_consistency(
    dependencies: Sequence[RecoveryReaderDependency],
    *,
    drift_tolerance: float,
    previous: Sequence[RecoveryReaderCreditConsistency] = (),
) -> tuple[RecoveryReaderCreditConsistency, ...]:
    """Compare matching group attribution across all currently bound readers."""

    tolerance = _nonnegative_finite(
        drift_tolerance,
        "recovery reader cross-reader credit drift tolerance",
    )
    dependency_by_kind = {dependency.reader_kind: dependency for dependency in dependencies}
    reader_kinds = tuple(sorted(dependency_by_kind))
    if not reader_kinds:
        return ()
    groups_by_key: dict[frozenset[str], dict[str, RecoveryReaderInteractionGroup]] = {}
    for dependency in dependencies:
        for group in dependency.interaction_groups:
            groups_by_key.setdefault(frozenset(group.strategy_rollout_ids), {})[
                dependency.reader_kind
            ] = group
    previous_by_key = {frozenset(item.strategy_rollout_ids): item for item in previous}
    audits: list[RecoveryReaderCreditConsistency] = []
    for group_key in sorted(groups_by_key, key=lambda item: tuple(sorted(item))):
        groups = groups_by_key[group_key]
        ordered_ids = tuple(sorted(group_key))
        current_groups = tuple(groups.get(reader_kind) for reader_kind in reader_kinds)
        structure_digests = tuple(
            "" if group is None else recovery_reader_credit_structure_digest(group)
            for group in current_groups
        )
        profiles = tuple(
            () if group is None else recovery_reader_credit_profile(group)
            for group in current_groups
        )
        base_checkpoint_digests = tuple(
            dependency_by_kind[reader_kind].base_checkpoint_digest for reader_kind in reader_kinds
        )
        state_digests = tuple(
            "" if group is None else group.replay_digest for group in current_groups
        )
        profiles_valid = bool(profiles) and all(
            profile and len(profile) == len(profiles[0]) for profile in profiles
        )
        max_drift = 0.0
        if profiles_valid:
            for index, first in enumerate(profiles):
                for second in profiles[index + 1 :]:
                    max_drift = max(
                        max_drift,
                        sum(abs(left - right) for left, right in zip(first, second, strict=True)),
                    )
        else:
            max_drift = 2.0
        group_complete = bool(
            current_groups
            and all(
                group is not None and group.credit_decomposition_safe for group in current_groups
            )
        )
        structure_consistent = bool(
            group_complete and all(structure_digests) and len(set(structure_digests)) == 1
        )
        within_tolerance = bool(profiles_valid and max_drift <= tolerance)
        checkpoint_complete = bool(all(base_checkpoint_digests) and all(state_digests))
        coverage_complete = set(reader_kinds) == set(
            RECOVERY_READER_CREDIT_CONSISTENCY_READER_KINDS
        )
        previous_audit = previous_by_key.get(group_key)
        changed_reader_kinds: list[str] = []
        if previous_audit is None:
            changed_reader_kinds.extend(reader_kinds)
        else:
            previous_by_reader = {
                reader_kind: index for index, reader_kind in enumerate(previous_audit.reader_kinds)
            }
            for index, reader_kind in enumerate(reader_kinds):
                previous_index = previous_by_reader.get(reader_kind)
                if previous_index is None:
                    changed_reader_kinds.append(reader_kind)
                    continue
                profile_changed = len(profiles[index]) != len(
                    previous_audit.credit_profiles[previous_index]
                ) or any(
                    abs(left - right) > 1e-12
                    for left, right in zip(
                        profiles[index],
                        previous_audit.credit_profiles[previous_index],
                        strict=True,
                    )
                )
                if (
                    structure_digests[index]
                    != previous_audit.credit_structure_digests[previous_index]
                    or profile_changed
                    or base_checkpoint_digests[index]
                    != previous_audit.base_checkpoint_digests[previous_index]
                    or state_digests[index] != previous_audit.state_digests[previous_index]
                ):
                    changed_reader_kinds.append(reader_kind)
        audit_valid = bool(
            coverage_complete
            and group_complete
            and structure_consistent
            and within_tolerance
            and checkpoint_complete
        )
        if audit_valid:
            reader_attribution_safe = (True,) * len(reader_kinds)
        elif previous_audit is not None and changed_reader_kinds:
            reader_attribution_safe = tuple(
                reader_kind not in changed_reader_kinds for reader_kind in reader_kinds
            )
        else:
            reader_attribution_safe = (False,) * len(reader_kinds)
        audits.append(
            RecoveryReaderCreditConsistency(
                strategy_rollout_ids=ordered_ids,
                reader_kinds=reader_kinds,
                credit_structure_digests=structure_digests,
                credit_profiles=profiles,
                base_checkpoint_digests=base_checkpoint_digests,
                state_digests=state_digests,
                max_credit_drift_l1=max_drift,
                credit_drift_tolerance=tolerance,
                coverage_complete=coverage_complete,
                structure_consistent=structure_consistent,
                within_tolerance=within_tolerance,
                checkpoint_complete=checkpoint_complete,
                reader_attribution_safe=reader_attribution_safe,
                changed_reader_kinds=tuple(changed_reader_kinds),
            )
        )
    return tuple(audits)


@dataclass(frozen=True)
class RolloutDecision:
    goal_id: str
    plan: PlanState
    selected: ImaginedRollout


class GoalPlanner:
    """Rank real candidates against a goal and learn progress from outcomes."""

    def __init__(self, config: PlanningConfig | None = None) -> None:
        self.config = config or PlanningConfig()
        self._calibration: dict[str, tuple[int, int]] = {}
        self._world_error_calibration: tuple[float, ...] = ()

    def calibrate_world_prediction_errors(self, errors: Sequence[float]) -> float:
        """Install a data-derived baseline for world prediction error policy."""

        values = tuple(float(error) for error in errors)
        if not values:
            raise ValueError("world error calibration requires at least one sample")
        if any(not math.isfinite(error) or error < 0.0 for error in values):
            raise ValueError("world error calibration samples must be finite and non-negative")
        self._world_error_calibration = values
        return self.world_prediction_error_threshold()

    def world_prediction_error_threshold(
        self,
        *,
        recovery: bool = False,
        trigger_error: float | None = None,
    ) -> float:
        """Return a threshold on the world-error scale, not a probability scale."""

        threshold = (
            self.config.recovery_error_threshold if recovery else self.config.replan_error_threshold
        )
        if self._world_error_calibration:
            ordered = sorted(self._world_error_calibration)
            index = min(
                len(ordered) - 1,
                max(0, math.ceil((len(ordered) - 1) * self.config.world_calibration_quantile)),
            )
            threshold = max(
                threshold,
                ordered[index]
                + self.config.world_calibration_std_multiplier
                * math.sqrt(
                    sum((error - sum(ordered) / len(ordered)) ** 2 for error in ordered)
                    / len(ordered)
                )
                + self.config.world_calibration_margin,
            )
        if recovery and trigger_error is not None:
            trigger_error = float(trigger_error)
            if not math.isfinite(trigger_error) or trigger_error < 0.0:
                raise ValueError("trigger_error must be finite and non-negative")
            threshold = max(threshold, trigger_error)
        return float(threshold)

    @property
    def world_error_calibration_sample_count(self) -> int:
        return len(self._world_error_calibration)

    def _uncertainty_penalty(self, candidate: PlanningCandidate) -> float:
        multiplier = {
            "unseen": self.config.unseen_uncertainty_multiplier,
            "stochastic": self.config.stochastic_uncertainty_multiplier,
            "conflicted": self.config.conflicted_uncertainty_multiplier,
        }.get(candidate.uncertainty_mode, 1.0)
        return self.config.uncertainty_weight * multiplier * candidate.uncertainty

    def plan(
        self,
        goals: GoalState,
        candidates: tuple[PlanningCandidate, ...],
        *,
        tick: int,
        goal_id: str | None = None,
    ) -> PlanningDecision:
        if not candidates:
            raise ValueError("goal planning requires executable candidates")
        goal = self._select_goal(goals, goal_id)
        if int(tick) < 0:
            raise ValueError("planning tick cannot be negative")
        residual = 1.0 - goal.progress
        scored: list[PlanCandidate] = []
        for candidate in candidates:
            expected_value = (
                self.config.reward_weight * candidate.predicted_reward
                + self.config.progress_weight * residual * candidate.expected_progress
                + self.config.success_weight * candidate.success_probability
                - self._uncertainty_penalty(candidate)
                - self.config.resource_weight * candidate.resource_cost
                - self.config.conflict_weight * candidate.conflict
                + self.config.concept_weight * candidate.concept_affinity
            )
            risk = max(candidate.uncertainty, candidate.resource_cost, candidate.conflict)
            scored.append(
                PlanCandidate(
                    plan_id=candidate.candidate_id,
                    action_kind=candidate.action.kind,
                    expected_value=expected_value,
                    risk=risk,
                )
            )
        selected_index = max(range(len(scored)), key=lambda index: scored[index].expected_value)
        plan = PlanState(
            tick=int(tick),
            candidates=tuple(scored),
            selected_plan_id=scored[selected_index].plan_id,
        )
        return PlanningDecision(goal.goal_id, plan, candidates[selected_index])

    def apply_outcome(self, goals: GoalState, outcome: Outcome) -> GoalState:
        """Advance goal progress from an experienced outcome, not a plan promise."""

        if not goals.goals:
            return goals
        gain = self.config.outcome_progress_gain * max(0.0, float(outcome.reward))
        if outcome.success is False:
            gain = 0.0
        updated = tuple(
            Goal(
                goal_id=goal.goal_id,
                description=goal.description,
                priority=goal.priority,
                progress=max(0.0, min(1.0, goal.progress + gain)),
                version=goal.version,
            )
            for goal in goals.goals
        )
        return GoalState(tick=outcome.tick, goals=updated, version=goals.version)

    def plan_rollouts(
        self,
        goals: GoalState,
        rollouts: tuple[ImaginedRollout, ...],
        *,
        tick: int,
        goal_id: str | None = None,
    ) -> RolloutDecision:
        """Select among multi-step imagined world-model trajectories."""

        if not rollouts:
            raise ValueError("rollout planning requires imagined rollouts")
        goal = self._select_goal(goals, goal_id)
        if any(rollout.goal_id != goal.goal_id for rollout in rollouts):
            raise ValueError("rollout goal ids must match the selected goal")
        residual = 1.0 - goal.progress
        scored: list[PlanCandidate] = []
        for rollout in rollouts:
            expected_value = 0.0
            risk = 0.0
            for index, step in enumerate(rollout.steps):
                discount = self.config.discount**index
                expected_value += discount * (
                    self.config.reward_weight * step.predicted_reward
                    + self.config.success_weight * step.success_probability
                    - self._uncertainty_penalty(step)
                    - self.config.resource_weight * step.resource_cost
                    - self.config.conflict_weight * step.conflict
                    + self.config.concept_weight * step.concept_affinity
                )
                risk = max(risk, step.uncertainty, step.resource_cost, step.conflict)
            expected_value += (
                self.config.progress_weight
                * residual
                * rollout.steps[-1].expected_progress
                * rollout.confidence
            )
            expected_value += (
                self.config.concept_sequence_weight * rollout.concept_sequence_affinity
            )
            scored.append(
                PlanCandidate(
                    plan_id=rollout.rollout_id,
                    action_kind=rollout.steps[0].action.kind,
                    expected_value=expected_value,
                    risk=risk,
                )
            )
        selected_index = max(range(len(scored)), key=lambda index: scored[index].expected_value)
        plan = PlanState(
            tick=int(tick),
            candidates=tuple(scored),
            selected_plan_id=scored[selected_index].plan_id,
        )
        return RolloutDecision(goal.goal_id, plan, rollouts[selected_index])

    def rollout_prediction_error(self, rollout: ImaginedRollout, outcome: Outcome) -> float:
        """Score the first imagined step against the experienced outcome."""

        return abs(float(rollout.steps[0].predicted_reward) - float(outcome.reward))

    def record_rollout_outcome(self, rollout: ImaginedRollout, outcome: Outcome) -> float:
        """Update empirical success calibration for an imagined rollout."""

        attempts, successes = self._calibration.get(rollout.rollout_id, (0, 0))
        attempts += 1
        successes += int(
            outcome.success is True or (outcome.success is None and outcome.reward > 0.0)
        )
        self._calibration[rollout.rollout_id] = (attempts, successes)
        return successes / attempts

    def calibrated_confidence(self, rollout_id: str) -> float | None:
        stats = self._calibration.get(rollout_id)
        return None if stats is None else stats[1] / stats[0]

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": PLANNING_CHECKPOINT_FORMAT,
            "config": self.config.to_payload(),
            "calibration": {
                rollout_id: [attempts, successes]
                for rollout_id, (attempts, successes) in self._calibration.items()
            },
            "world_error_calibration": list(self._world_error_calibration),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> GoalPlanner:
        if payload.get("format") != PLANNING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported planning checkpoint format")
        planner = cls(PlanningConfig.from_payload(dict(payload.get("config", {}))))
        planner._calibration = {
            str(rollout_id): (int(stats[0]), int(stats[1]))
            for rollout_id, stats in dict(payload.get("calibration", {})).items()
        }
        planner._world_error_calibration = tuple(
            float(error) for error in payload.get("world_error_calibration", ())
        )
        return planner

    @staticmethod
    def _select_goal(goals: GoalState, goal_id: str | None) -> Goal:
        if not goals.goals:
            raise ValueError("goal planning requires at least one goal")
        if goal_id is not None:
            for goal in goals.goals:
                if goal.goal_id == goal_id:
                    return goal
            raise ValueError("requested planning goal is not registered")
        return max(goals.goals, key=lambda goal: (goal.priority, -goal.progress, goal.goal_id))
