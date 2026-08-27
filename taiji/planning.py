"""Goal-directed candidate scoring and outcome-driven progress updates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import Goal, GoalState, Outcome, PlanCandidate, PlanState, WorldAction

PLANNING_CHECKPOINT_FORMAT = "taiji-planning-v1"
RECOVERY_ARCHIVE_CHECKPOINT_FORMAT = "taiji-recovery-archive-v1"
RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT = "taiji-recovery-strategy-ledger-v1"


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
        _unit(self.resource_cost, "recovery archive resource_cost")
        if self.outcome_reward is not None and not math.isfinite(float(self.outcome_reward)):
            raise ValueError("recovery archive outcome_reward must be finite")
        if int(self.evidence_count) < 0:
            raise ValueError("recovery archive evidence_count cannot be negative")
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
            revision=int(payload.get("revision", 0)),
        )


@dataclass(frozen=True)
class RecoveryStrategyLedger:
    """Admission and revocation ledger for recovery-derived long-term memory."""

    evidence_threshold: int = 2
    approvals: tuple[RecoveryStrategyApproval, ...] = ()
    revoked_rollout_ids: tuple[str, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if int(self.evidence_threshold) <= 0:
            raise ValueError("recovery strategy evidence_threshold must be positive")
        approval_ids = tuple(approval.approval_id for approval in self.approvals)
        if len(set(approval_ids)) != len(approval_ids):
            raise ValueError("recovery strategy approvals must be unique")
        if any(not rollout_id for rollout_id in self.revoked_rollout_ids):
            raise ValueError("recovery strategy revoked rollout ids cannot be empty")
        if len(set(self.revoked_rollout_ids)) != len(self.revoked_rollout_ids):
            raise ValueError("recovery strategy revoked rollout ids must be unique")
        if int(self.revision) < 0:
            raise ValueError("recovery strategy ledger revision cannot be negative")

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

    @property
    def active_memory_ids(self) -> tuple[str, ...]:
        return tuple(approval.memory_id for approval in self.active_approvals())

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT,
            "evidence_threshold": self.evidence_threshold,
            "approvals": [approval.to_payload() for approval in self.approvals],
            "revoked_rollout_ids": list(self.revoked_rollout_ids),
            "revision": self.revision,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecoveryStrategyLedger:
        if payload.get("format", RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT) != (
            RECOVERY_STRATEGY_LEDGER_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported recovery strategy ledger checkpoint format")
        return cls(
            evidence_threshold=int(payload.get("evidence_threshold", 2)),
            approvals=tuple(
                RecoveryStrategyApproval.from_payload(dict(item))
                for item in payload.get("approvals", ())
            ),
            revoked_rollout_ids=tuple(str(item) for item in payload.get("revoked_rollout_ids", ())),
            revision=int(payload.get("revision", 0)),
        )


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
