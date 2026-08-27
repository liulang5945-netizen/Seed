"""Goal-directed candidate scoring and outcome-driven progress updates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import Goal, GoalState, Outcome, PlanCandidate, PlanState, WorldAction

PLANNING_CHECKPOINT_FORMAT = "taiji-planning-v1"


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
