"""Goal-directed candidate scoring and outcome-driven progress updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    outcome_progress_gain: float = 0.40
    discount: float = 0.90
    replan_error_threshold: float = 0.25
    recovery_error_threshold: float = 1.00

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if float(value) < 0.0:
                raise ValueError(f"planning {name} cannot be negative")
        _unit(self.outcome_progress_gain, "outcome_progress_gain")
        _unit(self.discount, "discount")
        _nonnegative_finite(self.replan_error_threshold, "replan_error_threshold")
        _nonnegative_finite(self.recovery_error_threshold, "recovery_error_threshold")

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
    resource_cost: float = 0.0
    conflict: float = 0.0
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
        if not self.prediction_provenance:
            raise ValueError("prediction provenance cannot be empty")


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

    def __post_init__(self) -> None:
        if not self.rollout_id or not self.goal_id:
            raise ValueError("imagined rollout ids cannot be empty")
        if not self.steps:
            raise ValueError("imagined rollout requires at least one step")
        _unit(self.confidence, "rollout confidence")
        if self.provenance != "imagined":
            raise ValueError("rollout provenance must be imagined")

    def to_payload(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "goal_id": self.goal_id,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "steps": [
                {
                    "candidate_id": step.candidate_id,
                    "action": step.action.to_payload(),
                    "predicted_reward": step.predicted_reward,
                    "success_probability": step.success_probability,
                    "expected_progress": step.expected_progress,
                    "uncertainty": step.uncertainty,
                    "resource_cost": step.resource_cost,
                    "conflict": step.conflict,
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
            steps=tuple(
                PlanningCandidate(
                    candidate_id=str(item["candidate_id"]),
                    action=WorldAction.from_payload(item["action"]),
                    predicted_reward=float(item["predicted_reward"]),
                    success_probability=float(item["success_probability"]),
                    expected_progress=float(item["expected_progress"]),
                    uncertainty=float(item.get("uncertainty", 0.0)),
                    resource_cost=float(item.get("resource_cost", 0.0)),
                    conflict=float(item.get("conflict", 0.0)),
                    prediction_provenance=str(item.get("prediction_provenance", "planner")),
                )
                for item in payload.get("steps", ())
            ),
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
                - self.config.uncertainty_weight * candidate.uncertainty
                - self.config.resource_weight * candidate.resource_cost
                - self.config.conflict_weight * candidate.conflict
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
                    - self.config.uncertainty_weight * step.uncertainty
                    - self.config.resource_weight * step.resource_cost
                    - self.config.conflict_weight * step.conflict
                )
                risk = max(risk, step.uncertainty, step.resource_cost, step.conflict)
            expected_value += (
                self.config.progress_weight
                * residual
                * rollout.steps[-1].expected_progress
                * rollout.confidence
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
        successes += int(outcome.success is True or (outcome.success is None and outcome.reward > 0.0))
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
