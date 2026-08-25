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


@dataclass(frozen=True)
class PlanningConfig:
    reward_weight: float = 0.60
    progress_weight: float = 1.00
    success_weight: float = 0.80
    uncertainty_weight: float = 1.20
    resource_weight: float = 0.40
    conflict_weight: float = 0.80
    outcome_progress_gain: float = 0.40

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if float(value) < 0.0:
                raise ValueError(f"planning {name} cannot be negative")
        _unit(self.outcome_progress_gain, "outcome_progress_gain")

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


@dataclass(frozen=True)
class PlanningDecision:
    """The scored executable plans and the currently selected action."""

    goal_id: str
    plan: PlanState
    selected: PlanningCandidate


class GoalPlanner:
    """Rank real candidates against a goal and learn progress from outcomes."""

    def __init__(self, config: PlanningConfig | None = None) -> None:
        self.config = config or PlanningConfig()

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

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": PLANNING_CHECKPOINT_FORMAT,
            "config": self.config.to_payload(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> GoalPlanner:
        if payload.get("format") != PLANNING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported planning checkpoint format")
        return cls(PlanningConfig.from_payload(dict(payload.get("config", {}))))

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
