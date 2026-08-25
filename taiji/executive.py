"""Taiji executive contract for learned candidate selection.

The executive does not decode text or invent semantic labels from bytes.  A
world/affordance subsystem supplies structured candidates that already carry
an ``ActionIntent`` and its corresponding ``ContentPlan``.  The executive
learns which candidate is useful in the current percept/world/goal state from
experienced outcome feedback, then returns the selected structured pair.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import (
    ActionIntent,
    CognitiveState,
    Goal,
    PerceptEvent,
    WorldAction,
    WorldAffordance,
)
from .generation import ContentPlan

EXECUTIVE_CHECKPOINT_FORMAT = "taiji-executive-v1"
EXECUTIVE_CANDIDATE_FEATURE_NAMES = (
    "goal_alignment",
    "world_relevance",
    "information_gain",
    "confidence",
    "uncertainty",
    "resource_cost",
)
EXECUTIVE_CONTEXT_FEATURE_NAMES = (
    "percept_mean",
    "percept_std",
    "percept_norm",
    "percept_peak",
    "world_mean",
    "world_std",
    "world_norm",
    "world_peak",
    "memory_mean",
    "memory_std",
    "memory_norm",
    "memory_peak",
    "goal_residual",
    "world_uncertainty",
    "percept_prediction_error",
    "percept_boundary_score",
    "percept_confidence",
    "episodic_confidence",
    "self_confidence",
    "resource_fraction",
    "curiosity",
    "fatigue",
    "stress",
    "novelty",
    "resource_budget",
)


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _tensor_summary(value: torch.Tensor) -> tuple[float, float, float, float]:
    """Compress a learned vector into a stable, bounded context signal."""

    flat = value.detach().flatten().to(dtype=torch.float32)
    if flat.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    scale = math.sqrt(float(flat.numel()))
    summary = torch.stack(
        (
            flat.mean(),
            flat.std(unbiased=False),
            torch.linalg.vector_norm(flat) / max(1.0, scale),
            flat.abs().max(),
        )
    )
    return tuple(float(item) for item in torch.tanh(summary))


@dataclass(frozen=True)
class ExecutiveContext:
    """Fixed-shape learned context derived from Taiji cognitive state."""

    features: torch.Tensor
    tick: int
    goal_id: str | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 1:
            raise ValueError("executive context features must be a vector")
        if self.features.numel() != len(EXECUTIVE_CONTEXT_FEATURE_NAMES):
            raise ValueError("executive context feature contract mismatch")
        if int(self.tick) < 0:
            raise ValueError("executive context tick cannot be negative")
        if self.goal_id is not None and not str(self.goal_id):
            raise ValueError("executive context goal_id cannot be empty")

    @classmethod
    def from_state(
        cls,
        state: CognitiveState,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ExecutiveContext:
        """Build context from perception, world, memory, goals and drives."""

        _unit(novelty, "executive novelty")
        _unit(resource_budget, "executive resource budget")
        percept = state.percept
        goal = max(state.goals.goals, key=lambda item: (item.priority, -item.progress, item.goal_id), default=None)
        percept_summary = _tensor_summary(
            torch.empty(0) if percept is None else percept.features
        )
        world_summary = _tensor_summary(state.world.latent)
        memory_summary = _tensor_summary(state.memory.semantic_context)
        values = (
            *percept_summary,
            *world_summary,
            *memory_summary,
            1.0 - (0.0 if goal is None else goal.progress),
            state.world.uncertainty,
            0.0 if percept is None else percept.prediction_error,
            0.0 if percept is None else percept.boundary_score,
            0.0 if percept is None else percept.confidence,
            state.memory.episodic_confidence,
            state.self_state.confidence,
            state.self_state.resource_fraction,
            state.homeostasis.curiosity,
            state.homeostasis.fatigue,
            state.homeostasis.stress,
            novelty,
            resource_budget,
        )
        return cls(
            features=torch.tensor(values, dtype=torch.float32),
            tick=state.tick,
            goal_id=None if goal is None else goal.goal_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": EXECUTIVE_CHECKPOINT_FORMAT,
            "features": self.features.detach().cpu().clone(),
            "tick": self.tick,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> ExecutiveContext:
        if payload.get("format") != EXECUTIVE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported executive context format")
        return cls(
            features=payload["features"].detach().to(device).clone(),
            tick=int(payload["tick"]),
            goal_id=payload.get("goal_id"),
        )


@dataclass(frozen=True)
class ExecutiveCandidate:
    """One structured affordance candidate offered to the executive."""

    candidate_id: str
    action_intent: ActionIntent
    content_plan: ContentPlan
    features: tuple[float, ...]
    provenance: str = "external"
    source_percept_id: str | None = None
    source_affordance_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id):
            raise ValueError("executive candidate_id cannot be empty")
        if not isinstance(self.action_intent, ActionIntent):
            raise TypeError("executive candidate action_intent must be ActionIntent")
        if not isinstance(self.content_plan, ContentPlan):
            raise TypeError("executive candidate content_plan must be ContentPlan")
        if self.action_intent.intent_id != self.content_plan.intent_id:
            raise ValueError("executive candidate action/content intent ids must match")
        if not str(self.provenance):
            raise ValueError("executive candidate provenance cannot be empty")
        if self.source_percept_id is not None and not str(self.source_percept_id):
            raise ValueError("executive candidate source_percept_id cannot be empty")
        if self.source_affordance_id is not None and not str(self.source_affordance_id):
            raise ValueError("executive candidate source_affordance_id cannot be empty")
        values = tuple(float(value) for value in self.features)
        if not values:
            raise ValueError("executive candidate features cannot be empty")
        for index, value in enumerate(values):
            _finite(value, f"executive candidate feature {index}")
        object.__setattr__(self, "features", values)

    @classmethod
    def from_world_affordance(
        cls,
        affordance: WorldAffordance,
        *,
        tick: int,
        goal: Goal | None = None,
        percept: PerceptEvent | None = None,
        features: Sequence[float] | None = None,
    ) -> ExecutiveCandidate:
        """Convert a learned/world affordance without an action lookup table."""

        if not isinstance(affordance, WorldAffordance):
            raise TypeError("affordance must be a WorldAffordance")
        candidate_id = (
            f"{percept.event_id}:{affordance.affordance_id}"
            if percept is not None
            else affordance.affordance_id
        )
        intent_id = f"{candidate_id}:intent"
        parameters = dict(affordance.parameters)
        if affordance.actor_id:
            parameters.setdefault("actor_id", affordance.actor_id)
        if affordance.target_id:
            parameters.setdefault("target_id", affordance.target_id)
        goal_id = None if goal is None else goal.goal_id
        confidence = affordance.confidence
        if features is None:
            raise ValueError(
                "world affordance candidate synthesis requires learned continuous features"
            )
        candidate_features = tuple(float(value) for value in features)
        intent = ActionIntent(
            intent_id=intent_id,
            kind=affordance.action_kind,
            parameters=parameters,
            source_goal_id=goal_id,
            confidence=confidence,
            tick=int(tick),
        )
        content = ContentPlan(
            content_id=f"{intent_id}:content",
            intent_id=intent_id,
            intent_kind=affordance.action_kind,
            semantic_slots={
                "actor_id": affordance.actor_id,
                "target_id": affordance.target_id,
                "parameters": parameters,
            },
            source_goal_id=goal_id,
            confidence=confidence,
            provenance="affordance-derived",
            tick=int(tick),
        )
        return cls(
            candidate_id=candidate_id,
            action_intent=intent,
            content_plan=content,
            features=candidate_features,
            provenance="affordance-derived/learned",
            source_percept_id=None if percept is None else percept.event_id,
            source_affordance_id=affordance.affordance_id,
        )

    @classmethod
    def synthesize_from_state(
        cls,
        state: CognitiveState,
        *,
        features_by_affordance: Mapping[str, Sequence[float]] | None = None,
    ) -> tuple[ExecutiveCandidate, ...]:
        """Produce candidates from current Taiji affordances and active goal."""

        active_goal = max(
            state.goals.goals,
            key=lambda item: (item.priority, -item.progress, item.goal_id),
            default=None,
        )
        if features_by_affordance is None:
            raise ValueError(
                "world affordance candidate synthesis requires a learned feature source"
            )
        feature_map = dict(features_by_affordance)
        missing = [
            affordance.affordance_id
            for affordance in state.world.affordances
            if affordance.affordance_id not in feature_map
        ]
        if missing:
            raise ValueError(f"missing learned affordance features: {missing}")
        return tuple(
            cls.from_world_affordance(
                affordance,
                tick=state.tick,
                goal=active_goal,
                percept=state.percept,
                features=feature_map.get(affordance.affordance_id),
            )
            for affordance in state.world.affordances
        )

    def feature_tensor(self) -> torch.Tensor:
        return torch.tensor(self.features, dtype=torch.float32)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": EXECUTIVE_CHECKPOINT_FORMAT,
            "candidate_id": self.candidate_id,
            "action_intent": self.action_intent.to_payload(),
            "content_plan": self.content_plan.to_payload(),
            "features": list(self.features),
            "provenance": self.provenance,
            "source_percept_id": self.source_percept_id,
            "source_affordance_id": self.source_affordance_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> ExecutiveCandidate:
        if payload.get("format") != EXECUTIVE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported executive candidate format")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            action_intent=ActionIntent.from_payload(payload["action_intent"], device=device),
            content_plan=ContentPlan.from_payload(payload["content_plan"]),
            features=tuple(float(value) for value in payload["features"]),
            provenance=str(payload.get("provenance", "external")),
            source_percept_id=payload.get("source_percept_id"),
            source_affordance_id=payload.get("source_affordance_id"),
        )


@dataclass(frozen=True)
class ExecutiveDecision:
    """Auditable selection result returned before an organ executes it."""

    selected: ExecutiveCandidate
    scores: Mapping[str, float]
    context: ExecutiveContext

    def __post_init__(self) -> None:
        if self.selected.candidate_id not in self.scores:
            raise ValueError("executive decision must score its selected candidate")
        if any(not math.isfinite(float(value)) for value in self.scores.values()):
            raise ValueError("executive decision scores must be finite")

    @property
    def action_intent(self) -> ActionIntent:
        return self.selected.action_intent

    @property
    def content_plan(self) -> ContentPlan:
        return self.selected.content_plan

    def to_world_action(
        self,
        *,
        tick: int | None = None,
        provenance: str = "planned",
    ) -> WorldAction:
        """Materialize the selected intent as a world-action contract."""

        return WorldAction(
            action_id=self.action_intent.intent_id,
            kind=self.action_intent.kind,
            tick=self.context.tick if tick is None else int(tick),
            parameters=self.action_intent.parameters,
            provenance=provenance,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": EXECUTIVE_CHECKPOINT_FORMAT,
            "selected": self.selected.to_payload(),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
            "context": self.context.to_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> ExecutiveDecision:
        if payload.get("format") != EXECUTIVE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported executive decision format")
        return cls(
            selected=ExecutiveCandidate.from_payload(payload["selected"], device=device),
            scores={str(key): float(value) for key, value in payload["scores"].items()},
            context=ExecutiveContext.from_payload(payload["context"], device=device),
        )


@dataclass(frozen=True)
class ExecutiveTrainingExample:
    candidate: ExecutiveCandidate
    context: ExecutiveContext
    reward: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reward)):
            raise ValueError("executive feedback reward must be finite")


class ExecutiveController:
    """Learn a state-conditioned utility over structured executive candidates."""

    def __init__(self, *, candidate_feature_dim: int = 6, seed: int = 0) -> None:
        del seed
        if int(candidate_feature_dim) <= 0:
            raise ValueError("executive candidate_feature_dim must be positive")
        self.candidate_feature_dim = int(candidate_feature_dim)
        feature_dim = self.candidate_feature_dim + len(EXECUTIVE_CONTEXT_FEATURE_NAMES)
        self._model = nn.Linear(feature_dim, 1, bias=True)
        with torch.no_grad():
            self._model.weight.zero_()
            self._model.bias.zero_()
        self.training_steps = 0

    def to(self, device: torch.device | str) -> ExecutiveController:
        self._model.to(device)
        return self

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._model.parameters())

    @property
    def feature_names(self) -> tuple[str, ...]:
        candidate_names = (
            EXECUTIVE_CANDIDATE_FEATURE_NAMES
            if self.candidate_feature_dim == len(EXECUTIVE_CANDIDATE_FEATURE_NAMES)
            else tuple(f"candidate_feature_{index}" for index in range(self.candidate_feature_dim))
        )
        return candidate_names + EXECUTIVE_CONTEXT_FEATURE_NAMES

    def _feature_matrix(
        self,
        candidates: Sequence[ExecutiveCandidate],
        context: ExecutiveContext,
    ) -> torch.Tensor:
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("executive selection requires candidates")
        if any(item.feature_tensor().numel() != self.candidate_feature_dim for item in candidates):
            raise ValueError("executive candidate feature dimension does not match controller")
        candidate_features = torch.stack([item.feature_tensor() for item in candidates])
        context_features = context.features.detach().to(dtype=torch.float32)
        return torch.cat(
            (
                candidate_features,
                context_features.unsqueeze(0).expand(len(candidates), -1),
            ),
            dim=1,
        ).to(self._model.weight.device)

    def scores(
        self,
        candidates: Sequence[ExecutiveCandidate],
        context: ExecutiveContext,
    ) -> tuple[float, ...]:
        features = self._feature_matrix(candidates, context)
        with torch.no_grad():
            return tuple(float(value) for value in self._model(features).flatten())

    def select(
        self,
        candidates: Sequence[ExecutiveCandidate],
        context: ExecutiveContext,
    ) -> ExecutiveDecision:
        candidates = tuple(candidates)
        values = self.scores(candidates, context)
        selected_index = max(range(len(candidates)), key=lambda index: values[index])
        return ExecutiveDecision(
            selected=candidates[selected_index],
            scores={candidate.candidate_id: values[index] for index, candidate in enumerate(candidates)},
            context=context,
        )

    def fit(
        self,
        examples: Sequence[ExecutiveTrainingExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
    ) -> float:
        examples = tuple(examples)
        if not examples:
            raise ValueError("executive fit requires examples")
        if int(epochs) <= 0:
            raise ValueError("executive epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("executive learning_rate must be positive")
        features = torch.cat(
            [self._feature_matrix((example.candidate,), example.context) for example in examples],
            dim=0,
        )
        targets = torch.tensor(
            [float(example.reward) for example in examples],
            dtype=features.dtype,
            device=features.device,
        )
        optimizer = torch.optim.SGD(self._model.parameters(), lr=float(learning_rate))
        final_loss = 0.0
        for _ in range(int(epochs)):
            prediction = self._model(features).flatten()
            loss = torch.mean((prediction - targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
        self.training_steps += int(epochs) * len(examples)
        return final_loss

    def update(
        self,
        decision: ExecutiveDecision,
        reward: float,
        *,
        learning_rate: float = 0.05,
    ) -> float:
        """Apply one online update from an experienced environment outcome."""

        _finite(reward, "executive reward")
        if float(learning_rate) <= 0.0:
            raise ValueError("executive online learning_rate must be positive")
        features = self._feature_matrix((decision.selected,), decision.context)
        target = torch.tensor(float(reward), dtype=features.dtype, device=features.device)
        prediction = self._model(features).reshape(())
        loss = (prediction - target) ** 2
        optimizer = torch.optim.SGD(self._model.parameters(), lr=float(learning_rate))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.training_steps += 1
        return float((prediction.detach() - target).item())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": EXECUTIVE_CHECKPOINT_FORMAT,
            "feature_names": list(self.feature_names),
            "candidate_feature_dim": self.candidate_feature_dim,
            "training_steps": self.training_steps,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._model.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> ExecutiveController:
        if payload.get("format") != EXECUTIVE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported executive checkpoint format")
        controller = cls(
            candidate_feature_dim=int(
                payload.get("candidate_feature_dim", len(EXECUTIVE_CANDIDATE_FEATURE_NAMES))
            )
        )
        if tuple(payload.get("feature_names", ())) != controller.feature_names:
            raise ValueError("executive feature contract mismatch")
        controller._model.load_state_dict(payload["state_dict"])
        controller.training_steps = int(payload.get("training_steps", 0))
        return controller
