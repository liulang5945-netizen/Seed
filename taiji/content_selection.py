"""Learned, context-conditioned selection of semantic content plans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .contracts import GoalState, WorldState
from .generation import ContentPlan

CONTENT_SELECTION_CHECKPOINT_FORMAT = "taiji-content-selection-v1"
FEATURE_NAMES = (
    "goal_signal",
    "world_signal",
    "information_gain",
    "confidence",
    "uncertainty",
    "resource_cost",
)


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class ContentSelectionContext:
    """State-derived signals presented to the content selector."""

    goal_residual: float
    world_uncertainty: float
    novelty: float = 0.0
    resource_budget: float = 1.0

    def __post_init__(self) -> None:
        _unit(self.goal_residual, "goal residual")
        _unit(self.world_uncertainty, "world uncertainty")
        _unit(self.novelty, "novelty")
        _unit(self.resource_budget, "resource budget")

    @classmethod
    def from_state(
        cls,
        goals: GoalState,
        world: WorldState,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ContentSelectionContext:
        progress = max((goal.progress for goal in goals.goals), default=0.0)
        return cls(
            goal_residual=1.0 - progress,
            world_uncertainty=world.uncertainty,
            novelty=novelty,
            resource_budget=resource_budget,
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "goal_residual": self.goal_residual,
            "world_uncertainty": self.world_uncertainty,
            "novelty": self.novelty,
            "resource_budget": self.resource_budget,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContentSelectionContext:
        return cls(
            goal_residual=float(payload["goal_residual"]),
            world_uncertainty=float(payload["world_uncertainty"]),
            novelty=float(payload.get("novelty", 0.0)),
            resource_budget=float(payload.get("resource_budget", 1.0)),
        )


@dataclass(frozen=True)
class ContentCandidate:
    """One candidate semantic message/action considered by the selector."""

    candidate_id: str
    intent_id: str
    intent_kind: str
    semantic_slots: Mapping[str, Any] = field(default_factory=dict)
    goal_id: str | None = None
    expected_outcome: str = ""
    goal_alignment: float = 0.0
    world_relevance: float = 0.0
    information_gain: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    resource_cost: float = 0.0
    tick: int = 0

    def __post_init__(self) -> None:
        for name in ("candidate_id", "intent_id", "intent_kind"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} cannot be empty")
        if self.goal_id is not None and not str(self.goal_id):
            raise ValueError("goal_id cannot be empty when provided")
        for name in (
            "goal_alignment",
            "world_relevance",
            "information_gain",
            "confidence",
            "uncertainty",
            "resource_cost",
        ):
            _unit(getattr(self, name), name)
        if int(self.tick) < 0:
            raise ValueError("content candidate tick cannot be negative")

    def features(self, context: ContentSelectionContext) -> torch.Tensor:
        return torch.tensor(
            (
                self.goal_alignment * context.goal_residual,
                self.world_relevance * context.world_uncertainty,
                self.information_gain * max(context.world_uncertainty, context.novelty),
                self.confidence,
                self.uncertainty,
                self.resource_cost * (1.0 - context.resource_budget),
            ),
            dtype=torch.float32,
        )

    def to_content_plan(self, *, provenance: str = "selected") -> ContentPlan:
        return ContentPlan(
            content_id=f"{self.intent_id}:content:{self.candidate_id}",
            intent_id=self.intent_id,
            intent_kind=self.intent_kind,
            semantic_slots=dict(self.semantic_slots),
            source_goal_id=self.goal_id,
            expected_outcome=self.expected_outcome,
            confidence=self.confidence,
            provenance=provenance,
            tick=self.tick,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "intent_id": self.intent_id,
            "intent_kind": self.intent_kind,
            "semantic_slots": dict(self.semantic_slots),
            "goal_id": self.goal_id,
            "expected_outcome": self.expected_outcome,
            "goal_alignment": self.goal_alignment,
            "world_relevance": self.world_relevance,
            "information_gain": self.information_gain,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "resource_cost": self.resource_cost,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContentCandidate:
        slots = payload.get("semantic_slots", {})
        if not isinstance(slots, Mapping):
            raise ValueError("content candidate semantic_slots must be a mapping")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            intent_id=str(payload["intent_id"]),
            intent_kind=str(payload["intent_kind"]),
            semantic_slots=dict(slots),
            goal_id=payload.get("goal_id"),
            expected_outcome=str(payload.get("expected_outcome", "")),
            goal_alignment=float(payload.get("goal_alignment", 0.0)),
            world_relevance=float(payload.get("world_relevance", 0.0)),
            information_gain=float(payload.get("information_gain", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            uncertainty=float(payload.get("uncertainty", 0.0)),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            tick=int(payload.get("tick", 0)),
        )


@dataclass(frozen=True)
class ContentTrainingExample:
    candidate: ContentCandidate
    context: ContentSelectionContext
    reward: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reward)):
            raise ValueError("content feedback reward must be finite")


@dataclass(frozen=True)
class ContentSelectionDecision:
    selected: ContentCandidate
    scores: Mapping[str, float]
    context: ContentSelectionContext

    def to_payload(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_payload(),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
            "context": self.context.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContentSelectionDecision:
        scores = payload.get("scores", {})
        if not isinstance(scores, Mapping):
            raise ValueError("content selection scores must be a mapping")
        return cls(
            selected=ContentCandidate.from_payload(payload["selected"]),
            scores={str(key): float(value) for key, value in scores.items()},
            context=ContentSelectionContext.from_payload(payload["context"]),
        )


class ContentSelector:
    """A small learned utility over context-conditioned content candidates."""

    def __init__(self, *, seed: int = 0) -> None:
        del seed
        self._model = nn.Linear(len(FEATURE_NAMES), 1, bias=True)
        with torch.no_grad():
            self._model.weight.zero_()
            self._model.bias.zero_()
        self.training_steps = 0

    def _feature_matrix(
        self,
        candidates: Sequence[ContentCandidate],
        context: ContentSelectionContext,
    ) -> torch.Tensor:
        if not candidates:
            raise ValueError("content selection requires candidates")
        return torch.stack([candidate.features(context) for candidate in candidates])

    def scores(
        self,
        candidates: Sequence[ContentCandidate],
        context: ContentSelectionContext,
    ) -> tuple[float, ...]:
        features = self._feature_matrix(candidates, context)
        with torch.no_grad():
            return tuple(float(value) for value in self._model(features).flatten())

    def select(
        self,
        candidates: Sequence[ContentCandidate],
        context: ContentSelectionContext,
    ) -> ContentSelectionDecision:
        candidates = tuple(candidates)
        values = self.scores(candidates, context)
        selected_index = max(range(len(candidates)), key=lambda index: values[index])
        return ContentSelectionDecision(
            selected=candidates[selected_index],
            scores={candidate.candidate_id: values[index] for index, candidate in enumerate(candidates)},
            context=context,
        )

    def fit(
        self,
        examples: Sequence[ContentTrainingExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
    ) -> float:
        examples = tuple(examples)
        if not examples:
            raise ValueError("content selector fit requires examples")
        if int(epochs) <= 0:
            raise ValueError("content selector epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("content selector learning_rate must be positive")
        features = torch.stack(
            [example.candidate.features(example.context) for example in examples]
        )
        targets = torch.tensor([example.reward for example in examples], dtype=torch.float32)
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

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": CONTENT_SELECTION_CHECKPOINT_FORMAT,
            "feature_names": list(FEATURE_NAMES),
            "training_steps": self.training_steps,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._model.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> ContentSelector:
        if payload.get("format") != CONTENT_SELECTION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported content selection checkpoint format")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("content selection feature contract mismatch")
        selector = cls()
        selector._model.load_state_dict(payload["state_dict"])
        selector.training_steps = int(payload.get("training_steps", 0))
        return selector
