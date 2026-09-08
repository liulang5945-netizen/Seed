"""Native event-to-world transition learning for M2.R3.R3.

The static structured semantic learner proves that a percept can be mapped to
facts, a goal, and a content plan.  This module adds the missing temporal
owner: it learns fact deltas from ``WorldState + PerceptEvent`` and only then
derives the next Goal and ContentPlan.  It is deliberately separate from the
static fact head so a world-transition lesion has an observable causal effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import Goal, PerceptEvent, WorldState
from .generation import ContentPlan
from .internalization import content_digest
from .local_learning import (
    apply_linear_delta,
    freeze_parameters,
    mean_squared_error_delta,
    softmax_error_delta,
)
from .semantic_training import SEMANTIC_METADATA_DIM, semantic_fact_key

SEMANTIC_TRANSITION_CHECKPOINT_FORMAT = "taiji-structured-semantic-transition-v2"
SEMANTIC_TRANSITION_VERSION = 2


def _required_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _relation_from_fact_key(key: str) -> tuple[str, str, str]:
    parts = str(key).split("::")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(f"invalid transition fact key: {key!r}")
    return parts[0], parts[1], parts[2]


def _event_input_payload(event: PerceptEvent) -> dict[str, Any]:
    return {
        "modality": event.modality,
        "features": event.features.detach().cpu().clone(),
        "boundary_score": event.boundary_score,
        "prediction_error": event.prediction_error,
        "boundary": event.boundary,
        "confidence": event.confidence,
        "duration": event.duration,
    }


@dataclass(frozen=True)
class StructuredSemanticTransitionExample:
    """One supervised fact-delta transition with semantic readout targets."""

    example_id: str
    family_id: str
    before: WorldState
    event: PerceptEvent
    after: WorldState
    goal: Goal
    content: ContentPlan

    def __post_init__(self) -> None:
        _required_text(self.example_id, "transition example_id")
        _required_text(self.family_id, "transition family_id")
        if not isinstance(self.before, WorldState) or not isinstance(self.after, WorldState):
            raise TypeError("transition before/after must be WorldState values")
        if not isinstance(self.event, PerceptEvent):
            raise TypeError("transition event must be a PerceptEvent")
        if not isinstance(self.goal, Goal):
            raise TypeError("transition goal must be a Goal")
        if not isinstance(self.content, ContentPlan):
            raise TypeError("transition content must be a ContentPlan")
        if int(self.after.tick) != int(self.before.tick) + 1:
            raise ValueError("transition after tick must follow before tick by one")
        if int(self.event.observation_tick) != int(self.after.tick):
            raise ValueError("transition event tick must match after tick")
        if int(self.content.tick) != int(self.after.tick):
            raise ValueError("transition content tick must match after tick")
        if self.content.source_goal_id not in {None, self.goal.goal_id}:
            raise ValueError("transition content must reference the transition goal")
        for world in (self.before, self.after):
            for relation in world.relations:
                if len(relation) != 3:
                    raise ValueError("transition world relations must be triples")
                semantic_fact_key(*relation)

    @property
    def before_fact_keys(self) -> tuple[str, ...]:
        return tuple(sorted(semantic_fact_key(*relation) for relation in self.before.relations))

    @property
    def after_fact_keys(self) -> tuple[str, ...]:
        return tuple(sorted(semantic_fact_key(*relation) for relation in self.after.relations))

    @property
    def input_digest(self) -> str:
        return content_digest(
            {
                "before_fact_keys": list(self.before_fact_keys),
                "event": _event_input_payload(self.event),
            }
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "before": self.before.to_payload(),
            "event": self.event.to_payload(),
            "after": self.after.to_payload(),
            "goal": self.goal.to_payload(),
            "content": self.content.to_payload(),
        }


def _ordered_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_required_text(value, name) for value in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


@dataclass(frozen=True)
class StructuredSemanticTransitionCorpus:
    """Record-disjoint transition corpus and its semantic output catalog."""

    train: tuple[StructuredSemanticTransitionExample, ...]
    dev: tuple[StructuredSemanticTransitionExample, ...]
    test: tuple[StructuredSemanticTransitionExample, ...]
    event_dim: int
    fact_keys: tuple[str, ...]
    goal_ids: tuple[str, ...]
    content_ids: tuple[str, ...]
    goals: tuple[Goal, ...]
    content_plans: tuple[ContentPlan, ...]
    source_digest: str
    runtime_only: bool = False

    def __post_init__(self) -> None:
        for name in ("train", "dev", "test"):
            values = tuple(getattr(self, name))
            if not self.runtime_only and not values:
                raise ValueError(f"transition corpus {name} split cannot be empty")
            if any(not isinstance(item, StructuredSemanticTransitionExample) for item in values):
                raise TypeError(f"transition corpus {name} contains an invalid example")
        if int(self.event_dim) <= 0:
            raise ValueError("transition event_dim must be positive")
        if any(item.event.features.numel() != int(self.event_dim) for item in self.all_examples):
            raise ValueError("transition event dimensions must match")
        if tuple(self.fact_keys) != tuple(sorted(self.fact_keys)):
            raise ValueError("transition fact_keys must be sorted")
        if tuple(self.goal_ids) != tuple(sorted(self.goal_ids)):
            raise ValueError("transition goal_ids must be sorted")
        if tuple(self.content_ids) != tuple(sorted(self.content_ids)):
            raise ValueError("transition content_ids must be sorted")
        if not self.fact_keys or not self.goal_ids or not self.content_ids:
            raise ValueError("transition output vocabulary cannot be empty")
        if tuple(goal.goal_id for goal in self.goals) != self.goal_ids:
            raise ValueError("transition goal catalog does not match goal_ids")
        if tuple(plan.content_id for plan in self.content_plans) != self.content_ids:
            raise ValueError("transition content catalog does not match content_ids")
        if len(self.source_digest) != 64 or not self.source_digest:
            raise ValueError("transition source_digest is invalid")
        if self.runtime_only and self.all_examples:
            raise ValueError("runtime-only transition corpus cannot contain examples")

    @property
    def all_examples(self) -> tuple[StructuredSemanticTransitionExample, ...]:
        return self.train + self.dev + self.test

    @classmethod
    def from_splits(
        cls,
        *,
        train: Sequence[StructuredSemanticTransitionExample],
        dev: Sequence[StructuredSemanticTransitionExample],
        test: Sequence[StructuredSemanticTransitionExample],
    ) -> StructuredSemanticTransitionCorpus:
        splits = {"train": tuple(train), "dev": tuple(dev), "test": tuple(test)}
        if any(not values for values in splits.values()):
            raise ValueError("transition train/dev/test splits cannot be empty")
        all_examples = tuple(item for values in splits.values() for item in values)
        if any(not isinstance(item, StructuredSemanticTransitionExample) for item in all_examples):
            raise TypeError("transition corpus contains a non-example value")
        example_ids = [item.example_id for item in all_examples]
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("transition example_id values must be globally unique")
        split_names = tuple(splits)
        for index, left_name in enumerate(split_names):
            for right_name in split_names[index + 1 :]:
                left = splits[left_name]
                right = splits[right_name]
                family_overlap = {item.family_id for item in left} & {
                    item.family_id for item in right
                }
                if family_overlap:
                    raise ValueError(
                        f"transition family leakage between {left_name}/{right_name}"
                    )
                input_overlap = {item.input_digest for item in left} & {
                    item.input_digest for item in right
                }
                if input_overlap:
                    raise ValueError(
                        f"transition input leakage between {left_name}/{right_name}"
                    )
        event_dims = {int(item.event.features.numel()) for item in all_examples}
        if len(event_dims) != 1:
            raise ValueError("transition event dimensions must be uniform")
        raw_facts = tuple(
            fact
            for item in splits["train"]
            for fact in item.before_fact_keys + item.after_fact_keys
        )
        for fact in raw_facts:
            _relation_from_fact_key(fact)
        fact_keys = tuple(sorted(set(raw_facts)))
        if not fact_keys:
            raise ValueError("transition fact vocabulary cannot be empty")
        goal_ids = _ordered_unique(
            (item.goal.goal_id for item in splits["train"]), "transition goal id"
        )
        content_ids = _ordered_unique(
            (item.content.content_id for item in splits["train"]), "transition content id"
        )
        for split_name in ("dev", "test"):
            unknown_facts = {
                fact
                for item in splits[split_name]
                for fact in item.before_fact_keys + item.after_fact_keys
                if fact not in fact_keys
            }
            if unknown_facts:
                raise ValueError(f"{split_name} contains unseen transition facts")
            unknown_goals = {
                item.goal.goal_id for item in splits[split_name] if item.goal.goal_id not in goal_ids
            }
            if unknown_goals:
                raise ValueError(f"{split_name} contains unseen transition goals")
            unknown_content = {
                item.content.content_id
                for item in splits[split_name]
                if item.content.content_id not in content_ids
            }
            if unknown_content:
                raise ValueError(f"{split_name} contains unseen transition content")
        goals_by_id = {item.goal.goal_id: item.goal for item in splits["train"]}
        content_by_id = {item.content.content_id: item.content for item in splits["train"]}
        source_digest = content_digest(
            {
                "format": SEMANTIC_TRANSITION_CHECKPOINT_FORMAT,
                "version": SEMANTIC_TRANSITION_VERSION,
                "train": [item.to_payload() for item in splits["train"]],
                "dev": [item.to_payload() for item in splits["dev"]],
                "test": [item.to_payload() for item in splits["test"]],
            }
        )
        return cls(
            train=splits["train"],
            dev=splits["dev"],
            test=splits["test"],
            event_dim=next(iter(event_dims)),
            fact_keys=fact_keys,
            goal_ids=goal_ids,
            content_ids=content_ids,
            goals=tuple(goals_by_id[goal_id] for goal_id in goal_ids),
            content_plans=tuple(content_by_id[content_id] for content_id in content_ids),
            source_digest=source_digest,
        )

    @classmethod
    def from_catalog_payload(
        cls, payload: Mapping[str, Any]
    ) -> StructuredSemanticTransitionCorpus:
        corpus = cls(
            train=(),
            dev=(),
            test=(),
            event_dim=int(payload["event_dim"]),
            fact_keys=tuple(str(item) for item in payload.get("fact_keys", ())),
            goal_ids=tuple(str(item) for item in payload.get("goal_ids", ())),
            content_ids=tuple(str(item) for item in payload.get("content_ids", ())),
            goals=tuple(Goal.from_payload(item) for item in payload.get("goal_catalog", ())),
            content_plans=tuple(
                ContentPlan.from_payload(item) for item in payload.get("content_catalog", ())
            ),
            source_digest=str(payload.get("source_digest", "")),
            runtime_only=True,
        )
        return corpus

    def manifest(self) -> dict[str, Any]:
        return {
            "format": SEMANTIC_TRANSITION_CHECKPOINT_FORMAT,
            "version": SEMANTIC_TRANSITION_VERSION,
            "source_digest": self.source_digest,
            "event_dim": self.event_dim,
            "split_sizes": {
                "train": len(self.train),
                "dev": len(self.dev),
                "test": len(self.test),
            },
            "fact_keys": list(self.fact_keys),
            "goal_ids": list(self.goal_ids),
            "content_ids": list(self.content_ids),
            "record_disjoint": True,
            "provider_attached": False,
            "runtime_only": self.runtime_only,
        }


@dataclass(frozen=True)
class StructuredSemanticTransitionResult:
    status: str
    world: WorldState | None
    goal: Goal | None
    content_plan: ContentPlan | None
    fact_scores: Mapping[str, float]
    delta_scores: Mapping[str, float]
    goal_scores: Mapping[str, float]
    content_scores: Mapping[str, float]
    confidence: float
    ambiguity: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "world": None if self.world is None else self.world.to_payload(),
            "goal": None if self.goal is None else self.goal.to_payload(),
            "content_plan": (
                None if self.content_plan is None else self.content_plan.to_payload()
            ),
            "fact_scores": {str(key): float(value) for key, value in self.fact_scores.items()},
            "delta_scores": {str(key): float(value) for key, value in self.delta_scores.items()},
            "goal_scores": {str(key): float(value) for key, value in self.goal_scores.items()},
            "content_scores": {
                str(key): float(value) for key, value in self.content_scores.items()
            },
            "confidence": float(self.confidence),
            "ambiguity": float(self.ambiguity),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> StructuredSemanticTransitionResult:
        world_payload = payload.get("world")
        goal_payload = payload.get("goal")
        content_payload = payload.get("content_plan")
        return cls(
            status=str(payload["status"]),
            world=None
            if world_payload is None
            else WorldState.from_payload(world_payload, device=device),
            goal=None if goal_payload is None else Goal.from_payload(goal_payload),
            content_plan=None
            if content_payload is None
            else ContentPlan.from_payload(content_payload),
            fact_scores={str(key): float(value) for key, value in payload.get("fact_scores", {}).items()},
            delta_scores={str(key): float(value) for key, value in payload.get("delta_scores", {}).items()},
            goal_scores={str(key): float(value) for key, value in payload.get("goal_scores", {}).items()},
            content_scores={
                str(key): float(value)
                for key, value in payload.get("content_scores", {}).items()
            },
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
        )


def _one_hot(indices: torch.Tensor, width: int) -> torch.Tensor:
    result = torch.zeros((indices.shape[0], int(width)), dtype=torch.float32, device=indices.device)
    result[torch.arange(indices.shape[0], device=indices.device), indices] = 1.0
    return result


class StructuredSemanticTransitionLearner(nn.Module):
    """Learn persistent fact deltas and read out the resulting semantic state."""

    CHECKPOINT_FORMAT = SEMANTIC_TRANSITION_CHECKPOINT_FORMAT
    CHECKPOINT_VERSION = SEMANTIC_TRANSITION_VERSION

    def __init__(
        self,
        corpus: StructuredSemanticTransitionCorpus,
        *,
        fact_threshold: float = 0.55,
        confidence_floor: float = 0.55,
        ambiguity_ceiling: float = 0.12,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(corpus, StructuredSemanticTransitionCorpus):
            raise TypeError("transition learner requires a StructuredSemanticTransitionCorpus")
        self.source_digest = corpus.source_digest
        self.event_dim = int(corpus.event_dim)
        self.fact_keys = tuple(corpus.fact_keys)
        self.goal_ids = tuple(corpus.goal_ids)
        self.content_ids = tuple(corpus.content_ids)
        self._goals = {goal.goal_id: goal for goal in corpus.goals}
        self._content_plans = {plan.content_id: plan for plan in corpus.content_plans}
        self.fact_threshold = _unit(fact_threshold, "transition fact threshold")
        self.confidence_floor = _unit(confidence_floor, "transition confidence floor")
        ambiguity_ceiling = float(ambiguity_ceiling)
        if not 0.0 <= ambiguity_ceiling <= 1.0:
            raise ValueError("transition ambiguity ceiling must be in [0, 1]")
        self.ambiguity_ceiling = ambiguity_ceiling
        self.transition_context_dim = self.event_dim + SEMANTIC_METADATA_DIM
        self.transition_input_dim = (
            len(self.fact_keys)
            + self.transition_context_dim
            + len(self.fact_keys) * self.transition_context_dim
        )
        self.transition_head = nn.Linear(self.transition_input_dim, len(self.fact_keys), bias=True)
        self.goal_head = nn.Linear(len(self.fact_keys), len(self.goal_ids), bias=True)
        self.content_head = nn.Linear(
            len(self.fact_keys) + len(self.goal_ids),
            len(self.content_ids),
            bias=True,
        )
        with torch.no_grad():
            for layer in (self.transition_head, self.goal_head, self.content_head):
                layer.weight.zero_()
                layer.bias.zero_()
        self.to(device)
        freeze_parameters(self)
        self.training_steps = 0

    @property
    def parameter_count(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.parameter_tensors())

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.parameters())

    def _event_vector(self, event: PerceptEvent) -> torch.Tensor:
        if not isinstance(event, PerceptEvent):
            raise TypeError("transition prediction requires a PerceptEvent")
        if event.features.numel() != self.event_dim:
            raise ValueError("transition event feature dimension does not match the learner")
        metadata = torch.tensor(
            (
                event.boundary_score,
                event.prediction_error,
                event.confidence,
                min(int(event.duration), 32) / 32.0,
            ),
            dtype=torch.float32,
            device=event.features.device,
        )
        return torch.cat((event.features.detach().to(dtype=torch.float32), metadata)).to(
            self.transition_head.weight.device
        )

    def _fact_vector(self, world: WorldState) -> torch.Tensor:
        if not isinstance(world, WorldState):
            raise TypeError("transition state must be a WorldState")
        active = {semantic_fact_key(*relation) for relation in world.relations}
        unknown = active.difference(self.fact_keys)
        if unknown:
            raise ValueError(f"transition world contains unknown facts: {sorted(unknown)}")
        return torch.tensor(
            [1.0 if key in active else 0.0 for key in self.fact_keys],
            dtype=torch.float32,
            device=self.transition_head.weight.device,
        )

    def _transition_input(
        self, current: torch.Tensor, event_context: torch.Tensor
    ) -> torch.Tensor:
        """Build additive and pairwise state/event features for local learning.

        The pairwise block is a generic interaction basis, not a semantic lookup
        table.  It lets one event express a different learned delta depending on
        which fact is currently active, which is required for compositional
        transitions such as toggles and state-conditioned updates.
        """

        if current.ndim != 2 or event_context.ndim != 2:
            raise ValueError("transition inputs must be two-dimensional batches")
        if current.shape[0] != event_context.shape[0]:
            raise ValueError("transition inputs need matching batch rows")
        if current.shape[1] != len(self.fact_keys):
            raise ValueError("transition fact input width does not match the learner")
        if event_context.shape[1] != self.transition_context_dim:
            raise ValueError("transition event input width does not match the learner")
        interaction = (current.unsqueeze(2) * event_context.unsqueeze(1)).reshape(
            current.shape[0], -1
        )
        return torch.cat((current, event_context, interaction), dim=1)

    def _training_batch(
        self, examples: Sequence[StructuredSemanticTransitionExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        examples = tuple(examples)
        if not examples:
            raise ValueError("transition fit requires examples")
        current = torch.stack([self._fact_vector(item.before) for item in examples])
        events = torch.stack([self._event_vector(item.event) for item in examples])
        inputs = self._transition_input(current, events)
        next_state = torch.stack([self._fact_vector(item.after) for item in examples])
        delta = next_state - current
        goal_index = {goal_id: index for index, goal_id in enumerate(self.goal_ids)}
        content_index = {content_id: index for index, content_id in enumerate(self.content_ids)}
        try:
            goals = torch.tensor(
                [goal_index[item.goal.goal_id] for item in examples],
                dtype=torch.long,
                device=inputs.device,
            )
            content = torch.tensor(
                [content_index[item.content.content_id] for item in examples],
                dtype=torch.long,
                device=inputs.device,
            )
        except KeyError as exc:
            raise ValueError("transition target is outside the training vocabulary") from exc
        return inputs, delta, next_state, goals, content

    def fit(
        self,
        examples: Sequence[StructuredSemanticTransitionExample],
        *,
        epochs: int = 220,
        learning_rate: float = 0.5,
    ) -> dict[str, float]:
        if int(epochs) <= 0:
            raise ValueError("transition epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("transition learning_rate must be positive")
        examples = tuple(examples)
        inputs, delta, next_state, goals, content = self._training_batch(examples)
        one_hot_goals = _one_hot(goals, len(self.goal_ids))
        final = {"transition_loss": 0.0, "goal_loss": 0.0, "content_loss": 0.0}
        for _ in range(int(epochs)):
            predicted_delta = self.transition_head(inputs)
            apply_linear_delta(
                self.transition_head,
                inputs,
                mean_squared_error_delta(predicted_delta, delta),
                float(learning_rate),
            )
            goal_logits = self.goal_head(next_state)
            apply_linear_delta(
                self.goal_head,
                next_state,
                softmax_error_delta(goal_logits, goals),
                float(learning_rate),
            )
            content_inputs = torch.cat((next_state, one_hot_goals), dim=1)
            content_logits = self.content_head(content_inputs)
            apply_linear_delta(
                self.content_head,
                content_inputs,
                softmax_error_delta(content_logits, content),
                float(learning_rate),
            )
            with torch.no_grad():
                final = {
                    "transition_loss": float(torch.mean((predicted_delta - delta) ** 2)),
                    "goal_loss": float(
                        torch.mean((torch.softmax(goal_logits, dim=-1) - one_hot_goals) ** 2)
                    ),
                    "content_loss": float(
                        torch.mean(
                            (
                                torch.softmax(content_logits, dim=-1)
                                - _one_hot(content, len(self.content_ids))
                            )
                            ** 2
                        )
                    ),
                }
        self.training_steps += int(epochs) * len(examples)
        return final

    @staticmethod
    def _conflict_groups(fact_keys: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        groups: dict[tuple[str, str], list[str]] = {}
        for key in fact_keys:
            subject, predicate, _ = _relation_from_fact_key(key)
            groups.setdefault((subject, predicate), []).append(key)
        return tuple(tuple(sorted(values)) for values in groups.values() if len(values) > 1)

    def _materialize_world(
        self,
        previous: WorldState,
        event: PerceptEvent,
        fact_scores: Mapping[str, float],
    ) -> WorldState:
        active = tuple(
            sorted(key for key, score in fact_scores.items() if score >= self.fact_threshold)
        )
        relations = tuple(_relation_from_fact_key(key) for key in active)
        entities = tuple(sorted({item for relation in relations for item in (relation[0], relation[2])}))
        uncertainty = sum(
            1.0 - max(score, 1.0 - score) for score in fact_scores.values()
        ) / max(1, len(fact_scores))
        return WorldState(
            tick=int(event.observation_tick),
            latent=previous.latent.detach().clone(),
            entities=entities,
            relations=relations,
            uncertainty=max(0.0, min(1.0, float(uncertainty))),
            percept_event_id=event.event_id,
            percept_assembly_id=event.assembly_id,
        )

    @torch.no_grad()
    def predict(
        self,
        previous: WorldState,
        event: PerceptEvent,
    ) -> StructuredSemanticTransitionResult:
        current = self._fact_vector(previous)
        current_scores = {
            key: float(value)
            for key, value in zip(self.fact_keys, current, strict=True)
        }
        if event.confidence < self.confidence_floor:
            return StructuredSemanticTransitionResult(
                status="unknown",
                world=previous,
                goal=None,
                content_plan=None,
                fact_scores=current_scores,
                delta_scores={},
                goal_scores={},
                content_scores={},
                confidence=float(event.confidence),
                ambiguity=1.0,
            )
        current_active = {
            key for key, score in current_scores.items() if score >= self.fact_threshold
        }
        if any(len(current_active.intersection(group)) > 1 for group in self._conflict_groups(self.fact_keys)):
            return StructuredSemanticTransitionResult(
                status="conflict",
                world=previous,
                goal=None,
                content_plan=None,
                fact_scores=current_scores,
                delta_scores={},
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        transition_input = self._transition_input(
            current.reshape(1, -1), self._event_vector(event).reshape(1, -1)
        )
        delta_values = self.transition_head(transition_input).reshape(-1)
        next_values = torch.clamp(current + delta_values, 0.0, 1.0)
        fact_scores = {
            key: float(value)
            for key, value in zip(self.fact_keys, next_values, strict=True)
        }
        delta_scores = {
            key: float(value)
            for key, value in zip(self.fact_keys, delta_values, strict=True)
        }
        world = self._materialize_world(previous, event, fact_scores)
        active = {key for key, score in fact_scores.items() if score >= self.fact_threshold}
        conflicts = any(
            len(active.intersection(group)) > 1
            or (
                len(group) > 1
                and all(0.45 <= fact_scores[key] <= 0.55 for key in group)
            )
            for group in self._conflict_groups(self.fact_keys)
        )
        if conflicts:
            return StructuredSemanticTransitionResult(
                status="conflict",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                delta_scores=delta_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        if not active:
            return StructuredSemanticTransitionResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                delta_scores=delta_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        goal_probabilities = torch.softmax(self.goal_head(next_values.reshape(1, -1)), dim=-1).reshape(-1)
        goal_scores = {
            goal_id: float(value)
            for goal_id, value in zip(self.goal_ids, goal_probabilities, strict=True)
        }
        order = torch.argsort(goal_probabilities, descending=True)
        goal_index = int(order[0])
        goal_confidence = float(goal_probabilities[goal_index])
        second_goal = float(goal_probabilities[order[1]]) if len(order) > 1 else 0.0
        goal_ambiguity = 1.0 - max(0.0, goal_confidence - second_goal)
        goal = self._goals[self.goal_ids[goal_index]]
        if goal_confidence < self.confidence_floor or goal_confidence - second_goal < self.ambiguity_ceiling:
            return StructuredSemanticTransitionResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                delta_scores=delta_scores,
                goal_scores=goal_scores,
                content_scores={},
                confidence=goal_confidence,
                ambiguity=goal_ambiguity,
            )
        content_input = torch.cat((next_values, goal_probabilities)).reshape(1, -1)
        content_probabilities = torch.softmax(self.content_head(content_input), dim=-1).reshape(-1)
        content_scores = {
            content_id: float(value)
            for content_id, value in zip(
                self.content_ids, content_probabilities, strict=True
            )
        }
        content_index = int(torch.argmax(content_probabilities))
        content_confidence = float(content_probabilities[content_index])
        content = self._content_plans[self.content_ids[content_index]]
        if content_confidence < self.confidence_floor:
            return StructuredSemanticTransitionResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                delta_scores=delta_scores,
                goal_scores=goal_scores,
                content_scores=content_scores,
                confidence=content_confidence,
                ambiguity=1.0 - content_confidence,
            )
        status = "clarify" if content.intent_kind == "request_information" else "resolved"
        return StructuredSemanticTransitionResult(
            status=status,
            world=world,
            goal=goal,
            content_plan=content,
            fact_scores=fact_scores,
            delta_scores=delta_scores,
            goal_scores=goal_scores,
            content_scores=content_scores,
            confidence=min(goal_confidence, content_confidence),
            ambiguity=min(goal_ambiguity, 1.0 - content_confidence),
        )

    def owner_digests(self) -> dict[str, str]:
        return {
            "semantic_transition": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.transition_head.state_dict().items()
                }
            ),
            "semantic_transition_goal": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.goal_head.state_dict().items()
                }
            ),
            "semantic_transition_content": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.content_head.state_dict().items()
                }
            ),
        }

    @torch.no_grad()
    def zero_transition_head(self) -> tuple[str, str]:
        before = self.owner_digests()["semantic_transition"]
        self.transition_head.weight.zero_()
        self.transition_head.bias.zero_()
        return before, self.owner_digests()["semantic_transition"]

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "source_digest": self.source_digest,
            "event_dim": self.event_dim,
            "fact_keys": list(self.fact_keys),
            "goal_ids": list(self.goal_ids),
            "content_ids": list(self.content_ids),
            "goal_catalog": [goal.to_payload() for goal in self._goals.values()],
            "content_catalog": [plan.to_payload() for plan in self._content_plans.values()],
            "fact_threshold": self.fact_threshold,
            "confidence_floor": self.confidence_floor,
            "ambiguity_ceiling": self.ambiguity_ceiling,
            "training_steps": self.training_steps,
            "state_dict": {
                name: value.detach().cpu().clone() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        corpus: StructuredSemanticTransitionCorpus | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> StructuredSemanticTransitionLearner:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported structured semantic transition checkpoint format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported structured semantic transition checkpoint version")
        if corpus is None:
            corpus = StructuredSemanticTransitionCorpus.from_catalog_payload(payload)
        if payload.get("source_digest") != corpus.source_digest:
            raise ValueError("structured semantic transition corpus digest mismatch")
        for key, expected in (
            ("event_dim", corpus.event_dim),
            ("fact_keys", list(corpus.fact_keys)),
            ("goal_ids", list(corpus.goal_ids)),
            ("content_ids", list(corpus.content_ids)),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"structured semantic transition checkpoint {key} mismatch")
        learner = cls(
            corpus,
            fact_threshold=float(payload.get("fact_threshold", 0.55)),
            confidence_floor=float(payload.get("confidence_floor", 0.55)),
            ambiguity_ceiling=float(payload.get("ambiguity_ceiling", 0.12)),
            device=device,
        )
        learner.load_state_dict(payload["state_dict"])
        learner.training_steps = int(payload.get("training_steps", 0))
        return learner


__all__ = [
    "SEMANTIC_TRANSITION_CHECKPOINT_FORMAT",
    "SEMANTIC_TRANSITION_VERSION",
    "StructuredSemanticTransitionCorpus",
    "StructuredSemanticTransitionExample",
    "StructuredSemanticTransitionLearner",
    "StructuredSemanticTransitionResult",
]
