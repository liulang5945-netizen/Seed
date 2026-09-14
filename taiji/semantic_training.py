"""Native structured semantic training between Taiji-owned contracts.

This module is the first M2.R3 training seam.  It does not parse text, call a
provider, choose a tool, or execute a workbench operation.  Instead it trains a
small, checkpointed chain over existing Taiji contracts::

    PerceptEvent -> relation facts / WorldState -> Goal -> ContentPlan

The vocabulary is derived from the training corpus.  Relation facts are
materialized into a ``WorldState`` only after the learned fact head has made a
prediction; the goal and content heads consume the preceding predictions.  A
structured canary can therefore distinguish a learned semantic bridge from a
fixed lookup table while keeping the language organ outside the measurement.

Learning uses the same detached local-delta rules as the rest of the native
peripheral learners.  There is deliberately no autograd or optimizer state in
this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    logistic_error_delta,
    softmax_error_delta,
)

SEMANTIC_TRAINING_CHECKPOINT_FORMAT = "taiji-structured-semantic-training-v1"
SEMANTIC_TRAINING_VERSION = 2
SEMANTIC_METADATA_DIM = 4
SEMANTIC_FACT_SEPARATOR = "::"


def _required_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    if SEMANTIC_FACT_SEPARATOR in normalized:
        raise ValueError(f"{name} cannot contain {SEMANTIC_FACT_SEPARATOR!r}")
    return normalized


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def semantic_fact_key(subject: str, predicate: str, object_id: str) -> str:
    """Return the stable corpus key for one relation fact."""

    return SEMANTIC_FACT_SEPARATOR.join(
        (
            _required_text(subject, "fact subject"),
            _required_text(predicate, "fact predicate"),
            _required_text(object_id, "fact object"),
        )
    )


def _relation_from_fact_key(key: str) -> tuple[str, str, str]:
    parts = str(key).split(SEMANTIC_FACT_SEPARATOR)
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(f"invalid semantic fact key: {key!r}")
    return parts[0], parts[1], parts[2]


def _semantic_input_payload(percept: PerceptEvent) -> dict[str, Any]:
    """Build an identity payload that excludes transport-only event identity."""

    return {
        "modality": percept.modality,
        "features": percept.features.detach().cpu().clone(),
        "boundary_score": percept.boundary_score,
        "prediction_error": percept.prediction_error,
        "boundary": percept.boundary,
        "confidence": percept.confidence,
        "duration": percept.duration,
    }


def semantic_input_digest(percept: PerceptEvent) -> str:
    """Return a digest for semantic content, independent of event transport IDs."""

    if not isinstance(percept, PerceptEvent):
        raise TypeError("semantic input digest requires a PerceptEvent")
    return content_digest(_semantic_input_payload(percept))


@dataclass(frozen=True)
class StructuredSemanticExample:
    """One supervised semantic bridge example with explicit owner targets."""

    example_id: str
    family_id: str
    percept: PerceptEvent
    world: WorldState
    goal: Goal
    content: ContentPlan

    def __post_init__(self) -> None:
        _required_text(self.example_id, "semantic example_id")
        _required_text(self.family_id, "semantic family_id")
        if not isinstance(self.percept, PerceptEvent):
            raise TypeError("semantic example percept must be a PerceptEvent")
        if not isinstance(self.world, WorldState):
            raise TypeError("semantic example world must be a WorldState")
        if not isinstance(self.goal, Goal):
            raise TypeError("semantic example goal must be a Goal")
        if not isinstance(self.content, ContentPlan):
            raise TypeError("semantic example content must be a ContentPlan")
        if int(self.world.tick) != int(self.percept.observation_tick):
            raise ValueError("semantic world and percept ticks must match")
        if int(self.content.tick) != int(self.percept.observation_tick):
            raise ValueError("semantic content and percept ticks must match")
        if self.content.source_goal_id not in {None, self.goal.goal_id}:
            raise ValueError("semantic content must reference the example goal")
        for relation in self.world.relations:
            if len(relation) != 3:
                raise ValueError("semantic world relations must be triples")
            semantic_fact_key(*relation)

    @property
    def fact_keys(self) -> tuple[str, ...]:
        return tuple(sorted(semantic_fact_key(*relation) for relation in self.world.relations))

    @property
    def input_digest(self) -> str:
        return semantic_input_digest(self.percept)

    def to_payload(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "percept": self.percept.to_payload(),
            "world": self.world.to_payload(),
            "goal": self.goal.to_payload(),
            "content": self.content.to_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> StructuredSemanticExample:
        return cls(
            example_id=str(payload["example_id"]),
            family_id=str(payload["family_id"]),
            percept=PerceptEvent.from_payload(payload["percept"], device=device),
            world=WorldState.from_payload(payload["world"], device=device),
            goal=Goal.from_payload(payload["goal"]),
            content=ContentPlan.from_payload(payload["content"]),
        )


def _ordered_unique(values: Iterable[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_required_text(value, name) for value in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _normalized_fact_feature_masks(
    fact_keys: tuple[str, ...],
    feature_dim: int,
    masks: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    """Validate a typed fact->feature binding into per-fact index tuples.

    The binding is the K1.1 composition contract: each semantic fact may only
    read the features that indicate its own attribute value, so a linear fact
    head cannot bind state attributes to correlated identity features.
    """

    if masks is None:
        return None
    if set(masks) != set(fact_keys):
        raise ValueError("fact feature masks must cover exactly the corpus fact keys")
    normalized: list[tuple[int, ...]] = []
    for key in fact_keys:
        indices = tuple(int(value) for value in masks[key])
        if not indices:
            raise ValueError(f"fact feature mask cannot be empty: {key}")
        if len(set(indices)) != len(indices):
            raise ValueError(f"fact feature mask repeats a feature: {key}")
        if any(not 0 <= index < feature_dim for index in indices):
            raise ValueError(f"fact feature mask index out of range: {key}")
        normalized.append(indices)
    return tuple(normalized)


def _normalized_readout_exclusions(
    fact_keys: tuple[str, ...],
    excluded: Sequence[str] | None,
) -> tuple[str, ...]:
    if excluded is None:
        return ()
    keys = tuple(str(key) for key in excluded)
    if len(set(keys)) != len(keys):
        raise ValueError("readout fact exclusions repeat a fact")
    unknown = [key for key in keys if key not in fact_keys]
    if unknown:
        raise ValueError(f"readout fact exclusions outside the corpus vocabulary: {unknown}")
    return keys


@dataclass(frozen=True)
class StructuredSemanticCorpus:
    """Record-disjoint train/dev/test corpus and its learned output vocabulary."""

    train: tuple[StructuredSemanticExample, ...]
    dev: tuple[StructuredSemanticExample, ...]
    test: tuple[StructuredSemanticExample, ...]
    feature_dim: int
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
                raise ValueError(f"semantic corpus {name} split cannot be empty")
            if any(not isinstance(item, StructuredSemanticExample) for item in values):
                raise TypeError(f"semantic corpus {name} split contains an invalid example")
        if int(self.feature_dim) <= 0:
            raise ValueError("semantic corpus feature_dim must be positive")
        if any(
            item.percept.features.numel() != int(self.feature_dim) for item in self.all_examples
        ):
            raise ValueError("semantic corpus percept feature dimensions must match")
        if tuple(self.fact_keys) != tuple(sorted(self.fact_keys)):
            raise ValueError("semantic corpus fact_keys must be sorted")
        if tuple(self.goal_ids) != tuple(sorted(self.goal_ids)):
            raise ValueError("semantic corpus goal_ids must be sorted")
        if tuple(self.content_ids) != tuple(sorted(self.content_ids)):
            raise ValueError("semantic corpus content_ids must be sorted")
        if not self.fact_keys or not self.goal_ids or not self.content_ids:
            raise ValueError("semantic corpus output vocabulary cannot be empty")
        if tuple(goal.goal_id for goal in self.goals) != self.goal_ids:
            raise ValueError("semantic corpus goal catalog does not match goal_ids")
        if tuple(plan.content_id for plan in self.content_plans) != self.content_ids:
            raise ValueError("semantic corpus content catalog does not match content_ids")
        if len(self.source_digest) != 64 or not self.source_digest:
            raise ValueError("semantic corpus source_digest is invalid")
        if self.runtime_only and self.all_examples:
            raise ValueError("runtime-only semantic corpus cannot contain examples")

    @property
    def all_examples(self) -> tuple[StructuredSemanticExample, ...]:
        return self.train + self.dev + self.test

    @classmethod
    def from_splits(
        cls,
        *,
        train: Sequence[StructuredSemanticExample],
        dev: Sequence[StructuredSemanticExample],
        test: Sequence[StructuredSemanticExample],
    ) -> StructuredSemanticCorpus:
        splits = {
            "train": tuple(train),
            "dev": tuple(dev),
            "test": tuple(test),
        }
        if any(not values for values in splits.values()):
            raise ValueError("semantic corpus train/dev/test splits cannot be empty")
        all_examples = tuple(item for values in splits.values() for item in values)
        if any(not isinstance(item, StructuredSemanticExample) for item in all_examples):
            raise TypeError("semantic corpus contains a non-example value")
        example_ids = [item.example_id for item in all_examples]
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("semantic example_id values must be globally unique")
        family_by_split: dict[str, set[str]] = {
            name: {item.family_id for item in values} for name, values in splits.items()
        }
        split_names = tuple(splits)
        for index, left_name in enumerate(split_names):
            for right_name in split_names[index + 1 :]:
                overlap = family_by_split[left_name] & family_by_split[right_name]
                if overlap:
                    raise ValueError(
                        f"semantic family leakage between {left_name}/{right_name}: {sorted(overlap)}"
                    )
                input_overlap = {item.input_digest for item in splits[left_name]} & {
                    item.input_digest for item in splits[right_name]
                }
                if input_overlap:
                    raise ValueError(f"semantic input leakage between {left_name}/{right_name}")
        feature_dims = {int(item.percept.features.numel()) for item in all_examples}
        if len(feature_dims) != 1:
            raise ValueError("semantic corpus feature dimensions must be uniform")
        raw_train_facts = tuple(fact for item in splits["train"] for fact in item.fact_keys)
        for fact in raw_train_facts:
            _relation_from_fact_key(fact)
        train_facts = tuple(sorted(set(raw_train_facts)))
        if not train_facts:
            raise ValueError("semantic fact key cannot be empty")
        train_goals = _ordered_unique(
            (item.goal.goal_id for item in splits["train"]),
            "semantic goal id",
        )
        train_content = _ordered_unique(
            (item.content.content_id for item in splits["train"]),
            "semantic content id",
        )
        for split_name in ("dev", "test"):
            split_values = splits[split_name]
            unknown_facts = {
                fact for item in split_values for fact in item.fact_keys if fact not in train_facts
            }
            if unknown_facts:
                raise ValueError(
                    f"{split_name} contains unseen semantic facts: {sorted(unknown_facts)}"
                )
            unknown_goals = {
                item.goal.goal_id for item in split_values if item.goal.goal_id not in train_goals
            }
            if unknown_goals:
                raise ValueError(
                    f"{split_name} contains unseen semantic goals: {sorted(unknown_goals)}"
                )
            unknown_content = {
                item.content.content_id
                for item in split_values
                if item.content.content_id not in train_content
            }
            if unknown_content:
                raise ValueError(
                    f"{split_name} contains unseen semantic content: {sorted(unknown_content)}"
                )
        goals_by_id = {item.goal.goal_id: item.goal for item in splits["train"]}
        content_by_id = {item.content.content_id: item.content for item in splits["train"]}
        source_payload = {
            "format": SEMANTIC_TRAINING_CHECKPOINT_FORMAT,
            "version": SEMANTIC_TRAINING_VERSION,
            "train": [item.to_payload() for item in splits["train"]],
            "dev": [item.to_payload() for item in splits["dev"]],
            "test": [item.to_payload() for item in splits["test"]],
        }
        digest = content_digest(source_payload)
        return cls(
            train=splits["train"],
            dev=splits["dev"],
            test=splits["test"],
            feature_dim=next(iter(feature_dims)),
            fact_keys=train_facts,
            goal_ids=train_goals,
            content_ids=train_content,
            goals=tuple(goals_by_id[goal_id] for goal_id in train_goals),
            content_plans=tuple(content_by_id[content_id] for content_id in train_content),
            source_digest=digest,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "format": SEMANTIC_TRAINING_CHECKPOINT_FORMAT,
            "version": SEMANTIC_TRAINING_VERSION,
            "source_digest": self.source_digest,
            "feature_dim": self.feature_dim,
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

    @classmethod
    def from_catalog_payload(cls, payload: Mapping[str, Any]) -> StructuredSemanticCorpus:
        """Reconstruct the inference vocabulary from a learner checkpoint."""

        goals = tuple(Goal.from_payload(item) for item in payload.get("goal_catalog", ()))
        content_plans = tuple(
            ContentPlan.from_payload(item) for item in payload.get("content_catalog", ())
        )
        source_digest = str(payload.get("source_digest", ""))
        corpus = cls(
            train=(),
            dev=(),
            test=(),
            feature_dim=int(payload["feature_dim"]),
            fact_keys=tuple(str(item) for item in payload.get("fact_keys", ())),
            goal_ids=tuple(str(item) for item in payload.get("goal_ids", ())),
            content_ids=tuple(str(item) for item in payload.get("content_ids", ())),
            goals=goals,
            content_plans=content_plans,
            source_digest=source_digest,
            runtime_only=True,
        )
        return corpus


@dataclass(frozen=True)
class StructuredSemanticResult:
    """A learned semantic result with explicit uncertainty and no execution."""

    status: str
    world: WorldState | None
    goal: Goal | None
    content_plan: ContentPlan | None
    fact_scores: Mapping[str, float]
    goal_scores: Mapping[str, float]
    content_scores: Mapping[str, float]
    confidence: float
    ambiguity: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "world": None if self.world is None else self.world.to_payload(),
            "goal": None if self.goal is None else self.goal.to_payload(),
            "content_plan": None if self.content_plan is None else self.content_plan.to_payload(),
            "fact_scores": {str(key): float(value) for key, value in self.fact_scores.items()},
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
    ) -> StructuredSemanticResult:
        world_payload = payload.get("world")
        goal_payload = payload.get("goal")
        content_payload = payload.get("content_plan")
        return cls(
            status=str(payload["status"]),
            world=(
                None
                if world_payload is None
                else WorldState.from_payload(world_payload, device=device)
            ),
            goal=None if goal_payload is None else Goal.from_payload(goal_payload),
            content_plan=(
                None if content_payload is None else ContentPlan.from_payload(content_payload)
            ),
            fact_scores={
                str(key): float(value) for key, value in payload.get("fact_scores", {}).items()
            },
            goal_scores={
                str(key): float(value) for key, value in payload.get("goal_scores", {}).items()
            },
            content_scores={
                str(key): float(value) for key, value in payload.get("content_scores", {}).items()
            },
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
        )


def _one_hot(indices: torch.Tensor, width: int) -> torch.Tensor:
    result = torch.zeros((indices.shape[0], int(width)), dtype=torch.float32, device=indices.device)
    result[torch.arange(indices.shape[0], device=indices.device), indices] = 1.0
    return result


class StructuredSemanticLearner(nn.Module):
    """Trainable native bridge from percept features to semantic contracts."""

    CHECKPOINT_FORMAT = SEMANTIC_TRAINING_CHECKPOINT_FORMAT
    CHECKPOINT_VERSION = SEMANTIC_TRAINING_VERSION

    def __init__(
        self,
        corpus: StructuredSemanticCorpus,
        *,
        fact_threshold: float = 0.65,
        confidence_floor: float = 0.55,
        ambiguity_ceiling: float = 0.12,
        fact_feature_masks: Mapping[str, Sequence[int]] | None = None,
        readout_excluded_facts: Sequence[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(corpus, StructuredSemanticCorpus):
            raise TypeError("structured semantic learner requires a StructuredSemanticCorpus")
        self.source_digest = corpus.source_digest
        self.feature_dim = int(corpus.feature_dim)
        self.fact_keys = tuple(corpus.fact_keys)
        self.goal_ids = tuple(corpus.goal_ids)
        self.content_ids = tuple(corpus.content_ids)
        self._goals = {goal.goal_id: goal for goal in corpus.goals}
        self._content_plans = {plan.content_id: plan for plan in corpus.content_plans}
        self.fact_threshold = _unit(fact_threshold, "semantic fact threshold")
        self.confidence_floor = _unit(confidence_floor, "semantic confidence floor")
        ambiguity_ceiling = float(ambiguity_ceiling)
        if not 0.0 <= ambiguity_ceiling <= 1.0:
            raise ValueError("semantic ambiguity ceiling must be in [0, 1]")
        self.ambiguity_ceiling = ambiguity_ceiling
        self._fact_feature_masks = _normalized_fact_feature_masks(
            self.fact_keys, self.feature_dim, fact_feature_masks
        )
        self.readout_excluded_facts = _normalized_readout_exclusions(
            self.fact_keys, readout_excluded_facts
        )
        self.input_dim = self.feature_dim + SEMANTIC_METADATA_DIM
        self.fact_head = nn.Linear(self.input_dim, len(self.fact_keys), bias=True)
        self.goal_head = nn.Linear(len(self.fact_keys), len(self.goal_ids), bias=True)
        self.content_head = nn.Linear(
            len(self.fact_keys) + len(self.goal_ids),
            len(self.content_ids),
            bias=True,
        )
        with torch.no_grad():
            for layer in (self.fact_head, self.goal_head, self.content_head):
                layer.weight.zero_()
                layer.bias.zero_()
        self.to(device)
        self._readout_input_mask: torch.Tensor | None
        if self.readout_excluded_facts:
            mask = torch.ones(len(self.fact_keys), device=self.fact_head.weight.device)
            for key in self.readout_excluded_facts:
                mask[self.fact_keys.index(key)] = 0.0
            self._readout_input_mask = mask
        else:
            self._readout_input_mask = None
        self._apply_fact_feature_masks()
        freeze_parameters(self)
        self.training_steps = 0

    def _masked_readout_input(self, fact_probabilities: torch.Tensor) -> torch.Tensor:
        """Drop identity facts from the goal/content readout inputs."""

        if self._readout_input_mask is None:
            return fact_probabilities
        return fact_probabilities * self._readout_input_mask

    @torch.no_grad()
    def _apply_fact_feature_masks(self) -> None:
        """Re-enforce the typed fact->feature binding on the fact head."""

        if self._fact_feature_masks is None:
            return
        allowed = torch.zeros_like(self.fact_head.weight)
        for row, indices in enumerate(self._fact_feature_masks):
            allowed[row, list(indices)] = 1.0
        self.fact_head.weight.mul_(allowed)

    @property
    def parameter_count(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.parameter_tensors())

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.parameters())

    def _percept_input(self, percept: PerceptEvent) -> torch.Tensor:
        if not isinstance(percept, PerceptEvent):
            raise TypeError("semantic prediction requires a PerceptEvent")
        if percept.features.numel() != self.feature_dim:
            raise ValueError("semantic percept feature dimension does not match the learner")
        metadata = torch.tensor(
            (
                percept.boundary_score,
                percept.prediction_error,
                percept.confidence,
                min(int(percept.duration), 32) / 32.0,
            ),
            dtype=torch.float32,
            device=percept.features.device,
        )
        return torch.cat((percept.features.detach().to(dtype=torch.float32), metadata)).to(
            self.fact_head.weight.device
        )

    def _training_batch(
        self, examples: Sequence[StructuredSemanticExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        examples = tuple(examples)
        if not examples:
            raise ValueError("structured semantic fit requires examples")
        inputs = torch.stack([self._percept_input(item.percept) for item in examples])
        facts = torch.zeros(
            (len(examples), len(self.fact_keys)),
            dtype=torch.float32,
            device=inputs.device,
        )
        fact_index = {key: index for index, key in enumerate(self.fact_keys)}
        for row, example in enumerate(examples):
            for key in example.fact_keys:
                if key not in fact_index:
                    raise ValueError(f"semantic example uses an unknown fact key: {key}")
                facts[row, fact_index[key]] = 1.0
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
            raise ValueError("semantic example target is outside the training vocabulary") from exc
        return inputs, facts, goals, content

    def fit(
        self,
        examples: Sequence[StructuredSemanticExample],
        *,
        epochs: int = 160,
        learning_rate: float = 2.0,
    ) -> dict[str, float]:
        """Fit each native stage with teacher-forced upstream targets."""

        if int(epochs) <= 0:
            raise ValueError("structured semantic epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("structured semantic learning_rate must be positive")
        examples = tuple(examples)
        inputs, facts, goals, content = self._training_batch(examples)
        one_hot_goals = _one_hot(goals, len(self.goal_ids))
        readout_inputs = self._masked_readout_input(facts)
        final = {"fact_loss": 0.0, "goal_loss": 0.0, "content_loss": 0.0}
        for _ in range(int(epochs)):
            fact_logits = self.fact_head(inputs)
            fact_error = logistic_error_delta(fact_logits, facts)
            apply_linear_delta(self.fact_head, inputs, fact_error, float(learning_rate))
            self._apply_fact_feature_masks()
            goal_logits = self.goal_head(readout_inputs)
            goal_error = softmax_error_delta(goal_logits, goals)
            apply_linear_delta(self.goal_head, readout_inputs, goal_error, float(learning_rate))
            content_inputs = torch.cat((readout_inputs, one_hot_goals), dim=1)
            content_logits = self.content_head(content_inputs)
            content_error = softmax_error_delta(content_logits, content)
            apply_linear_delta(
                self.content_head,
                content_inputs,
                content_error,
                float(learning_rate),
            )
            with torch.no_grad():
                final = {
                    "fact_loss": float(torch.mean((torch.sigmoid(fact_logits) - facts) ** 2)),
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
        percept: PerceptEvent,
        fact_scores: Mapping[str, float],
    ) -> WorldState:
        active = tuple(
            sorted(key for key, score in fact_scores.items() if score >= self.fact_threshold)
        )
        relations = tuple(_relation_from_fact_key(key) for key in active)
        entities = tuple(
            sorted({item for relation in relations for item in (relation[0], relation[2])})
        )
        uncertainty = 1.0
        if fact_scores:
            uncertainty = float(
                sum(1.0 - max(score, 1.0 - score) for score in fact_scores.values())
                / len(fact_scores)
            )
        return WorldState(
            tick=int(percept.observation_tick),
            latent=torch.empty(0, device=self.fact_head.weight.device),
            entities=entities,
            relations=relations,
            uncertainty=max(0.0, min(1.0, uncertainty)),
            percept_event_id=percept.event_id,
            percept_assembly_id=percept.assembly_id,
        )

    @torch.no_grad()
    def predict(self, percept: PerceptEvent) -> StructuredSemanticResult:
        """Run the full learned chain without mutating the learner."""

        inputs = self._percept_input(percept).reshape(1, -1)
        fact_probabilities = torch.sigmoid(self.fact_head(inputs)).reshape(-1)
        fact_scores = {
            key: float(fact_probabilities[index]) for index, key in enumerate(self.fact_keys)
        }
        world = self._materialize_world(percept, fact_scores)
        if percept.confidence < self.confidence_floor:
            return StructuredSemanticResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=float(percept.confidence),
                ambiguity=1.0,
            )
        active = {key for key, score in fact_scores.items() if score >= self.fact_threshold}
        conflicts = any(
            len(active.intersection(group)) > 1
            or (len(group) > 1 and all(0.45 <= fact_scores[key] <= 0.55 for key in group))
            for group in self._conflict_groups(self.fact_keys)
        )
        if conflicts:
            return StructuredSemanticResult(
                status="conflict",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        if not active:
            return StructuredSemanticResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        goal_logits = self.goal_head(self._masked_readout_input(fact_probabilities.reshape(1, -1)))
        goal_probabilities = torch.softmax(goal_logits, dim=-1).reshape(-1)
        goal_scores = {
            goal_id: float(goal_probabilities[index]) for index, goal_id in enumerate(self.goal_ids)
        }
        goal_order = torch.argsort(goal_probabilities, descending=True)
        goal_index = int(goal_order[0])
        goal_confidence = float(goal_probabilities[goal_index])
        second_goal = float(goal_probabilities[goal_order[1]]) if len(goal_order) > 1 else 0.0
        goal_ambiguity = 1.0 - max(0.0, goal_confidence - second_goal)
        goal = self._goals[self.goal_ids[goal_index]]
        if (
            goal_confidence < self.confidence_floor
            or goal_confidence - second_goal < self.ambiguity_ceiling
        ):
            return StructuredSemanticResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores=goal_scores,
                content_scores={},
                confidence=goal_confidence,
                ambiguity=goal_ambiguity,
            )
        content_input = torch.cat(
            (self._masked_readout_input(fact_probabilities), goal_probabilities)
        ).reshape(1, -1)
        content_probabilities = torch.softmax(self.content_head(content_input), dim=-1).reshape(-1)
        content_scores = {
            content_id: float(content_probabilities[index])
            for index, content_id in enumerate(self.content_ids)
        }
        content_index = int(torch.argmax(content_probabilities))
        content_confidence = float(content_probabilities[content_index])
        content = self._content_plans[self.content_ids[content_index]]
        if content_confidence < self.confidence_floor:
            return StructuredSemanticResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores=goal_scores,
                content_scores=content_scores,
                confidence=content_confidence,
                ambiguity=1.0 - content_confidence,
            )
        status = "clarify" if content.intent_kind == "request_information" else "resolved"
        return StructuredSemanticResult(
            status=status,
            world=world,
            goal=goal,
            content_plan=content,
            fact_scores=fact_scores,
            goal_scores=goal_scores,
            content_scores=content_scores,
            confidence=min(goal_confidence, content_confidence),
            ambiguity=min(goal_ambiguity, 1.0 - content_confidence),
        )

    def owner_digests(self) -> dict[str, str]:
        return {
            "semantic_fact": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.fact_head.state_dict().items()
                }
            ),
            "semantic_goal": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.goal_head.state_dict().items()
                }
            ),
            "semantic_content": content_digest(
                {
                    name: value.detach().cpu().clone()
                    for name, value in self.content_head.state_dict().items()
                }
            ),
        }

    @torch.no_grad()
    def zero_fact_head(self) -> tuple[str, str]:
        """Lesion the percept-to-fact owner for causal attribution."""

        before = self.owner_digests()["semantic_fact"]
        self.fact_head.weight.zero_()
        self.fact_head.bias.zero_()
        return before, self.owner_digests()["semantic_fact"]

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "source_digest": self.source_digest,
            "feature_dim": self.feature_dim,
            "fact_keys": list(self.fact_keys),
            "goal_ids": list(self.goal_ids),
            "content_ids": list(self.content_ids),
            "goal_catalog": [goal.to_payload() for goal in self._goals.values()],
            "content_catalog": [plan.to_payload() for plan in self._content_plans.values()],
            "fact_threshold": self.fact_threshold,
            "confidence_floor": self.confidence_floor,
            "ambiguity_ceiling": self.ambiguity_ceiling,
            "fact_feature_masks": (
                None
                if self._fact_feature_masks is None
                else [list(indices) for indices in self._fact_feature_masks]
            ),
            "readout_excluded_facts": list(self.readout_excluded_facts) or None,
            "training_steps": self.training_steps,
            "state_dict": {
                name: value.detach().cpu().clone() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        corpus: StructuredSemanticCorpus | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> StructuredSemanticLearner:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported structured semantic checkpoint format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported structured semantic checkpoint version")
        if corpus is None:
            corpus = StructuredSemanticCorpus.from_catalog_payload(payload)
        if payload.get("source_digest") != corpus.source_digest:
            raise ValueError("structured semantic checkpoint corpus digest mismatch")
        for key, expected in (
            ("feature_dim", corpus.feature_dim),
            ("fact_keys", list(corpus.fact_keys)),
            ("goal_ids", list(corpus.goal_ids)),
            ("content_ids", list(corpus.content_ids)),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"structured semantic checkpoint {key} mismatch")
        raw_masks = payload.get("fact_feature_masks")
        fact_feature_masks = (
            {
                str(key): tuple(int(index) for index in indices)
                for key, indices in zip(payload["fact_keys"], raw_masks, strict=True)
            }
            if raw_masks is not None
            else None
        )
        raw_excluded = payload.get("readout_excluded_facts")
        learner = cls(
            corpus,
            fact_threshold=float(payload.get("fact_threshold", 0.65)),
            confidence_floor=float(payload.get("confidence_floor", 0.55)),
            ambiguity_ceiling=float(payload.get("ambiguity_ceiling", 0.12)),
            fact_feature_masks=fact_feature_masks,
            readout_excluded_facts=None if raw_excluded is None else tuple(raw_excluded),
            device=device,
        )
        learner.load_state_dict(payload["state_dict"])
        learner._apply_fact_feature_masks()
        learner.training_steps = int(payload.get("training_steps", 0))
        return learner


__all__ = [
    "SEMANTIC_FACT_SEPARATOR",
    "SEMANTIC_TRAINING_CHECKPOINT_FORMAT",
    "SEMANTIC_TRAINING_VERSION",
    "StructuredSemanticCorpus",
    "StructuredSemanticExample",
    "StructuredSemanticLearner",
    "StructuredSemanticResult",
    "semantic_fact_key",
    "semantic_input_digest",
]
