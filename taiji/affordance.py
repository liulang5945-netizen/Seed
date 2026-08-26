"""Taiji-owned continuous features for structured world affordances.

The world organ supplies a numeric grounding vector for each affordance.  This
module learns a small continuous projection from that grounding; it never
indexes ``action_kind``, ``affordance_id`` or a fixed action table.  Executive
selection consumes the projected features, while the outcome head provides a
development-time objective for learning the projection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from .contracts import WorldAffordance, WorldState
from .local_learning import (
    LocalAdam,
    apply_sgd_step,
    backproject_linear,
    clip_gradient_norm,
    freeze_parameters,
    linear_gradients,
    mean_squared_error_delta,
    tanh_delta,
)

AFFORDANCE_FEATURE_CHECKPOINT_FORMAT = "taiji-affordance-features-v1"


def _attribute_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        scalar = float(value.detach().cpu())
        return scalar if math.isfinite(scalar) else None
    return None


def _summary(values: Sequence[float]) -> torch.Tensor:
    if not values:
        return torch.zeros(4, dtype=torch.float32)
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    return torch.stack(
        (
            tensor.mean(),
            tensor.std(unbiased=False),
            torch.linalg.vector_norm(tensor) / math.sqrt(float(tensor.numel())),
            tensor.max() - tensor.min(),
        )
    )


class WorldAffordanceGroundingProducer:
    """Create raw affordance grounding from object/relation world lineage.

    This is a world-organ boundary, not a symbolic action encoder.  It uses
    only numeric state summaries and binding structure; ``action_kind`` and
    ``affordance_id`` never become feature-table indices.
    """

    BASE_FEATURE_DIM = 17

    def __init__(self, grounding_dim: int) -> None:
        if int(grounding_dim) <= 0:
            raise ValueError("world affordance grounding_dim must be positive")
        self.grounding_dim = int(grounding_dim)

    def _base_features(
        self,
        state: WorldState,
        affordance: WorldAffordance,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        objects = {item.object_id: item for item in state.objects}
        lineage = {f"world-state:{state.tick}"}

        def object_summary(object_id: str) -> torch.Tensor:
            if not object_id:
                return torch.zeros(4, dtype=torch.float32)
            lineage.add(f"object:{object_id}")
            obj = objects.get(object_id)
            if obj is None:
                lineage.add(f"object-missing:{object_id}")
                return torch.zeros(4, dtype=torch.float32)
            values = [
                number
                for _, value in obj.attributes
                if (number := _attribute_number(value)) is not None
            ]
            return _summary(values)

        actor = object_summary(affordance.actor_id)
        target = object_summary(affordance.target_id)
        relevant_relations = tuple(
            relation
            for relation in state.relations
            if affordance.actor_id
            and affordance.target_id
            and (
                relation[0] in (affordance.actor_id, affordance.target_id)
                or relation[2] in (affordance.actor_id, affordance.target_id)
            )
        )
        for subject_id, predicate, object_id in relevant_relations:
            lineage.add(f"relation:{subject_id}:{predicate}:{object_id}")
        direct_relations = tuple(
            relation
            for relation in relevant_relations
            if (relation[0] == affordance.actor_id and relation[2] == affordance.target_id)
            or (relation[0] == affordance.target_id and relation[2] == affordance.actor_id)
        )
        relation_features = torch.tensor(
            (
                float(len(state.relations)),
                float(len(relevant_relations)),
                float(len(direct_relations)),
                float(len({relation[1] for relation in relevant_relations})),
            ),
            dtype=torch.float32,
        )
        latent = _summary(
            [
                number
                for value in state.latent.detach().flatten().cpu().tolist()
                if (number := _attribute_number(value)) is not None
            ]
        )
        base = torch.cat(
            (
                actor,
                target,
                relation_features,
                latent,
                torch.tensor((affordance.confidence,), dtype=torch.float32),
            )
        )
        return base, tuple(sorted(lineage))

    def ground(self, state: WorldState, affordance: WorldAffordance) -> WorldAffordance:
        if not isinstance(state, WorldState):
            raise TypeError("state must be a WorldState")
        if not isinstance(affordance, WorldAffordance):
            raise TypeError("affordance must be a WorldAffordance")
        base, lineage = self._base_features(state, affordance)
        pooled = torch.stack(
            tuple(
                (
                    base[index % base.numel() :: self.grounding_dim].mean()
                    if index < base.numel()
                    else base[index % base.numel()]
                )
                for index in range(self.grounding_dim)
            )
        )
        return replace(
            affordance,
            features=pooled,
            feature_provenance="world-state-grounding",
            grounding_lineage=lineage,
        )


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(value: torch.Tensor, *, name: str, dimension: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if dimension is not None and value.numel() != int(dimension):
        raise ValueError(f"{name} dimension does not match the feature contract")
    if value.numel() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")
    return value.detach().clone().to(dtype=torch.float32)


@dataclass(frozen=True)
class AffordanceFeatureTrainingExample:
    """One grounded affordance outcome used to learn the feature source."""

    example_id: str
    affordance_id: str
    action_kind: str
    grounding: torch.Tensor
    reward: float
    percept_features: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    world_latent: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    world_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.example_id):
            raise ValueError("affordance feature example_id cannot be empty")
        if not str(self.affordance_id):
            raise ValueError("affordance feature affordance_id cannot be empty")
        if not str(self.action_kind):
            raise ValueError("affordance feature action_kind cannot be empty")
        _finite(self.reward, "affordance feature reward")
        if not 0.0 <= float(self.world_uncertainty) <= 1.0:
            raise ValueError("affordance feature world_uncertainty must be in [0, 1]")
        object.__setattr__(
            self,
            "grounding",
            _vector(self.grounding, name="affordance feature grounding"),
        )
        object.__setattr__(
            self,
            "percept_features",
            _vector(self.percept_features, name="affordance feature percept_features"),
        )
        object.__setattr__(
            self,
            "world_latent",
            _vector(self.world_latent, name="affordance feature world_latent"),
        )


class LearnedAffordanceFeatures(nn.Module):
    """Learn a reusable continuous projection for world affordance groundings."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        *,
        context_dim: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("affordance feature input_dim must be positive")
        if int(feature_dim) <= 0:
            raise ValueError("affordance feature feature_dim must be positive")
        if int(context_dim) < 0:
            raise ValueError("affordance feature context_dim cannot be negative")
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim)
        producer_input_dim = self.input_dim + (2 * self.context_dim) + 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.grounding_producer = nn.Linear(producer_input_dim, self.input_dim)
            self.encoder = nn.Linear(self.input_dim, self.feature_dim)
            self.outcome_head = nn.Linear(self.feature_dim, 1)
        freeze_parameters(self)
        self.fit_updates = 0
        self.online_updates = 0

    @property
    def _trainable(self) -> tuple[torch.Tensor, ...]:
        layers = (self.grounding_producer, self.encoder, self.outcome_head)
        return tuple(tensor for layer in layers for tensor in (layer.weight, layer.bias))

    def _local_pass(
        self, producer_inputs: torch.Tensor | None, groundings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Derive the mean-squared-error gradients of the whole reward stack.

        ``groundings`` is the ``(rows, input_dim)`` producer output (or the raw
        grounding when ``context_dim`` is zero and the producer is bypassed).
        ``producer_inputs`` is the matrix that produced it, or ``None`` when the
        producer took no part in the forward pass, in which case its gradients
        are reported as zeros so the parameter tuple stays stable.

        The prediction is returned rather than a scalar loss because the two
        callers need different scalars from it: ``fit`` reports the mean squared
        error while ``online_update`` reports the signed residual, and a loss
        alone has already discarded the sign.
        """

        with torch.no_grad():
            features = torch.tanh(self.encoder(groundings))
            prediction = self.outcome_head(features)
        error = mean_squared_error_delta(prediction, targets)
        head_gradients = linear_gradients(self.outcome_head, features, error)
        feature_error = tanh_delta(backproject_linear(self.outcome_head, error), features)
        encoder_gradients = linear_gradients(self.encoder, groundings, feature_error)
        if producer_inputs is None:
            producer_gradients: tuple[torch.Tensor, ...] = (
                torch.zeros_like(self.grounding_producer.weight),
                torch.zeros_like(self.grounding_producer.bias),
            )
        else:
            grounding_error = tanh_delta(
                backproject_linear(self.encoder, feature_error), groundings
            )
            producer_gradients = linear_gradients(
                self.grounding_producer, producer_inputs, grounding_error
            )
        gradients = clip_gradient_norm(
            (*producer_gradients, *encoder_gradients, *head_gradients), max_norm=1.0
        )
        return prediction, gradients

    def _produced_grounding(
        self,
        grounding: torch.Tensor,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> torch.Tensor:
        """Build the Taiji-owned grounding without symbolic action lookup."""

        return self._grounding_with_producer_input(
            grounding,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )[1]

    def _grounding_with_producer_input(
        self,
        grounding: torch.Tensor,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return the producer input alongside the grounding it produced.

        Local credit assignment needs the producer's own input to build its
        weight gradient, which a plain forward pass would otherwise discard.
        The input is ``None`` when ``context_dim`` is zero, because the producer
        is then bypassed entirely.
        """

        vector = _vector(grounding, name="affordance grounding", dimension=self.input_dim)
        device = self.encoder.weight.device
        vector = vector.to(device)
        if self.context_dim == 0:
            return None, vector
        if percept_features is None or world_latent is None:
            raise ValueError("contextual affordance grounding requires percept and world features")
        percept = _vector(
            percept_features,
            name="affordance percept features",
            dimension=self.context_dim,
        ).to(device)
        world = _vector(
            world_latent,
            name="affordance world latent",
            dimension=self.context_dim,
        ).to(device)
        if not 0.0 <= float(world_uncertainty) <= 1.0:
            raise ValueError("affordance world_uncertainty must be in [0, 1]")
        producer_input = torch.cat(
            (vector, percept, world, torch.tensor([float(world_uncertainty)], device=device))
        )
        with torch.no_grad():
            produced = torch.tanh(self.grounding_producer(producer_input))
        return producer_input, produced

    def encode(
        self,
        grounding: torch.Tensor,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> torch.Tensor:
        """Return a learned continuous vector without symbolic identifiers."""

        produced = self._produced_grounding(
            grounding,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )
        features = torch.tanh(self.encoder(produced))
        return features

    @torch.no_grad()
    def features_for(
        self,
        affordance: WorldAffordance,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> torch.Tensor:
        if not isinstance(affordance, WorldAffordance):
            raise TypeError("affordance must be a WorldAffordance")
        if affordance.features.numel() == 0:
            raise ValueError(
                "WorldAffordance requires numeric grounding before learned feature synthesis"
            )
        return (
            self.encode(
                affordance.features,
                percept_features=percept_features,
                world_latent=world_latent,
                world_uncertainty=world_uncertainty,
            )
            .detach()
            .clone()
        )

    @torch.no_grad()
    def predict_reward(
        self,
        grounding: torch.Tensor,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> float:
        features = self.encode(
            grounding,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )
        return float(self.outcome_head(features).reshape(()).detach().cpu())

    @torch.no_grad()
    def predict_affordance_reward(
        self,
        affordance: WorldAffordance,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> float:
        return self.predict_reward(
            affordance.features,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )

    def fit(
        self,
        examples: Sequence[AffordanceFeatureTrainingExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
    ) -> list[float]:
        examples = tuple(examples)
        if not examples:
            raise ValueError("affordance feature fit requires examples")
        if int(epochs) <= 0:
            raise ValueError("affordance feature epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("affordance feature learning_rate must be positive")
        targets = torch.tensor(
            [item.reward for item in examples],
            dtype=torch.float32,
            device=self.encoder.weight.device,
        ).reshape(-1, 1)
        optimizer = LocalAdam(self._trainable, learning_rate=float(learning_rate))
        losses: list[float] = []
        self.train()
        for _ in range(int(epochs)):
            produced = [
                self._grounding_with_producer_input(
                    item.grounding,
                    percept_features=(item.percept_features if self.context_dim else None),
                    world_latent=(item.world_latent if self.context_dim else None),
                    world_uncertainty=item.world_uncertainty,
                )
                for item in examples
            ]
            groundings = torch.stack([grounding for _, grounding in produced])
            producer_inputs = (
                torch.stack([inputs for inputs, _ in produced]) if self.context_dim else None
            )
            prediction, gradients = self._local_pass(producer_inputs, groundings, targets)
            losses.append(float(torch.mean((prediction - targets) ** 2)))
            optimizer.apply(gradients)
        self.eval()
        self.fit_updates += int(epochs) * len(examples)
        return losses

    def online_update(
        self,
        affordance: WorldAffordance,
        reward: float,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
        learning_rate: float = 0.01,
        repeats: int = 1,
    ) -> float:
        """Correct the source from one experienced world outcome."""

        if not isinstance(affordance, WorldAffordance):
            raise TypeError("affordance must be a WorldAffordance")
        if float(learning_rate) <= 0.0 or int(repeats) <= 0:
            raise ValueError("affordance online learning_rate and repeats must be positive")
        producer_inputs, grounding = self._grounding_with_producer_input(
            affordance.features,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )
        target = torch.tensor(
            float(_finite(reward, "affordance online reward")),
            dtype=torch.float32,
            device=grounding.device,
        ).reshape(1, 1)
        rows = grounding.reshape(1, -1)
        producer_rows = None if producer_inputs is None else producer_inputs.reshape(1, -1)
        self.train()
        error = 0.0
        for _ in range(int(repeats)):
            prediction, gradients = self._local_pass(producer_rows, rows, target)
            error = float((prediction - target).reshape(()).cpu())
            apply_sgd_step(self._trainable, gradients, float(learning_rate))
        self.eval()
        self.online_updates += 1
        return error

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": AFFORDANCE_FEATURE_CHECKPOINT_FORMAT,
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "context_dim": self.context_dim,
            "fit_updates": self.fit_updates,
            "online_updates": self.online_updates,
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> LearnedAffordanceFeatures:
        if payload.get("format") != AFFORDANCE_FEATURE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported affordance feature checkpoint format")
        learner = cls(
            int(payload["input_dim"]),
            int(payload["feature_dim"]),
            context_dim=int(payload.get("context_dim", 0)),
            seed=0,
        )
        learner.load_state_dict(payload["state_dict"])
        learner.fit_updates = int(payload.get("fit_updates", 0))
        learner.online_updates = int(payload.get("online_updates", 0))
        learner.eval()
        return learner
