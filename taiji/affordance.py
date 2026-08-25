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
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .contracts import WorldAffordance

AFFORDANCE_FEATURE_CHECKPOINT_FORMAT = "taiji-affordance-features-v1"


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
        self.fit_updates = 0
        self.online_updates = 0

    def _produced_grounding(
        self,
        grounding: torch.Tensor,
        *,
        percept_features: torch.Tensor | None = None,
        world_latent: torch.Tensor | None = None,
        world_uncertainty: float = 0.0,
    ) -> torch.Tensor:
        """Build the Taiji-owned grounding without symbolic action lookup."""

        vector = _vector(grounding, name="affordance grounding", dimension=self.input_dim)
        device = self.encoder.weight.device
        vector = vector.to(device)
        if self.context_dim == 0:
            return vector
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
        return torch.tanh(self.grounding_producer(producer_input))

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
        return self.encode(
            affordance.features,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        ).detach().clone()

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
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        losses: list[float] = []
        self.train()
        for _ in range(int(epochs)):
            groundings = torch.stack(
                [
                    self._produced_grounding(
                        item.grounding,
                        percept_features=(item.percept_features if self.context_dim else None),
                        world_latent=(item.world_latent if self.context_dim else None),
                        world_uncertainty=item.world_uncertainty,
                    )
                    for item in examples
                ]
            )
            prediction = self.outcome_head(torch.tanh(self.encoder(groundings))).flatten()
            loss = torch.mean((prediction - targets) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
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
        grounding = self._produced_grounding(
            affordance.features,
            percept_features=percept_features,
            world_latent=world_latent,
            world_uncertainty=world_uncertainty,
        )
        target = torch.tensor(
            float(_finite(reward, "affordance online reward")),
            dtype=torch.float32,
            device=grounding.device,
        )
        optimizer = torch.optim.SGD(self.parameters(), lr=float(learning_rate))
        self.train()
        error = 0.0
        for _ in range(int(repeats)):
            prediction = self.outcome_head(torch.tanh(self.encoder(grounding))).reshape(())
            error = float((prediction.detach() - target).cpu())
            loss = (prediction - target) ** 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
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
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
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
