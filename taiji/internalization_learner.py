"""Native R5A-S1 consolidation for grounded Taiji feature examples.

This learner is intentionally smaller than a general-purpose model.  It proves
the lifecycle needed before a grounded description can be considered for
internalization: a parent checkpoint is captured, a detached trial state is
updated from train-only examples, holdout and lesion measurements remain read
only, and only then is the trial atomically adopted.  There is no optimizer,
provider, action table, or text path in this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .internalization import GroundedFeatureExample, content_digest

INTERNALIZATION_LEARNER_CHECKPOINT_FORMAT = "taiji-internalization-learner-v1"


def _examples(
    value: Iterable[GroundedFeatureExample], name: str
) -> tuple[GroundedFeatureExample, ...]:
    items = tuple(value)
    if not items:
        raise ValueError(f"{name} must contain at least one grounded example")
    if any(not isinstance(item, GroundedFeatureExample) for item in items):
        raise TypeError(f"{name} must contain GroundedFeatureExample values")
    if len({item.example_id for item in items}) != len(items):
        raise ValueError(f"{name} cannot contain duplicate example IDs")
    return tuple(sorted(items, key=lambda item: item.example_id))


def _finite_rate(value: float, name: str) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return rate


def _finite_non_negative_rate(value: float, name: str) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return rate


@dataclass(frozen=True)
class InternalizationLearningReport:
    """Immutable measurements for one checkpointed consolidation trial."""

    parent_checkpoint_digest: str
    child_checkpoint_digest: str
    replay_digest: str
    train_examples: int
    holdout_examples: int
    train_loss_before: float
    train_loss_after: float
    holdout_loss_before: float
    holdout_loss_after: float
    holdout_internalized_lesion_loss: float
    holdout_grounding_lesion_loss: float
    retention_loss_before: float
    retention_loss_after: float
    fit_updates: int
    ranking_updates: int
    online_updates: int
    lineage_depth: int

    @property
    def holdout_gain(self) -> float:
        return self.holdout_loss_before - self.holdout_loss_after

    @property
    def passed(self) -> bool:
        return bool(
            self.train_loss_after < self.train_loss_before
            and self.holdout_loss_after < self.holdout_loss_before
            and self.holdout_internalized_lesion_loss > self.holdout_loss_after
            and self.holdout_grounding_lesion_loss > self.holdout_loss_after
            and self.retention_loss_after <= self.retention_loss_before + 0.05
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_LEARNER_CHECKPOINT_FORMAT,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "child_checkpoint_digest": self.child_checkpoint_digest,
            "replay_digest": self.replay_digest,
            "train_examples": self.train_examples,
            "holdout_examples": self.holdout_examples,
            "train_loss_before": self.train_loss_before,
            "train_loss_after": self.train_loss_after,
            "holdout_loss_before": self.holdout_loss_before,
            "holdout_loss_after": self.holdout_loss_after,
            "holdout_internalized_lesion_loss": self.holdout_internalized_lesion_loss,
            "holdout_grounding_lesion_loss": self.holdout_grounding_lesion_loss,
            "retention_loss_before": self.retention_loss_before,
            "retention_loss_after": self.retention_loss_after,
            "holdout_gain": self.holdout_gain,
            "fit_updates": self.fit_updates,
            "ranking_updates": self.ranking_updates,
            "online_updates": self.online_updates,
            "lineage_depth": self.lineage_depth,
            "passed": self.passed,
        }


class InternalizedFeatureLearner:
    """A checkpointed native readout over grounded feature vectors.

    The update is a normalized local delta rule.  It has no fixed action or
    language vocabulary: any grounded feature dimension can participate, and
    the scalar target is the experienced outcome carried by the example.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        learning_rate: float = 0.5,
        bias_learning_rate: float | None = None,
        reward_bounds: tuple[float, float] = (-1.0, 1.0),
        pairwise_margin: float = 0.0,
        manifest_revision: str = "taiji-w7-r5-internalization-v1",
        device: torch.device | str = "cpu",
    ) -> None:
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("internalization learner feature_dim must be positive")
        self.learning_rate = _finite_rate(learning_rate, "internalization learner learning_rate")
        self.bias_learning_rate = _finite_non_negative_rate(
            self.learning_rate if bias_learning_rate is None else bias_learning_rate,
            "internalization learner bias_learning_rate",
        )
        if len(reward_bounds) != 2 or not reward_bounds[0] <= reward_bounds[1]:
            raise ValueError("internalization learner reward_bounds must be ordered")
        if not all(math.isfinite(float(item)) for item in reward_bounds):
            raise ValueError("internalization learner reward_bounds must be finite")
        self.reward_bounds = (float(reward_bounds[0]), float(reward_bounds[1]))
        self.pairwise_margin = float(pairwise_margin)
        if not math.isfinite(self.pairwise_margin) or not 0.0 <= self.pairwise_margin <= 1.0:
            raise ValueError("internalization learner pairwise_margin must be within [0, 1]")
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("internalization learner manifest_revision cannot be empty")
        self.weights = torch.zeros(self.feature_dim, dtype=torch.float32, device=device)
        self.bias = torch.zeros((), dtype=torch.float32, device=device)
        self.fit_updates = 0
        self.ranking_updates = 0
        self.online_updates = 0
        self.revision = 0
        self.replay_digest = ""
        self.parent_checkpoint_digest = ""
        self.lineage: tuple[str, ...] = ()

    def _features(self, example: GroundedFeatureExample) -> torch.Tensor:
        if example.grounding.numel() != self.feature_dim:
            raise ValueError("grounded example feature dimension does not match the learner")
        features = example.grounding.detach().to(
            device=self.weights.device, dtype=self.weights.dtype
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("grounded example features must be finite")
        return features

    def _target(self, example: GroundedFeatureExample) -> float:
        target = float(example.target_reward)
        lower, upper = self.reward_bounds
        if not lower <= target <= upper:
            raise ValueError("grounded example target_reward is outside learner bounds")
        return target

    def score(
        self,
        example: GroundedFeatureExample,
        *,
        internalized_enabled: bool = True,
        grounding_enabled: bool = True,
    ) -> float:
        """Predict an experienced outcome, with explicit lesion switches."""

        features = self._features(example)
        if not grounding_enabled:
            features = torch.zeros_like(features)
        if not internalized_enabled:
            return 0.0
        with torch.no_grad():
            return float(torch.dot(self.weights, features) + self.bias)

    def mean_squared_error(
        self,
        examples: Iterable[GroundedFeatureExample],
        *,
        internalized_enabled: bool = True,
        grounding_enabled: bool = True,
    ) -> float:
        """Measure a partition without changing weights, counters, or lineage."""

        items = _examples(examples, "evaluation examples")
        errors = []
        for item in items:
            errors.append(
                (
                    self.score(
                        item,
                        internalized_enabled=internalized_enabled,
                        grounding_enabled=grounding_enabled,
                    )
                    - self._target(item)
                )
                ** 2
            )
        return float(sum(errors) / len(errors))

    def _apply_update(self, example: GroundedFeatureExample) -> None:
        features = self._features(example)
        target = self._target(example)
        prediction = self.score(example)
        error = prediction - target
        normalizer = float(torch.dot(features, features)) + 1.0
        step = self.learning_rate / normalizer
        with torch.no_grad():
            self.weights.add_(features, alpha=-step * error)
            self.bias.add_(-self.bias_learning_rate / normalizer * error)

    def _apply_pairwise_update(
        self,
        preferred: GroundedFeatureExample,
        other: GroundedFeatureExample,
    ) -> bool:
        """Apply a local margin update using only two observed train rewards.

        The caller establishes ``preferred`` from its observed reward, never
        from an external description or a holdout answer.  Bias cancels from a
        pairwise comparison, so the update changes only the grounded feature
        direction needed to separate the two experienced outcomes.
        """

        if self.pairwise_margin <= 0.0:
            return False
        if not self._target(preferred) > self._target(other):
            raise ValueError("pairwise preference must follow strictly higher train reward")
        difference = self._features(preferred) - self._features(other)
        squared_norm = float(torch.dot(difference, difference))
        if squared_norm <= 1e-12:
            raise ValueError("pairwise preference requires distinguishable grounded features")
        observed_margin = self.score(preferred) - self.score(other)
        deficit = self.pairwise_margin - observed_margin
        if deficit <= 0.0:
            return False
        with torch.no_grad():
            self.weights.add_(difference, alpha=self.learning_rate * deficit / squared_norm)
        return True

    def online_update(self, example: GroundedFeatureExample) -> None:
        """Apply one post-checkpoint online update and retain its counter."""

        if not isinstance(example, GroundedFeatureExample):
            raise TypeError("online update accepts a GroundedFeatureExample")
        self._apply_update(example)
        self.online_updates += 1
        self.revision += 1

    def _ranking_pairs(
        self,
        value: Iterable[tuple[GroundedFeatureExample, GroundedFeatureExample]],
        train_ids: set[str],
    ) -> tuple[tuple[GroundedFeatureExample, GroundedFeatureExample], ...]:
        """Validate deterministic train-only preference pairs for one trial."""

        pairs = tuple(value)
        seen: set[tuple[str, str]] = set()
        validated: list[tuple[GroundedFeatureExample, GroundedFeatureExample]] = []
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("ranking_pairs must contain (preferred, other) tuples")
            preferred, other = pair
            if not isinstance(preferred, GroundedFeatureExample) or not isinstance(
                other, GroundedFeatureExample
            ):
                raise TypeError("ranking_pairs must contain GroundedFeatureExample values")
            if preferred.example_id not in train_ids or other.example_id not in train_ids:
                raise ValueError("ranking pairs cannot use holdout or retention-only examples")
            if preferred.example_id == other.example_id:
                raise ValueError("ranking pair examples must be distinct")
            if not self._target(preferred) > self._target(other):
                raise ValueError("ranking pairs must follow strictly higher train reward")
            identity = (preferred.example_id, other.example_id)
            if identity in seen:
                raise ValueError("ranking_pairs cannot contain duplicates")
            seen.add(identity)
            validated.append((preferred, other))
        return tuple(sorted(validated, key=lambda pair: (pair[0].example_id, pair[1].example_id)))

    def consolidate(
        self,
        train_examples: Iterable[GroundedFeatureExample],
        *,
        holdout_examples: Iterable[GroundedFeatureExample],
        retention_examples: Iterable[GroundedFeatureExample],
        replay_digest: str = "",
        passes: int = 4,
        ranking_pairs: Iterable[tuple[GroundedFeatureExample, GroundedFeatureExample]] = (),
    ) -> InternalizationLearningReport:
        """Run an atomic parent-to-child native consolidation trial.

        The parent payload is captured before the trial clone is mutated.  The
        caller receives a report only after train, holdout, retention, and two
        lesion measurements are available.  Holdout and retention examples are
        evaluated but never sent through ``_apply_update``.
        """

        train = _examples(train_examples, "train examples")
        holdout = _examples(holdout_examples, "holdout examples")
        retention = _examples(retention_examples, "retention examples")
        train_ids = {item.example_id for item in train}
        if train_ids.intersection(item.example_id for item in holdout):
            raise ValueError("train and holdout partitions must be disjoint")
        if int(passes) <= 0:
            raise ValueError("consolidation passes must be positive")
        pairs = self._ranking_pairs(ranking_pairs, train_ids)
        digest = str(replay_digest).strip() or content_digest(
            {"partition": "train", "examples": [item.to_payload() for item in train]}
        )
        parent_payload = self.checkpoint()
        parent_digest = content_digest(parent_payload)
        trial = type(self).from_checkpoint(parent_payload, device=self.weights.device)
        trial.parent_checkpoint_digest = parent_digest
        trial.lineage = (*self.lineage, parent_digest)
        train_before = self.mean_squared_error(train)
        holdout_before = self.mean_squared_error(holdout)
        retention_before = self.mean_squared_error(retention)
        for _ in range(int(passes)):
            for item in train:
                trial._apply_update(item)
                trial.fit_updates += 1
            for preferred, other in pairs:
                if trial._apply_pairwise_update(preferred, other):
                    trial.ranking_updates += 1
        trial.replay_digest = digest
        trial.revision += 1
        train_after = trial.mean_squared_error(train)
        holdout_after = trial.mean_squared_error(holdout)
        retention_after = trial.mean_squared_error(retention)
        internalized_lesion = trial.mean_squared_error(holdout, internalized_enabled=False)
        grounding_lesion = trial.mean_squared_error(holdout, grounding_enabled=False)
        child_digest = content_digest(trial.checkpoint())
        self._adopt(trial)
        return InternalizationLearningReport(
            parent_checkpoint_digest=parent_digest,
            child_checkpoint_digest=child_digest,
            replay_digest=digest,
            train_examples=len(train),
            holdout_examples=len(holdout),
            train_loss_before=train_before,
            train_loss_after=train_after,
            holdout_loss_before=holdout_before,
            holdout_loss_after=holdout_after,
            holdout_internalized_lesion_loss=internalized_lesion,
            holdout_grounding_lesion_loss=grounding_lesion,
            retention_loss_before=retention_before,
            retention_loss_after=retention_after,
            fit_updates=self.fit_updates,
            ranking_updates=self.ranking_updates,
            online_updates=self.online_updates,
            lineage_depth=len(self.lineage),
        )

    def _adopt(self, trial: InternalizedFeatureLearner) -> None:
        self.weights = trial.weights.detach().clone()
        self.bias = trial.bias.detach().clone()
        self.fit_updates = trial.fit_updates
        self.ranking_updates = trial.ranking_updates
        self.online_updates = trial.online_updates
        self.revision = trial.revision
        self.replay_digest = trial.replay_digest
        self.parent_checkpoint_digest = trial.parent_checkpoint_digest
        self.lineage = trial.lineage

    def checkpoint(self) -> dict[str, Any]:
        """Return the complete continuation state, including lineage."""

        return {
            "format": INTERNALIZATION_LEARNER_CHECKPOINT_FORMAT,
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "bias_learning_rate": self.bias_learning_rate,
            "reward_bounds": list(self.reward_bounds),
            "pairwise_margin": self.pairwise_margin,
            "manifest_revision": self.manifest_revision,
            "weights": self.weights.detach().cpu().clone(),
            "bias": self.bias.detach().cpu().clone(),
            "fit_updates": self.fit_updates,
            "ranking_updates": self.ranking_updates,
            "online_updates": self.online_updates,
            "revision": self.revision,
            "replay_digest": self.replay_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> InternalizedFeatureLearner:
        if payload.get("format") != INTERNALIZATION_LEARNER_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization learner checkpoint format")
        bounds = tuple(float(item) for item in payload["reward_bounds"])
        learner = cls(
            int(payload["feature_dim"]),
            learning_rate=float(payload["learning_rate"]),
            bias_learning_rate=float(payload.get("bias_learning_rate", payload["learning_rate"])),
            reward_bounds=bounds,  # type: ignore[arg-type]
            pairwise_margin=float(payload.get("pairwise_margin", 0.0)),
            manifest_revision=str(payload["manifest_revision"]),
            device=device,
        )
        learner.weights.copy_(payload["weights"].detach().to(device=device, dtype=torch.float32))
        learner.bias.copy_(payload["bias"].detach().to(device=device, dtype=torch.float32))
        learner.fit_updates = int(payload.get("fit_updates", 0))
        learner.ranking_updates = int(payload.get("ranking_updates", 0))
        learner.online_updates = int(payload.get("online_updates", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.replay_digest = str(payload.get("replay_digest", ""))
        learner.parent_checkpoint_digest = str(payload.get("parent_checkpoint_digest", ""))
        learner.lineage = tuple(str(item) for item in payload.get("lineage", ()))
        return learner


__all__ = [
    "INTERNALIZATION_LEARNER_CHECKPOINT_FORMAT",
    "InternalizationLearningReport",
    "InternalizedFeatureLearner",
]
