"""First native learner for governed evolution experiences.

The learner consumes only normalized :class:`EvolutionExperience` DTOs.  It
does not import Seed, providers, tools, or client runtime code.  The first
training object is route/interaction credit: a Taiji-owned local readout
learns which content-addressed capability context predicts a useful outcome.
This is deliberately smaller than a full predictive-fabric update so that the
first real mutation has an isolated holdout, retention, lesion, and rollback
gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .evolution_experience import EvolutionExperience
from .internalization import GroundedFeatureExample, content_digest
from .internalization_learner import (
    InternalizationLearningReport,
    InternalizedFeatureLearner,
)

EVOLUTION_TRAINING_FORMAT = "taiji-native-evolution-training-v1"
EVOLUTION_TRAINING_VERSION = 1
EVOLUTION_TRAINING_MANIFEST_REVISION = "taiji-w7-e3-1-route-credit-v1"


def _finite_bound(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _experience_tuple(
    experiences: Iterable[EvolutionExperience],
    *,
    partition: str,
) -> tuple[EvolutionExperience, ...]:
    items = tuple(experiences)
    if not items:
        raise ValueError(f"{partition} experiences must contain at least one item")
    if any(not isinstance(item, EvolutionExperience) for item in items):
        raise TypeError(f"{partition} experiences must contain EvolutionExperience values")
    if any(item.partition != partition for item in items):
        raise ValueError(f"{partition} experiences contain a different partition")
    if len({item.experience_id for item in items}) != len(items):
        raise ValueError(f"{partition} experiences cannot contain duplicate IDs")
    return tuple(sorted(items, key=lambda item: item.experience_id))


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], float(value)))


@dataclass(frozen=True)
class NativeEvolutionLearningReport:
    """Measured result of one train-only route-credit trial."""

    parent_learner_digest: str
    child_learner_digest: str
    dataset_digest: str
    train_experiences: int
    holdout_experiences: int
    retention_experiences: int
    frozen_holdout_loss: float
    replay_only_holdout_loss: float
    native_holdout_loss: float
    native_retention_loss_before: float
    native_retention_loss_after: float
    admitted: bool
    rolled_back: bool
    consumed_experience_ids: tuple[str, ...]
    internalization: InternalizationLearningReport

    @property
    def holdout_gain_vs_frozen(self) -> float:
        return self.frozen_holdout_loss - self.native_holdout_loss

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_TRAINING_FORMAT,
            "parent_learner_digest": self.parent_learner_digest,
            "child_learner_digest": self.child_learner_digest,
            "dataset_digest": self.dataset_digest,
            "train_experiences": self.train_experiences,
            "holdout_experiences": self.holdout_experiences,
            "retention_experiences": self.retention_experiences,
            "frozen_holdout_loss": self.frozen_holdout_loss,
            "replay_only_holdout_loss": self.replay_only_holdout_loss,
            "native_holdout_loss": self.native_holdout_loss,
            "native_retention_loss_before": self.native_retention_loss_before,
            "native_retention_loss_after": self.native_retention_loss_after,
            "holdout_gain_vs_frozen": self.holdout_gain_vs_frozen,
            "admitted": self.admitted,
            "rolled_back": self.rolled_back,
            "consumed_experience_ids": list(self.consumed_experience_ids),
            "internalization": self.internalization.to_payload(),
        }


class EvolutionExperienceEncoder:
    """Create a deterministic, non-leaking route-context feature vector.

    The encoder uses hashing rather than a fixed vocabulary or manually named
    neuron types.  It intentionally excludes outcome-bearing fields such as
    success, status, reward components, result digest, and error code.  The
    feature dimension is a capacity setting; it is not a semantic taxonomy.
    """

    _IDENTITY_FIELDS = ("source_kind", "source_id", "source_version", "route_identity")

    def __init__(self, feature_dim: int, *, namespace: str = "taiji-route-credit-v1") -> None:
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("evolution feature_dim must be positive")
        self.namespace = str(namespace).strip()
        if not self.namespace:
            raise ValueError("evolution feature namespace cannot be empty")

    def _identity_payload(self, experience: EvolutionExperience) -> tuple[tuple[str, str], ...]:
        route_identity = (
            experience.capability_id
            or experience.skill_digest
            or experience.mcp_schema_digest
            or experience.plugin_digest
            or experience.source_id
        )
        return (
            ("route_identity", "|".join((experience.source_kind, experience.source_version, route_identity))),
        )

    def encode(self, experience: EvolutionExperience) -> torch.Tensor:
        if not isinstance(experience, EvolutionExperience):
            raise TypeError("evolution encoder accepts EvolutionExperience values")
        vector = torch.zeros(self.feature_dim, dtype=torch.float32)
        identity = self._identity_payload(experience)
        if not identity:
            raise ValueError("evolution experience has no stable identity fields")
        for field, value in identity:
            seed = f"{self.namespace}\0{field}\0{value}".encode()
            for probe in range(2):
                digest = hashlib.sha256(seed + bytes((probe,))).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.feature_dim
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[bucket] += sign
        norm = float(vector.norm().item())
        if norm <= 1e-8:
            raise ValueError("evolution feature encoder produced an empty vector")
        return vector / norm

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_TRAINING_FORMAT,
            "version": EVOLUTION_TRAINING_VERSION,
            "feature_dim": self.feature_dim,
            "namespace": self.namespace,
            "identity_fields": list(self._IDENTITY_FIELDS),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> EvolutionExperienceEncoder:
        if payload.get("format") != EVOLUTION_TRAINING_FORMAT:
            raise ValueError("unsupported evolution encoder format")
        if int(payload.get("version", -1)) != EVOLUTION_TRAINING_VERSION:
            raise ValueError("unsupported evolution encoder version")
        if tuple(payload.get("identity_fields", ())) != cls._IDENTITY_FIELDS:
            raise ValueError("evolution encoder identity field contract drift")
        return cls(int(payload["feature_dim"]), namespace=str(payload["namespace"]))


class NativeEvolutionTrainer:
    """Train and atomically admit Taiji-owned route credit."""

    def __init__(
        self,
        feature_dim: int,
        *,
        learning_rate: float = 0.5,
        bias_learning_rate: float = 0.0,
        reward_bounds: tuple[float, float] = (-1.0, 1.0),
        feature_namespace: str = "taiji-route-credit-v1",
        manifest_revision: str = EVOLUTION_TRAINING_MANIFEST_REVISION,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
    ) -> None:
        if len(reward_bounds) != 2:
            raise ValueError("evolution reward_bounds must contain two values")
        bounds = (_finite_bound(reward_bounds[0], "reward lower bound"), _finite_bound(reward_bounds[1], "reward upper bound"))
        if bounds[0] > bounds[1]:
            raise ValueError("evolution reward_bounds must be ordered")
        self.encoder = EvolutionExperienceEncoder(feature_dim, namespace=feature_namespace)
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("evolution training manifest_revision cannot be empty")
        self.reward_bounds = bounds
        self.success_reward = _finite_bound(success_reward, "success_reward")
        self.failure_reward = _finite_bound(failure_reward, "failure_reward")
        self.learner = InternalizedFeatureLearner(
            self.encoder.feature_dim,
            learning_rate=learning_rate,
            bias_learning_rate=bias_learning_rate,
            reward_bounds=bounds,
            manifest_revision=self.manifest_revision,
        )
        self.consumed_experience_ids: tuple[str, ...] = ()
        self.last_dataset_digest = ""
        self.revision = 0

    def _reward(self, experience: EvolutionExperience) -> tuple[float, tuple[tuple[str, float], ...]]:
        if experience.reward_components:
            terms = tuple(sorted((str(name), float(value)) for name, value in experience.reward_components))
            raw = sum(value for _, value in terms)
        else:
            raw = self.success_reward if experience.success else self.failure_reward
            terms = (("outcome", raw),)
        return _clamp(raw, self.reward_bounds), terms

    def example(self, experience: EvolutionExperience) -> GroundedFeatureExample:
        if not isinstance(experience, EvolutionExperience):
            raise TypeError("native evolution training accepts EvolutionExperience values")
        target, reward_terms = self._reward(experience)
        grounding = self.encoder.encode(experience)
        provenance = tuple(
            sorted(
                (
                    ("encoder", self.encoder.namespace),
                    ("experience", experience.experience_id),
                    ("partition", experience.partition),
                    ("source", experience.source_digest),
                )
            )
        )
        feature_payload_digest = content_digest(
            {
                "encoder": self.encoder.checkpoint(),
                "grounding": grounding,
                "provenance": provenance,
            }
        )
        example_id = content_digest(
            {
                "format": EVOLUTION_TRAINING_FORMAT,
                "experience_id": experience.experience_id,
                "partition": experience.partition,
                "feature_payload_digest": feature_payload_digest,
            }
        )
        return GroundedFeatureExample(
            example_id=example_id,
            evidence_id=experience.experience_id,
            outcome_id=experience.outcome_id or experience.experience_id,
            affordance_id=experience.capability_id or experience.source_id,
            action_kind=experience.capability_id or experience.source_id,
            grounding=grounding,
            capability_snapshot_digest=experience.capability_snapshot_id or experience.source_digest,
            parent_checkpoint_id=experience.parent_checkpoint_digest,
            feature_payload_digest=feature_payload_digest,
            reward_terms=reward_terms,
            provenance=provenance,
            target_reward=target,
            manifest_revision=self.manifest_revision,
        )

    def _examples(
        self,
        experiences: Iterable[EvolutionExperience],
        *,
        partition: str,
    ) -> tuple[GroundedFeatureExample, ...]:
        return tuple(self.example(item) for item in _experience_tuple(experiences, partition=partition))

    def consolidate(
        self,
        train_experiences: Iterable[EvolutionExperience],
        *,
        holdout_experiences: Iterable[EvolutionExperience],
        retention_experiences: Iterable[EvolutionExperience],
        passes: int = 4,
    ) -> NativeEvolutionLearningReport:
        train_items = _experience_tuple(train_experiences, partition="train")
        holdout_items = _experience_tuple(holdout_experiences, partition="holdout")
        retention_items = _experience_tuple(retention_experiences, partition="retention")
        partitions = [train_items, holdout_items, retention_items]
        seen: set[str] = set()
        for items in partitions:
            ids = {item.experience_id for item in items}
            if seen.intersection(ids):
                raise ValueError("experience partitions must be disjoint")
            seen.update(ids)
        pending_items = tuple(
            item for item in train_items if item.experience_id not in self.consumed_experience_ids
        )
        if not pending_items:
            raise ValueError("native evolution training has no new train experiences")

        train = self._examples(pending_items, partition="train")
        holdout = self._examples(holdout_items, partition="holdout")
        retention = self._examples(retention_items, partition="retention")
        dataset_digest = content_digest(
            {
                "train": [item.experience_digest for item in pending_items],
                "holdout": [item.experience_digest for item in holdout_items],
                "retention": [item.experience_digest for item in retention_items],
                "encoder": self.encoder.checkpoint(),
            }
        )
        parent_payload = self.learner.checkpoint()
        parent_digest = content_digest(parent_payload)
        frozen = InternalizedFeatureLearner.from_checkpoint(parent_payload)
        frozen_loss = frozen.mean_squared_error(holdout)
        replay_only = InternalizedFeatureLearner.from_checkpoint(parent_payload)
        replay_loss = replay_only.mean_squared_error(holdout)

        trial = type(self).from_checkpoint(self.checkpoint())
        internalization = trial.learner.consolidate(
            train,
            holdout_examples=holdout,
            retention_examples=retention,
            replay_digest=dataset_digest,
            passes=passes,
        )
        admitted = bool(internalization.passed and internalization.holdout_loss_after < frozen_loss)
        if admitted:
            self.learner = trial.learner
            self.consumed_experience_ids = tuple(
                sorted((*self.consumed_experience_ids, *(item.experience_id for item in pending_items)))
            )
            self.last_dataset_digest = dataset_digest
            self.revision += 1
            child_digest = content_digest(self.learner.checkpoint())
        else:
            child_digest = content_digest(trial.learner.checkpoint())
        return NativeEvolutionLearningReport(
            parent_learner_digest=parent_digest,
            child_learner_digest=child_digest,
            dataset_digest=dataset_digest,
            train_experiences=len(pending_items),
            holdout_experiences=len(holdout_items),
            retention_experiences=len(retention_items),
            frozen_holdout_loss=frozen_loss,
            replay_only_holdout_loss=replay_loss,
            native_holdout_loss=internalization.holdout_loss_after,
            native_retention_loss_before=internalization.retention_loss_before,
            native_retention_loss_after=internalization.retention_loss_after,
            admitted=admitted,
            rolled_back=not admitted,
            consumed_experience_ids=tuple(item.experience_id for item in pending_items)
            if admitted
            else (),
            internalization=internalization,
        )

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": EVOLUTION_TRAINING_FORMAT,
            "version": EVOLUTION_TRAINING_VERSION,
            "encoder": self.encoder.checkpoint(),
            "manifest_revision": self.manifest_revision,
            "reward_bounds": list(self.reward_bounds),
            "success_reward": self.success_reward,
            "failure_reward": self.failure_reward,
            "learner": self.learner.checkpoint(),
            "consumed_experience_ids": list(self.consumed_experience_ids),
            "last_dataset_digest": self.last_dataset_digest,
            "revision": self.revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeEvolutionTrainer:
        if payload.get("format") != EVOLUTION_TRAINING_FORMAT:
            raise ValueError("unsupported evolution training format")
        if int(payload.get("version", -1)) != EVOLUTION_TRAINING_VERSION:
            raise ValueError("unsupported evolution training version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("evolution training checkpoint digest mismatch")
        encoder = EvolutionExperienceEncoder.from_checkpoint(payload["encoder"])
        bounds = tuple(float(item) for item in payload["reward_bounds"])
        trainer = cls(
            encoder.feature_dim,
            learning_rate=float(payload["learner"]["learning_rate"]),
            bias_learning_rate=float(payload["learner"].get("bias_learning_rate", 0.0)),
            reward_bounds=bounds,  # type: ignore[arg-type]
            feature_namespace=encoder.namespace,
            manifest_revision=str(payload["manifest_revision"]),
            success_reward=float(payload["success_reward"]),
            failure_reward=float(payload["failure_reward"]),
        )
        trainer.learner = InternalizedFeatureLearner.from_checkpoint(payload["learner"])
        if trainer.learner.feature_dim != encoder.feature_dim:
            raise ValueError("evolution learner and encoder dimensions differ")
        if trainer.learner.manifest_revision != trainer.manifest_revision:
            raise ValueError("evolution learner manifest revision drift")
        trainer.consumed_experience_ids = tuple(
            sorted(str(item) for item in payload.get("consumed_experience_ids", ()))
        )
        trainer.last_dataset_digest = str(payload.get("last_dataset_digest", ""))
        trainer.revision = int(payload.get("revision", 0))
        return trainer


__all__ = [
    "EVOLUTION_TRAINING_FORMAT",
    "EVOLUTION_TRAINING_MANIFEST_REVISION",
    "EVOLUTION_TRAINING_VERSION",
    "EvolutionExperienceEncoder",
    "NativeEvolutionLearningReport",
    "NativeEvolutionTrainer",
]
