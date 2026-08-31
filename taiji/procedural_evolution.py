"""Governed procedural-memory intake from native evolution experiences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import ActionIntent, EpisodicMemoryRecord
from .evolution_experience import EvolutionExperience
from .evolution_training import EvolutionExperienceEncoder
from .internalization import content_digest
from .procedural_memory import ProceduralMemoryLearner

PROCEDURAL_EVOLUTION_FORMAT = "taiji-native-procedural-evolution-v1"
PROCEDURAL_EVOLUTION_VERSION = 1
PROCEDURAL_EVOLUTION_MANIFEST_REVISION = "taiji-w7-e3-2-procedural-memory-v1"


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


def _action_kind(experience: EvolutionExperience) -> str:
    action_kind = str(experience.capability_id or experience.source_id).strip()
    if not action_kind:
        raise ValueError("procedural experience must identify an action kind")
    return action_kind


@dataclass(frozen=True)
class NativeProceduralLearningReport:
    """Measured result of one train-only procedural-memory trial."""

    parent_learner_digest: str
    child_learner_digest: str
    dataset_digest: str
    train_experiences: int
    excluded_train_experiences: int
    holdout_experiences: int
    retention_experiences: int
    frozen_holdout_accuracy: float
    replay_only_holdout_accuracy: float
    native_holdout_accuracy: float
    frozen_retention_accuracy: float
    native_retention_accuracy: float
    native_consolidation_loss: float
    admitted: bool
    rolled_back: bool
    consumed_experience_ids: tuple[str, ...]
    excluded_experience_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": PROCEDURAL_EVOLUTION_FORMAT,
            "parent_learner_digest": self.parent_learner_digest,
            "child_learner_digest": self.child_learner_digest,
            "dataset_digest": self.dataset_digest,
            "train_experiences": self.train_experiences,
            "excluded_train_experiences": self.excluded_train_experiences,
            "holdout_experiences": self.holdout_experiences,
            "retention_experiences": self.retention_experiences,
            "frozen_holdout_accuracy": self.frozen_holdout_accuracy,
            "replay_only_holdout_accuracy": self.replay_only_holdout_accuracy,
            "native_holdout_accuracy": self.native_holdout_accuracy,
            "frozen_retention_accuracy": self.frozen_retention_accuracy,
            "native_retention_accuracy": self.native_retention_accuracy,
            "native_consolidation_loss": self.native_consolidation_loss,
            "admitted": self.admitted,
            "rolled_back": self.rolled_back,
            "consumed_experience_ids": list(self.consumed_experience_ids),
            "excluded_experience_ids": list(self.excluded_experience_ids),
        }


class NativeProceduralMemoryTrainer:
    """Convert successful experiences into a dynamic procedural readout."""

    def __init__(
        self,
        cue_dim: int,
        *,
        epochs: int = 120,
        learning_rate: float = 0.1,
        feature_namespace: str = "taiji-route-credit-v1",
        manifest_revision: str = PROCEDURAL_EVOLUTION_MANIFEST_REVISION,
    ) -> None:
        if int(epochs) <= 0:
            raise ValueError("procedural evolution epochs must be positive")
        if float(learning_rate) <= 0.0:
            raise ValueError("procedural evolution learning_rate must be positive")
        self.encoder = EvolutionExperienceEncoder(cue_dim, namespace=feature_namespace)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("procedural evolution manifest_revision cannot be empty")
        self.learner = ProceduralMemoryLearner(cue_dim)
        self.consumed_experience_ids: tuple[str, ...] = ()
        self.excluded_experience_ids: tuple[str, ...] = ()
        self.last_dataset_digest = ""
        self.revision = 0

    def record(self, experience: EvolutionExperience) -> EpisodicMemoryRecord:
        if not isinstance(experience, EvolutionExperience):
            raise TypeError("procedural intake accepts EvolutionExperience values")
        kind = _action_kind(experience)
        intent = ActionIntent(
            intent_id=experience.intent_id or experience.experience_id,
            kind=kind,
            confidence=max(0.0, min(1.0, 1.0 - float(experience.uncertainty))),
            tick=experience.tick,
        )
        return EpisodicMemoryRecord(
            memory_id=f"procedural:{experience.experience_id}",
            episode_id=experience.episode_id or experience.experience_id,
            tick=experience.tick,
            cue=self.encoder.encode(experience),
            action_intent=intent,
            provenance="experienced",
            event_ids=(experience.experience_id,),
        )

    def _accuracy(
        self,
        records: tuple[EpisodicMemoryRecord, ...],
        learner: ProceduralMemoryLearner,
    ) -> float:
        if not records:
            raise ValueError("procedural accuracy needs records")
        if learner.readout is None:
            raise RuntimeError("procedural learner must be prepared before measurement")
        correct = 0
        for record in records:
            if record.action_intent is None:
                raise ValueError("procedural record is missing action intent")
            if record.action_intent.kind not in learner.action_kinds:
                raise ValueError("measurement action kind was not observed in train")
            correct += int(learner.predict(record.cue) == record.action_intent.kind)
        return correct / len(records)

    def consolidate(
        self,
        train_experiences: Iterable[EvolutionExperience],
        *,
        holdout_experiences: Iterable[EvolutionExperience],
        retention_experiences: Iterable[EvolutionExperience],
    ) -> NativeProceduralLearningReport:
        train_items = _experience_tuple(train_experiences, partition="train")
        holdout_items = _experience_tuple(holdout_experiences, partition="holdout")
        retention_items = _experience_tuple(retention_experiences, partition="retention")
        seen: set[str] = set()
        for items in (train_items, holdout_items, retention_items):
            ids = {item.experience_id for item in items}
            if seen.intersection(ids):
                raise ValueError("procedural experience partitions must be disjoint")
            seen.update(ids)
        successful = tuple(
            item
            for item in train_items
            if item.success and item.status == "success"
        )
        excluded = tuple(item for item in train_items if item not in successful)
        pending = tuple(
            item for item in successful if item.experience_id not in self.consumed_experience_ids
        )
        if not pending:
            raise ValueError("procedural evolution training has no new successful experiences")
        train_records = tuple(self.record(item) for item in pending)
        holdout_records = tuple(self.record(item) for item in holdout_items)
        retention_records = tuple(self.record(item) for item in retention_items)
        action_kinds = tuple(
            dict.fromkeys(
                (
                    *self.learner.action_kinds,
                    *sorted({record.action_intent.kind for record in train_records if record.action_intent}),
                )
            )
        )
        if not action_kinds:
            raise ValueError("procedural evolution found no train action kinds")
        for record in (*holdout_records, *retention_records):
            if record.action_intent is None or record.action_intent.kind not in action_kinds:
                raise ValueError("holdout or retention action kind was not observed in train")
        parent_payload = self.learner.checkpoint()
        parent_digest = content_digest(parent_payload)
        frozen = ProceduralMemoryLearner.from_checkpoint(parent_payload)
        frozen.prepare(action_kinds)
        frozen_holdout = self._accuracy(holdout_records, frozen)
        frozen_retention = self._accuracy(retention_records, frozen)
        replay_only = ProceduralMemoryLearner.from_checkpoint(parent_payload)
        replay_only.prepare(action_kinds)
        replay_holdout = self._accuracy(holdout_records, replay_only)

        trial = type(self).from_checkpoint(self.checkpoint())
        trial.learner.prepare(action_kinds)
        consolidation_loss = trial.learner.consolidate(
            train_records,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            action_kinds=action_kinds,
        )
        native_holdout = self._accuracy(holdout_records, trial.learner)
        native_retention = self._accuracy(retention_records, trial.learner)
        admitted = bool(
            native_holdout > frozen_holdout
            and native_retention >= frozen_retention
        )
        if admitted:
            self.learner = trial.learner
            self.consumed_experience_ids = tuple(
                sorted((*self.consumed_experience_ids, *(item.experience_id for item in pending)))
            )
            self.excluded_experience_ids = tuple(
                sorted((*self.excluded_experience_ids, *(item.experience_id for item in excluded)))
            )
            self.last_dataset_digest = content_digest(
                {
                    "train": [item.experience_digest for item in pending],
                    "holdout": [item.experience_digest for item in holdout_items],
                    "retention": [item.experience_digest for item in retention_items],
                    "encoder": self.encoder.checkpoint(),
                }
            )
            self.revision += 1
            child_digest = content_digest(self.learner.checkpoint())
        else:
            child_digest = content_digest(trial.learner.checkpoint())
        dataset_digest = content_digest(
            {
                "train": [item.experience_digest for item in pending],
                "holdout": [item.experience_digest for item in holdout_items],
                "retention": [item.experience_digest for item in retention_items],
                "encoder": self.encoder.checkpoint(),
            }
        )
        return NativeProceduralLearningReport(
            parent_learner_digest=parent_digest,
            child_learner_digest=child_digest,
            dataset_digest=dataset_digest,
            train_experiences=len(pending),
            excluded_train_experiences=len(excluded),
            holdout_experiences=len(holdout_items),
            retention_experiences=len(retention_items),
            frozen_holdout_accuracy=frozen_holdout,
            replay_only_holdout_accuracy=replay_holdout,
            native_holdout_accuracy=native_holdout,
            frozen_retention_accuracy=frozen_retention,
            native_retention_accuracy=native_retention,
            native_consolidation_loss=consolidation_loss,
            admitted=admitted,
            rolled_back=not admitted,
            consumed_experience_ids=tuple(item.experience_id for item in pending)
            if admitted
            else (),
            excluded_experience_ids=tuple(item.experience_id for item in excluded)
            if admitted
            else (),
        )

    def predict(self, experience: EvolutionExperience) -> str:
        return self.learner.predict(self.record(experience).cue)

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": PROCEDURAL_EVOLUTION_FORMAT,
            "version": PROCEDURAL_EVOLUTION_VERSION,
            "encoder": self.encoder.checkpoint(),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "manifest_revision": self.manifest_revision,
            "learner": self.learner.checkpoint(),
            "consumed_experience_ids": list(self.consumed_experience_ids),
            "excluded_experience_ids": list(self.excluded_experience_ids),
            "last_dataset_digest": self.last_dataset_digest,
            "revision": self.revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeProceduralMemoryTrainer:
        if payload.get("format") != PROCEDURAL_EVOLUTION_FORMAT:
            raise ValueError("unsupported procedural evolution format")
        if int(payload.get("version", -1)) != PROCEDURAL_EVOLUTION_VERSION:
            raise ValueError("unsupported procedural evolution version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("procedural evolution checkpoint digest mismatch")
        encoder = EvolutionExperienceEncoder.from_checkpoint(payload["encoder"])
        trainer = cls(
            encoder.feature_dim,
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            feature_namespace=encoder.namespace,
            manifest_revision=str(payload["manifest_revision"]),
        )
        trainer.learner = ProceduralMemoryLearner.from_checkpoint(payload["learner"])
        if trainer.learner.cue_dim != encoder.feature_dim:
            raise ValueError("procedural learner and encoder dimensions differ")
        trainer.consumed_experience_ids = tuple(
            sorted(str(item) for item in payload.get("consumed_experience_ids", ()))
        )
        trainer.excluded_experience_ids = tuple(
            sorted(str(item) for item in payload.get("excluded_experience_ids", ()))
        )
        trainer.last_dataset_digest = str(payload.get("last_dataset_digest", ""))
        trainer.revision = int(payload.get("revision", 0))
        return trainer


__all__ = [
    "NativeProceduralLearningReport",
    "NativeProceduralMemoryTrainer",
    "PROCEDURAL_EVOLUTION_FORMAT",
    "PROCEDURAL_EVOLUTION_MANIFEST_REVISION",
    "PROCEDURAL_EVOLUTION_VERSION",
]
