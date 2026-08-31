"""Checkpointed native world-prediction intake from real transitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import WorldInterventionCase, WorldTransition
from .internalization import content_digest
from .world_learning import WorldDynamicsLearner, WorldSchema, WorldSchemaRegistry

WORLD_EVOLUTION_FORMAT = "taiji-native-world-evolution-v1"
WORLD_EVOLUTION_VERSION = 1
WORLD_EVOLUTION_MANIFEST_REVISION = "taiji-w7-e3-3-world-prediction-v1"


def _transitions(
    transitions: Iterable[WorldTransition],
    *,
    partition: str,
) -> tuple[WorldTransition, ...]:
    items = tuple(transitions)
    if not items:
        raise ValueError(f"{partition} transitions must contain at least one item")
    if any(not isinstance(item, WorldTransition) for item in items):
        raise TypeError(f"{partition} transitions must contain WorldTransition values")
    action_ids = tuple(item.action.action_id for item in items)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError(f"{partition} transitions cannot contain duplicate action IDs")
    return tuple(sorted(items, key=lambda item: item.action.action_id))


def transition_to_case(transition: WorldTransition, *, case_id: str = "") -> WorldInterventionCase:
    """Represent an observed transition as a non-executing training case."""

    if not isinstance(transition, WorldTransition):
        raise TypeError("world evolution accepts WorldTransition values")
    resolved_id = str(case_id).strip() or f"transition:{transition.action.action_id}"
    return WorldInterventionCase(
        case_id=resolved_id,
        initial=transition.before,
        action=transition.action,
        expected_state=transition.after,
        expected_outcome=transition.outcome,
    )


@dataclass(frozen=True)
class NativeWorldLearningReport:
    """Measured result of one local world-model trial."""

    parent_learner_digest: str
    child_learner_digest: str
    dataset_digest: str
    train_transitions: int
    holdout_transitions: int
    retention_transitions: int
    frozen_holdout_error: float
    replay_only_holdout_error: float
    native_holdout_error: float
    frozen_retention_error: float
    native_retention_error: float
    native_training_loss: float
    admitted: bool
    rolled_back: bool
    consumed_transition_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": WORLD_EVOLUTION_FORMAT,
            "parent_learner_digest": self.parent_learner_digest,
            "child_learner_digest": self.child_learner_digest,
            "dataset_digest": self.dataset_digest,
            "train_transitions": self.train_transitions,
            "holdout_transitions": self.holdout_transitions,
            "retention_transitions": self.retention_transitions,
            "frozen_holdout_error": self.frozen_holdout_error,
            "replay_only_holdout_error": self.replay_only_holdout_error,
            "native_holdout_error": self.native_holdout_error,
            "frozen_retention_error": self.frozen_retention_error,
            "native_retention_error": self.native_retention_error,
            "native_training_loss": self.native_training_loss,
            "admitted": self.admitted,
            "rolled_back": self.rolled_back,
            "consumed_transition_ids": list(self.consumed_transition_ids),
        }


class NativeWorldPredictionTrainer:
    """Train the existing local world model from real transition evidence."""

    def __init__(
        self,
        schema: WorldSchema,
        *,
        hidden_dim: int = 32,
        seed: int = 11,
        epochs: int = 250,
        learning_rate: float = 0.01,
        manifest_revision: str = WORLD_EVOLUTION_MANIFEST_REVISION,
        schema_registry: WorldSchemaRegistry | None = None,
    ) -> None:
        if not isinstance(schema, WorldSchema):
            raise TypeError("world evolution schema must be WorldSchema")
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("world evolution epochs and learning_rate must be positive")
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("world evolution manifest_revision cannot be empty")
        self.learner = WorldDynamicsLearner(
            schema,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
            schema_registry=schema_registry,
        )
        self.consumed_transition_ids: tuple[str, ...] = ()
        self.revision = 0
        self.last_dataset_digest = ""

    def _error(
        self,
        transitions: tuple[WorldTransition, ...],
        learner: WorldDynamicsLearner,
    ) -> float:
        errors = []
        for transition in transitions:
            prediction = learner.predict(
                transition.before,
                transition.action,
                register_parameters=False,
            )
            state_error = learner.schema.normalized_state_error(
                prediction.state,
                transition.after,
            )
            expected_success = float(
                transition.outcome.success
                if transition.outcome.success is not None
                else transition.outcome.reward > 0.0
            )
            errors.append(
                state_error
                + (prediction.reward - float(transition.outcome.reward)) ** 2
                + (prediction.success_probability - expected_success) ** 2
            )
        return sum(errors) / len(errors)

    def consolidate(
        self,
        train_transitions: Iterable[WorldTransition],
        *,
        holdout_transitions: Iterable[WorldTransition],
        retention_transitions: Iterable[WorldTransition],
    ) -> NativeWorldLearningReport:
        train = _transitions(train_transitions, partition="train")
        holdout = _transitions(holdout_transitions, partition="holdout")
        retention = _transitions(retention_transitions, partition="retention")
        seen: set[str] = set()
        for items in (train, holdout, retention):
            ids = {item.action.action_id for item in items}
            if seen.intersection(ids):
                raise ValueError("world transition partitions must be disjoint")
            seen.update(ids)
        pending = tuple(
            item for item in train if item.action.action_id not in self.consumed_transition_ids
        )
        if not pending:
            raise ValueError("world evolution training has no new transitions")
        train_cases = tuple(transition_to_case(item) for item in pending)
        parent_payload = self.checkpoint()
        parent_digest = content_digest(parent_payload)
        frozen = type(self).from_checkpoint(parent_payload)
        frozen_holdout = self._error(holdout, frozen.learner)
        frozen_retention = self._error(retention, frozen.learner)
        replay_only = type(self).from_checkpoint(parent_payload)
        replay_holdout = self._error(holdout, replay_only.learner)
        trial = type(self).from_checkpoint(parent_payload)
        losses = trial.learner.fit(
            train_cases,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        native_holdout = self._error(holdout, trial.learner)
        native_retention = self._error(retention, trial.learner)
        admitted = bool(
            native_holdout < frozen_holdout
            and native_retention <= frozen_retention + 0.05
        )
        dataset_digest = content_digest(
            {
                "schema": self.schema.payload(),
                "train": [item.to_payload() for item in pending],
                "holdout": [item.to_payload() for item in holdout],
                "retention": [item.to_payload() for item in retention],
            }
        )
        if admitted:
            self.learner = trial.learner
            self.schema = trial.schema
            self.consumed_transition_ids = tuple(
                sorted((*self.consumed_transition_ids, *(item.action.action_id for item in pending)))
            )
            self.last_dataset_digest = dataset_digest
            self.revision += 1
            child_digest = content_digest(self.checkpoint())
        else:
            child_digest = content_digest(trial.checkpoint())
        return NativeWorldLearningReport(
            parent_learner_digest=parent_digest,
            child_learner_digest=child_digest,
            dataset_digest=dataset_digest,
            train_transitions=len(pending),
            holdout_transitions=len(holdout),
            retention_transitions=len(retention),
            frozen_holdout_error=frozen_holdout,
            replay_only_holdout_error=replay_holdout,
            native_holdout_error=native_holdout,
            frozen_retention_error=frozen_retention,
            native_retention_error=native_retention,
            native_training_loss=float(losses[-1]),
            admitted=admitted,
            rolled_back=not admitted,
            consumed_transition_ids=tuple(item.action.action_id for item in pending)
            if admitted
            else (),
        )

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": WORLD_EVOLUTION_FORMAT,
            "version": WORLD_EVOLUTION_VERSION,
            "schema": self.schema.payload(),
            "schema_registry": self.learner.schema_registry.checkpoint(),
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "manifest_revision": self.manifest_revision,
            "model_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.learner.state_dict().items()
            },
            "counters": {
                "online_updates": self.learner.online_updates,
                "transition_acceptances": self.learner.transition_acceptances,
                "transition_rejections": self.learner.transition_rejections,
                "schema_evolution_count": self.learner.schema_evolution_count,
            },
            "schema_snapshots": {
                str(version): {
                    name: tensor.detach().cpu().clone() for name, tensor in snapshot.items()
                }
                for version, snapshot in self.learner._schema_snapshots.items()
            },
            "consumed_transition_ids": list(self.consumed_transition_ids),
            "last_dataset_digest": self.last_dataset_digest,
            "revision": self.revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeWorldPredictionTrainer:
        if payload.get("format") != WORLD_EVOLUTION_FORMAT:
            raise ValueError("unsupported world evolution format")
        if int(payload.get("version", -1)) != WORLD_EVOLUTION_VERSION:
            raise ValueError("unsupported world evolution version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("world evolution checkpoint digest mismatch")
        schema = WorldSchema.from_payload(dict(payload["schema"]))
        registry = WorldSchemaRegistry.from_checkpoint(payload["schema_registry"])
        trainer = cls(
            schema,
            hidden_dim=int(payload["hidden_dim"]),
            seed=int(payload.get("seed", 0)),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            manifest_revision=str(payload["manifest_revision"]),
            schema_registry=registry,
        )
        trainer.learner.load_state_dict(payload["model_state"])
        counters = payload.get("counters") or {}
        trainer.learner.online_updates = int(counters.get("online_updates", 0))
        trainer.learner.transition_acceptances = int(counters.get("transition_acceptances", 0))
        trainer.learner.transition_rejections = int(counters.get("transition_rejections", 0))
        trainer.learner.schema_evolution_count = int(counters.get("schema_evolution_count", 0))
        trainer.learner._schema_snapshots = {
            int(version): {
                str(name): tensor.detach().cpu().clone()
                for name, tensor in dict(snapshot).items()
            }
            for version, snapshot in dict(payload.get("schema_snapshots") or {}).items()
        }
        trainer.consumed_transition_ids = tuple(
            sorted(str(item) for item in payload.get("consumed_transition_ids", ()))
        )
        trainer.last_dataset_digest = str(payload.get("last_dataset_digest", ""))
        trainer.revision = int(payload.get("revision", 0))
        return trainer


__all__ = [
    "NativeWorldLearningReport",
    "NativeWorldPredictionTrainer",
    "WORLD_EVOLUTION_FORMAT",
    "WORLD_EVOLUTION_MANIFEST_REVISION",
    "WORLD_EVOLUTION_VERSION",
    "transition_to_case",
]
