"""Learned world-dynamics and intervention evaluation for the P3 slice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .contracts import (
    WorldEpisode,
    WorldEpisodeCorpus,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldState,
    WorldTransition,
)
from .local_learning import (
    LocalAdam,
    apply_sgd_step,
    backproject_linear,
    freeze_parameters,
    linear_gradients,
    logistic_error_delta,
    mean_squared_error_delta,
    tanh_delta,
)
from .world import TaijiWorldState, _world_state_equal

WORLD_SCHEMA_REGISTRY_CHECKPOINT_FORMAT = "taiji-world-schema-registry-v1"


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


@dataclass(frozen=True)
class WorldSchema:
    """A data-derived numeric interface for a world intervention corpus."""

    object_ids: tuple[str, ...]
    state_slots: tuple[tuple[str, str], ...]
    relation_slots: tuple[tuple[str, str, str], ...]
    action_kinds: tuple[str, ...]
    actor_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]
    state_scales: tuple[float, ...] = ()
    open_set: bool = False

    def __post_init__(self) -> None:
        scales = self.state_scales or (1.0,) * self.state_dim
        if len(scales) != self.state_dim:
            raise ValueError("world state scales must match state_dim")
        if any(not math.isfinite(float(scale)) or float(scale) <= 0.0 for scale in scales):
            raise ValueError("world state scales must be finite and positive")
        object.__setattr__(self, "state_scales", tuple(float(scale) for scale in scales))

    @classmethod
    def from_corpus(cls, corpus: WorldInterventionCorpus) -> WorldSchema:
        if not corpus.train:
            raise ValueError("world dynamics requires at least one training case")
        object_ids: set[str] = set()
        state_slots: set[tuple[str, str]] = set()
        relation_slots: set[tuple[str, str, str]] = set()
        action_kinds: set[str] = set()
        actor_ids: set[str] = set()
        target_ids: set[str] = set()
        parameter_names: set[str] = set()
        for case in corpus.train:
            for obj in case.initial.objects:
                object_ids.add(obj.object_id)
                for name, value in obj.attributes:
                    _numeric(value, f"{obj.object_id}.{name}")
                    state_slots.add((obj.object_id, name))
            relation_slots.update(case.initial.relations)
            relation_slots.update(case.expected_state.relations)
            action_kinds.add(case.action.kind)
            actor_ids.add(case.action.actor_id)
            target_ids.add(case.action.target_id)
            for name, value in case.action.parameters:
                try:
                    _numeric(value, f"action parameter {name}")
                except ValueError:
                    continue
                parameter_names.add(name)
        if not state_slots:
            raise ValueError("world dynamics requires numeric object attributes")
        sorted_state_slots = tuple(sorted(state_slots))
        state_scales = []
        for object_id, name in sorted_state_slots:
            values = []
            for case in corpus.train:
                for state in (case.initial, case.expected_state):
                    item = next(
                        (
                            candidate
                            for candidate in state.objects
                            if candidate.object_id == object_id
                        ),
                        None,
                    )
                    if item is not None:
                        values.append(
                            abs(_numeric(item.attribute(name, 0.0), f"{object_id}.{name}"))
                        )
            state_scales.append(max(1.0, *values))
        return cls(
            object_ids=tuple(sorted(object_ids)),
            state_slots=sorted_state_slots,
            relation_slots=tuple(sorted(relation_slots)),
            action_kinds=tuple(sorted(action_kinds)),
            actor_ids=tuple(sorted(actor_ids)),
            target_ids=tuple(sorted(target_ids)),
            parameter_names=tuple(sorted(parameter_names)),
            state_scales=tuple(state_scales) + (1.0,) * len(relation_slots),
        )

    @property
    def state_dim(self) -> int:
        return len(self.state_slots) + len(self.relation_slots)

    @property
    def object_state_dim(self) -> int:
        return len(self.state_slots)

    @property
    def action_dim(self) -> int:
        return (
            len(self.action_kinds)
            + len(self.actor_ids)
            + len(self.target_ids)
            + len(self.parameter_names)
            + len(self.target_ids) * len(self.parameter_names)
        )

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.action_dim

    @property
    def state_feature_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(("state", object_id, name) for object_id, name in self.state_slots) + tuple(
            ("relation", subject, predicate, object_id)
            for subject, predicate, object_id in self.relation_slots
        )

    @property
    def input_feature_keys(self) -> tuple[tuple[str, ...], ...]:
        return (
            self.state_feature_keys
            + tuple(("action-kind", kind) for kind in self.action_kinds)
            + tuple(("actor", actor_id) for actor_id in self.actor_ids)
            + tuple(("target", target_id) for target_id in self.target_ids)
            + tuple(("parameter", name) for name in self.parameter_names)
            + tuple(
                ("interaction", target_id, name)
                for target_id in self.target_ids
                for name in self.parameter_names
            )
        )

    def evolve_open_set(
        self,
        *,
        states: tuple[WorldState, ...] = (),
        actions: tuple[Any, ...] = (),
        include_action_parameters: bool = True,
    ) -> WorldSchema:
        """Register observed world identities while retaining the old schema.

        The returned schema is data-derived from the observed state/action
        boundary.  Existing feature order remains addressable by semantic
        keys, while new object attributes, relations, action kinds, actors,
        targets, and numeric parameters receive new slots.
        """

        if not states and not actions:
            raise ValueError("open-set schema evolution needs an observed state or action")
        object_ids = set(self.object_ids)
        state_slots = set(self.state_slots)
        relation_slots = set(self.relation_slots)
        action_kinds = set(self.action_kinds)
        actor_ids = set(self.actor_ids)
        target_ids = set(self.target_ids)
        parameter_names = set(self.parameter_names)
        for state in states:
            for obj in state.objects:
                object_ids.add(obj.object_id)
                for name, value in obj.attributes:
                    try:
                        _numeric(value, f"{obj.object_id}.{name}")
                    except ValueError:
                        continue
                    state_slots.add((obj.object_id, name))
            relation_slots.update(state.relations)
        for action in actions:
            action_kinds.add(str(action.kind))
            actor_ids.add(str(action.actor_id))
            target_ids.add(str(action.target_id))
            if include_action_parameters:
                for name, value in action.parameters:
                    try:
                        _numeric(value, f"action parameter {name}")
                    except ValueError:
                        continue
                    parameter_names.add(str(name))
        changed = bool(
            object_ids != set(self.object_ids)
            or state_slots != set(self.state_slots)
            or relation_slots != set(self.relation_slots)
            or action_kinds != set(self.action_kinds)
            or actor_ids != set(self.actor_ids)
            or target_ids != set(self.target_ids)
            or parameter_names != set(self.parameter_names)
        )
        if not changed and not self.open_set:
            return self
        sorted_state_slots = tuple(sorted(state_slots))
        old_scales = {
            key: float(self.state_scales[index]) for index, key in enumerate(self.state_slots)
        }
        state_scales = []
        for object_id, name in sorted_state_slots:
            if (object_id, name) in old_scales:
                state_scales.append(old_scales[(object_id, name)])
                continue
            values = []
            for state in states:
                candidate_object = next(
                    (item for item in state.objects if item.object_id == object_id), None
                )
                if candidate_object is not None:
                    values.append(
                        abs(
                            _numeric(
                                candidate_object.attribute(name, 0.0),
                                f"{object_id}.{name}",
                            )
                        )
                    )
            state_scales.append(max(1.0, *values))
        return WorldSchema(
            object_ids=tuple(sorted(object_ids)),
            state_slots=sorted_state_slots,
            relation_slots=tuple(sorted(relation_slots)),
            action_kinds=tuple(sorted(action_kinds)),
            actor_ids=tuple(sorted(actor_ids)),
            target_ids=tuple(sorted(target_ids)),
            parameter_names=tuple(sorted(parameter_names)),
            state_scales=tuple(state_scales) + (1.0,) * len(relation_slots),
            open_set=self.open_set or changed,
        )

    def state_values(self, state: WorldState) -> torch.Tensor:
        objects = {item.object_id: item for item in state.objects}
        values = []
        for object_id, name in self.state_slots:
            if object_id not in objects:
                if not self.open_set:
                    raise ValueError(f"world state is missing object: {object_id}")
                values.append(0.0)
            else:
                values.append(
                    _numeric(objects[object_id].attribute(name, 0.0), f"{object_id}.{name}")
                )
        relation_set = set(state.relations)
        values.extend(float(relation in relation_set) for relation in self.relation_slots)
        return torch.tensor(values, dtype=torch.float32)

    def normalized_state_error(self, predicted: WorldState, actual: WorldState) -> float:
        """Compare world states after applying schema-owned per-slot scales."""

        predicted_values = self.state_values(predicted)
        actual_values = self.state_values(actual)
        scales = torch.tensor(self.state_scales, dtype=torch.float32)
        return float(torch.mean(((predicted_values - actual_values) / scales) ** 2))

    def encode(self, state: WorldState, action: Any, *, bind_target: bool = True) -> torch.Tensor:
        state_values = self.state_values(state)
        action_values = torch.zeros(self.action_dim, dtype=torch.float32)
        offset = 0

        if action.kind not in self.action_kinds:
            raise ValueError(f"unknown action kind: {action.kind}")
        action_values[offset + self.action_kinds.index(action.kind)] = 1.0
        offset += len(self.action_kinds)

        if action.actor_id not in self.actor_ids:
            raise ValueError(f"unknown action actor: {action.actor_id}")
        action_values[offset + self.actor_ids.index(action.actor_id)] = 1.0
        offset += len(self.actor_ids)

        if bind_target:
            if action.target_id not in self.target_ids:
                raise ValueError(f"unknown action target: {action.target_id}")
            action_values[offset + self.target_ids.index(action.target_id)] = 1.0
        offset += len(self.target_ids)

        parameters = dict(action.parameters)
        parameter_values = []
        for name in self.parameter_names:
            value = _numeric(parameters.get(name, 0.0), f"action parameter {name}")
            action_values[offset] = value
            parameter_values.append(value)
            offset += 1
        if bind_target:
            target_index = self.target_ids.index(action.target_id)
            interaction_offset = offset + target_index * len(self.parameter_names)
            for parameter_index, value in enumerate(parameter_values):
                action_values[interaction_offset + parameter_index] = value
        return torch.cat((state_values, action_values))

    def delta_target(self, case: WorldInterventionCase) -> torch.Tensor:
        return self.state_values(case.expected_state) - self.state_values(case.initial)

    def payload(self) -> dict[str, Any]:
        return {
            "object_ids": list(self.object_ids),
            "state_slots": [list(item) for item in self.state_slots],
            "relation_slots": [list(item) for item in self.relation_slots],
            "action_kinds": list(self.action_kinds),
            "actor_ids": list(self.actor_ids),
            "target_ids": list(self.target_ids),
            "parameter_names": list(self.parameter_names),
            "target_parameter_interactions": len(self.target_ids) * len(self.parameter_names),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "input_dim": self.input_dim,
            "state_scales": list(self.state_scales),
            "open_set": self.open_set,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WorldSchema:
        return cls(
            object_ids=tuple(str(item) for item in payload["object_ids"]),
            state_slots=tuple((str(item[0]), str(item[1])) for item in payload["state_slots"]),
            relation_slots=tuple(
                (str(item[0]), str(item[1]), str(item[2]))
                for item in payload.get("relation_slots", ())
            ),
            action_kinds=tuple(str(item) for item in payload["action_kinds"]),
            actor_ids=tuple(str(item) for item in payload["actor_ids"]),
            target_ids=tuple(str(item) for item in payload["target_ids"]),
            parameter_names=tuple(str(item) for item in payload["parameter_names"]),
            state_scales=tuple(float(item) for item in payload.get("state_scales", ())),
            open_set=bool(payload.get("open_set", False)),
        )

    def without_features(self, feature_keys: tuple[tuple[str, ...], ...]) -> WorldSchema:
        """Remove active feature slots while preserving semantic scale metadata."""

        requested = set(feature_keys)
        state_slots = set(self.state_slots)
        relation_slots = set(self.relation_slots)
        action_kinds = set(self.action_kinds)
        actor_ids = set(self.actor_ids)
        target_ids = set(self.target_ids)
        parameter_names = set(self.parameter_names)
        active = set(self.input_feature_keys) | set(self.state_feature_keys)
        unknown = requested - active
        if unknown:
            raise ValueError(f"cannot prune unknown world schema features: {sorted(unknown)}")
        for key in requested:
            kind = key[0]
            if kind == "state" and len(key) == 3:
                state_slots.discard((key[1], key[2]))
            elif kind == "relation" and len(key) == 4:
                relation_slots.discard((key[1], key[2], key[3]))
            elif kind == "action-kind" and len(key) == 2:
                action_kinds.discard(key[1])
            elif kind == "actor" and len(key) == 2:
                actor_ids.discard(key[1])
            elif kind == "target" and len(key) == 2:
                target_ids.discard(key[1])
            elif kind == "parameter" and len(key) == 2:
                parameter_names.discard(key[1])
            elif kind == "interaction":
                raise ValueError(
                    "interaction features are derived; prune their target or parameter"
                )
            else:
                raise ValueError(f"invalid world schema feature key: {key}")
        if not state_slots and not relation_slots:
            raise ValueError("world schema must retain at least one state feature")
        old_scales = {
            key: float(self.state_scales[index])
            for index, key in enumerate(self.state_feature_keys)
        }
        sorted_state_slots = tuple(sorted(state_slots))
        sorted_relation_slots = tuple(sorted(relation_slots))
        state_scales = tuple(
            old_scales[key]
            for key in (
                tuple(("state", object_id, name) for object_id, name in sorted_state_slots)
                + tuple(
                    ("relation", subject, predicate, object_id)
                    for subject, predicate, object_id in sorted_relation_slots
                )
            )
        )
        return WorldSchema(
            object_ids=self.object_ids,
            state_slots=sorted_state_slots,
            relation_slots=sorted_relation_slots,
            action_kinds=tuple(sorted(action_kinds)),
            actor_ids=tuple(sorted(actor_ids)),
            target_ids=tuple(sorted(target_ids)),
            parameter_names=tuple(sorted(parameter_names)),
            state_scales=state_scales,
            open_set=True,
        )


class WorldSchemaRegistryError(ValueError):
    """Base error for fail-closed world schema lifecycle operations."""


class WorldSchemaConflictError(WorldSchemaRegistryError):
    """Raised when a schema operation would silently contradict known evidence."""


class WorldSchemaBudgetError(WorldSchemaRegistryError):
    """Raised when schema growth exceeds the registry resource budget."""


@dataclass(frozen=True)
class WorldSchemaProposal:
    """An auditable schema change waiting for learner/network commit."""

    base_version: int
    schema: WorldSchema
    operation: str
    added_features: tuple[tuple[str, ...], ...] = ()
    removed_features: tuple[tuple[str, ...], ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.base_version) < 0:
            raise ValueError("schema proposal base_version must be non-negative")
        if not str(self.operation):
            raise ValueError("schema proposal operation cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "schema": self.schema.payload(),
            "operation": self.operation,
            "added_features": [list(key) for key in self.added_features],
            "removed_features": [list(key) for key in self.removed_features],
            "evidence_ids": list(self.evidence_ids),
        }


class WorldSchemaRegistry:
    """Versioned, checkpointable identity and feature lifecycle for world learning."""

    def __init__(
        self,
        schema: WorldSchema,
        *,
        max_feature_count: int = 4096,
        history_limit: int = 64,
        outcome_min_lead: int = 1,
        outcome_min_leader_share: float = 0.6,
        stochastic_min_support: int = 2,
    ) -> None:
        if not isinstance(schema, WorldSchema):
            raise TypeError("world schema registry requires a WorldSchema")
        if int(max_feature_count) <= 0:
            raise ValueError("world schema registry max_feature_count must be positive")
        if int(history_limit) <= 0:
            raise ValueError("world schema registry history_limit must be positive")
        if int(outcome_min_lead) <= 0:
            raise ValueError("world schema registry outcome_min_lead must be positive")
        if not 0.5 < float(outcome_min_leader_share) <= 1.0:
            raise ValueError("world schema registry outcome_min_leader_share must be in (0.5, 1.0]")
        if int(stochastic_min_support) <= 0:
            raise ValueError("world schema registry stochastic_min_support must be positive")
        self._schema = schema
        self._max_feature_count = int(max_feature_count)
        self._history_limit = int(history_limit)
        self._outcome_min_lead = int(outcome_min_lead)
        self._outcome_min_leader_share = float(outcome_min_leader_share)
        self._stochastic_min_support = int(stochastic_min_support)
        self._revisions: dict[int, WorldSchema] = {0: schema}
        self._active_version = 0
        self._next_version = 1
        self._aliases: dict[str, str] = {}
        self._confidence: dict[tuple[str, ...], float] = {
            key: 0.0 for key in self._known_feature_keys(schema)
        }
        self._feedback: dict[tuple[str, ...], float] = {}
        self._tombstones: set[tuple[str, ...]] = set()
        self._conflicts: list[dict[str, Any]] = []
        self._lineage: list[dict[str, Any]] = []
        self._transition_outcomes: dict[tuple[str, ...], dict[str, int]] = {}
        self._transition_confidence: dict[tuple[str, ...], float] = {}

    @staticmethod
    def _known_feature_keys(schema: WorldSchema) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(set(schema.input_feature_keys) | set(schema.state_feature_keys)))

    @staticmethod
    def _feature_count(schema: WorldSchema) -> int:
        return int(schema.input_dim + schema.state_dim)

    @property
    def schema(self) -> WorldSchema:
        return self._schema

    @property
    def active_version(self) -> int:
        return self._active_version

    @property
    def revision_versions(self) -> tuple[int, ...]:
        return tuple(sorted(self._revisions))

    @property
    def max_feature_count(self) -> int:
        return self._max_feature_count

    @property
    def aliases(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._aliases.items()))

    @property
    def tombstones(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(self._tombstones))

    @property
    def feature_confidence(self) -> dict[tuple[str, ...], float]:
        return dict(self._confidence)

    @property
    def conflicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._conflicts)

    @property
    def lineage(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._lineage)

    @property
    def contradiction_count(self) -> int:
        return len(self._conflicts)

    @property
    def transition_outcome_count(self) -> int:
        return len(self._transition_outcomes)

    @property
    def transition_confidence(self) -> dict[tuple[str, ...], float]:
        return dict(self._transition_confidence)

    @property
    def transition_hypotheses(self) -> dict[tuple[str, ...], tuple[dict[str, Any], ...]]:
        result: dict[tuple[str, ...], tuple[dict[str, Any], ...]] = {}
        for key, hypotheses in self._transition_outcomes.items():
            total = sum(hypotheses.values())
            result[key] = tuple(
                {
                    "signature": signature,
                    "evidence_count": count,
                    "probability": count / float(total),
                }
                for signature, count in sorted(hypotheses.items())
            )
        return result

    def transition_outcome_mode(self, key: tuple[str, ...]) -> str:
        hypotheses = self._transition_outcomes.get(tuple(str(item) for item in key), {})
        if len(hypotheses) <= 1:
            return "deterministic"
        if all(count >= self._stochastic_min_support for count in hypotheses.values()):
            return "stochastic"
        return "conflicted"

    def schema_at(self, version: int) -> WorldSchema:
        try:
            return self._revisions[int(version)]
        except KeyError as exc:
            raise WorldSchemaRegistryError(
                f"world schema revision is not available: {version}"
            ) from exc

    def canonical_object_id(self, object_id: str) -> str:
        current = str(object_id)
        visited: set[str] = set()
        while current in self._aliases:
            if current in visited:
                raise WorldSchemaConflictError("world schema alias cycle detected")
            visited.add(current)
            current = self._aliases[current]
        return current

    def register_alias(self, alias: str, canonical: str) -> bool:
        alias = str(alias)
        canonical = self.canonical_object_id(str(canonical))
        if not alias or not canonical:
            raise ValueError("world schema aliases cannot be empty")
        if alias == canonical:
            return False
        if canonical not in self._schema.object_ids:
            raise WorldSchemaConflictError(
                f"world schema alias target is not registered: {canonical}"
            )
        existing = self._aliases.get(alias)
        if existing is not None:
            if self.canonical_object_id(existing) != canonical:
                raise WorldSchemaConflictError(
                    f"world schema alias already points to {existing}: {alias}"
                )
            return False
        if alias in self._schema.object_ids:
            raise WorldSchemaConflictError(
                f"world schema alias collides with canonical object: {alias}"
            )
        self._aliases[alias] = canonical
        self._lineage.append(
            {
                "operation": "merge-alias",
                "version": self._active_version,
                "alias": alias,
                "canonical": canonical,
            }
        )
        return True

    def normalize_state(self, state: WorldState) -> WorldState:
        if not isinstance(state, WorldState):
            raise TypeError("world schema registry requires a WorldState")
        objects = []
        seen: set[str] = set()
        for obj in state.objects:
            canonical = self.canonical_object_id(obj.object_id)
            if canonical in seen:
                raise WorldSchemaConflictError(
                    f"world schema alias merge produced duplicate object: {canonical}"
                )
            seen.add(canonical)
            objects.append(replace(obj, object_id=canonical))
        relations = tuple(
            sorted(
                {
                    (
                        self.canonical_object_id(subject),
                        str(predicate),
                        self.canonical_object_id(object_id),
                    )
                    for subject, predicate, object_id in state.relations
                }
            )
        )
        entities = tuple(self.canonical_object_id(entity) for entity in state.entities)
        return replace(state, entities=entities, objects=tuple(objects), relations=relations)

    def normalize_action(self, action: Any) -> Any:
        if not hasattr(action, "actor_id") or not hasattr(action, "target_id"):
            raise TypeError("world schema registry requires a structured world action")
        return replace(
            action,
            actor_id=self.canonical_object_id(action.actor_id),
            target_id=self.canonical_object_id(action.target_id),
        )

    def _record_conflict(self, key: tuple[str, ...], expected: Any, observed: Any) -> None:
        self._conflicts.append(
            {
                "key": list(key),
                "expected": expected,
                "observed": observed,
                "version": self._active_version,
            }
        )

    @staticmethod
    def _stable_state_signature(state: WorldState) -> str:
        """Serialize semantic state only, excluding tick/event noise."""

        objects = tuple(
            (
                str(obj.object_id),
                tuple(sorted((str(name), repr(value)) for name, value in obj.attributes)),
            )
            for obj in sorted(state.objects, key=lambda item: item.object_id)
        )
        relations = tuple(
            sorted(
                (
                    str(subject),
                    str(predicate),
                    str(object_id),
                )
                for subject, predicate, object_id in state.relations
            )
        )
        entities = tuple(sorted(str(entity) for entity in state.entities))
        return repr((objects, relations, entities))

    @staticmethod
    def _stable_action_signature(action: Any) -> str:
        parameters = tuple(sorted((str(name), repr(value)) for name, value in action.parameters))
        return repr(
            (
                str(action.kind),
                str(action.actor_id),
                str(action.target_id),
                parameters,
            )
        )

    def transition_evidence_key(self, transition: WorldTransition) -> tuple[str, ...]:
        """Return a cross-episode key for the same semantic intervention context."""

        if not isinstance(transition, WorldTransition):
            raise TypeError("world schema registry requires a WorldTransition")
        before = self.normalize_state(transition.before)
        action = self.normalize_action(transition.action)
        return (
            "transition",
            self._stable_state_signature(before),
            self._stable_action_signature(action),
        )

    def _transition_outcome_signature(self, transition: WorldTransition) -> str:
        after = self.normalize_state(transition.after)
        reward = float(transition.outcome.reward)
        success = (
            bool(transition.outcome.success)
            if transition.outcome.success is not None
            else reward > 0.0
        )
        return repr((self._stable_state_signature(after), round(reward, 6), success))

    def record_transition_outcome(self, transition: WorldTransition) -> bool:
        """Adjudicate repeated real outcomes before allowing local learning."""

        if not isinstance(transition, WorldTransition):
            raise TypeError("world schema registry requires a WorldTransition")
        key = self.transition_evidence_key(transition)
        signature = self._transition_outcome_signature(transition)
        hypotheses = self._transition_outcomes.setdefault(key, {})
        if hypotheses and signature not in hypotheses:
            ranked_before = sorted(hypotheses.items(), key=lambda item: (-item[1], item[0]))
            self._record_conflict(key, ranked_before[0][0], signature)
        hypotheses[signature] = hypotheses.get(signature, 0) + 1
        ranked = sorted(hypotheses.items(), key=lambda item: (-item[1], item[0]))
        leader_signature, leader_count = ranked[0]
        total = sum(hypotheses.values())
        self._transition_confidence[key] = leader_count / float(total)
        runner_count = 0 if len(ranked) == 1 else ranked[1][1]
        clear_leader = bool(
            len(ranked) == 1
            or (
                leader_count - runner_count >= self._outcome_min_lead
                and leader_count / float(total) >= self._outcome_min_leader_share
            )
        )
        return bool(signature == leader_signature and clear_leader)

    def record_feedback(
        self,
        feature_key: tuple[str, ...],
        observed: float,
        *,
        expected: float | None = None,
        tolerance: float = 0.0,
    ) -> bool:
        key = tuple(str(item) for item in feature_key)
        value = float(observed)
        if not math.isfinite(value) or float(tolerance) < 0.0:
            raise ValueError("world schema feedback must be finite with non-negative tolerance")
        if expected is not None:
            expected_value = float(expected)
            if not math.isfinite(expected_value):
                raise ValueError("world schema feedback expected value must be finite")
            if abs(value - expected_value) > float(tolerance):
                self._record_conflict(key, expected_value, value)
                return False
        previous = self._feedback.get(key)
        if previous is not None and abs(previous - value) > float(tolerance):
            self._record_conflict(key, previous, value)
            return False
        self._feedback[key] = value
        self._confidence[key] = min(1.0, self._confidence.get(key, 0.0) + 0.1)
        return True

    def propose_open_set(
        self,
        *,
        states: tuple[WorldState, ...] = (),
        actions: tuple[Any, ...] = (),
        register_parameters: bool = True,
        evidence_ids: tuple[str, ...] = (),
    ) -> WorldSchemaProposal | None:
        if not states and not actions:
            raise ValueError("world schema proposal needs an observed state or action")
        normalized_states = tuple(self.normalize_state(state) for state in states)
        normalized_actions = tuple(self.normalize_action(action) for action in actions)
        candidate = self._schema.evolve_open_set(
            states=normalized_states,
            actions=normalized_actions,
            include_action_parameters=register_parameters,
        )
        if candidate == self._schema:
            return None
        old_features = set(self._known_feature_keys(self._schema))
        new_features = set(self._known_feature_keys(candidate))
        added = tuple(sorted(new_features - old_features))
        blocked = set(added) & self._tombstones
        if blocked:
            key = tuple(sorted(blocked))[0]
            self._record_conflict(key, 0.0, 1.0)
            raise WorldSchemaConflictError(
                f"world schema feature is tombstoned and needs explicit reactivation: {key}"
            )
        if self._feature_count(candidate) > self._max_feature_count:
            raise WorldSchemaBudgetError(
                "world schema feature budget would be exceeded; prune or merge before growth"
            )
        return WorldSchemaProposal(
            base_version=self._active_version,
            schema=candidate,
            operation="open-set-grow",
            added_features=added,
            evidence_ids=tuple(str(item) for item in evidence_ids),
        )

    def propose_prune(
        self,
        feature_keys: tuple[tuple[str, ...], ...],
        *,
        evidence_ids: tuple[str, ...] = (),
    ) -> WorldSchemaProposal:
        requested = tuple(sorted(set(tuple(str(item) for item in key) for key in feature_keys)))
        if not requested:
            raise ValueError("world schema prune needs at least one feature")
        candidate = self._schema.without_features(requested)
        return WorldSchemaProposal(
            base_version=self._active_version,
            schema=candidate,
            operation="prune-tombstone",
            removed_features=requested,
            evidence_ids=tuple(str(item) for item in evidence_ids),
        )

    def commit(self, proposal: WorldSchemaProposal) -> int:
        if not isinstance(proposal, WorldSchemaProposal):
            raise TypeError("world schema registry commits WorldSchemaProposal values")
        if proposal.base_version != self._active_version:
            raise WorldSchemaConflictError("world schema proposal is based on a stale revision")
        if proposal.schema == self._schema:
            return self._active_version
        version = self._next_version
        self._next_version += 1
        self._schema = proposal.schema
        self._active_version = version
        self._revisions[version] = proposal.schema
        for key in proposal.added_features:
            self._confidence.setdefault(key, 0.0)
        self._tombstones.update(proposal.removed_features)
        self._lineage.append(
            {
                "operation": proposal.operation,
                "from_version": proposal.base_version,
                "to_version": version,
                "added_features": [list(key) for key in proposal.added_features],
                "removed_features": [list(key) for key in proposal.removed_features],
                "evidence_ids": list(proposal.evidence_ids),
            }
        )
        while len(self._revisions) > self._history_limit:
            oldest = min(self._revisions)
            if oldest == self._active_version:
                break
            del self._revisions[oldest]
        return version

    def rollback(self, version: int) -> bool:
        target_version = int(version)
        target = self.schema_at(target_version)
        if target_version == self._active_version:
            return False
        previous = self._active_version
        self._active_version = target_version
        self._schema = target
        self._lineage.append(
            {
                "operation": "rollback",
                "from_version": previous,
                "to_version": target_version,
            }
        )
        return True

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": WORLD_SCHEMA_REGISTRY_CHECKPOINT_FORMAT,
            "schema": self._schema.payload(),
            "active_version": self._active_version,
            "next_version": self._next_version,
            "max_feature_count": self._max_feature_count,
            "history_limit": self._history_limit,
            "outcome_min_lead": self._outcome_min_lead,
            "outcome_min_leader_share": self._outcome_min_leader_share,
            "stochastic_min_support": self._stochastic_min_support,
            "revisions": [
                {"version": version, "schema": schema.payload()}
                for version, schema in sorted(self._revisions.items())
            ],
            "aliases": [[alias, canonical] for alias, canonical in self.aliases],
            "confidence": [
                {"key": list(key), "value": value}
                for key, value in sorted(self._confidence.items())
            ],
            "feedback": [
                {"key": list(key), "value": value} for key, value in sorted(self._feedback.items())
            ],
            "tombstones": [list(key) for key in self.tombstones],
            "conflicts": [dict(item) for item in self._conflicts],
            "lineage": [dict(item) for item in self._lineage],
            "transition_outcomes": [
                {
                    "key": list(key),
                    "hypotheses": [
                        {"signature": signature, "evidence_count": count}
                        for signature, count in sorted(hypotheses.items())
                    ],
                }
                for key, hypotheses in sorted(self._transition_outcomes.items())
            ],
            "transition_confidence": [
                {"key": list(key), "value": value}
                for key, value in sorted(self._transition_confidence.items())
            ],
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> WorldSchemaRegistry:
        if payload.get("format") != WORLD_SCHEMA_REGISTRY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported world schema registry checkpoint format")
        registry = cls(
            WorldSchema.from_payload(dict(payload["schema"])),
            max_feature_count=int(payload.get("max_feature_count", 4096)),
            history_limit=int(payload.get("history_limit", 64)),
            outcome_min_lead=int(payload.get("outcome_min_lead", 1)),
            outcome_min_leader_share=float(payload.get("outcome_min_leader_share", 0.6)),
            stochastic_min_support=int(payload.get("stochastic_min_support", 2)),
        )
        revisions = payload.get("revisions", ())
        if not isinstance(revisions, (list, tuple)):
            raise ValueError("world schema registry revisions must be a sequence")
        registry._revisions = {
            int(item["version"]): WorldSchema.from_payload(dict(item["schema"]))
            for item in revisions
        }
        if not registry._revisions:
            raise ValueError("world schema registry checkpoint has no revisions")
        registry._active_version = int(payload["active_version"])
        if registry._active_version not in registry._revisions:
            raise ValueError("world schema registry active revision is missing")
        registry._schema = registry._revisions[registry._active_version]
        if (
            registry._schema.payload()
            != WorldSchema.from_payload(dict(payload["schema"])).payload()
        ):
            raise ValueError("world schema registry active schema does not match checkpoint")
        registry._next_version = int(payload.get("next_version", max(registry._revisions) + 1))
        aliases = payload.get("aliases", ())
        registry._aliases = {str(item[0]): str(item[1]) for item in aliases}
        registry._confidence = {
            tuple(str(value) for value in item["key"]): float(item["value"])
            for item in payload.get("confidence", ())
        }
        registry._feedback = {
            tuple(str(value) for value in item["key"]): float(item["value"])
            for item in payload.get("feedback", ())
        }
        registry._tombstones = {
            tuple(str(value) for value in key) for key in payload.get("tombstones", ())
        }
        registry._conflicts = [dict(item) for item in payload.get("conflicts", ())]
        registry._lineage = [dict(item) for item in payload.get("lineage", ())]
        registry._transition_outcomes = {}
        for item in payload.get("transition_outcomes", ()):
            key = tuple(str(value) for value in item["key"])
            hypotheses = item.get("hypotheses")
            if hypotheses is None:
                signature = str(item["signature"])
                registry._transition_outcomes[key] = {signature: 1}
            else:
                registry._transition_outcomes[key] = {
                    str(hypothesis["signature"]): int(hypothesis["evidence_count"])
                    for hypothesis in hypotheses
                }
        registry._transition_confidence = {
            tuple(str(value) for value in item["key"]): float(item["value"])
            for item in payload.get("transition_confidence", ())
        }
        return registry


@dataclass(frozen=True)
class WorldPrediction:
    state: WorldState
    reward: float
    success_probability: float


@dataclass(frozen=True)
class WorldEpisodeRollout:
    episode_id: str
    predictions: tuple[WorldPrediction, ...]

    @property
    def final_state(self) -> WorldState:
        return self.predictions[-1].state


def _replace_numeric_state(
    state: WorldState, schema: WorldSchema, values: torch.Tensor
) -> WorldState:
    updates: dict[str, dict[str, float]] = {}
    for index, (object_id, name) in enumerate(schema.state_slots):
        updates.setdefault(object_id, {})[name] = float(values[index].detach().cpu())
    objects = []
    for obj in state.objects:
        attributes = dict(obj.attributes)
        attributes.update(updates.get(obj.object_id, {}))
        objects.append(
            WorldObject(obj.object_id, attributes=tuple(sorted(attributes.items())), tags=obj.tags)
        )
    known_relations = set(schema.relation_slots)
    relations = {relation for relation in state.relations if relation not in known_relations}
    relations.update(
        relation
        for index, relation in enumerate(schema.relation_slots)
        if float(values[schema.object_state_dim + index].detach().cpu()) >= 0.5
    )
    return WorldState(
        tick=state.tick + 1,
        latent=state.latent.detach().clone(),
        entities=state.entities,
        relations=tuple(sorted(relations)),
        objects=tuple(objects),
        events=state.events,
        affordances=state.affordances,
        uncertainty=state.uncertainty,
        percept_event_id=state.percept_event_id,
        percept_assembly_id=state.percept_assembly_id,
        percept_boundary_closed=state.percept_boundary_closed,
    )


class WorldDynamicsLearner(nn.Module):
    """A compact learned transition model with no sequence-model dependency."""

    def __init__(
        self,
        schema: WorldSchema,
        *,
        hidden_dim: int = 64,
        seed: int = 0,
        schema_registry: WorldSchemaRegistry | None = None,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        torch.manual_seed(int(seed))
        if schema_registry is not None and schema_registry.schema != schema:
            raise ValueError("world schema registry schema must match learner schema")
        self.schema = schema
        self.schema_registry = (
            WorldSchemaRegistry(schema) if schema_registry is None else schema_registry
        )
        self.hidden_dim = int(hidden_dim)
        self.online_updates = 0
        self.transition_acceptances = 0
        self.transition_rejections = 0
        self.schema_evolution_count = 0
        self.network = nn.Sequential(
            nn.Linear(schema.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, schema.state_dim + 2),
        )
        freeze_parameters(self)
        self._schema_snapshots: dict[int, dict[str, torch.Tensor]] = {
            self.schema_registry.active_version: self._snapshot_state_dict()
        }

    def _snapshot_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}

    def _resize_for_schema(self, schema: WorldSchema) -> None:
        old_schema = self.schema
        old_input = self._input_layer
        old_output = self._output_layer
        new_input = nn.Linear(
            schema.input_dim,
            self.hidden_dim,
            device=old_input.weight.device,
            dtype=old_input.weight.dtype,
        )
        new_output = nn.Linear(
            self.hidden_dim,
            schema.state_dim + 2,
            device=old_output.weight.device,
            dtype=old_output.weight.dtype,
        )
        with torch.no_grad():
            new_input.bias.copy_(old_input.bias)
            old_input_indices = {
                key: index for index, key in enumerate(old_schema.input_feature_keys)
            }
            new_input_indices = {key: index for index, key in enumerate(schema.input_feature_keys)}
            for key, old_index in old_input_indices.items():
                new_index = new_input_indices.get(key)
                if new_index is not None:
                    new_input.weight[:, new_index].copy_(old_input.weight[:, old_index])

            new_output.bias.zero_()
            old_output_indices = {
                key: index for index, key in enumerate(old_schema.state_feature_keys)
            }
            new_output_indices = {key: index for index, key in enumerate(schema.state_feature_keys)}
            for key, old_index in old_output_indices.items():
                new_index = new_output_indices.get(key)
                if new_index is not None:
                    new_output.weight[new_index].copy_(old_output.weight[old_index])
                    new_output.bias[new_index].copy_(old_output.bias[old_index])
            new_output.weight[schema.state_dim].copy_(old_output.weight[old_schema.state_dim])
            new_output.bias[schema.state_dim].copy_(old_output.bias[old_schema.state_dim])
            new_output.weight[schema.state_dim + 1].copy_(
                old_output.weight[old_schema.state_dim + 1]
            )
            new_output.bias[schema.state_dim + 1].copy_(old_output.bias[old_schema.state_dim + 1])
        self.network = nn.Sequential(new_input, nn.Tanh(), new_output)
        freeze_parameters(self)

    def register_open_set(
        self,
        *states: WorldState,
        action: Any | None = None,
        register_parameters: bool = True,
    ) -> bool:
        """Grow the schema and network in place from observed identities."""

        if not states and action is None:
            raise ValueError("open-set registration needs an observed state or action")
        observed_states = tuple(states)
        observed_actions = () if action is None else (action,)
        proposal = self.schema_registry.propose_open_set(
            states=observed_states,
            actions=observed_actions,
            register_parameters=register_parameters,
        )
        if proposal is None:
            return False
        self._resize_for_schema(proposal.schema)
        version = self.schema_registry.commit(proposal)
        self.schema = self.schema_registry.schema
        self._schema_snapshots[version] = self._snapshot_state_dict()
        self.schema_evolution_count += 1
        return True

    def register_schema_alias(self, alias: str, canonical: str) -> bool:
        return self.schema_registry.register_alias(alias, canonical)

    def prune_schema(
        self,
        *feature_keys: tuple[str, ...],
        evidence_ids: tuple[str, ...] = (),
    ) -> bool:
        proposal = self.schema_registry.propose_prune(
            tuple(feature_keys), evidence_ids=evidence_ids
        )
        self._resize_for_schema(proposal.schema)
        version = self.schema_registry.commit(proposal)
        self.schema = self.schema_registry.schema
        self._schema_snapshots[version] = self._snapshot_state_dict()
        self.schema_evolution_count += 1
        return True

    def rollback_schema(self, version: int) -> bool:
        target_version = int(version)
        if target_version == self.schema_registry.active_version:
            return False
        snapshot = self._schema_snapshots.get(target_version)
        if snapshot is None:
            raise WorldSchemaRegistryError(
                f"world learner has no network checkpoint for schema revision {target_version}"
            )
        target_schema = self.schema_registry.schema_at(target_version)
        self._resize_for_schema(target_schema)
        self.schema_registry.rollback(target_version)
        self.schema = target_schema
        self.load_state_dict(snapshot)
        freeze_parameters(self)
        return True

    def record_schema_feedback(
        self,
        feature_key: tuple[str, ...],
        observed: float,
        *,
        expected: float | None = None,
        tolerance: float = 0.0,
    ) -> bool:
        return self.schema_registry.record_feedback(
            feature_key,
            observed,
            expected=expected,
            tolerance=tolerance,
        )

    @property
    def _input_layer(self) -> nn.Linear:
        layer = self.network[0]
        assert isinstance(layer, nn.Linear)
        return layer

    @property
    def _output_layer(self) -> nn.Linear:
        layer = self.network[2]
        assert isinstance(layer, nn.Linear)
        return layer

    @property
    def _trainable(self) -> tuple[torch.Tensor, ...]:
        return (
            self._input_layer.weight,
            self._input_layer.bias,
            self._output_layer.weight,
            self._output_layer.bias,
        )

    def _local_pass(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, tuple[torch.Tensor, ...]]:
        """Run one detached forward pass and derive the composite-loss gradients.

        The objective is ``mse(delta) + mse(reward) + bce_with_logits(success)``
        over disjoint output slices, so the output error is the sum of three
        slice-local deltas.  It is then carried one hop back through the
        ``Tanh`` to reach the input layer, which reproduces exactly what
        ``loss.backward()`` used to compute.
        """

        state_dim = self.schema.state_dim
        input_layer = self._input_layer
        output_layer = self._output_layer
        with torch.no_grad():
            hidden = self.network[1](input_layer(features))
            output = output_layer(hidden)
            delta_loss = torch.nn.functional.mse_loss(output[:, :state_dim], targets[:, :state_dim])
            reward_loss = torch.nn.functional.mse_loss(output[:, -2], targets[:, -2])
            success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, -1], targets[:, -1]
            )
            loss = float(delta_loss + reward_loss + success_loss)
        error = torch.zeros_like(output)
        error[:, :state_dim] = mean_squared_error_delta(
            output[:, :state_dim], targets[:, :state_dim]
        )
        error[:, -2] += mean_squared_error_delta(output[:, -2], targets[:, -2])
        error[:, -1] += logistic_error_delta(output[:, -1], targets[:, -1])
        output_gradients = linear_gradients(output_layer, hidden, error)
        hidden_error = tanh_delta(backproject_linear(output_layer, error), hidden)
        input_gradients = linear_gradients(input_layer, features, hidden_error)
        return loss, (*input_gradients, *output_gradients)

    def predict(
        self,
        state: WorldState,
        action: Any,
        *,
        bind_target: bool = True,
        register_parameters: bool = True,
    ) -> WorldPrediction:
        normalized_state = self.schema_registry.normalize_state(state)
        normalized_action = self.schema_registry.normalize_action(action)
        self.register_open_set(
            normalized_state,
            action=normalized_action,
            register_parameters=register_parameters,
        )
        self.eval()
        with torch.no_grad():
            output = self.network(
                self.schema.encode(normalized_state, normalized_action, bind_target=bind_target)
            )
        delta = output[: self.schema.state_dim]
        values = self.schema.state_values(normalized_state) + delta
        return WorldPrediction(
            state=_replace_numeric_state(normalized_state, self.schema, values),
            reward=float(output[-2]),
            success_probability=float(torch.sigmoid(output[-1])),
        )

    def fit(
        self,
        cases: tuple[WorldInterventionCase, ...],
        *,
        epochs: int = 250,
        learning_rate: float = 0.01,
    ) -> list[float]:
        if not cases:
            raise ValueError("world dynamics requires training cases")
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        features = torch.stack([self.schema.encode(case.initial, case.action) for case in cases])
        targets = torch.stack(
            [
                torch.cat(
                    (
                        self.schema.delta_target(case),
                        torch.tensor(
                            [
                                float(case.expected_outcome.reward),
                                float(
                                    case.expected_outcome.success
                                    if case.expected_outcome.success is not None
                                    else case.expected_outcome.reward > 0.0
                                ),
                            ],
                            dtype=torch.float32,
                        ),
                    )
                )
                for case in cases
            ]
        )
        optimizer = LocalAdam(
            self._trainable,
            learning_rate=float(learning_rate),
        )
        losses = []
        self.train()
        for _ in range(int(epochs)):
            loss, gradients = self._local_pass(features, targets)
            optimizer.apply(gradients)
            losses.append(loss)
        return losses

    def online_update(
        self,
        transition: WorldTransition,
        *,
        learning_rate: float = 0.005,
        repeats: int = 1,
        register_parameters: bool = True,
    ) -> list[float]:
        """Apply local error-driven correction from one experienced transition."""

        if float(learning_rate) <= 0.0 or int(repeats) <= 0:
            raise ValueError("learning_rate and repeats must be positive")
        normalized_transition = replace(
            transition,
            before=self.schema_registry.normalize_state(transition.before),
            action=self.schema_registry.normalize_action(transition.action),
            after=self.schema_registry.normalize_state(transition.after),
        )
        self.register_open_set(
            normalized_transition.before,
            normalized_transition.after,
            action=normalized_transition.action,
            register_parameters=register_parameters,
        )
        if not self.schema_registry.record_transition_outcome(normalized_transition):
            self.transition_rejections += 1
            return []
        self.transition_acceptances += 1
        features = self.schema.encode(
            normalized_transition.before, normalized_transition.action
        ).unsqueeze(0)
        target = torch.cat(
            (
                self.schema.state_values(normalized_transition.after)
                - self.schema.state_values(normalized_transition.before),
                torch.tensor(
                    [
                        float(normalized_transition.outcome.reward),
                        float(
                            normalized_transition.outcome.success
                            if normalized_transition.outcome.success is not None
                            else normalized_transition.outcome.reward > 0.0
                        ),
                    ],
                    dtype=torch.float32,
                ),
            )
        ).unsqueeze(0)
        rate = float(learning_rate)
        losses = []
        self.train()
        for _ in range(int(repeats)):
            loss, gradients = self._local_pass(features, target)
            apply_sgd_step(self._trainable, gradients, rate)
            losses.append(loss)
        self.online_updates += 1
        return losses


@dataclass(frozen=True)
class WorldInterventionEvaluationConfig:
    seeds: tuple[int, ...] = (11, 29, 47)
    hidden_dim: int = 64
    epochs: int = 250
    learning_rate: float = 0.01
    minimum_state_gain: float = 0.05
    minimum_outcome_gain: float = 0.02
    minimum_binding_drop: float = 0.05
    maximum_state_mse: float = 0.25
    maximum_outcome_error: float = 1.0
    maximum_cross_seed_std: float = 0.2


def _outcome_success(case: WorldInterventionCase) -> float:
    return float(
        case.expected_outcome.success
        if case.expected_outcome.success is not None
        else case.expected_outcome.reward > 0.0
    )


def _metrics(
    predictions: tuple[WorldPrediction, ...],
    cases: tuple[WorldInterventionCase, ...],
    schema: WorldSchema,
) -> dict[str, float]:
    state_errors = []
    reward_errors = []
    success_losses = []
    success_correct = []
    for prediction, case in zip(predictions, cases, strict=True):
        expected_values = schema.state_values(case.expected_state)
        actual_values = schema.state_values(prediction.state)
        state_errors.append(float(torch.mean((actual_values - expected_values) ** 2)))
        reward_errors.append((prediction.reward - float(case.expected_outcome.reward)) ** 2)
        expected_success = _outcome_success(case)
        probability = min(max(prediction.success_probability, 1e-6), 1.0 - 1e-6)
        success_losses.append(
            -(
                expected_success * math.log(probability)
                + (1.0 - expected_success) * math.log(1.0 - probability)
            )
        )
        success_correct.append(
            float((prediction.success_probability >= 0.5) == bool(expected_success))
        )
    state_mse = sum(state_errors) / len(state_errors)
    reward_mse = sum(reward_errors) / len(reward_errors)
    success_bce = sum(success_losses) / len(success_losses)
    return {
        "state_mse": state_mse,
        "reward_mse": reward_mse,
        "success_bce": success_bce,
        "outcome_error": reward_mse + success_bce,
        "success_accuracy": sum(success_correct) / len(success_correct),
    }


def _mean_targets(
    cases: tuple[WorldInterventionCase, ...], schema: WorldSchema
) -> tuple[torch.Tensor, float, float]:
    deltas = torch.stack([schema.delta_target(case) for case in cases])
    rewards = sum(float(case.expected_outcome.reward) for case in cases) / len(cases)
    successes = sum(_outcome_success(case) for case in cases) / len(cases)
    return deltas.mean(dim=0), rewards, successes


def _baseline_predictions(
    cases: tuple[WorldInterventionCase, ...],
    schema: WorldSchema,
    *,
    deltas: dict[str, torch.Tensor] | None,
    rewards: dict[str, float] | None,
    successes: dict[str, float] | None,
) -> tuple[WorldPrediction, ...]:
    global_delta, global_reward, global_success = _mean_targets(cases, schema)
    predictions = []
    for case in cases:
        key = case.action.kind
        delta = global_delta if deltas is None else deltas.get(key, global_delta)
        reward = global_reward if rewards is None else rewards.get(key, global_reward)
        success = global_success if successes is None else successes.get(key, global_success)
        values = schema.state_values(case.initial) + delta
        predictions.append(
            WorldPrediction(
                state=_replace_numeric_state(case.initial, schema, values),
                reward=reward,
                success_probability=success,
            )
        )
    return tuple(predictions)


def _action_only_statistics(
    cases: tuple[WorldInterventionCase, ...], schema: WorldSchema
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    grouped: dict[str, list[WorldInterventionCase]] = {}
    for case in cases:
        grouped.setdefault(case.action.kind, []).append(case)
    deltas = {}
    rewards = {}
    successes = {}
    for kind, group in grouped.items():
        delta, reward, success = _mean_targets(tuple(group), schema)
        deltas[kind] = delta
        rewards[kind] = reward
        successes[kind] = success
    return deltas, rewards, successes


class WorldInterventionEvaluator:
    """Train and score A2's first causal intervention gate."""

    FORMAT = "taiji-a2-world-intervention-v1"

    def __init__(self, config: WorldInterventionEvaluationConfig | None = None) -> None:
        self.config = config or WorldInterventionEvaluationConfig()

    def evaluate(self, corpus: WorldInterventionCorpus) -> dict[str, Any]:
        if not corpus.train or not corpus.holdout:
            raise ValueError("A2 evaluation requires non-empty train and holdout splits")
        schema = WorldSchema.from_corpus(corpus)
        action_deltas, action_rewards, action_successes = _action_only_statistics(
            corpus.train, schema
        )
        seed_results: list[dict[str, Any]] = []
        for seed in self.config.seeds:
            learner = WorldDynamicsLearner(
                schema,
                hidden_dim=self.config.hidden_dim,
                seed=int(seed),
            )
            losses = learner.fit(
                corpus.train,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
            )
            learned = tuple(learner.predict(case.initial, case.action) for case in corpus.holdout)
            lesion = tuple(
                learner.predict(case.initial, case.action, bind_target=False)
                for case in corpus.holdout
            )
            frequency = _baseline_predictions(
                corpus.holdout,
                schema,
                deltas=None,
                rewards=None,
                successes=None,
            )
            action_only = _baseline_predictions(
                corpus.holdout,
                schema,
                deltas=action_deltas,
                rewards=action_rewards,
                successes=action_successes,
            )
            seed_result = {
                "seed": int(seed),
                "training_loss": float(losses[-1]),
                "learned": _metrics(learned, corpus.holdout, schema),
                "target_binding_lesion": _metrics(lesion, corpus.holdout, schema),
                "frequency_baseline": _metrics(frequency, corpus.holdout, schema),
                "action_only_baseline": _metrics(action_only, corpus.holdout, schema),
            }
            if corpus.time_shuffled:
                shuffled = tuple(
                    learner.predict(case.initial, case.action) for case in corpus.time_shuffled
                )
                seed_result["time_shuffled"] = _metrics(shuffled, corpus.time_shuffled, schema)
            seed_results.append(seed_result)
        state_gains = [
            item["frequency_baseline"]["state_mse"] - item["learned"]["state_mse"]
            for item in seed_results
        ]
        outcome_gains = [
            item["frequency_baseline"]["outcome_error"] - item["learned"]["outcome_error"]
            for item in seed_results
        ]
        binding_drops = [
            item["target_binding_lesion"]["state_mse"] - item["learned"]["state_mse"]
            for item in seed_results
        ]
        state_errors = [item["learned"]["state_mse"] for item in seed_results]
        outcome_errors = [item["learned"]["outcome_error"] for item in seed_results]
        state_error_max = max(state_errors)
        outcome_error_max = max(outcome_errors)
        gate = {
            "state_gain_min": min(state_gains),
            "outcome_gain_min": min(outcome_gains),
            "binding_drop_min": min(binding_drops),
            "state_error_max": state_error_max,
            "outcome_error_max": outcome_error_max,
            "state_gain_std": float(torch.tensor(state_gains).std(unbiased=False)),
            "passed": (
                min(state_gains) >= self.config.minimum_state_gain
                and min(outcome_gains) >= self.config.minimum_outcome_gain
                and min(binding_drops) >= self.config.minimum_binding_drop
                and state_error_max <= self.config.maximum_state_mse
                and outcome_error_max <= self.config.maximum_outcome_error
                and float(torch.tensor(state_gains).std(unbiased=False))
                <= self.config.maximum_cross_seed_std
            ),
        }
        return {
            "format": self.FORMAT,
            "config": {
                "seeds": list(self.config.seeds),
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "minimum_state_gain": self.config.minimum_state_gain,
                "minimum_outcome_gain": self.config.minimum_outcome_gain,
                "minimum_binding_drop": self.config.minimum_binding_drop,
                "maximum_state_mse": self.config.maximum_state_mse,
                "maximum_outcome_error": self.config.maximum_outcome_error,
                "maximum_cross_seed_std": self.config.maximum_cross_seed_std,
            },
            "schema": schema.payload(),
            "train_cases": len(corpus.train),
            "holdout_cases": len(corpus.holdout),
            "time_shuffled_cases": len(corpus.time_shuffled),
            "seeds": seed_results,
            "gate": gate,
        }


def _case_from_transition(transition: Any, case_id: str) -> WorldInterventionCase:
    return WorldInterventionCase(
        case_id=case_id,
        initial=transition.before,
        action=transition.action,
        expected_state=transition.after,
        expected_outcome=transition.outcome,
    )


def rollout_episode(
    learner: WorldDynamicsLearner,
    episode: WorldEpisode,
    *,
    bind_target: bool = True,
) -> WorldEpisodeRollout:
    current = episode.initial
    predictions = []
    for transition in episode.transitions:
        prediction = learner.predict(current, transition.action, bind_target=bind_target)
        predictions.append(prediction)
        current = prediction.state
    return WorldEpisodeRollout(episode_id=episode.episode_id, predictions=tuple(predictions))


def _episode_metrics(
    rollouts: tuple[WorldEpisodeRollout, ...],
    episodes: tuple[WorldEpisode, ...],
    schema: WorldSchema,
) -> dict[str, float]:
    state_errors = []
    final_errors = []
    reward_errors = []
    success_correct = []
    for rollout, episode in zip(rollouts, episodes, strict=True):
        episode_final_error = None
        for prediction, transition in zip(rollout.predictions, episode.transitions, strict=True):
            state_error = float(
                torch.mean(
                    (schema.state_values(prediction.state) - schema.state_values(transition.after))
                    ** 2
                )
            )
            state_errors.append(state_error)
            episode_final_error = state_error
            reward_errors.append((prediction.reward - transition.outcome.reward) ** 2)
            expected_success = bool(
                transition.outcome.success
                if transition.outcome.success is not None
                else transition.outcome.reward > 0.0
            )
            success_correct.append(
                float((prediction.success_probability >= 0.5) == expected_success)
            )
        if episode_final_error is None:
            raise ValueError("world episode must contain at least one transition")
        final_errors.append(episode_final_error)
    return {
        "rollout_state_mse": sum(state_errors) / len(state_errors),
        "final_state_mse": sum(final_errors) / len(final_errors),
        "reward_mse": sum(reward_errors) / len(reward_errors),
        "success_accuracy": sum(success_correct) / len(success_correct),
    }


def _checkpoint_recovery(episodes: tuple[WorldEpisode, ...]) -> bool:
    for episode in episodes:
        store = TaijiWorldState(episode.initial)
        midpoint = max(1, len(episode.transitions) // 2)
        restored = None
        checkpoint_state = None
        for index, transition in enumerate(episode.transitions):
            store.apply(transition)
            if index + 1 == midpoint:
                restored = TaijiWorldState.from_checkpoint(store.checkpoint())
                checkpoint_state = transition.after
        if restored is None:
            return False
        if checkpoint_state is None or not _world_state_equal(restored.state, checkpoint_state):
            return False
        for transition in episode.transitions[midpoint:]:
            restored.apply(transition)
        if not _world_state_equal(restored.state, episode.final_state):
            return False
    return True


@dataclass(frozen=True)
class WorldEpisodeEvaluationConfig:
    seeds: tuple[int, ...] = (11, 29, 47)
    hidden_dim: int = 64
    epochs: int = 250
    learning_rate: float = 0.01
    maximum_rollout_state_mse: float = 0.25
    maximum_final_state_mse: float = 0.5


class WorldEpisodeEvaluator:
    """Evaluate multi-step rollout, episode-ID independence and recovery."""

    FORMAT = "taiji-a2-world-episode-v1"

    def __init__(self, config: WorldEpisodeEvaluationConfig | None = None) -> None:
        self.config = config or WorldEpisodeEvaluationConfig()

    def evaluate(self, corpus: WorldEpisodeCorpus) -> dict[str, Any]:
        if not corpus.train or not corpus.holdout:
            raise ValueError("episode evaluation requires train and holdout episodes")
        train_cases = tuple(
            _case_from_transition(transition, f"{episode.episode_id}:{index}")
            for episode in corpus.train
            for index, transition in enumerate(episode.transitions)
        )
        holdout_cases = tuple(
            _case_from_transition(transition, f"{episode.episode_id}:{index}")
            for episode in corpus.holdout
            for index, transition in enumerate(episode.transitions)
        )
        intervention_corpus = WorldInterventionCorpus(train=train_cases, holdout=holdout_cases)
        schema = WorldSchema.from_corpus(intervention_corpus)
        seed_results: list[dict[str, Any]] = []
        for seed in self.config.seeds:
            learner = WorldDynamicsLearner(
                schema,
                hidden_dim=self.config.hidden_dim,
                seed=int(seed),
            )
            losses = learner.fit(
                train_cases,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
            )
            rollouts = tuple(rollout_episode(learner, episode) for episode in corpus.holdout)
            seed_results.append(
                {
                    "seed": int(seed),
                    "training_loss": float(losses[-1]),
                    "rollout": _episode_metrics(rollouts, corpus.holdout, schema),
                    "schema_uses_episode_id": False,
                }
            )
        rollout_errors = [item["rollout"]["rollout_state_mse"] for item in seed_results]
        final_errors = [item["rollout"]["final_state_mse"] for item in seed_results]
        checkpoint_passed = _checkpoint_recovery(corpus.holdout)
        gate = {
            "rollout_state_mse_max": max(rollout_errors),
            "final_state_mse_max": max(final_errors),
            "checkpoint_recovery": checkpoint_passed,
            "passed": (
                max(rollout_errors) <= self.config.maximum_rollout_state_mse
                and max(final_errors) <= self.config.maximum_final_state_mse
                and checkpoint_passed
            ),
        }
        return {
            "format": self.FORMAT,
            "config": {
                "seeds": list(self.config.seeds),
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "maximum_rollout_state_mse": self.config.maximum_rollout_state_mse,
                "maximum_final_state_mse": self.config.maximum_final_state_mse,
            },
            "schema": schema.payload(),
            "train_episodes": len(corpus.train),
            "holdout_episodes": len(corpus.holdout),
            "seeds": seed_results,
            "gate": gate,
        }
