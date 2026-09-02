"""Auditable key/value training data contract for native episodic learning."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

MEMORY_LEARNING_EXAMPLE_FORMAT = "taiji-native-memory-learning-example-v1"
MEMORY_LEARNING_EXAMPLE_VERSION = 1
MEMORY_LEARNING_CURRICULUM_FORMAT = "taiji-native-memory-learning-curriculum-v1"
MEMORY_LEARNING_CURRICULUM_VERSION = 1
MEMORY_LEARNING_ROLES = ("repeated_observation", "one_shot_experience")
MEMORY_LEARNING_PARTITIONS = ("train", "holdout", "retention")
MEMORY_LEARNING_QUERY_PARTITIONS = ("holdout", "retention")
MEMORY_LEARNING_BYPASS_FIELDS = ("time_index", "episode_tag", "provenance")
MEMORY_LEARNING_EXAMPLE_FIELDS = (
    "example_id",
    "partition",
    "cue_key",
    "action_value",
    "outcome_value",
    "reward",
    "role",
    *MEMORY_LEARNING_BYPASS_FIELDS,
)


@dataclass(frozen=True)
class MemoryLearningExample:
    """One cue_key -> (action_value, outcome_value) observation with bypass fields.

    The key and the two value channels are the only label-bearing fields.  Time,
    episode and provenance are declared bypass fields: they describe how the
    observation reached the model and must never determine the answer.
    """

    example_id: str
    partition: str
    cue_key: int
    action_value: int
    outcome_value: int
    reward: float = 1.0
    role: str = "repeated_observation"
    time_index: int = 2
    episode_tag: str = ""
    provenance: str = "experienced"

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("memory learning example id must be non-empty")
        if self.partition not in MEMORY_LEARNING_PARTITIONS:
            raise ValueError(
                f"unsupported memory learning partition: {self.partition}"
            )
        if self.role not in MEMORY_LEARNING_ROLES:
            raise ValueError(f"unsupported memory learning role: {self.role}")
        symbols = (self.cue_key, self.action_value, self.outcome_value)
        if any(int(value) < 0 for value in symbols):
            raise ValueError("memory learning symbols cannot be negative")
        if not math.isfinite(float(self.reward)):
            raise ValueError("memory learning reward must be finite")
        if self.time_index < 0:
            raise ValueError("memory learning time index cannot be negative")
        if not self.provenance.strip():
            raise ValueError("memory learning provenance cannot be empty")

    @property
    def value_key(self) -> str:
        return f"{self.action_value}/{self.outcome_value}"

    @property
    def is_query(self) -> bool:
        return self.partition in MEMORY_LEARNING_QUERY_PARTITIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MEMORY_LEARNING_EXAMPLE_FORMAT,
            "version": MEMORY_LEARNING_EXAMPLE_VERSION,
            "fields": list(MEMORY_LEARNING_EXAMPLE_FIELDS),
            "example_id": self.example_id,
            "partition": self.partition,
            "cue_key": int(self.cue_key),
            "action_value": int(self.action_value),
            "outcome_value": int(self.outcome_value),
            "reward": float(self.reward),
            "role": self.role,
            "time_index": int(self.time_index),
            "episode_tag": self.episode_tag,
            "provenance": self.provenance,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryLearningExample:
        if payload.get("format") != MEMORY_LEARNING_EXAMPLE_FORMAT:
            raise ValueError("unsupported memory learning example format")
        if int(payload.get("version", -1)) != MEMORY_LEARNING_EXAMPLE_VERSION:
            raise ValueError("unsupported memory learning example version")
        fields = tuple(str(item) for item in payload.get("fields", ()))
        if fields != MEMORY_LEARNING_EXAMPLE_FIELDS:
            raise ValueError("memory learning example fields do not match the contract")
        return cls(
            example_id=str(payload["example_id"]),
            partition=str(payload["partition"]),
            cue_key=int(payload["cue_key"]),
            action_value=int(payload["action_value"]),
            outcome_value=int(payload["outcome_value"]),
            reward=float(payload["reward"]),
            role=str(payload["role"]),
            time_index=int(payload["time_index"]),
            episode_tag=str(payload["episode_tag"]),
            provenance=str(payload["provenance"]),
        )


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("memory learning summary needs at least one value")
    numbers = [float(value) for value in values]
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _bypass_field_leaks(examples: Sequence[MemoryLearningExample]) -> dict[str, bool]:
    """Flag bypass fields that group examples and still determine the answer.

    Leak detection is scoped to a single partition on purpose: a shortcut is
    exploitable exactly where it is observed, and mixing partitions lets an
    unrelated default value hide a real train-side shortcut. A field whose value
    is unique per example cannot generalise and is not a leak; a field that
    groups several examples yet always maps to one value_key is.
    """

    leaks: dict[str, bool] = {}
    answers = {example.value_key for example in examples}
    for name in MEMORY_LEARNING_BYPASS_FIELDS:
        observed: dict[Any, set[str]] = defaultdict(set)
        for example in examples:
            observed[getattr(example, name)].add(example.value_key)
        groups_examples = 1 < len(observed) < len(examples)
        deterministic = all(len(values) == 1 for values in observed.values())
        leaks[name] = groups_examples and deterministic and len(answers) > 1
    return leaks


@dataclass(frozen=True)
class MemoryLearningCurriculum:
    """A train/holdout/retention curriculum whose key/value relations are auditable."""

    name: str
    train: tuple[MemoryLearningExample, ...]
    holdout: tuple[MemoryLearningExample, ...]
    retention: tuple[MemoryLearningExample, ...]
    declared_key_value_deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("memory learning curriculum name must be non-empty")
        for partition in MEMORY_LEARNING_PARTITIONS:
            examples = getattr(self, partition)
            if not examples:
                raise ValueError(
                    f"memory learning curriculum partition cannot be empty: {partition}"
                )
            for example in examples:
                if example.partition != partition:
                    raise ValueError(
                        "memory learning example partition does not match its slot: "
                        f"{example.example_id} declares {example.partition}, stored in {partition}"
                    )
        identifiers = [example.example_id for example in self.examples]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("memory learning example ids must be unique")
        train_keys = {example.cue_key for example in self.train}
        for partition in MEMORY_LEARNING_QUERY_PARTITIONS:
            for example in getattr(self, partition):
                if example.cue_key not in train_keys:
                    raise ValueError(
                        "memory learning query targets an untrained cue key: "
                        f"{example.example_id}"
                    )
        if self.declared_key_value_deterministic and self._conflicting_train_keys():
            raise ValueError(
                "memory learning curriculum declares determinism but train keys conflict"
            )

    @property
    def examples(self) -> tuple[MemoryLearningExample, ...]:
        return (*self.train, *self.holdout, *self.retention)

    def _values_by_key(
        self, examples: Sequence[MemoryLearningExample]
    ) -> dict[int, set[str]]:
        values: dict[int, set[str]] = defaultdict(set)
        for example in examples:
            values[example.cue_key].add(example.value_key)
        return values

    def _conflicting_train_keys(self) -> tuple[int, ...]:
        values = self._values_by_key(self.train)
        return tuple(sorted(key for key, items in values.items() if len(items) > 1))

    def _partition_audit(
        self, examples: Sequence[MemoryLearningExample]
    ) -> dict[str, Any]:
        observations = Counter(example.cue_key for example in examples)
        values = self._values_by_key(examples)
        roles = Counter(example.role for example in examples)
        role_consistent = all(
            (example.role == "repeated_observation") == (observations[example.cue_key] > 1)
            for example in examples
        )
        actions = {example.action_value for example in examples}
        outcomes = {example.outcome_value for example in examples}
        combinations = Counter(example.value_key for example in examples)
        conflicting = tuple(sorted(key for key, items in values.items() if len(items) > 1))
        bypass_leaks = _bypass_field_leaks(examples)
        return {
            "example_count": len(examples),
            "key_count": len(observations),
            "value_count": len(combinations),
            "action_count": len(actions),
            "outcome_count": len(outcomes),
            "factorial_complete": len(combinations) == len(actions) * len(outcomes),
            "observations_per_key": _summary(tuple(observations.values())),
            "roles": {name: int(roles.get(name, 0)) for name in MEMORY_LEARNING_ROLES},
            "role_labels_match_observation_counts": role_consistent,
            "values_per_key": _summary(
                tuple(float(len(items)) for items in values.values())
            ),
            "conflicting_key_count": len(conflicting),
            "conflicting_keys": list(conflicting),
            "key_value_deterministic": not conflicting,
            "reward": _summary(tuple(example.reward for example in examples)),
            "combination_counts": dict(sorted(combinations.items())),
            "bypass_field_value_leak": bypass_leaks,
            "any_bypass_field_leak": any(bypass_leaks.values()),
        }

    def audit(self) -> dict[str, Any]:
        partitions = {
            partition: self._partition_audit(getattr(self, partition))
            for partition in MEMORY_LEARNING_PARTITIONS
        }
        train_values = self._values_by_key(self.train)
        query_relations: dict[str, Any] = {}
        for partition in MEMORY_LEARNING_QUERY_PARTITIONS:
            examples = getattr(self, partition)
            answered_by_train = sum(
                int(example.value_key in train_values[example.cue_key])
                for example in examples
            )
            ambiguous = sum(
                int(len(train_values[example.cue_key]) > 1) for example in examples
            )
            query_relations[partition] = {
                "query_count": len(examples),
                "keys_covered_by_train": True,
                "answers_present_in_train": answered_by_train,
                "answers_present_in_train_ratio": answered_by_train / len(examples),
                "answer_matches_train_taught_value": answered_by_train == len(examples),
                "ambiguous_key_query_count": ambiguous,
            }
        bypass_leaks = {
            name: any(
                partitions[partition]["bypass_field_value_leak"][name]
                for partition in MEMORY_LEARNING_PARTITIONS
            )
            for name in MEMORY_LEARNING_BYPASS_FIELDS
        }
        train_ids = {example.example_id for example in self.train}
        query_ids = {
            example.example_id
            for partition in MEMORY_LEARNING_QUERY_PARTITIONS
            for example in getattr(self, partition)
        }
        leakage = {
            "train_query_id_overlap": len(train_ids & query_ids),
            "bypass_field_value_leak": bypass_leaks,
            "any_bypass_field_leak": any(bypass_leaks.values()),
        }
        well_formed = (
            leakage["train_query_id_overlap"] == 0
            and not leakage["any_bypass_field_leak"]
            and all(
                partitions[partition]["factorial_complete"]
                for partition in MEMORY_LEARNING_PARTITIONS
            )
            and all(
                partitions[partition]["role_labels_match_observation_counts"]
                for partition in MEMORY_LEARNING_PARTITIONS
            )
            and all(
                query_relations[partition]["answer_matches_train_taught_value"]
                for partition in MEMORY_LEARNING_QUERY_PARTITIONS
            )
        )
        return {
            "format": MEMORY_LEARNING_CURRICULUM_FORMAT,
            "version": MEMORY_LEARNING_CURRICULUM_VERSION,
            "name": self.name,
            "declared_key_value_deterministic": self.declared_key_value_deterministic,
            "observed_key_value_deterministic": not self._conflicting_train_keys(),
            "declaration_matches_observation": (
                self.declared_key_value_deterministic
                == (not self._conflicting_train_keys())
            ),
            "partitions": partitions,
            "query_relations": query_relations,
            "leakage": leakage,
            "well_formed": well_formed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MEMORY_LEARNING_CURRICULUM_FORMAT,
            "version": MEMORY_LEARNING_CURRICULUM_VERSION,
            "name": self.name,
            "declared_key_value_deterministic": self.declared_key_value_deterministic,
            "partitions": {
                partition: [
                    example.to_dict() for example in getattr(self, partition)
                ]
                for partition in MEMORY_LEARNING_PARTITIONS
            },
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryLearningCurriculum:
        if payload.get("format") != MEMORY_LEARNING_CURRICULUM_FORMAT:
            raise ValueError("unsupported memory learning curriculum format")
        if int(payload.get("version", -1)) != MEMORY_LEARNING_CURRICULUM_VERSION:
            raise ValueError("unsupported memory learning curriculum version")
        partitions = payload.get("partitions", {})
        missing = set(MEMORY_LEARNING_PARTITIONS) - set(partitions)
        if missing:
            raise ValueError(
                f"memory learning curriculum is missing partitions: {sorted(missing)}"
            )
        return cls(
            name=str(payload["name"]),
            train=tuple(
                MemoryLearningExample.from_dict(item) for item in partitions["train"]
            ),
            holdout=tuple(
                MemoryLearningExample.from_dict(item) for item in partitions["holdout"]
            ),
            retention=tuple(
                MemoryLearningExample.from_dict(item) for item in partitions["retention"]
            ),
            declared_key_value_deterministic=bool(
                payload.get("declared_key_value_deterministic", True)
            ),
        )
