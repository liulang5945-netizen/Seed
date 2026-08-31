"""Native internalization of admitted Skill/MCP knowledge artifacts.

This module is the E4 boundary between governed corpus material and Taiji's
existing semantic, procedural, and affordance organs.  It never imports a
provider or executes a capability.  Artifact identity is used for provenance
only; the local encoder consumes redacted structural content and schema
metadata, while runtime experiences supply the observed outcome signal.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .affordance import AffordanceFeatureTrainingExample, LearnedAffordanceFeatures
from .contracts import ActionIntent, EpisodicMemoryRecord, Outcome
from .evolution_experience import EvolutionCorpusArtifact, EvolutionExperience
from .internalization import GroundedFeatureExample, content_digest
from .internalization_learner import InternalizationLearningReport, InternalizedFeatureLearner
from .procedural_memory import ProceduralSequenceLearner

ARTIFACT_INTERNALIZATION_FORMAT = "taiji-native-artifact-internalization-v1"
ARTIFACT_INTERNALIZATION_VERSION = 1
ARTIFACT_INTERNALIZATION_MANIFEST_REVISION = "taiji-w7-e4-artifact-internalization-v1"
_ADMITTED_SOURCE_KINDS = frozenset({"skill_artifact", "mcp_artifact"})
_LEARNABLE_UNIT_KINDS = frozenset({"knowledge", "procedure", "affordance"})
_IDENTITY_ONLY_KEYS = frozenset(
    {
        "artifact_digest",
        "content_digest",
        "corpus_id",
        "publisher",
        "relation_digests",
        "scope_id",
        "server_id",
        "skill_id",
        "source_digest",
        "source_id",
    }
)


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _reward(experience: EvolutionExperience) -> float:
    if experience.reward_components:
        value = sum(float(item) for _, item in experience.reward_components)
    else:
        value = 1.0 if experience.success else -1.0
    return max(-1.0, min(1.0, _finite(value, "artifact internalization reward")))


def _reward_terms(experience: EvolutionExperience, reward: float) -> tuple[tuple[str, float], ...]:
    if experience.reward_components:
        return tuple(sorted((str(name), float(value)) for name, value in experience.reward_components))
    return (("outcome", reward),)


def _artifacts(
    artifacts: Iterable[EvolutionCorpusArtifact],
    *,
    partition: str,
) -> tuple[EvolutionCorpusArtifact, ...]:
    items = tuple(artifacts)
    if not items:
        raise ValueError(f"{partition} artifacts must contain at least one item")
    if any(not isinstance(item, EvolutionCorpusArtifact) for item in items):
        raise TypeError(f"{partition} artifacts must contain EvolutionCorpusArtifact values")
    if any(item.partition != partition for item in items):
        raise ValueError(f"{partition} artifacts contain a different partition")
    if any(item.status != "admitted" for item in items):
        raise ValueError(f"{partition} artifacts must be admitted before internalization")
    if any(item.source_kind not in _ADMITTED_SOURCE_KINDS for item in items):
        raise ValueError("artifact internalization only accepts Skill/MCP artifacts")
    if len({item.artifact_digest for item in items}) != len(items):
        raise ValueError(f"{partition} artifacts cannot contain duplicate digests")
    return tuple(sorted(items, key=lambda item: item.artifact_digest))


def _experiences(
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
    if any(item.source_kind not in {"skill", "mcp"} for item in items):
        raise ValueError("artifact internalization only accepts Skill/MCP experiences")
    if len({item.experience_id for item in items}) != len(items):
        raise ValueError(f"{partition} experiences cannot contain duplicate IDs")
    return tuple(sorted(items, key=lambda item: item.experience_id))


def _flatten(value: Any, *, path: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        tokens: list[str] = []
        for raw_key, child in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key).strip()
            if not key or key.lower() in _IDENTITY_ONLY_KEYS:
                continue
            child_path = f"{path}.{key}" if path else key
            tokens.extend(_flatten(child, path=child_path))
        return tuple(tokens)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tokens = []
        for index, child in enumerate(value):
            tokens.extend(_flatten(child, path=f"{path}[]" if path else "[]"))
        return tuple(tokens)
    if isinstance(value, bytes):
        return (f"{path}=bytes",)
    if value is None:
        return (f"{path}=null",) if path else ("null",)
    return (f"{path}={value}",) if path else (str(value),)


class ArtifactKnowledgeEncoder:
    """Encode redacted artifact structure without a source-identity table."""

    _FIELDS = (
        "source_kind",
        "unit_kind",
        "content",
        "capability_semantics",
        "input_schema_digest",
        "output_schema_digest",
        "constraint_digests",
        "language",
    )

    def __init__(self, feature_dim: int = 64, *, namespace: str = "taiji-artifact-knowledge-v1") -> None:
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("artifact knowledge feature_dim must be positive")
        self.namespace = str(namespace).strip()
        if not self.namespace:
            raise ValueError("artifact knowledge namespace cannot be empty")

    def _tokens(self, artifact: EvolutionCorpusArtifact) -> tuple[str, ...]:
        payload = {
            "source_kind": artifact.source_kind,
            "unit_kind": artifact.unit_kind,
            "content": artifact.content,
            "capability_semantics": artifact.capability_semantics,
            "input_schema_digest": artifact.input_schema_digest,
            "output_schema_digest": artifact.output_schema_digest,
            "constraint_digests": artifact.constraint_digests,
            "language": artifact.language,
        }
        return _flatten({key: payload[key] for key in self._FIELDS})

    def encode(self, artifact: EvolutionCorpusArtifact) -> torch.Tensor:
        if not isinstance(artifact, EvolutionCorpusArtifact):
            raise TypeError("artifact knowledge encoder accepts EvolutionCorpusArtifact values")
        if artifact.source_kind not in _ADMITTED_SOURCE_KINDS:
            raise ValueError("artifact knowledge encoder accepts only Skill/MCP artifacts")
        tokens = self._tokens(artifact)
        if not tokens:
            raise ValueError("artifact knowledge encoder received empty structural content")
        vector = torch.zeros(self.feature_dim, dtype=torch.float32)
        for token in tokens:
            seed = f"{self.namespace}\0{token}".encode()
            digest = hashlib.sha256(seed).digest()
            for probe in range(4):
                bucket = int.from_bytes(digest[probe * 4 : probe * 4 + 4], "big") % self.feature_dim
                sign = 1.0 if digest[16 + probe] & 1 else -1.0
                vector[bucket] += sign
        norm = float(vector.norm().item())
        if norm <= 1e-8:
            raise ValueError("artifact knowledge encoder produced an empty vector")
        return vector / norm

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_INTERNALIZATION_FORMAT,
            "version": ARTIFACT_INTERNALIZATION_VERSION,
            "feature_dim": self.feature_dim,
            "namespace": self.namespace,
            "identity_only_keys": sorted(_IDENTITY_ONLY_KEYS),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> ArtifactKnowledgeEncoder:
        if payload.get("format") != ARTIFACT_INTERNALIZATION_FORMAT:
            raise ValueError("unsupported artifact knowledge encoder format")
        if int(payload.get("version", -1)) != ARTIFACT_INTERNALIZATION_VERSION:
            raise ValueError("unsupported artifact knowledge encoder version")
        if tuple(sorted(str(item) for item in payload.get("identity_only_keys", ()))) != tuple(
            sorted(_IDENTITY_ONLY_KEYS)
        ):
            raise ValueError("artifact knowledge identity boundary drift")
        return cls(int(payload["feature_dim"]), namespace=str(payload["namespace"]))


def _procedure_unit(
    artifacts: tuple[EvolutionCorpusArtifact, ...], source_digest: str
) -> EvolutionCorpusArtifact | None:
    candidates = [
        item
        for item in artifacts
        if item.source_digest == source_digest and item.unit_kind == "procedure"
    ]
    if candidates:
        return sorted(candidates, key=lambda item: item.artifact_digest)[0]
    candidates = [item for item in artifacts if item.source_digest == source_digest]
    return sorted(candidates, key=lambda item: item.artifact_digest)[0] if candidates else None


@dataclass(frozen=True)
class ArtifactInternalizationReport:
    """Independent measurements for one artifact-to-organ consolidation."""

    dataset_digest: str
    parent_checkpoint_digest: str
    child_checkpoint_digest: str
    train_artifacts: int
    holdout_artifacts: int
    retention_artifacts: int
    train_experiences: int
    holdout_experiences: int
    retention_experiences: int
    semantic: InternalizationLearningReport
    procedural_train_accuracy: float
    procedural_holdout_accuracy: float
    procedural_retention_accuracy: float
    procedural_lesion_holdout_accuracy: float
    affordance_frozen_holdout_mse: float
    affordance_native_holdout_mse: float
    affordance_native_retention_mse: float
    admitted: bool
    rolled_back: bool
    consumed_artifact_digests: tuple[str, ...]
    consumed_experience_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(
            self.admitted
            and self.semantic.passed
            and self.procedural_holdout_accuracy > self.procedural_lesion_holdout_accuracy
            and self.procedural_retention_accuracy >= 0.5
            and self.affordance_native_holdout_mse < self.affordance_frozen_holdout_mse
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_INTERNALIZATION_FORMAT,
            "dataset_digest": self.dataset_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "child_checkpoint_digest": self.child_checkpoint_digest,
            "train_artifacts": self.train_artifacts,
            "holdout_artifacts": self.holdout_artifacts,
            "retention_artifacts": self.retention_artifacts,
            "train_experiences": self.train_experiences,
            "holdout_experiences": self.holdout_experiences,
            "retention_experiences": self.retention_experiences,
            "semantic": self.semantic.to_payload(),
            "procedural_train_accuracy": self.procedural_train_accuracy,
            "procedural_holdout_accuracy": self.procedural_holdout_accuracy,
            "procedural_retention_accuracy": self.procedural_retention_accuracy,
            "procedural_lesion_holdout_accuracy": self.procedural_lesion_holdout_accuracy,
            "affordance_frozen_holdout_mse": self.affordance_frozen_holdout_mse,
            "affordance_native_holdout_mse": self.affordance_native_holdout_mse,
            "affordance_native_retention_mse": self.affordance_native_retention_mse,
            "admitted": self.admitted,
            "rolled_back": self.rolled_back,
            "passed": self.passed,
            "consumed_artifact_digests": list(self.consumed_artifact_digests),
            "consumed_experience_ids": list(self.consumed_experience_ids),
        }


class ArtifactInternalizationTrainer:
    """Atomically consolidate admitted Skill/MCP artifacts into three organs."""

    def __init__(
        self,
        *,
        feature_dim: int = 64,
        procedural_hidden_dim: int = 16,
        affordance_feature_dim: int = 12,
        seed: int = 17,
        semantic_learning_rate: float = 0.5,
        semantic_passes: int = 12,
        procedural_epochs: int = 250,
        procedural_learning_rate: float = 0.05,
        affordance_epochs: int = 200,
        affordance_learning_rate: float = 0.05,
        manifest_revision: str = ARTIFACT_INTERNALIZATION_MANIFEST_REVISION,
    ) -> None:
        if int(procedural_hidden_dim) <= 0 or int(affordance_feature_dim) <= 0:
            raise ValueError("artifact internalization learner dimensions must be positive")
        if min(int(semantic_passes), int(procedural_epochs), int(affordance_epochs)) <= 0:
            raise ValueError("artifact internalization epochs must be positive")
        if min(
            float(semantic_learning_rate),
            float(procedural_learning_rate),
            float(affordance_learning_rate),
        ) <= 0.0:
            raise ValueError("artifact internalization learning rates must be positive")
        self.encoder = ArtifactKnowledgeEncoder(feature_dim)
        self.procedural_hidden_dim = int(procedural_hidden_dim)
        self.affordance_feature_dim = int(affordance_feature_dim)
        self.seed = int(seed)
        self.semantic_learning_rate = float(semantic_learning_rate)
        self.semantic_passes = int(semantic_passes)
        self.procedural_epochs = int(procedural_epochs)
        self.procedural_learning_rate = float(procedural_learning_rate)
        self.affordance_epochs = int(affordance_epochs)
        self.affordance_learning_rate = float(affordance_learning_rate)
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("artifact internalization manifest_revision cannot be empty")
        self.semantic = InternalizedFeatureLearner(
            feature_dim,
            learning_rate=self.semantic_learning_rate,
            bias_learning_rate=0.0,
            manifest_revision=self.manifest_revision,
        )
        self.procedural = ProceduralSequenceLearner(
            feature_dim,
            hidden_dim=self.procedural_hidden_dim,
            seed=self.seed,
        )
        self.affordance = LearnedAffordanceFeatures(
            feature_dim,
            self.affordance_feature_dim,
            seed=self.seed,
        )
        self.consumed_artifact_digests: tuple[str, ...] = ()
        self.consumed_experience_ids: tuple[str, ...] = ()
        self.revision = 0

    def _example(
        self,
        artifact: EvolutionCorpusArtifact,
        experience: EvolutionExperience,
    ) -> GroundedFeatureExample:
        feature = self.encoder.encode(artifact)
        reward = _reward(experience)
        feature_digest = content_digest(
            {"artifact_digest": artifact.artifact_digest, "feature": feature}
        )
        provenance = tuple(
            sorted(
                (
                    ("artifact", artifact.artifact_digest),
                    ("experience", experience.experience_id),
                    ("source", artifact.source_digest),
                    ("organ", artifact.unit_kind),
                )
            )
        )
        return GroundedFeatureExample(
            example_id=content_digest(
                {
                    "artifact": artifact.artifact_digest,
                    "experience": experience.experience_digest,
                    "feature": feature_digest,
                }
            ),
            evidence_id=experience.experience_id,
            outcome_id=experience.outcome_id or experience.experience_id,
            affordance_id=artifact.artifact_digest,
            action_kind=experience.capability_id or artifact.unit_kind,
            grounding=feature,
            capability_snapshot_digest=experience.capability_snapshot_id or artifact.source_digest,
            parent_checkpoint_id=experience.parent_checkpoint_digest,
            feature_payload_digest=feature_digest,
            reward_terms=_reward_terms(experience, reward),
            provenance=provenance,
            target_reward=reward,
            manifest_revision=self.manifest_revision,
        )

    def _examples(
        self,
        artifacts: tuple[EvolutionCorpusArtifact, ...],
        experiences: tuple[EvolutionExperience, ...],
    ) -> tuple[GroundedFeatureExample, ...]:
        result = []
        for experience in experiences:
            matching = [
                artifact
                for artifact in artifacts
                if artifact.source_digest == experience.source_digest
                and artifact.unit_kind in _LEARNABLE_UNIT_KINDS
            ]
            if not matching:
                raise ValueError("artifact internalization experience has no matching artifact")
            result.extend(self._example(artifact, experience) for artifact in matching)
        if not result:
            raise ValueError("artifact internalization produced no grounded examples")
        return tuple(sorted(result, key=lambda item: item.example_id))

    def _procedural_records(
        self,
        artifacts: tuple[EvolutionCorpusArtifact, ...],
        experiences: tuple[EvolutionExperience, ...],
    ) -> tuple[EpisodicMemoryRecord, ...]:
        records: list[EpisodicMemoryRecord] = []
        for experience in experiences:
            if not experience.capability_id:
                continue
            artifact = _procedure_unit(artifacts, experience.source_digest)
            if artifact is None:
                raise ValueError("procedural experience has no matching artifact unit")
            cue = self.encoder.encode(artifact)
            tick = max(1, int(experience.tick))
            intent_id = f"artifact-intent:{experience.experience_id}"
            records.append(
                EpisodicMemoryRecord(
                    memory_id=f"artifact-memory:{experience.experience_id}",
                    episode_id=experience.episode_id or f"artifact:{experience.source_digest}",
                    tick=tick,
                    cue=cue,
                    action_intent=ActionIntent(
                        intent_id,
                        experience.capability_id,
                        tick=max(0, tick - 1),
                    ),
                    outcome=Outcome(
                        intent_id,
                        reward=_reward(experience),
                        success=experience.success,
                        tick=tick,
                        provenance="artifact-outcome",
                    ),
                    provenance="artifact-internalized",
                )
            )
        if not records:
            raise ValueError("artifact internalization needs capability-bearing experiences")
        return tuple(sorted(records, key=lambda item: item.memory_id))

    @staticmethod
    def _sequence_accuracy(
        learner: ProceduralSequenceLearner,
        records: tuple[EpisodicMemoryRecord, ...],
    ) -> float:
        grouped: dict[str, list[EpisodicMemoryRecord]] = {}
        for record in records:
            grouped.setdefault(record.episode_id, []).append(record)
        correct = 0
        total = 0
        for episode in grouped.values():
            ordered = tuple(sorted(episode, key=lambda item: (item.tick, item.memory_id)))
            actual = tuple(
                record.action_intent.kind
                for record in ordered
                if record.action_intent is not None
            )
            predicted = learner.predict_episode(tuple(record.cue for record in ordered))
            correct += sum(left == right for left, right in zip(predicted, actual, strict=True))
            total += len(actual)
        return correct / float(total)

    @staticmethod
    def _affordance_mse(
        learner: LearnedAffordanceFeatures,
        examples: tuple[AffordanceFeatureTrainingExample, ...],
    ) -> float:
        errors = [
            (learner.predict_reward(example.grounding) - float(example.reward)) ** 2
            for example in examples
        ]
        return sum(errors) / len(errors)

    def consolidate(
        self,
        train_artifacts: Iterable[EvolutionCorpusArtifact],
        *,
        holdout_artifacts: Iterable[EvolutionCorpusArtifact],
        retention_artifacts: Iterable[EvolutionCorpusArtifact],
        train_experiences: Iterable[EvolutionExperience],
        holdout_experiences: Iterable[EvolutionExperience],
        retention_experiences: Iterable[EvolutionExperience],
    ) -> ArtifactInternalizationReport:
        train_artifacts_tuple = _artifacts(train_artifacts, partition="train")
        holdout_artifacts_tuple = _artifacts(holdout_artifacts, partition="holdout")
        retention_artifacts_tuple = _artifacts(retention_artifacts, partition="retention")
        train_experiences_tuple = _experiences(train_experiences, partition="train")
        holdout_experiences_tuple = _experiences(holdout_experiences, partition="holdout")
        retention_experiences_tuple = _experiences(retention_experiences, partition="retention")
        artifact_digests = set()
        for items in (
            train_artifacts_tuple,
            holdout_artifacts_tuple,
            retention_artifacts_tuple,
        ):
            current = {item.artifact_digest for item in items}
            if artifact_digests.intersection(current):
                raise ValueError("artifact partitions must be disjoint")
            artifact_digests.update(current)
        experience_ids = set()
        for items in (
            train_experiences_tuple,
            holdout_experiences_tuple,
            retention_experiences_tuple,
        ):
            current = {item.experience_id for item in items}
            if experience_ids.intersection(current):
                raise ValueError("experience partitions must be disjoint")
            experience_ids.update(current)

        train_examples = self._examples(train_artifacts_tuple, train_experiences_tuple)
        holdout_examples = self._examples(holdout_artifacts_tuple, holdout_experiences_tuple)
        retention_examples = self._examples(retention_artifacts_tuple, retention_experiences_tuple)
        dataset_digest = content_digest(
            {
                "train_artifacts": [item.artifact_digest for item in train_artifacts_tuple],
                "holdout_artifacts": [item.artifact_digest for item in holdout_artifacts_tuple],
                "retention_artifacts": [item.artifact_digest for item in retention_artifacts_tuple],
                "train_experiences": [item.experience_digest for item in train_experiences_tuple],
                "holdout_experiences": [item.experience_digest for item in holdout_experiences_tuple],
                "retention_experiences": [item.experience_digest for item in retention_experiences_tuple],
                "encoder": self.encoder.checkpoint(),
            }
        )
        parent_payload = self.checkpoint()
        parent_digest = content_digest(parent_payload)
        semantic_trial = InternalizedFeatureLearner.from_checkpoint(self.semantic.checkpoint())
        semantic_report = semantic_trial.consolidate(
            train_examples,
            holdout_examples=holdout_examples,
            retention_examples=retention_examples,
            replay_digest=dataset_digest,
            passes=self.semantic_passes,
        )
        procedural_train = self._procedural_records(train_artifacts_tuple, train_experiences_tuple)
        procedural_holdout = self._procedural_records(
            holdout_artifacts_tuple, holdout_experiences_tuple
        )
        procedural_retention = self._procedural_records(
            retention_artifacts_tuple, retention_experiences_tuple
        )
        procedural_trial = ProceduralSequenceLearner.from_checkpoint(self.procedural.checkpoint())
        procedural_trial.consolidate(
            procedural_train,
            epochs=self.procedural_epochs,
            learning_rate=self.procedural_learning_rate,
        )
        procedural_train_accuracy = self._sequence_accuracy(procedural_trial, procedural_train)
        procedural_holdout_accuracy = self._sequence_accuracy(procedural_trial, procedural_holdout)
        procedural_retention_accuracy = self._sequence_accuracy(procedural_trial, procedural_retention)
        procedural_lesion = ProceduralSequenceLearner.from_checkpoint(procedural_trial.checkpoint())
        with torch.no_grad():
            for parameter in procedural_lesion.parameters():
                parameter.zero_()
        procedural_lesion_holdout_accuracy = self._sequence_accuracy(
            procedural_lesion, procedural_holdout
        )

        affordance_train = tuple(
            AffordanceFeatureTrainingExample(
                example_id=example.example_id,
                affordance_id=example.affordance_id,
                action_kind=example.action_kind,
                grounding=example.grounding,
                reward=example.target_reward,
            )
            for example in train_examples
            if next(
                artifact
                for artifact in train_artifacts_tuple
                if artifact.artifact_digest == example.affordance_id
            ).unit_kind
            == "affordance"
        )
        affordance_holdout = tuple(
            AffordanceFeatureTrainingExample(
                example_id=example.example_id,
                affordance_id=example.affordance_id,
                action_kind=example.action_kind,
                grounding=example.grounding,
                reward=example.target_reward,
            )
            for example in holdout_examples
            if next(
                artifact
                for artifact in holdout_artifacts_tuple
                if artifact.artifact_digest == example.affordance_id
            ).unit_kind
            == "affordance"
        )
        affordance_retention = tuple(
            AffordanceFeatureTrainingExample(
                example_id=example.example_id,
                affordance_id=example.affordance_id,
                action_kind=example.action_kind,
                grounding=example.grounding,
                reward=example.target_reward,
            )
            for example in retention_examples
            if next(
                artifact
                for artifact in retention_artifacts_tuple
                if artifact.artifact_digest == example.affordance_id
            ).unit_kind
            == "affordance"
        )
        if not affordance_train or not affordance_holdout or not affordance_retention:
            raise ValueError("artifact internalization requires affordance units in all partitions")
        affordance_trial = LearnedAffordanceFeatures.from_checkpoint(self.affordance.checkpoint())
        frozen_affordance_holdout_mse = self._affordance_mse(affordance_trial, affordance_holdout)
        affordance_trial.fit(
            affordance_train,
            epochs=self.affordance_epochs,
            learning_rate=self.affordance_learning_rate,
        )
        affordance_native_holdout_mse = self._affordance_mse(affordance_trial, affordance_holdout)
        affordance_native_retention_mse = self._affordance_mse(
            affordance_trial, affordance_retention
        )
        admitted = bool(
            semantic_report.passed
            and procedural_holdout_accuracy > procedural_lesion_holdout_accuracy
            and procedural_retention_accuracy >= 0.5
            and affordance_native_holdout_mse < frozen_affordance_holdout_mse
        )
        if admitted:
            self.semantic = semantic_trial
            self.procedural = procedural_trial
            self.affordance = affordance_trial
            self.consumed_artifact_digests = tuple(
                sorted((*self.consumed_artifact_digests, *(artifact_digests)))
            )
            self.consumed_experience_ids = tuple(
                sorted((*self.consumed_experience_ids, *(experience_ids)))
            )
            self.revision += 1
            child_digest = content_digest(self.checkpoint())
        else:
            child_digest = content_digest(
                {
                    "semantic": semantic_trial.checkpoint(),
                    "procedural": procedural_trial.checkpoint(),
                    "affordance": affordance_trial.checkpoint(),
                }
            )
        return ArtifactInternalizationReport(
            dataset_digest=dataset_digest,
            parent_checkpoint_digest=parent_digest,
            child_checkpoint_digest=child_digest,
            train_artifacts=len(train_artifacts_tuple),
            holdout_artifacts=len(holdout_artifacts_tuple),
            retention_artifacts=len(retention_artifacts_tuple),
            train_experiences=len(train_experiences_tuple),
            holdout_experiences=len(holdout_experiences_tuple),
            retention_experiences=len(retention_experiences_tuple),
            semantic=semantic_report,
            procedural_train_accuracy=procedural_train_accuracy,
            procedural_holdout_accuracy=procedural_holdout_accuracy,
            procedural_retention_accuracy=procedural_retention_accuracy,
            procedural_lesion_holdout_accuracy=procedural_lesion_holdout_accuracy,
            affordance_frozen_holdout_mse=frozen_affordance_holdout_mse,
            affordance_native_holdout_mse=affordance_native_holdout_mse,
            affordance_native_retention_mse=affordance_native_retention_mse,
            admitted=admitted,
            rolled_back=not admitted,
            consumed_artifact_digests=tuple(sorted(artifact_digests)) if admitted else (),
            consumed_experience_ids=tuple(sorted(experience_ids)) if admitted else (),
        )

    def semantic_value_from_feature(self, feature: torch.Tensor) -> float:
        """Read internalized semantic value without accepting an external artifact."""

        example = GroundedFeatureExample(
            example_id="internal-query",
            evidence_id="internal-query",
            outcome_id="internal-query",
            affordance_id="internal-query",
            action_kind="internal-query",
            grounding=feature,
            capability_snapshot_digest="0" * 64,
            parent_checkpoint_id="0" * 64,
            feature_payload_digest=content_digest(feature),
            reward_terms=(("query", 0.0),),
            provenance=(("organ", "semantic"),),
        )
        return self.semantic.score(example)

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": ARTIFACT_INTERNALIZATION_FORMAT,
            "version": ARTIFACT_INTERNALIZATION_VERSION,
            "encoder": self.encoder.checkpoint(),
            "procedural_hidden_dim": self.procedural_hidden_dim,
            "affordance_feature_dim": self.affordance_feature_dim,
            "seed": self.seed,
            "semantic_learning_rate": self.semantic_learning_rate,
            "semantic_passes": self.semantic_passes,
            "procedural_epochs": self.procedural_epochs,
            "procedural_learning_rate": self.procedural_learning_rate,
            "affordance_epochs": self.affordance_epochs,
            "affordance_learning_rate": self.affordance_learning_rate,
            "manifest_revision": self.manifest_revision,
            "semantic": self.semantic.checkpoint(),
            "procedural": self.procedural.checkpoint(),
            "affordance": self.affordance.checkpoint(),
            "consumed_artifact_digests": list(self.consumed_artifact_digests),
            "consumed_experience_ids": list(self.consumed_experience_ids),
            "revision": self.revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> ArtifactInternalizationTrainer:
        if payload.get("format") != ARTIFACT_INTERNALIZATION_FORMAT:
            raise ValueError("unsupported artifact internalization format")
        if int(payload.get("version", -1)) != ARTIFACT_INTERNALIZATION_VERSION:
            raise ValueError("unsupported artifact internalization version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("artifact internalization checkpoint digest mismatch")
        encoder = ArtifactKnowledgeEncoder.from_checkpoint(payload["encoder"])
        trainer = cls(
            feature_dim=encoder.feature_dim,
            procedural_hidden_dim=int(payload["procedural_hidden_dim"]),
            affordance_feature_dim=int(payload["affordance_feature_dim"]),
            seed=int(payload.get("seed", 0)),
            semantic_learning_rate=float(payload["semantic_learning_rate"]),
            semantic_passes=int(payload["semantic_passes"]),
            procedural_epochs=int(payload["procedural_epochs"]),
            procedural_learning_rate=float(payload["procedural_learning_rate"]),
            affordance_epochs=int(payload["affordance_epochs"]),
            affordance_learning_rate=float(payload["affordance_learning_rate"]),
            manifest_revision=str(payload["manifest_revision"]),
        )
        trainer.encoder = encoder
        trainer.semantic = InternalizedFeatureLearner.from_checkpoint(payload["semantic"])
        trainer.procedural = ProceduralSequenceLearner.from_checkpoint(payload["procedural"])
        trainer.affordance = LearnedAffordanceFeatures.from_checkpoint(payload["affordance"])
        if trainer.semantic.feature_dim != encoder.feature_dim:
            raise ValueError("artifact internalization semantic dimension drift")
        if trainer.procedural.cue_dim != encoder.feature_dim:
            raise ValueError("artifact internalization procedural dimension drift")
        if trainer.affordance.input_dim != encoder.feature_dim:
            raise ValueError("artifact internalization affordance dimension drift")
        trainer.consumed_artifact_digests = tuple(
            sorted(str(item) for item in payload.get("consumed_artifact_digests", ()))
        )
        trainer.consumed_experience_ids = tuple(
            sorted(str(item) for item in payload.get("consumed_experience_ids", ()))
        )
        trainer.revision = int(payload.get("revision", 0))
        if trainer.revision < 0:
            raise ValueError("artifact internalization revision cannot be negative")
        return trainer


__all__ = [
    "ARTIFACT_INTERNALIZATION_FORMAT",
    "ARTIFACT_INTERNALIZATION_MANIFEST_REVISION",
    "ARTIFACT_INTERNALIZATION_VERSION",
    "ArtifactInternalizationReport",
    "ArtifactInternalizationTrainer",
    "ArtifactKnowledgeEncoder",
]
