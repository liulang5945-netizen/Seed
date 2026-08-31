"""Checkpoint envelope used before any Taiji-native training mutation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from seed_platform.evolution_ledger import EvolutionExperienceLedger
from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest

TRAINING_CHECKPOINT_FORMAT = "seed-taiji-native-training-checkpoint-v1"
TRAINING_CHECKPOINT_VERSION = 1
TRAINING_CHECKPOINT_KINDS = ("parent", "trial", "admitted")
_PARTITIONS = ("train", "holdout", "retention", "security")


def _numeric_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for key, raw_number in value.items():
        if isinstance(raw_number, bool):
            raise TypeError(f"{name}.{key} must be numeric")
        number = float(raw_number)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name}.{key} must be finite and non-negative")
        result[str(key)] = number
    return {key: result[key] for key in sorted(result)}


def _dataset_digest(ledger: EvolutionExperienceLedger) -> str:
    corpus, experiences = ledger.training_view()
    return content_digest(
        {
            "corpus": [item.artifact_digest for item in corpus],
            "experiences": [item.experience_digest for item in experiences],
        }
    )


def _partition_manifest(ledger: EvolutionExperienceLedger) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for partition in _PARTITIONS:
        corpus = [item for item in ledger.corpus if item.partition == partition]
        experiences = [item for item in ledger.experiences if item.partition == partition]
        manifest[partition] = {
            "corpus_count": len(corpus),
            "corpus_digest": content_digest([item.artifact_digest for item in corpus]),
            "experience_count": len(experiences),
            "experience_digest": content_digest(
                [item.experience_digest for item in experiences]
            ),
        }
    return manifest


def _ledger_cursor(ledger: EvolutionExperienceLedger) -> dict[str, Any]:
    return {
        "revision": int(ledger.revision),
        "tail_event_digest": ledger.tail_event_digest,
        "next_event_sequence": len(ledger.experiences) + 1,
    }


@dataclass(frozen=True)
class NativeTrainingCheckpoint:
    checkpoint_kind: str
    checkpoint_id: str
    parent_checkpoint_digest: str
    model_checkpoint: Mapping[str, Any]
    model_digest: str
    ledger_checkpoint: Mapping[str, Any]
    dataset_digest: str
    ledger_cursor: Mapping[str, Any]
    partition_manifest: Mapping[str, Any]
    learner_state: Mapping[str, Any]
    random_state: torch.Tensor
    resource_ledger: Mapping[str, float]
    checkpoint_digest: str = ""
    version: int = TRAINING_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.version != TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported training checkpoint version")
        if self.checkpoint_kind not in TRAINING_CHECKPOINT_KINDS:
            raise ValueError("unsupported training checkpoint kind")
        if not str(self.checkpoint_id).strip():
            raise ValueError("training checkpoint_id cannot be empty")
        if self.checkpoint_kind != "parent" and not self.parent_checkpoint_digest:
            raise ValueError("trial/admitted checkpoint requires parent_checkpoint_digest")
        if self.parent_checkpoint_digest and not self._is_digest(self.parent_checkpoint_digest):
            raise ValueError("parent_checkpoint_digest must be a SHA-256 digest")
        if not self._is_digest(self.model_digest):
            raise ValueError("model_digest must be a SHA-256 digest")
        if not isinstance(self.model_checkpoint, Mapping):
            raise TypeError("model_checkpoint must be a mapping")
        if not isinstance(self.ledger_checkpoint, Mapping):
            raise TypeError("ledger_checkpoint must be a mapping")
        if not isinstance(self.ledger_cursor, Mapping):
            raise TypeError("ledger_cursor must be a mapping")
        if not isinstance(self.partition_manifest, Mapping):
            raise TypeError("partition_manifest must be a mapping")
        if not isinstance(self.learner_state, Mapping):
            raise TypeError("learner_state must be a mapping")
        ledger_digest = str(self.ledger_checkpoint.get("checkpoint_digest", ""))
        if not self._is_digest(ledger_digest):
            raise ValueError("ledger checkpoint must contain a SHA-256 digest")
        if not self._is_digest(self.dataset_digest):
            raise ValueError("dataset_digest must be a SHA-256 digest")
        if self.checkpoint_digest and not self._is_digest(self.checkpoint_digest):
            raise ValueError("checkpoint_digest must be a SHA-256 digest")
        if not isinstance(self.random_state, torch.Tensor):
            raise TypeError("random_state must be a tensor")
        object.__setattr__(self, "checkpoint_id", str(self.checkpoint_id).strip())
        object.__setattr__(self, "parent_checkpoint_digest", str(self.parent_checkpoint_digest).strip())
        object.__setattr__(self, "model_digest", str(self.model_digest).strip())
        object.__setattr__(self, "dataset_digest", str(self.dataset_digest).strip())
        object.__setattr__(self, "checkpoint_digest", str(self.checkpoint_digest).strip())

        expected_model_digest = content_digest(self.model_checkpoint)
        if expected_model_digest != self.model_digest:
            raise ValueError("training model checkpoint digest mismatch")
        expected_ledger_digest = str(self.ledger_checkpoint.get("checkpoint_digest", ""))
        if content_digest(
            {key: value for key, value in self.ledger_checkpoint.items() if key != "checkpoint_digest"}
        ) != expected_ledger_digest:
            raise ValueError("training ledger checkpoint digest mismatch")
        if self.checkpoint_digest and self.checkpoint_digest != content_digest(self._header_payload()):
            raise ValueError("training checkpoint digest mismatch")
        object.__setattr__(self, "random_state", self.random_state.detach().cpu().clone())
        object.__setattr__(self, "resource_ledger", _numeric_mapping(self.resource_ledger, "resource_ledger"))

    @classmethod
    def create(
        cls,
        model: Taiji,
        ledger: EvolutionExperienceLedger,
        *,
        checkpoint_kind: str,
        checkpoint_id: str,
        parent_checkpoint_digest: str = "",
        learner_state: Mapping[str, Any] | None = None,
        resource_ledger: Mapping[str, Any] | None = None,
    ) -> NativeTrainingCheckpoint:
        if not isinstance(model, Taiji):
            raise TypeError("training checkpoint model must be Taiji")
        if not isinstance(ledger, EvolutionExperienceLedger):
            raise TypeError("training checkpoint ledger must be EvolutionExperienceLedger")
        model_checkpoint = model.checkpoint()
        ledger_checkpoint = ledger.checkpoint()
        record = cls(
            checkpoint_kind=checkpoint_kind,
            checkpoint_id=checkpoint_id,
            parent_checkpoint_digest=parent_checkpoint_digest,
            model_checkpoint=model_checkpoint,
            model_digest=content_digest(model_checkpoint),
            ledger_checkpoint=ledger_checkpoint,
            dataset_digest=_dataset_digest(ledger),
            ledger_cursor=_ledger_cursor(ledger),
            partition_manifest=_partition_manifest(ledger),
            learner_state={} if learner_state is None else dict(learner_state),
            random_state=torch.random.get_rng_state(),
            resource_ledger={} if resource_ledger is None else resource_ledger,
        )
        return cls(**{**record.__dict__, "checkpoint_digest": content_digest(record._header_payload())})

    def _header_payload(self) -> dict[str, Any]:
        return {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "version": self.version,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "model_digest": self.model_digest,
            "ledger_digest": self.ledger_checkpoint["checkpoint_digest"],
            "dataset_digest": self.dataset_digest,
            "ledger_cursor": dict(self.ledger_cursor),
            "partition_manifest": dict(self.partition_manifest),
            "learner_state": dict(self.learner_state),
            "random_state": self.random_state,
            "resource_ledger": dict(self.resource_ledger),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._header_payload(),
            "model_checkpoint": dict(self.model_checkpoint),
            "ledger_checkpoint": dict(self.ledger_checkpoint),
            "checkpoint_digest": self.checkpoint_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NativeTrainingCheckpoint:
        if payload.get("format") != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported training checkpoint format")
        if int(payload.get("version", -1)) != TRAINING_CHECKPOINT_VERSION:
            raise ValueError("unsupported training checkpoint version")
        return cls(
            checkpoint_kind=str(payload["checkpoint_kind"]),
            checkpoint_id=str(payload["checkpoint_id"]),
            parent_checkpoint_digest=str(payload.get("parent_checkpoint_digest", "")),
            model_checkpoint=payload["model_checkpoint"],
            model_digest=str(payload["model_digest"]),
            ledger_checkpoint=payload["ledger_checkpoint"],
            dataset_digest=str(payload["dataset_digest"]),
            ledger_cursor=payload["ledger_cursor"],
            partition_manifest=payload["partition_manifest"],
            learner_state=payload.get("learner_state") or {},
            random_state=payload["random_state"],
            resource_ledger=payload.get("resource_ledger") or {},
            checkpoint_digest=str(payload["checkpoint_digest"]),
            version=int(payload["version"]),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> NativeTrainingCheckpoint:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError("training checkpoint file must contain a mapping")
        return cls.from_payload(payload)

    def validate_against(
        self,
        model: Taiji,
        ledger: EvolutionExperienceLedger,
    ) -> None:
        if not isinstance(model, Taiji) or not isinstance(ledger, EvolutionExperienceLedger):
            raise TypeError("training checkpoint validation requires Taiji and evolution ledger")
        actual_config = TaijiConfig.from_dict(dict(self.model_checkpoint["config"]))
        if actual_config != model.config:
            raise ValueError("training checkpoint architecture drift")
        current_ledger = ledger.checkpoint()
        if current_ledger.get("checkpoint_digest") != self.ledger_checkpoint.get("checkpoint_digest"):
            raise ValueError("training checkpoint ledger drift")
        if _dataset_digest(ledger) != self.dataset_digest:
            raise ValueError("training checkpoint dataset drift")
        if _ledger_cursor(ledger) != dict(self.ledger_cursor):
            raise ValueError("training checkpoint ledger cursor drift")
        if _partition_manifest(ledger) != dict(self.partition_manifest):
            raise ValueError("training checkpoint partition manifest drift")
        Taiji.from_checkpoint(self.model_checkpoint, device=model.device)
        EvolutionExperienceLedger.from_checkpoint(self.ledger_checkpoint)

    def restore_into(
        self,
        model: Taiji,
        ledger: EvolutionExperienceLedger,
    ) -> EvolutionExperienceLedger:
        self.validate_against(model, ledger)
        restored_ledger = EvolutionExperienceLedger.from_checkpoint(self.ledger_checkpoint)
        model.restore(self.model_checkpoint)
        torch.random.set_rng_state(self.random_state.clone())
        return restored_ledger

    @staticmethod
    def _is_digest(value: Any) -> bool:
        text = str(value).strip()
        return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


__all__ = [
    "NativeTrainingCheckpoint",
    "TRAINING_CHECKPOINT_FORMAT",
    "TRAINING_CHECKPOINT_KINDS",
    "TRAINING_CHECKPOINT_VERSION",
]
