"""Content-addressed continuation-learning contracts for the native K runtime.

The K adapter already proves that a real observation can cross the K1/K2/K3
boundary and that a candidate namespace can be rolled back.  This module adds
the missing learning boundary: a completed episode becomes a typed experience,
train and holdout experiences are record/family disjoint, and an update receipt
states exactly which owners changed.

Version 1 deliberately mirrors the workers that exist today.  K1 and K2 use
Taiji's detached local-delta rules and therefore carry no optimizer state.  K3
is a deterministic outcome projector.  The parent Taiji checkpoint and K3 are
immutable; only the K1/K2 worker checkpoints may change in the candidate
namespace.  A later optimizer-backed implementation must introduce a new
contract version instead of silently changing this one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .continual_k_adapter import KAdapterExchange
from .internalization import content_digest
from .k_worker_manifest import K_WORKER_IDS
from .outcome_dependency import OutcomeDependencyProjection
from .semantic_training import StructuredSemanticExample
from .semantic_transition import StructuredSemanticTransitionExample

TAIJI_K_CONTINUATION_CONTRACT_FORMAT = "taiji-k-continuation-learning-v1"
TAIJI_K_CONTINUATION_CONTRACT_VERSION = 1
K_CONTINUATION_SPLITS = ("train", "holdout")
K_CONTINUATION_LEARNABLE_WORKERS = ("k1.semantic", "k2.transition")
K_CONTINUATION_FROZEN_WORKERS = ("k3.outcome_projection",)


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _pairs(values: Any, name: str) -> tuple[tuple[str, str], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of pairs")
    normalized = tuple(
        (_text(item[0], f"{name} worker id"), _digest(item[1], f"{name} digest"))
        for item in values
    )
    if tuple(worker_id for worker_id, _ in normalized) != K_WORKER_IDS:
        raise ValueError(f"{name} must contain exactly K1/K2/K3 in order")
    return normalized


def _tags(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_text(value, f"{name} item") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class KContinuationExperience:
    """One completed K episode with explicit supervised continuation targets."""

    experience_id: str
    family_id: str
    split: str
    parent_checkpoint_digest: str
    worker_bundle_digest: str
    source_manifest_digest: str
    observation_digest: str
    semantic_example: StructuredSemanticExample
    transition_example: StructuredSemanticTransitionExample
    projection: OutcomeDependencyProjection
    exchange: KAdapterExchange
    format: str = TAIJI_K_CONTINUATION_CONTRACT_FORMAT
    version: int = TAIJI_K_CONTINUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_CONTINUATION_CONTRACT_FORMAT:
            raise ValueError("unsupported K continuation experience format")
        if int(self.version) != TAIJI_K_CONTINUATION_CONTRACT_VERSION:
            raise ValueError("unsupported K continuation experience version")
        object.__setattr__(self, "experience_id", _text(self.experience_id, "experience_id"))
        object.__setattr__(self, "family_id", _text(self.family_id, "family_id"))
        if self.split not in K_CONTINUATION_SPLITS:
            raise ValueError(f"unsupported K continuation split: {self.split}")
        for field_name in (
            "parent_checkpoint_digest",
            "worker_bundle_digest",
            "source_manifest_digest",
            "observation_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        if not isinstance(self.semantic_example, StructuredSemanticExample):
            raise TypeError("K continuation semantic target is invalid")
        if not isinstance(self.transition_example, StructuredSemanticTransitionExample):
            raise TypeError("K continuation transition target is invalid")
        if not isinstance(self.projection, OutcomeDependencyProjection):
            raise TypeError("K continuation projection target is invalid")
        if not isinstance(self.exchange, KAdapterExchange):
            raise TypeError("K continuation exchange is invalid")
        if not self.projection.accepted:
            raise ValueError("K continuation experience requires an accepted K3 projection")
        if self.exchange.input.parent_checkpoint_digest != self.parent_checkpoint_digest:
            raise ValueError("K continuation exchange crosses the parent checkpoint")
        if self.exchange.input.source_manifest_digest != self.source_manifest_digest:
            raise ValueError("K continuation exchange crosses the source manifest")
        if self.exchange.input.observation_digest != self.observation_digest:
            raise ValueError("K continuation observation digest does not match the exchange")
        if self.exchange.output.outcome_signature != self.projection.outcome_signature:
            raise ValueError("K continuation exchange does not echo the projection outcome")
        if self.exchange.output.dependency_digest != self.projection.dependency_digest:
            raise ValueError("K continuation exchange does not echo the dependency")
        if self.exchange.output.dependency_projection_digest != self.projection.projection_digest:
            raise ValueError("K continuation exchange does not echo the projection digest")
        if self.exchange.scope_id != self.projection.scope_id:
            raise ValueError("K continuation exchange and projection scopes differ")

        semantic = self.semantic_example
        transition = self.transition_example
        if semantic.percept.event_id != transition.event.event_id:
            raise ValueError("K continuation semantic and transition events differ")
        if int(semantic.percept.observation_tick) != int(transition.event.observation_tick):
            raise ValueError("K continuation semantic and transition ticks differ")
        if int(semantic.world.tick) != int(transition.after.tick):
            raise ValueError("K continuation semantic and transition target ticks differ")
        if semantic.goal.goal_id != transition.goal.goal_id:
            raise ValueError("K continuation semantic and transition goals differ")
        if semantic.content.content_id != transition.content.content_id:
            raise ValueError("K continuation semantic and transition contents differ")
        if not torch.equal(
            semantic.percept.features.detach().cpu(),
            transition.event.features.detach().cpu(),
        ):
            raise ValueError("K continuation semantic and transition features differ")
        if int(self.exchange.input.tick) != int(transition.event.observation_tick):
            raise ValueError("K continuation exchange tick does not match the target")
        if int(self.projection.parent_tick) != int(transition.after.tick):
            raise ValueError("K continuation projection tick does not match the target")
        success = dict(self.projection.event.attributes).get("success")
        if not isinstance(success, bool):
            raise ValueError("K continuation projection has no boolean success outcome")
        if bool(self.exchange.output.success) != success:
            raise ValueError("K continuation exchange success does not match the projection")

    @property
    def experience_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    @property
    def target_digest(self) -> str:
        return content_digest(
            {
                "semantic": self.semantic_example.to_payload(),
                "transition": self.transition_example.to_payload(),
                "projection": self.projection.to_payload(),
            }
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": int(self.version),
            "kind": "experience",
            "experience_id": self.experience_id,
            "family_id": self.family_id,
            "split": self.split,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "worker_bundle_digest": self.worker_bundle_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "observation_digest": self.observation_digest,
            "semantic_example": self.semantic_example.to_payload(),
            "transition_example": self.transition_example.to_payload(),
            "projection": self.projection.to_payload(),
            "exchange": self.exchange.to_payload(),
            "target_digest": self.target_digest,
        }
        if include_digest:
            payload["experience_digest"] = self.experience_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KContinuationExperience:
        semantic_payload = payload.get("semantic_example")
        transition_payload = payload.get("transition_example")
        projection_payload = payload.get("projection")
        exchange_payload = payload.get("exchange")
        if not all(
            isinstance(item, Mapping)
            for item in (semantic_payload, transition_payload, projection_payload, exchange_payload)
        ):
            raise ValueError("K continuation experience contains an invalid nested payload")
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            experience_id=str(payload["experience_id"]),
            family_id=str(payload["family_id"]),
            split=str(payload["split"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            worker_bundle_digest=str(payload["worker_bundle_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            observation_digest=str(payload["observation_digest"]),
            semantic_example=StructuredSemanticExample.from_payload(semantic_payload),
            transition_example=StructuredSemanticTransitionExample.from_payload(
                transition_payload
            ),
            projection=OutcomeDependencyProjection.from_payload(projection_payload),
            exchange=KAdapterExchange.from_payload(exchange_payload),
        )
        if str(payload.get("target_digest", "")) != item.target_digest:
            raise ValueError("K continuation target digest mismatch")
        if str(payload.get("experience_digest", "")) != item.experience_digest:
            raise ValueError("K continuation experience digest mismatch")
        return item


@dataclass(frozen=True)
class KContinuationCourse:
    """A sealed continuation course with disjoint train and holdout records."""

    course_id: str
    parent_checkpoint_digest: str
    worker_bundle_digest: str
    source_manifest_digest: str
    train: tuple[KContinuationExperience, ...]
    holdout: tuple[KContinuationExperience, ...]
    learnable_workers: tuple[str, ...] = K_CONTINUATION_LEARNABLE_WORKERS
    frozen_workers: tuple[str, ...] = K_CONTINUATION_FROZEN_WORKERS
    format: str = TAIJI_K_CONTINUATION_CONTRACT_FORMAT
    version: int = TAIJI_K_CONTINUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_CONTINUATION_CONTRACT_FORMAT:
            raise ValueError("unsupported K continuation course format")
        if int(self.version) != TAIJI_K_CONTINUATION_CONTRACT_VERSION:
            raise ValueError("unsupported K continuation course version")
        object.__setattr__(self, "course_id", _text(self.course_id, "course_id"))
        for field_name in (
            "parent_checkpoint_digest",
            "worker_bundle_digest",
            "source_manifest_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        train = tuple(self.train)
        holdout = tuple(self.holdout)
        if not train or not holdout:
            raise ValueError("K continuation course requires train and holdout records")
        if any(not isinstance(item, KContinuationExperience) for item in (*train, *holdout)):
            raise TypeError("K continuation course contains an invalid experience")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "holdout", holdout)
        for item in (*train, *holdout):
            if item.parent_checkpoint_digest != self.parent_checkpoint_digest:
                raise ValueError("K continuation course crosses parent checkpoints")
            if item.worker_bundle_digest != self.worker_bundle_digest:
                raise ValueError("K continuation course crosses worker bundles")
            if item.source_manifest_digest != self.source_manifest_digest:
                raise ValueError("K continuation course crosses source manifests")
        if tuple(item.split for item in train) != ("train",) * len(train):
            raise ValueError("K continuation train records must use the train split")
        if tuple(item.split for item in holdout) != ("holdout",) * len(holdout):
            raise ValueError("K continuation holdout records must use the holdout split")
        all_items = (*train, *holdout)
        for field_name in ("experience_id", "observation_digest"):
            values = [str(getattr(item, field_name)) for item in all_items]
            if len(set(values)) != len(values):
                raise ValueError(f"K continuation {field_name} values must be globally unique")
        train_families = {item.family_id for item in train}
        holdout_families = {item.family_id for item in holdout}
        if train_families & holdout_families:
            raise ValueError("K continuation family leakage between train and holdout")
        train_inputs = self._input_identities(train)
        holdout_inputs = self._input_identities(holdout)
        if train_inputs & holdout_inputs:
            raise ValueError("K continuation input leakage between train and holdout")
        object.__setattr__(
            self,
            "learnable_workers",
            _tags(self.learnable_workers, "learnable_workers"),
        )
        object.__setattr__(self, "frozen_workers", _tags(self.frozen_workers, "frozen_workers"))
        if self.learnable_workers != K_CONTINUATION_LEARNABLE_WORKERS:
            raise ValueError("K continuation v1 only permits K1/K2 updates")
        if self.frozen_workers != K_CONTINUATION_FROZEN_WORKERS:
            raise ValueError("K continuation v1 requires K3 to remain frozen")

    @staticmethod
    def _input_identities(
        experiences: Sequence[KContinuationExperience],
    ) -> set[str]:
        return {
            identity
            for item in experiences
            for identity in (
                item.observation_digest,
                item.exchange.input.input_digest,
                item.semantic_example.input_digest,
                item.transition_example.input_digest,
            )
        }

    @property
    def train_experience_digests(self) -> tuple[str, ...]:
        return tuple(item.experience_digest for item in self.train)

    @property
    def holdout_experience_digests(self) -> tuple[str, ...]:
        return tuple(item.experience_digest for item in self.holdout)

    @property
    def course_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": int(self.version),
            "kind": "course",
            "course_id": self.course_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "worker_bundle_digest": self.worker_bundle_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "train": [item.to_payload() for item in self.train],
            "holdout": [item.to_payload() for item in self.holdout],
            "learnable_workers": list(self.learnable_workers),
            "frozen_workers": list(self.frozen_workers),
        }
        if include_digest:
            payload["course_digest"] = self.course_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KContinuationCourse:
        train_payload = payload.get("train")
        holdout_payload = payload.get("holdout")
        if not isinstance(train_payload, Sequence) or isinstance(
            train_payload, (str, bytes, bytearray)
        ):
            raise ValueError("K continuation train payload is invalid")
        if not isinstance(holdout_payload, Sequence) or isinstance(
            holdout_payload, (str, bytes, bytearray)
        ):
            raise ValueError("K continuation holdout payload is invalid")
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            course_id=str(payload["course_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            worker_bundle_digest=str(payload["worker_bundle_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            train=tuple(KContinuationExperience.from_payload(value) for value in train_payload),
            holdout=tuple(
                KContinuationExperience.from_payload(value) for value in holdout_payload
            ),
            learnable_workers=tuple(str(value) for value in payload["learnable_workers"]),
            frozen_workers=tuple(str(value) for value in payload["frozen_workers"]),
        )
        if str(payload.get("course_digest", "")) != item.course_digest:
            raise ValueError("K continuation course digest mismatch")
        return item


@dataclass(frozen=True)
class KContinuationUpdateReceipt:
    """Audit receipt for one candidate update against a sealed course."""

    course_digest: str
    parent_checkpoint_digest: str
    parent_worker_bundle_digest: str
    candidate_worker_bundle_digest: str
    parent_worker_checkpoint_digests: tuple[tuple[str, str], ...]
    candidate_worker_checkpoint_digests: tuple[tuple[str, str], ...]
    train_experience_digests: tuple[str, ...]
    holdout_experience_digests: tuple[str, ...]
    updated_workers: tuple[str, ...]
    frozen_workers: tuple[str, ...]
    candidate_namespace: str
    training_steps: int
    optimizer_state_present: bool
    fresh_restore_verified: bool
    parent_unchanged: bool
    holdout_untrained: bool
    rollback_restored: bool
    format: str = TAIJI_K_CONTINUATION_CONTRACT_FORMAT
    version: int = TAIJI_K_CONTINUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_CONTINUATION_CONTRACT_FORMAT:
            raise ValueError("unsupported K continuation receipt format")
        if int(self.version) != TAIJI_K_CONTINUATION_CONTRACT_VERSION:
            raise ValueError("unsupported K continuation receipt version")
        for field_name in (
            "course_digest",
            "parent_checkpoint_digest",
            "parent_worker_bundle_digest",
            "candidate_worker_bundle_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "parent_worker_checkpoint_digests",
            _pairs(self.parent_worker_checkpoint_digests, "parent_worker_checkpoint_digests"),
        )
        object.__setattr__(
            self,
            "candidate_worker_checkpoint_digests",
            _pairs(
                self.candidate_worker_checkpoint_digests,
                "candidate_worker_checkpoint_digests",
            ),
        )
        for field_name in ("train_experience_digests", "holdout_experience_digests"):
            values = tuple(
                _digest(value, f"{field_name} item")
                for value in getattr(self, field_name)
            )
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be unique and non-empty")
            object.__setattr__(self, field_name, values)
        if set(self.train_experience_digests) & set(self.holdout_experience_digests):
            raise ValueError("K continuation receipt trains on holdout experiences")
        object.__setattr__(self, "updated_workers", _tags(self.updated_workers, "updated_workers"))
        object.__setattr__(self, "frozen_workers", _tags(self.frozen_workers, "frozen_workers"))
        if self.updated_workers != K_CONTINUATION_LEARNABLE_WORKERS:
            raise ValueError("K continuation receipt updated worker set is invalid")
        if self.frozen_workers != K_CONTINUATION_FROZEN_WORKERS:
            raise ValueError("K continuation receipt frozen worker set is invalid")
        object.__setattr__(self, "candidate_namespace", _text(self.candidate_namespace, "candidate_namespace"))
        if int(self.training_steps) <= 0:
            raise ValueError("K continuation receipt requires a positive training step count")
        if bool(self.optimizer_state_present):
            raise ValueError("K continuation v1 does not permit optimizer state")
        if self.parent_worker_bundle_digest == self.candidate_worker_bundle_digest:
            raise ValueError("K continuation candidate bundle must differ from its parent")
        parent = dict(self.parent_worker_checkpoint_digests)
        candidate = dict(self.candidate_worker_checkpoint_digests)
        if parent["k3.outcome_projection"] != candidate["k3.outcome_projection"]:
            raise ValueError("K continuation receipt changed deterministic K3")
        if all(parent[worker_id] == candidate[worker_id] for worker_id in K_CONTINUATION_LEARNABLE_WORKERS):
            raise ValueError("K continuation receipt changed no learnable worker")
        for field_name in (
            "optimizer_state_present",
            "fresh_restore_verified",
            "parent_unchanged",
            "holdout_untrained",
            "rollback_restored",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")

    @property
    def passed(self) -> bool:
        return all(
            (
                not self.optimizer_state_present,
                self.fresh_restore_verified,
                self.parent_unchanged,
                self.holdout_untrained,
                self.rollback_restored,
            )
        )

    @property
    def receipt_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": int(self.version),
            "kind": "update_receipt",
            "course_digest": self.course_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_worker_bundle_digest": self.parent_worker_bundle_digest,
            "candidate_worker_bundle_digest": self.candidate_worker_bundle_digest,
            "parent_worker_checkpoint_digests": [
                list(item) for item in self.parent_worker_checkpoint_digests
            ],
            "candidate_worker_checkpoint_digests": [
                list(item) for item in self.candidate_worker_checkpoint_digests
            ],
            "train_experience_digests": list(self.train_experience_digests),
            "holdout_experience_digests": list(self.holdout_experience_digests),
            "updated_workers": list(self.updated_workers),
            "frozen_workers": list(self.frozen_workers),
            "candidate_namespace": self.candidate_namespace,
            "training_steps": int(self.training_steps),
            "optimizer_state_present": bool(self.optimizer_state_present),
            "fresh_restore_verified": bool(self.fresh_restore_verified),
            "parent_unchanged": bool(self.parent_unchanged),
            "holdout_untrained": bool(self.holdout_untrained),
            "rollback_restored": bool(self.rollback_restored),
        }
        if include_digest:
            payload["receipt_digest"] = self.receipt_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KContinuationUpdateReceipt:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            course_digest=str(payload["course_digest"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            parent_worker_bundle_digest=str(payload["parent_worker_bundle_digest"]),
            candidate_worker_bundle_digest=str(payload["candidate_worker_bundle_digest"]),
            parent_worker_checkpoint_digests=tuple(
                tuple(value) for value in payload["parent_worker_checkpoint_digests"]
            ),
            candidate_worker_checkpoint_digests=tuple(
                tuple(value) for value in payload["candidate_worker_checkpoint_digests"]
            ),
            train_experience_digests=tuple(
                str(value) for value in payload["train_experience_digests"]
            ),
            holdout_experience_digests=tuple(
                str(value) for value in payload["holdout_experience_digests"]
            ),
            updated_workers=tuple(str(value) for value in payload["updated_workers"]),
            frozen_workers=tuple(str(value) for value in payload["frozen_workers"]),
            candidate_namespace=str(payload["candidate_namespace"]),
            training_steps=int(payload["training_steps"]),
            optimizer_state_present=bool(payload["optimizer_state_present"]),
            fresh_restore_verified=bool(payload["fresh_restore_verified"]),
            parent_unchanged=bool(payload["parent_unchanged"]),
            holdout_untrained=bool(payload["holdout_untrained"]),
            rollback_restored=bool(payload["rollback_restored"]),
        )
        if str(payload.get("receipt_digest", "")) != item.receipt_digest:
            raise ValueError("K continuation receipt digest mismatch")
        return item


__all__ = [
    "K_CONTINUATION_FROZEN_WORKERS",
    "K_CONTINUATION_LEARNABLE_WORKERS",
    "K_CONTINUATION_SPLITS",
    "KContinuationCourse",
    "KContinuationExperience",
    "KContinuationUpdateReceipt",
    "TAIJI_K_CONTINUATION_CONTRACT_FORMAT",
    "TAIJI_K_CONTINUATION_CONTRACT_VERSION",
]
