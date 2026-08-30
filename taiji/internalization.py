"""Taiji-owned grounded knowledge internalization primitives.

R5A-S0 is deliberately limited to pure conversion and deterministic replay
bookkeeping.  It does not execute an affordance, choose a capability, call a
provider, or mutate a learner.  The R5A-S1 learner consumes its examples only
through the separate checkpointed native learner module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import Outcome, WorldAffordance

INTERNALIZATION_CHECKPOINT_FORMAT = "taiji-internalization-v1"
INTERNALIZATION_MANIFEST_REVISION = "taiji-w7-r5-internalization-v1"
INTERNALIZATION_ALLOWED_STATUSES = (
    "external",
    "shadow",
    "internalized",
    "rejected",
    "rolled_back",
    "tombstoned",
)
INTERNALIZATION_TRAIN_PARTITION = "train"
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "capability_id",
        "capability_name",
        "prompt",
        "provider_text",
        "raw_transcript",
        "transcript",
    }
)


def _canonical(value: Any) -> Any:
    """Convert supported values into an order-independent JSON value."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": tensor.numpy().tobytes().hex(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in content digest: {type(value).__name__}")


def content_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a supported DTO payload."""

    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest_text(value: str, name: str, *, optional: bool = False) -> str:
    normalized = str(value).strip()
    if not normalized and optional:
        return ""
    return _required_text(normalized, name)


def _text_pairs(value: Mapping[str, Any], name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    pairs: list[tuple[str, str]] = []
    for key, item in value.items():
        normalized_key = _required_text(str(key), f"{name} key")
        if normalized_key in _FORBIDDEN_INPUT_KEYS:
            raise ValueError(f"{name} contains forbidden input: {normalized_key}")
        pairs.append((normalized_key, _required_text(str(item), f"{name}.{normalized_key}")))
    return tuple(sorted(pairs))


def _reward_pairs(
    value: Mapping[str, float] | None,
    *,
    lower: float,
    upper: float,
) -> tuple[tuple[str, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError("reward_terms must be a non-empty mapping")
    pairs: list[tuple[str, float]] = []
    for key, item in value.items():
        name = _required_text(str(key), "reward term name")
        if isinstance(item, bool):
            raise ValueError(f"reward term {name} must be numeric")
        number = float(item)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"reward term {name} is outside the configured bounds")
        pairs.append((name, number))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class InternalizationInput:
    """A grounded Workbench consequence eligible for DTO conversion.

    The object intentionally carries only digests for percept/world/recovery
    context.  It has no provider text, prompt, capability id, or transcript
    field, so those values cannot become a hidden learning input.
    """

    evidence_id: str
    outcome_id: str
    outcome: Outcome
    affordance: WorldAffordance
    capability_snapshot_digest: str
    parent_checkpoint_id: str
    owner_id: str
    manifest_revision: str = INTERNALIZATION_MANIFEST_REVISION
    reward_terms: Mapping[str, float] | None = None
    percept_digest: str = ""
    world_digest: str = ""
    recovery_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _required_text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "outcome_id", _required_text(self.outcome_id, "outcome_id"))
        if not isinstance(self.outcome, Outcome):
            raise TypeError("outcome must be a Taiji Outcome")
        if not isinstance(self.affordance, WorldAffordance):
            raise TypeError("affordance must be a grounded WorldAffordance")
        object.__setattr__(
            self,
            "capability_snapshot_digest",
            _digest_text(self.capability_snapshot_digest, "capability_snapshot_digest"),
        )
        object.__setattr__(
            self,
            "parent_checkpoint_id",
            _required_text(self.parent_checkpoint_id, "parent_checkpoint_id"),
        )
        object.__setattr__(self, "owner_id", _required_text(self.owner_id, "owner_id"))
        object.__setattr__(
            self,
            "manifest_revision",
            _required_text(self.manifest_revision, "manifest_revision"),
        )
        object.__setattr__(
            self,
            "reward_terms",
            None if self.reward_terms is None else dict(self.reward_terms),
        )
        for name in ("percept_digest", "world_digest", "recovery_digest"):
            object.__setattr__(
                self,
                name,
                _digest_text(getattr(self, name), name, optional=True),
            )
        object.__setattr__(self, "metadata", dict(self.metadata))
        _text_pairs(self.metadata, "metadata")

    def binding_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "outcome_id": self.outcome_id,
            "outcome": self.outcome.to_payload(),
            "affordance": self.affordance.to_payload(),
            "capability_snapshot_digest": self.capability_snapshot_digest,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "owner_id": self.owner_id,
            "manifest_revision": self.manifest_revision,
            "reward_terms": self.reward_terms,
            "percept_digest": self.percept_digest,
            "world_digest": self.world_digest,
            "recovery_digest": self.recovery_digest,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroundedFeatureExample:
    """Versioned, content-addressed feature material for bounded replay."""

    example_id: str
    evidence_id: str
    outcome_id: str
    affordance_id: str
    action_kind: str
    grounding: torch.Tensor
    capability_snapshot_digest: str
    parent_checkpoint_id: str
    feature_payload_digest: str
    reward_terms: tuple[tuple[str, float], ...]
    provenance: tuple[tuple[str, str], ...]
    target_reward: float = 0.0
    manifest_revision: str = INTERNALIZATION_MANIFEST_REVISION
    status: str = "external"
    tombstone_reason: str = ""

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "evidence_id",
            "outcome_id",
            "affordance_id",
            "action_kind",
            "capability_snapshot_digest",
            "parent_checkpoint_id",
            "feature_payload_digest",
            "manifest_revision",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.status not in INTERNALIZATION_ALLOWED_STATUSES:
            raise ValueError(f"unsupported internalization status: {self.status}")
        if self.status == "tombstoned" and not str(self.tombstone_reason).strip():
            raise ValueError("tombstoned example requires a reason")
        if self.grounding.ndim != 1 or not self.grounding.numel():
            raise ValueError("internalization grounding must be a non-empty vector")
        if not bool(torch.isfinite(self.grounding).all()):
            raise ValueError("internalization grounding must be finite")
        object.__setattr__(self, "grounding", self.grounding.detach().clone().to(torch.float32))
        if not math.isfinite(float(self.target_reward)):
            raise ValueError("internalization target_reward must be finite")
        object.__setattr__(self, "target_reward", float(self.target_reward))
        reward_pairs = tuple((str(name), float(value)) for name, value in tuple(self.reward_terms))
        if not reward_pairs or tuple(sorted(reward_pairs)) != reward_pairs:
            raise ValueError("internalization reward_terms must be sorted and non-empty")
        if len({name for name, _ in reward_pairs}) != len(reward_pairs):
            raise ValueError("internalization reward_terms must have unique names")
        if any(not math.isfinite(value) for _, value in reward_pairs):
            raise ValueError("internalization reward_terms must be finite")
        object.__setattr__(self, "reward_terms", reward_pairs)
        provenance = tuple((str(name), str(value)) for name, value in tuple(self.provenance))
        if tuple(sorted(provenance)) != provenance or any(
            not name or not value for name, value in provenance
        ):
            raise ValueError("internalization provenance must be sorted and non-empty")
        object.__setattr__(self, "provenance", provenance)

    @property
    def content_digest(self) -> str:
        return content_digest(self.content_payload())

    def content_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "outcome_id": self.outcome_id,
            "affordance_id": self.affordance_id,
            "action_kind": self.action_kind,
            "grounding": self.grounding,
            "capability_snapshot_digest": self.capability_snapshot_digest,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "feature_payload_digest": self.feature_payload_digest,
            "reward_terms": list(self.reward_terms),
            "provenance": list(self.provenance),
            "target_reward": self.target_reward,
            "manifest_revision": self.manifest_revision,
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self.content_payload()
        payload.update(
            {
                "format": INTERNALIZATION_CHECKPOINT_FORMAT,
                "example_id": self.example_id,
                "status": self.status,
                "tombstone_reason": self.tombstone_reason,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GroundedFeatureExample:
        if payload.get("format") != INTERNALIZATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization example format")
        return cls(
            example_id=str(payload["example_id"]),
            evidence_id=str(payload["evidence_id"]),
            outcome_id=str(payload["outcome_id"]),
            affordance_id=str(payload["affordance_id"]),
            action_kind=str(payload["action_kind"]),
            grounding=payload["grounding"].detach().clone(),
            capability_snapshot_digest=str(payload["capability_snapshot_digest"]),
            parent_checkpoint_id=str(payload["parent_checkpoint_id"]),
            feature_payload_digest=str(payload["feature_payload_digest"]),
            reward_terms=tuple(
                (str(name), float(value)) for name, value in payload["reward_terms"]
            ),
            provenance=tuple((str(name), str(value)) for name, value in payload["provenance"]),
            target_reward=float(payload.get("target_reward", 0.0)),
            manifest_revision=str(
                payload.get("manifest_revision", INTERNALIZATION_MANIFEST_REVISION)
            ),
            status=str(payload.get("status", "external")),
            tombstone_reason=str(payload.get("tombstone_reason", "")),
        )


@dataclass(frozen=True)
class InternalizationLifecycleRecord:
    """Auditable status and event lineage for one conversion attempt."""

    example_id: str
    evidence_id: str
    status: str
    events: tuple[str, ...]
    source_digest: str
    tombstone_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _required_text(self.example_id, "example_id"))
        object.__setattr__(self, "evidence_id", _required_text(self.evidence_id, "evidence_id"))
        if self.status not in INTERNALIZATION_ALLOWED_STATUSES:
            raise ValueError(f"unsupported lifecycle status: {self.status}")
        object.__setattr__(
            self, "events", tuple(_required_text(item, "lifecycle event") for item in self.events)
        )
        object.__setattr__(self, "source_digest", _digest_text(self.source_digest, "source_digest"))
        if self.status == "tombstoned" and not str(self.tombstone_reason).strip():
            raise ValueError("tombstoned lifecycle record requires a reason")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_CHECKPOINT_FORMAT,
            "example_id": self.example_id,
            "evidence_id": self.evidence_id,
            "status": self.status,
            "events": list(self.events),
            "source_digest": self.source_digest,
            "tombstone_reason": self.tombstone_reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InternalizationLifecycleRecord:
        if payload.get("format") != INTERNALIZATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization lifecycle format")
        return cls(
            example_id=str(payload["example_id"]),
            evidence_id=str(payload["evidence_id"]),
            status=str(payload["status"]),
            events=tuple(str(item) for item in payload["events"]),
            source_digest=str(payload["source_digest"]),
            tombstone_reason=str(payload.get("tombstone_reason", "")),
        )


@dataclass(frozen=True)
class InternalizationCausalGate:
    """The five independent checks required before external tombstoning."""

    external_sufficiency: bool
    internalization_necessity: bool
    grounding_necessity: bool
    checkpoint_recoverable: bool
    old_task_retention: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.external_sufficiency,
                self.internalization_necessity,
                self.grounding_necessity,
                self.checkpoint_recoverable,
                self.old_task_retention,
            )
        )

    def to_payload(self) -> dict[str, bool]:
        return {
            "external_sufficiency": self.external_sufficiency,
            "internalization_necessity": self.internalization_necessity,
            "grounding_necessity": self.grounding_necessity,
            "checkpoint_recoverable": self.checkpoint_recoverable,
            "old_task_retention": self.old_task_retention,
        }


@dataclass(frozen=True)
class InternalizationConversionResult:
    example: GroundedFeatureExample | None
    lifecycle: InternalizationLifecycleRecord
    accepted: bool
    reason: str = ""

    @property
    def status(self) -> str:
        return self.lifecycle.status

    @property
    def example_id(self) -> str:
        return self.lifecycle.example_id


class InternalizationConverter:
    """Convert only a fully bound, grounded outcome into feature material."""

    def __init__(
        self,
        *,
        manifest_revision: str = INTERNALIZATION_MANIFEST_REVISION,
        seed: int = 0,
        replay_budget: int = 64,
        reward_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.manifest_revision = _required_text(manifest_revision, "manifest_revision")
        self.seed = int(seed)
        self.replay_budget = int(replay_budget)
        if self.replay_budget <= 0:
            raise ValueError("replay_budget must be positive")
        if len(reward_bounds) != 2 or not reward_bounds[0] <= reward_bounds[1]:
            raise ValueError("reward_bounds must be an ordered pair")
        if not all(math.isfinite(float(item)) for item in reward_bounds):
            raise ValueError("reward_bounds must be finite")
        self.reward_bounds = (float(reward_bounds[0]), float(reward_bounds[1]))

    def _reject(self, source: InternalizationInput, reason: str) -> InternalizationConversionResult:
        source_digest = content_digest(source.binding_payload())
        lifecycle = InternalizationLifecycleRecord(
            example_id=f"rejected:{source_digest}",
            evidence_id=source.evidence_id,
            status="rejected",
            events=("outcome_bound", "conversion_rejected"),
            source_digest=source_digest,
            tombstone_reason=reason,
        )
        return InternalizationConversionResult(None, lifecycle, False, reason)

    def convert(self, source: InternalizationInput) -> InternalizationConversionResult:
        if not isinstance(source, InternalizationInput):
            raise TypeError("source must be an InternalizationInput")
        if source.manifest_revision != self.manifest_revision:
            return self._reject(source, "manifest_revision_mismatch")
        source_digest = content_digest(source.binding_payload())
        lower, upper = self.reward_bounds
        try:
            reward_terms = _reward_pairs(
                source.reward_terms,
                lower=lower,
                upper=upper,
            )
        except (TypeError, ValueError):
            return self._reject(source, "invalid_reward_terms")
        if reward_terms is None:
            return self._reject(source, "missing_reward_terms")
        if not lower <= float(source.outcome.reward) <= upper:
            return self._reject(source, "outcome_reward_out_of_bounds")
        affordance = source.affordance
        if (
            affordance.feature_provenance != "world-state-grounding"
            or not affordance.grounding_lineage
            or not any(item.startswith("world-state:") for item in affordance.grounding_lineage)
            or not affordance.features.numel()
        ):
            return self._reject(source, "missing_grounding")

        feature_payload = {
            "features": affordance.features,
            "feature_provenance": affordance.feature_provenance,
            "grounding_lineage": affordance.grounding_lineage,
        }
        feature_digest = content_digest(feature_payload)
        example_id = content_digest(
            {
                "kind": "grounded-feature-example",
                "seed": self.seed,
                "replay_budget": self.replay_budget,
                "evidence_id": source.evidence_id,
                "source_digest": source_digest,
                "affordance_id": affordance.affordance_id,
                "feature_payload_digest": feature_digest,
                "capability_snapshot_digest": source.capability_snapshot_digest,
                "parent_checkpoint_id": source.parent_checkpoint_id,
                "manifest_revision": source.manifest_revision,
            }
        )
        provenance = tuple(
            sorted(
                (
                    ("affordance", affordance.affordance_id),
                    ("capability_snapshot", source.capability_snapshot_digest),
                    ("evidence", source.evidence_id),
                    ("grounding", feature_digest),
                    ("outcome", source.outcome_id),
                    ("owner", source.owner_id),
                    ("parent_checkpoint", source.parent_checkpoint_id),
                )
            )
        )
        example = GroundedFeatureExample(
            example_id=example_id,
            evidence_id=source.evidence_id,
            outcome_id=source.outcome_id,
            affordance_id=affordance.affordance_id,
            action_kind=affordance.action_kind,
            grounding=affordance.features,
            capability_snapshot_digest=source.capability_snapshot_digest,
            parent_checkpoint_id=source.parent_checkpoint_id,
                feature_payload_digest=feature_digest,
                reward_terms=reward_terms,
                provenance=provenance,
                target_reward=source.outcome.reward,
                manifest_revision=source.manifest_revision,
        )
        lifecycle = InternalizationLifecycleRecord(
            example_id=example.example_id,
            evidence_id=source.evidence_id,
            status="external",
            events=("outcome_bound", "grounding_verified", "example_created"),
            source_digest=source_digest,
        )
        return InternalizationConversionResult(example, lifecycle, True)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_CHECKPOINT_FORMAT,
            "manifest_revision": self.manifest_revision,
            "seed": self.seed,
            "replay_budget": self.replay_budget,
            "reward_bounds": list(self.reward_bounds),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> InternalizationConverter:
        if payload.get("format") != INTERNALIZATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization converter format")
        bounds = tuple(float(item) for item in payload.get("reward_bounds", (-1.0, 1.0)))
        return cls(
            manifest_revision=str(payload["manifest_revision"]),
            seed=int(payload.get("seed", 0)),
            replay_budget=int(payload.get("replay_budget", 64)),
            reward_bounds=bounds,  # type: ignore[arg-type]
        )


class BoundedReplayBuffer:
    """A deterministic train-only replay partition with content deduplication."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay buffer capacity must be positive")
        self._examples: dict[str, GroundedFeatureExample] = {}
        self.deduplicated_count = 0

    def add(self, example: GroundedFeatureExample, *, partition: str = "train") -> bool:
        if partition != INTERNALIZATION_TRAIN_PARTITION:
            raise ValueError("holdout is read-only and cannot enter the training replay buffer")
        if not isinstance(example, GroundedFeatureExample):
            raise TypeError("replay buffer accepts GroundedFeatureExample values")
        existing = self._examples.get(example.example_id)
        if existing is not None:
            if existing.content_digest != example.content_digest:
                raise ValueError("example_id is already bound to different feature content")
            self.deduplicated_count += 1
            return False
        if len(self._examples) >= self.capacity:
            raise BufferError("internalization replay budget is exhausted")
        self._examples[example.example_id] = example
        return True

    def remove(self, example_id: str) -> GroundedFeatureExample | None:
        return self._examples.pop(str(example_id), None)

    @property
    def examples(self) -> tuple[GroundedFeatureExample, ...]:
        return tuple(self._examples[key] for key in sorted(self._examples))

    @property
    def replay_digest(self) -> str:
        return content_digest(
            {
                "partition": INTERNALIZATION_TRAIN_PARTITION,
                "examples": [item.to_payload() for item in self.examples],
            }
        )

    def replay(self, *, partition: str = "train") -> tuple[GroundedFeatureExample, ...]:
        if partition != INTERNALIZATION_TRAIN_PARTITION:
            raise ValueError("holdout replay is read-only and unavailable to the train partition")
        return self.examples

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "partition": INTERNALIZATION_TRAIN_PARTITION,
            "examples": [item.to_payload() for item in self.examples],
            "replay_digest": self.replay_digest,
            "deduplicated_count": self.deduplicated_count,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> BoundedReplayBuffer:
        if payload.get("format") != INTERNALIZATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization replay format")
        if payload.get("partition") != INTERNALIZATION_TRAIN_PARTITION:
            raise ValueError("internalization replay checkpoint has an invalid partition")
        buffer = cls(int(payload["capacity"]))
        for item in payload.get("examples", ()):
            example = GroundedFeatureExample.from_payload(item)
            if example.status in {"rejected", "rolled_back", "tombstoned"}:
                raise ValueError("terminal internalization content cannot enter replay")
            if not buffer.add(example):
                raise ValueError("duplicate example in internalization replay checkpoint")
        if buffer.replay_digest != str(payload.get("replay_digest", buffer.replay_digest)):
            raise ValueError("internalization replay digest mismatch")
        buffer.deduplicated_count = int(payload.get("deduplicated_count", 0))
        return buffer


class InternalizationLedger:
    """Pure S0 lifecycle ledger around conversion and bounded train replay."""

    def __init__(
        self,
        *,
        converter: InternalizationConverter | None = None,
        capacity: int | None = None,
    ) -> None:
        self.converter = converter or InternalizationConverter()
        self.replay_buffer = BoundedReplayBuffer(
            self.converter.replay_budget if capacity is None else int(capacity)
        )
        self._lifecycles: dict[str, InternalizationLifecycleRecord] = {}
        self._examples: dict[str, GroundedFeatureExample] = {}
        self._evidence_index: dict[str, str] = {}
        self.revision = 0

    def ingest(self, source: InternalizationInput) -> InternalizationConversionResult:
        result = self.converter.convert(source)
        prior_example_id = self._evidence_index.get(source.evidence_id)
        if prior_example_id is not None:
            if prior_example_id != result.lifecycle.example_id:
                return InternalizationConversionResult(
                    None,
                    InternalizationLifecycleRecord(
                        example_id=f"rejected:{content_digest(source.binding_payload())}",
                        evidence_id=source.evidence_id,
                        status="rejected",
                        events=("outcome_bound", "conversion_rejected"),
                        source_digest=content_digest(source.binding_payload()),
                        tombstone_reason="evidence_id_content_conflict",
                    ),
                    False,
                    "evidence_id_content_conflict",
                )
            existing_lifecycle = self._lifecycles[prior_example_id]
            if existing_lifecycle.status in {"rejected", "rolled_back", "tombstoned"}:
                return InternalizationConversionResult(
                    None,
                    existing_lifecycle,
                    False,
                    "terminal_content_cannot_be_resurrected",
                )
            if result.example is not None:
                existing_example = self._examples[result.example.example_id]
                if existing_example.content_digest != result.example.content_digest:
                    return InternalizationConversionResult(
                        None,
                        existing_lifecycle,
                        False,
                        "evidence_id_content_conflict",
                    )
                self.replay_buffer.add(result.example)
                lifecycle = existing_lifecycle
                deduped = InternalizationLifecycleRecord(
                    example_id=lifecycle.example_id,
                    evidence_id=lifecycle.evidence_id,
                    status=lifecycle.status,
                    events=(*lifecycle.events, "replay_deduplicated"),
                    source_digest=lifecycle.source_digest,
                    tombstone_reason=lifecycle.tombstone_reason,
                )
                self._lifecycles[result.example.example_id] = deduped
                return InternalizationConversionResult(
                    result.example, deduped, True, "replay_deduplicated"
                )
            return result
        self._lifecycles[result.lifecycle.example_id] = result.lifecycle
        self._evidence_index[source.evidence_id] = result.lifecycle.example_id
        if result.example is not None:
            self._examples[result.example.example_id] = result.example
            self.replay_buffer.add(result.example)
        self.revision += 1
        return result

    def example(self, example_id: str) -> GroundedFeatureExample:
        try:
            return self._examples[str(example_id)]
        except KeyError as exc:
            raise KeyError(f"unknown internalization example: {example_id}") from exc

    def lifecycle(self, example_id: str) -> InternalizationLifecycleRecord:
        try:
            return self._lifecycles[str(example_id)]
        except KeyError as exc:
            raise KeyError(f"unknown internalization lifecycle: {example_id}") from exc

    def advance_status(
        self,
        example_id: str,
        status: str,
        *,
        causal_gate: InternalizationCausalGate | None = None,
        reason: str = "",
    ) -> InternalizationLifecycleRecord:
        example_key = str(example_id)
        current = self.lifecycle(example_key)
        if status not in INTERNALIZATION_ALLOWED_STATUSES:
            raise ValueError(f"unsupported internalization status: {status}")
        if current.status in {"rejected", "rolled_back", "tombstoned"} and status != current.status:
            raise ValueError("terminal internalization content cannot be resurrected")
        if status == "internalized":
            if current.status not in {"external", "shadow"}:
                raise ValueError("only external or shadow content can be internalized")
            if causal_gate is None or not causal_gate.passed:
                raise ValueError("internalization requires all causal gate checks")
        if status == "tombstoned":
            if current.status != "internalized":
                raise ValueError(
                    "external description can be tombstoned only after internalization"
                )
            if causal_gate is None or not causal_gate.passed:
                raise ValueError("external tombstone requires all five causal gate checks")
            reason = _required_text(reason, "tombstone reason")
        if status == "rolled_back":
            reason = _required_text(reason, "rollback reason")
        if status == "rejected":
            reason = _required_text(reason, "rejection reason")
        event_by_status = {
            "shadow": "shadow_learned",
            "internalized": "internalization_admitted",
            "rejected": "conversion_rejected",
            "rolled_back": "rolled_back",
            "tombstoned": "external_description_tombstoned",
        }
        new_events = () if status == "external" else (event_by_status[status],)
        if status == "internalized":
            new_events = ("holdout_checked", "affordance_lesion_checked", *new_events)
        events = (*current.events, *new_events)
        updated = InternalizationLifecycleRecord(
            example_id=current.example_id,
            evidence_id=current.evidence_id,
            status=status,
            events=events,
            source_digest=current.source_digest,
            tombstone_reason=reason if status in {"rejected", "rolled_back", "tombstoned"} else "",
        )
        self._lifecycles[example_key] = updated
        if example_key in self._examples:
            old_example = self._examples[example_key]
            example_payload = old_example.to_payload()
            example_payload.pop("format", None)
            example_payload.update({"status": status, "tombstone_reason": updated.tombstone_reason})
            self._examples[example_key] = GroundedFeatureExample(**example_payload)
        if status in {"rejected", "rolled_back", "tombstoned"}:
            self.replay_buffer.remove(example_key)
        self.revision += 1
        return updated

    @property
    def replay_digest(self) -> str:
        return self.replay_buffer.replay_digest

    @property
    def replay_update_count(self) -> int:
        return len(self.replay_buffer.examples)

    @property
    def examples(self) -> tuple[GroundedFeatureExample, ...]:
        return tuple(self._examples[key] for key in sorted(self._examples))

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": INTERNALIZATION_CHECKPOINT_FORMAT,
            "converter": self.converter.checkpoint(),
            "replay_buffer": self.replay_buffer.checkpoint(),
            "examples": [self._examples[key].to_payload() for key in sorted(self._examples)],
            "lifecycles": [self._lifecycles[key].to_payload() for key in sorted(self._lifecycles)],
            "evidence_index": dict(sorted(self._evidence_index.items())),
            "revision": self.revision,
            "replay_digest": self.replay_digest,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> InternalizationLedger:
        if payload.get("format") != INTERNALIZATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported internalization ledger format")
        ledger = cls(converter=InternalizationConverter.from_checkpoint(payload["converter"]))
        ledger.replay_buffer = BoundedReplayBuffer.from_checkpoint(payload["replay_buffer"])
        ledger._examples = {
            example.example_id: example
            for example in (
                GroundedFeatureExample.from_payload(item) for item in payload.get("examples", ())
            )
        }
        ledger._lifecycles = {
            item.example_id: item
            for item in (
                InternalizationLifecycleRecord.from_payload(item)
                for item in payload.get("lifecycles", ())
            )
        }
        ledger._evidence_index = {
            str(key): str(value) for key, value in payload.get("evidence_index", {}).items()
        }
        ledger.revision = int(payload.get("revision", 0))
        if ledger.replay_digest != str(payload.get("replay_digest", ledger.replay_digest)):
            raise ValueError("internalization ledger replay digest mismatch")
        for example_id, lifecycle in ledger._lifecycles.items():
            if (
                lifecycle.status in {"rejected", "rolled_back", "tombstoned"}
                and example_id in ledger.replay_buffer._examples
            ):
                raise ValueError("terminal internalization lifecycle is present in replay")
        return ledger


# Short aliases keep the S0 surface readable for callers without creating a
# second implementation vocabulary.
GroundedOutcomeEvidence = InternalizationInput
InternalizationExample = GroundedFeatureExample
ReplayBuffer = BoundedReplayBuffer


__all__ = [
    "INTERNALIZATION_ALLOWED_STATUSES",
    "INTERNALIZATION_CHECKPOINT_FORMAT",
    "INTERNALIZATION_MANIFEST_REVISION",
    "INTERNALIZATION_TRAIN_PARTITION",
    "BoundedReplayBuffer",
    "GroundedFeatureExample",
    "GroundedOutcomeEvidence",
    "InternalizationCausalGate",
    "InternalizationConversionResult",
    "InternalizationConverter",
    "InternalizationExample",
    "InternalizationInput",
    "InternalizationLedger",
    "InternalizationLifecycleRecord",
    "ReplayBuffer",
    "content_digest",
]
