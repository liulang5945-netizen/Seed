"""Native content core v1: learnable content state, time-conditioned reading, step-wise generation.

This is the isolated R3 implementation of ``new_core_v1``. It is not the old
F/N177 graph: facts never enter the language observation ledger, content is
persisted as a versioned latent record, and every decoding step reads the
query plus the time intent plus the candidate content again from the current
decoder hidden. The byte decoder itself is the existing V3 backbone, so a
natural-language candidate can inherit the same content core later.

R3 does not train. The module only has to prove the interface and the credit
path: related swap changes the produced bytes, unrelated swap is measurable,
cut removes the candidate, rescue restores byte-identical output, teacher
forcing supervises the first target byte, and a zero-step bundle restores in a
fresh process with identical read/logits digests.
"""

from __future__ import annotations

import copy
import hashlib
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .conditional_language_v2 import _EvidenceCache
from .conditional_language_v3 import (
    ConditionalLanguageCandidateV3,
    TaskCoreConfigV3,
)
from .internalization import content_digest
from .persistence import atomic_save
from .utf8_state import utf8_allowed

CONTENT_RECORD_FORMAT = "taiji-native-content-record-v1"
CONTENT_STORE_FORMAT = "taiji-native-content-store-v1"
NATIVE_CORE_BUNDLE_FORMAT = "taiji-native-content-bundle-v1"

TIME_INTENTS = ("current", "previous", "unknown")
CONTENT_STATUSES = ("resolved", "ambiguous", "unresolved")
TIME_INTENT_INDEX = {name: index for index, name in enumerate(TIME_INTENTS)}
CONTENT_STATUS_INDEX = {name: index for index, name in enumerate(CONTENT_STATUSES)}

END_MARKER = b"<|end|>"

_HEX = frozenset(string.hexdigits.lower())


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value.lower())


def _require_bytes(value: bytes, *, label: str) -> None:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{label} must be nonempty bytes")


def _record_payload(
    *,
    content_id: str,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    raw_span: bytes,
    source_input_id: str,
    source_sha256: str,
    source_byte_span: tuple[int, int],
    owner: str,
    scope: str,
    session: str,
    turn_index: int,
    revision: int,
    valid_from: int,
    valid_to: int | None,
    superseded_by: str | None,
    encoder_revision: int,
    status: str,
    confidence: float,
    latent: Sequence[float],
) -> dict[str, Any]:
    return {
        "format": CONTENT_RECORD_FORMAT,
        "version": 1,
        "content_id": content_id,
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "raw_span_hex": raw_span.hex(),
        "source_input_id": source_input_id,
        "source_sha256": source_sha256,
        "source_byte_span": list(source_byte_span),
        "owner": owner,
        "scope": scope,
        "session": session,
        "turn_index": turn_index,
        "revision": revision,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "superseded_by": superseded_by,
        "encoder_revision": encoder_revision,
        "status": status,
        "confidence": float(confidence),
        "latent": list(latent),
    }


@dataclass(frozen=True)
class NativeContentRecordV1:
    """One sourced content record with a continuous latent and a version role."""

    content_id: str
    entity: str | None
    attribute: str | None
    value: str | None
    raw_span: bytes
    source_input_id: str
    source_sha256: str
    source_byte_span: tuple[int, int]
    owner: str
    scope: str
    session: str
    turn_index: int
    revision: int
    valid_from: int
    valid_to: int | None
    superseded_by: str | None
    encoder_revision: int
    status: str
    confidence: float
    latent: tuple[float, ...]
    record_sha256: str

    FORMAT = CONTENT_RECORD_FORMAT

    def __post_init__(self) -> None:
        if not self.content_id.startswith("nc1:"):
            raise ValueError("native content id is malformed")
        if self.status not in CONTENT_STATUSES:
            raise ValueError("native content status is outside the declared set")
        labels = (self.entity, self.attribute, self.value)
        if self.status == "resolved" and any(item is None for item in labels):
            raise ValueError("a resolved record requires entity/attribute/value labels")
        if any(item is not None and not str(item) for item in labels):
            raise ValueError("native content labels cannot be empty strings")
        _require_bytes(self.raw_span, label="native content raw span")
        if not self.source_input_id:
            raise ValueError("native content requires a source input id")
        if not _is_sha256(self.source_sha256):
            raise ValueError("native content requires a source sha256")
        if not all(
            isinstance(item, str) and item for item in (self.owner, self.scope, self.session)
        ):
            raise ValueError("native content requires explicit owner/scope/session")
        start, end = self.source_byte_span
        if start < 0 or end <= start:
            raise ValueError("native content source byte span is invalid")
        counters = (self.turn_index, self.revision, self.valid_from, self.encoder_revision)
        if any(int(item) < 0 for item in counters):
            raise ValueError("native content counters cannot be negative")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("native content valid_to precedes valid_from")
        if self.superseded_by is not None and not self.superseded_by.startswith("nc1:"):
            raise ValueError("native content supersession id is malformed")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("native content confidence must be in [0, 1]")
        if not self.latent or not all(math.isfinite(float(item)) for item in self.latent):
            raise ValueError("native content requires a finite latent vector")
        if self.record_sha256 != content_digest(self._digest_payload()):
            raise ValueError("native content record digest differs")

    def identity_payload(self) -> dict[str, Any]:
        """The part of the identity that a re-encoding must not change."""

        return {
            "source_input_id": self.source_input_id,
            "source_sha256": self.source_sha256,
            "source_byte_span": list(self.source_byte_span),
            "owner": self.owner,
            "scope": self.scope,
            "session": self.session,
            "turn_index": self.turn_index,
            "revision": self.revision,
        }

    def _digest_payload(self) -> dict[str, Any]:
        return _record_payload(
            content_id=self.content_id,
            entity=self.entity,
            attribute=self.attribute,
            value=self.value,
            raw_span=self.raw_span,
            source_input_id=self.source_input_id,
            source_sha256=self.source_sha256,
            source_byte_span=self.source_byte_span,
            owner=self.owner,
            scope=self.scope,
            session=self.session,
            turn_index=self.turn_index,
            revision=self.revision,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            superseded_by=self.superseded_by,
            encoder_revision=self.encoder_revision,
            status=self.status,
            confidence=self.confidence,
            latent=self.latent,
        )

    @staticmethod
    def make_content_id(
        *,
        source_input_id: str,
        source_sha256: str,
        source_byte_span: tuple[int, int],
        owner: str,
        scope: str,
        session: str,
        turn_index: int,
        revision: int,
    ) -> str:
        return (
            "nc1:"
            + content_digest(
                {
                    "source_input_id": source_input_id,
                    "source_sha256": source_sha256,
                    "source_byte_span": list(source_byte_span),
                    "owner": owner,
                    "scope": scope,
                    "session": session,
                    "turn_index": turn_index,
                    "revision": revision,
                }
            )[:24]
        )

    @classmethod
    def create(
        cls,
        *,
        entity: str | None,
        attribute: str | None,
        value: str | None,
        raw_span: bytes,
        source_input_id: str,
        source_sha256: str,
        source_byte_span: tuple[int, int],
        owner: str,
        scope: str,
        session: str,
        turn_index: int,
        revision: int,
        valid_from: int,
        encoder_revision: int,
        status: str,
        confidence: float,
        latent: Sequence[float],
    ) -> NativeContentRecordV1:
        content_id = cls.make_content_id(
            source_input_id=source_input_id,
            source_sha256=source_sha256,
            source_byte_span=source_byte_span,
            owner=owner,
            scope=scope,
            session=session,
            turn_index=turn_index,
            revision=revision,
        )
        latent_values = tuple(float(item) for item in latent)
        digest = content_digest(
            _record_payload(
                content_id=content_id,
                entity=entity,
                attribute=attribute,
                value=value,
                raw_span=raw_span,
                source_input_id=source_input_id,
                source_sha256=source_sha256,
                source_byte_span=source_byte_span,
                owner=owner,
                scope=scope,
                session=session,
                turn_index=turn_index,
                revision=revision,
                valid_from=valid_from,
                valid_to=None,
                superseded_by=None,
                encoder_revision=encoder_revision,
                status=status,
                confidence=float(confidence),
                latent=latent_values,
            )
        )
        return cls(
            content_id=content_id,
            entity=entity,
            attribute=attribute,
            value=value,
            raw_span=raw_span,
            source_input_id=source_input_id,
            source_sha256=source_sha256,
            source_byte_span=source_byte_span,
            owner=owner,
            scope=scope,
            session=session,
            turn_index=turn_index,
            revision=revision,
            valid_from=valid_from,
            valid_to=None,
            superseded_by=None,
            encoder_revision=encoder_revision,
            status=status,
            confidence=float(confidence),
            latent=latent_values,
            record_sha256=digest,
        )

    def supersede(self, *, valid_to: int, superseded_by: str) -> NativeContentRecordV1:
        if self.superseded_by is not None:
            raise ValueError("native content record is already superseded")
        if not superseded_by.startswith("nc1:"):
            raise ValueError("native content supersession id is malformed")
        digest = content_digest(
            _record_payload(
                content_id=self.content_id,
                entity=self.entity,
                attribute=self.attribute,
                value=self.value,
                raw_span=self.raw_span,
                source_input_id=self.source_input_id,
                source_sha256=self.source_sha256,
                source_byte_span=self.source_byte_span,
                owner=self.owner,
                scope=self.scope,
                session=self.session,
                turn_index=self.turn_index,
                revision=self.revision,
                valid_from=self.valid_from,
                valid_to=int(valid_to),
                superseded_by=superseded_by,
                encoder_revision=self.encoder_revision,
                status=self.status,
                confidence=self.confidence,
                latent=self.latent,
            )
        )
        return replace(
            self,
            valid_to=int(valid_to),
            superseded_by=superseded_by,
            record_sha256=digest,
        )

    def reencode(self, *, latent: Sequence[float], encoder_revision: int) -> NativeContentRecordV1:
        values = tuple(float(item) for item in latent)
        if not values or not all(math.isfinite(item) for item in values):
            raise ValueError("native content re-encoding needs a finite latent vector")
        digest = content_digest(
            _record_payload(
                content_id=self.content_id,
                entity=self.entity,
                attribute=self.attribute,
                value=self.value,
                raw_span=self.raw_span,
                source_input_id=self.source_input_id,
                source_sha256=self.source_sha256,
                source_byte_span=self.source_byte_span,
                owner=self.owner,
                scope=self.scope,
                session=self.session,
                turn_index=self.turn_index,
                revision=self.revision,
                valid_from=self.valid_from,
                valid_to=self.valid_to,
                superseded_by=self.superseded_by,
                encoder_revision=int(encoder_revision),
                status=self.status,
                confidence=self.confidence,
                latent=values,
            )
        )
        return replace(
            self,
            latent=values,
            encoder_revision=int(encoder_revision),
            record_sha256=digest,
        )

    def reopen(self) -> NativeContentRecordV1:
        """Return this record to the active set after its successor was removed."""

        if self.superseded_by is None and self.valid_to is None:
            return self
        digest = content_digest(
            _record_payload(
                content_id=self.content_id,
                entity=self.entity,
                attribute=self.attribute,
                value=self.value,
                raw_span=self.raw_span,
                source_input_id=self.source_input_id,
                source_sha256=self.source_sha256,
                source_byte_span=self.source_byte_span,
                owner=self.owner,
                scope=self.scope,
                session=self.session,
                turn_index=self.turn_index,
                revision=self.revision,
                valid_from=self.valid_from,
                valid_to=None,
                superseded_by=None,
                encoder_revision=self.encoder_revision,
                status=self.status,
                confidence=self.confidence,
                latent=self.latent,
            )
        )
        return replace(self, valid_to=None, superseded_by=None, record_sha256=digest)

    def to_payload(self) -> dict[str, Any]:
        return {**self._digest_payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NativeContentRecordV1:
        expected = {
            "format",
            "version",
            "content_id",
            "entity",
            "attribute",
            "value",
            "raw_span_hex",
            "source_input_id",
            "source_sha256",
            "source_byte_span",
            "owner",
            "scope",
            "session",
            "turn_index",
            "revision",
            "valid_from",
            "valid_to",
            "superseded_by",
            "encoder_revision",
            "status",
            "confidence",
            "latent",
            "record_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native content record payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native content record version")
        span = payload["source_byte_span"]
        latent = payload["latent"]
        if not isinstance(span, list) or len(span) != 2 or not isinstance(latent, list):
            raise ValueError("native content record spans must be a pair and a list")
        return cls(
            content_id=str(payload["content_id"]),
            entity=None if payload["entity"] is None else str(payload["entity"]),
            attribute=None if payload["attribute"] is None else str(payload["attribute"]),
            value=None if payload["value"] is None else str(payload["value"]),
            raw_span=bytes.fromhex(str(payload["raw_span_hex"])),
            source_input_id=str(payload["source_input_id"]),
            source_sha256=str(payload["source_sha256"]),
            source_byte_span=(int(span[0]), int(span[1])),
            owner=str(payload["owner"]),
            scope=str(payload["scope"]),
            session=str(payload["session"]),
            turn_index=int(payload["turn_index"]),
            revision=int(payload["revision"]),
            valid_from=int(payload["valid_from"]),
            valid_to=None if payload["valid_to"] is None else int(payload["valid_to"]),
            superseded_by=(
                None if payload["superseded_by"] is None else str(payload["superseded_by"])
            ),
            encoder_revision=int(payload["encoder_revision"]),
            status=str(payload["status"]),
            confidence=float(payload["confidence"]),
            latent=tuple(float(item) for item in latent),
            record_sha256=str(payload["record_sha256"]),
        )


@dataclass(frozen=True)
class NativeContentStoreV1:
    """Versioned content store; corrections supersede but never erase history."""

    records: tuple[NativeContentRecordV1, ...]

    FORMAT = CONTENT_STORE_FORMAT

    def __post_init__(self) -> None:
        if len({item.content_id for item in self.records}) != len(self.records):
            raise ValueError("native content store repeats a content id")
        active_by_key: dict[tuple[str, str], str] = {}
        for item in self.records:
            if item.status != "resolved" or item.valid_to is not None:
                continue
            key = (str(item.entity), str(item.attribute))
            if key in active_by_key:
                raise ValueError("native content store has two active values for one key")
            active_by_key[key] = item.content_id
        ids = {item.content_id for item in self.records}
        for item in self.records:
            if item.superseded_by is not None and item.superseded_by not in ids:
                raise ValueError("native content store supersession target is absent")

    def _keyed(self) -> dict[tuple[str, str], tuple[NativeContentRecordV1, ...]]:
        rows: dict[tuple[str, str], list[NativeContentRecordV1]] = {}
        for item in self.records:
            if item.entity is None or item.attribute is None:
                continue
            if item.status != "resolved":
                continue
            rows.setdefault((item.entity, item.attribute), []).append(item)
        return {
            key: tuple(sorted(values, key=lambda item: (item.revision, item.turn_index)))
            for key, values in rows.items()
        }

    def add(self, record: NativeContentRecordV1) -> NativeContentStoreV1:
        if any(item.content_id == record.content_id for item in self.records):
            raise ValueError("native content store already contains this record")
        if record.superseded_by is not None or record.valid_to is not None:
            raise ValueError("only a current record can be added")
        rows = list(self.records)
        if record.status == "resolved":
            previous = self.current(str(record.entity), str(record.attribute))
            if previous is not None:
                if record.revision <= previous.revision:
                    raise ValueError("native content correction revision must advance")
                rows = [
                    (
                        item.supersede(valid_to=record.valid_from, superseded_by=record.content_id)
                        if item.content_id == previous.content_id
                        else item
                    )
                    for item in rows
                ]
        rows.append(record)
        return NativeContentStoreV1(tuple(rows))

    def current(self, entity: str, attribute: str) -> NativeContentRecordV1 | None:
        for item in self.records:
            if (
                item.entity == entity
                and item.attribute == attribute
                and item.status == "resolved"
                and item.valid_to is None
            ):
                return item
        return None

    def history(self, entity: str, attribute: str) -> tuple[NativeContentRecordV1, ...]:
        return self._keyed().get((entity, attribute), ())

    def previous(self, entity: str, attribute: str) -> NativeContentRecordV1 | None:
        chain = self.history(entity, attribute)
        current = self.current(entity, attribute)
        if current is None:
            return None
        index = next(
            (
                position
                for position, item in enumerate(chain)
                if item.content_id == current.content_id
            ),
            None,
        )
        if index is None or index == 0:
            return None
        return chain[index - 1]

    def active_records(self) -> tuple[NativeContentRecordV1, ...]:
        return tuple(
            sorted(
                (item for item in self.records if item.valid_to is None),
                key=lambda item: item.content_id,
            )
        )

    def readable(
        self,
        *,
        owner: str,
        scope: str,
        session: str,
        time_intent: str,
        cross_session: bool = True,
    ) -> tuple[NativeContentRecordV1, ...]:
        if time_intent not in TIME_INTENT_INDEX:
            raise ValueError("native content readable set needs a declared time intent")
        seen: dict[str, NativeContentRecordV1] = {}

        def visible(candidate: NativeContentRecordV1) -> bool:
            return (
                candidate.owner == owner
                and candidate.scope == scope
                and (cross_session or candidate.session == session)
            )

        if time_intent == "previous":
            for key, chain in self._keyed().items():
                current = self.current(key[0], key[1])
                if current is None:
                    continue
                index = next(
                    (
                        position
                        for position, item in enumerate(chain)
                        if item.content_id == current.content_id
                    ),
                    None,
                )
                if index is None or index == 0:
                    continue
                candidate = chain[index - 1]
                if visible(candidate):
                    seen[candidate.content_id] = candidate
        else:
            for candidate in self.active_records():
                if not visible(candidate):
                    continue
                if candidate.status == "resolved" and candidate.valid_to is not None:
                    continue
                seen[candidate.content_id] = candidate
        return tuple(seen[key] for key in sorted(seen))

    def cut(self, content_id: str) -> NativeContentStoreV1:
        if not any(item.content_id == content_id for item in self.records):
            raise ValueError("native content cut target is absent")
        survivors = tuple(item for item in self.records if item.content_id != content_id)
        return NativeContentStoreV1(
            tuple(item.reopen() if item.superseded_by == content_id else item for item in survivors)
        )

    def swap(self, replacement: NativeContentRecordV1) -> NativeContentStoreV1:
        if not any(item.content_id == replacement.content_id for item in self.records):
            raise ValueError("native content swap target is absent")
        return NativeContentStoreV1(
            tuple(
                replacement if item.content_id == replacement.content_id else item
                for item in self.records
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "records": [item.to_payload() for item in self.records],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NativeContentStoreV1:
        required = {"format", "version", "records"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("native content store payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native content store version")
        rows = payload["records"]
        if not isinstance(rows, list):
            raise ValueError("native content store records must be a list")
        return cls(tuple(NativeContentRecordV1.from_payload(item) for item in rows))


@dataclass(frozen=True)
class NativeContentCoreConfigV1:
    width: int = 128
    embedding_width: int = 64
    key_width: int = 64
    slots: int = 16
    chunk_bytes: int = 32
    max_observation_bytes: int = 1024
    max_input_evidence_bytes: int = 1024
    max_output_position: int = 512
    max_generation_bytes: int = 256
    content_width: int = 64
    seed: int = 1770301
    position_extension_seed: int = 1770302
    read_seed: int = 1770303

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.width,
                self.embedding_width,
                self.key_width,
                self.slots,
                self.chunk_bytes,
                self.max_observation_bytes,
                self.max_input_evidence_bytes,
                self.max_output_position,
                self.max_generation_bytes,
                self.content_width,
                self.position_extension_seed,
                self.read_seed,
            )
        ):
            raise ValueError("native content core dimensions must be positive")
        if self.max_generation_bytes > self.max_output_position:
            raise ValueError("native content generation exceeds the output positions")
        if self.max_input_evidence_bytes > self.max_observation_bytes:
            raise ValueError("native content evidence exceeds the observation bound")

    def task_config(self) -> TaskCoreConfigV3:
        return TaskCoreConfigV3(
            width=self.width,
            embedding_width=self.embedding_width,
            key_width=self.key_width,
            slots=self.slots,
            chunk_bytes=self.chunk_bytes,
            max_observation_bytes=self.max_observation_bytes,
            max_input_evidence_bytes=self.max_input_evidence_bytes,
            max_output_position=self.max_output_position,
            max_generation_bytes=self.max_generation_bytes,
            seed=self.seed,
            position_extension_seed=self.position_extension_seed,
        )


@dataclass(frozen=True)
class NativeContentConditionV1:
    """One query plus the legal candidate set for a declared time intent."""

    query: bytes
    records: tuple[NativeContentRecordV1, ...]
    time_intent: str
    reencode: bool = False

    def __post_init__(self) -> None:
        _require_bytes(self.query, label="native content query")
        if self.time_intent not in TIME_INTENT_INDEX:
            raise ValueError("native content condition time intent is outside the declared set")
        ids = tuple(item.content_id for item in self.records)
        if len(set(ids)) != len(ids):
            raise ValueError("native content condition repeats a candidate")
        if ids != tuple(sorted(ids)):
            raise ValueError("native content condition candidates must be sorted by content id")


@dataclass(frozen=True)
class _NativeContentReadCache(_EvidenceCache):
    query_vector: torch.Tensor
    time_vector: torch.Tensor
    content_latents: torch.Tensor
    content_status: torch.Tensor
    content_mask: torch.Tensor
    content_ids: tuple[str, ...]


@dataclass(frozen=True)
class NativeContentAnswerV1:
    generated: bytes
    selected_ids: tuple[str, ...]
    step_scores: tuple[tuple[float, ...], ...]

    FORMAT = "taiji-native-content-answer-v1"

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "generated_hex": self.generated.hex(),
            "selected_ids": list(self.selected_ids),
            "step_scores": [list(row) for row in self.step_scores],
        }

    def answer_digest(self) -> str:
        return content_digest(self.to_payload())


class NativeContentCoreV1(ConditionalLanguageCandidateV3):
    """V3 byte decoder plus a per-step, time-conditioned content reader."""

    config: NativeContentCoreConfigV1  # type: ignore[assignment]

    def __init__(
        self,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v1",
    ):
        config = NativeContentCoreConfigV1() if config is None else config
        super().__init__(config.task_config(), bundle_id=bundle_id)
        self.config = config
        content = config.content_width
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.read_seed)
            self.content_encoder = nn.GRU(
                config.embedding_width, content, batch_first=True, dtype=torch.float32
            )
            self.query_encoder = nn.GRU(
                config.embedding_width, content, batch_first=True, dtype=torch.float32
            )
            self.time_embedding = nn.Embedding(len(TIME_INTENTS), content, dtype=torch.float32)
            self.status_embedding = nn.Embedding(
                len(CONTENT_STATUSES), content, dtype=torch.float32
            )
            self.content_score = nn.Sequential(
                nn.Linear(3 * content, content, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(content, 1, dtype=torch.float32),
            )
            self.hidden_projection = nn.Linear(
                config.width, content, bias=False, dtype=torch.float32
            )
            self.read_projection = nn.Linear(
                config.width + content, config.width, dtype=torch.float32
            )
            self.read_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    # --- content encoding -------------------------------------------------

    def encode_content(self, raw_span: bytes) -> torch.Tensor:
        _require_bytes(raw_span, label="native content span")
        tokens = torch.tensor(list(raw_span), dtype=torch.long)
        encoded, hidden = self.content_encoder(self.core.embedding(tokens.unsqueeze(0)))
        del encoded
        return hidden[0, 0]

    def encode_query(self, query: bytes) -> torch.Tensor:
        _require_bytes(query, label="native content query")
        tokens = torch.tensor(list(query), dtype=torch.long)
        encoded, hidden = self.query_encoder(self.core.embedding(tokens.unsqueeze(0)))
        del encoded
        return hidden[0, 0]

    # --- condition and read cache ----------------------------------------

    def new_condition(
        self,
        *,
        query: bytes,
        store: NativeContentStoreV1,
        owner: str,
        scope: str,
        session: str,
        time_intent: str,
        cross_session: bool = True,
        reencode: bool = False,
    ) -> NativeContentConditionV1:
        if len(query) > self.config.max_observation_bytes:
            raise ValueError("native content query exceeds the declared observation bound")
        records = store.readable(
            owner=owner,
            scope=scope,
            session=session,
            time_intent=time_intent,
            cross_session=cross_session,
        )
        return NativeContentConditionV1(
            query=query,
            records=records,
            time_intent=time_intent,
            reencode=reencode,
        )

    def content_read_cache(
        self, state: Any, condition: NativeContentConditionV1
    ) -> _NativeContentReadCache:
        self.validate(state)
        base = super().evidence_cache(state)
        device = base.byte_memory.device
        query_vector = self.encode_query(condition.query)
        time_vector = self.time_embedding(
            torch.tensor(TIME_INTENT_INDEX[condition.time_intent], dtype=torch.long, device=device)
        )
        latents: list[torch.Tensor] = []
        statuses: list[int] = []
        for record in condition.records:
            if condition.reencode:
                latent = self.encode_content(record.raw_span).to(device)
            else:
                if record.encoder_revision != self.parameter_revision:
                    raise ValueError("native content latent belongs to another encoder revision")
                if len(record.latent) != self.config.content_width:
                    raise ValueError("native content latent width differs from the candidate")
                latent = torch.tensor(record.latent, dtype=torch.float32, device=device)
            latents.append(latent)
            statuses.append(CONTENT_STATUS_INDEX[record.status])
        if latents:
            content_latents = torch.stack(latents)
            content_status = self.status_embedding(
                torch.tensor(statuses, dtype=torch.long, device=device)
            )
            content_mask = torch.ones(len(latents), dtype=torch.bool, device=device)
        else:
            content_latents = torch.zeros((0, self.config.content_width), device=device)
            content_status = torch.zeros((0, self.config.content_width), device=device)
            content_mask = torch.zeros(0, dtype=torch.bool, device=device)
        return _NativeContentReadCache(
            base.bundle_id,
            base.scope_id,
            base.session_id,
            base.parameter_revision,
            base.byte_memory,
            base.hidden,
            base.keys,
            base.values,
            query_vector,
            time_vector,
            content_latents,
            content_status,
            content_mask,
            tuple(record.content_id for record in condition.records),
        )

    def _read(
        self, cache: _NativeContentReadCache, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.config.content_width
        if len(cache.content_ids) == 0:
            scores = torch.zeros(0, device=hidden.device)
            return torch.zeros(width, device=hidden.device), scores
        context = cache.query_vector + cache.time_vector + self.hidden_projection(hidden)
        scoring = cache.content_latents + cache.content_status
        rows = torch.cat(
            (
                context.unsqueeze(0).expand(len(cache.content_ids), -1),
                scoring,
                context.unsqueeze(0) * scoring,
            ),
            dim=1,
        )
        scores = self.content_score(rows).squeeze(-1)
        scores = scores.masked_fill(~cache.content_mask, float("-inf"))
        weights = torch.softmax(scores, dim=0)
        read = weights @ cache.content_latents
        return read, scores

    # --- per-step decoding ------------------------------------------------

    def next_logits(  # type: ignore[override]
        self, state: Any, *, cache: _EvidenceCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate(state)
        if not isinstance(cache, _NativeContentReadCache):
            raise ValueError("native content decoding requires a content read cache")
        if (
            cache.bundle_id,
            cache.scope_id,
            cache.session_id,
            cache.parameter_revision,
        ) != (
            state.bundle_id,
            state.scope_id,
            state.session_id,
            state.parameter_revision,
        ):
            raise ValueError("native content read cache belongs to another state")
        if cache.byte_memory is not state.byte_memory or cache.hidden is not state.hidden:
            raise ValueError("native content read cache belongs to another task state")
        attention = self.attention_weights(state, cache=cache)
        evidence = attention @ cache.values
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        read, _ = self._read(cache, hidden)
        evidence = evidence + self.read_gate.sigmoid() * self.read_projection(
            torch.cat((hidden, read))
        )
        previous = self.core.embedding(
            torch.tensor(state.previous_byte, dtype=torch.long, device=hidden.device)
        )
        updated = self.decoder(torch.cat((previous, evidence)), hidden)
        return self.output(torch.cat((updated, evidence))), updated

    def conditioned_state(
        self,
        condition: NativeContentConditionV1,
        *,
        scope_id: str,
        session_id: str,
        native_state_ref: str,
        source_id: str = "native-content:query",
    ) -> Any:
        state = self.new_state(
            scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        if len(condition.query) > self.config.max_observation_bytes:
            raise ValueError("native content query exceeds the declared observation bound")
        return self.step_task(
            condition.query,
            state,
            scope_id=scope_id,
            session_id=session_id,
            source_id=source_id,
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": self.config.max_observation_bytes,
            },
        )

    def teacher_forced_conditioned(
        self,
        condition: NativeContentConditionV1,
        target: bytes,
        *,
        scope_id: str,
        session_id: str,
        native_state_ref: str = "native-content:probe",
    ) -> torch.Tensor:
        """One logit row per target byte; the first target byte is supervised."""

        if not target or len(target) > self.config.max_generation_bytes:
            raise ValueError("native content target is empty or exceeds the generation bound")
        state = self.conditioned_state(
            condition, scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        cache = self.content_read_cache(state, condition)
        logits: list[torch.Tensor] = []
        for symbol in target:
            current, hidden = self.next_logits(state, cache=cache)
            logits.append(current)
            state = self.observe_response(state, symbol, hidden)
        return torch.stack(logits)

    @torch.no_grad()
    def answer_conditioned(
        self,
        condition: NativeContentConditionV1,
        *,
        scope_id: str,
        session_id: str,
        native_state_ref: str = "native-content:probe",
        max_new_bytes: int | None = None,
        stop_on_marker: bool = True,
    ) -> NativeContentAnswerV1:
        allowance = self.config.max_generation_bytes if max_new_bytes is None else max_new_bytes
        if allowance <= 0 or allowance > self.config.max_generation_bytes:
            raise ValueError("native content generation allowance is outside the declared bound")
        state = self.conditioned_state(
            condition, scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        cache = self.content_read_cache(state, condition)
        generated: list[int] = []
        selected: list[str] = []
        rows: list[tuple[float, ...]] = []
        for _ in range(allowance):
            logits, hidden = self.next_logits(state, cache=cache)
            read, scores = self._read(cache, hidden)
            del read
            rows.append(tuple(float(item) for item in scores.tolist()))
            if cache.content_ids:
                selected.append(cache.content_ids[int(torch.argmax(scores))])
            else:
                selected.append("")
            allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
            choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            symbol = int(choices[logits[choices].argmax()])
            state = self.observe_response(state, symbol, hidden)
            generated.append(symbol)
            if stop_on_marker and bytes(generated).endswith(END_MARKER):
                break
        return NativeContentAnswerV1(
            generated=bytes(generated),
            selected_ids=tuple(selected),
            step_scores=tuple(rows),
        )

    # --- credit groups ----------------------------------------------------

    CreditGroups = Mapping[str, tuple[str, ...]]

    def credit_group_of(self, name: str) -> str:
        if name.startswith("core."):
            return "core"
        if name.startswith("content_encoder."):
            return "content_encoder"
        if name.startswith("query_encoder."):
            return "query_encoder"
        if name.startswith("decoder."):
            return "decoder"
        if name.startswith("output."):
            return "output"
        if name.startswith(("query.", "input_position.", "output_position.")):
            return "attention"
        return "reader"

    def credit_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for name, _ in self.named_parameters():
            groups.setdefault(self.credit_group_of(name), []).append(name)
        return {name: tuple(sorted(values)) for name, values in sorted(groups.items())}

    def set_credit_group(self, group: str, enabled: bool) -> None:
        declared = self.credit_groups()
        if group not in declared:
            raise ValueError("native content credit group is not declared")
        for name, parameter in self.named_parameters():
            if name in declared[group]:
                parameter.requires_grad_(enabled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class NativeContentBundleV1:
    """A complete recoverable native-content entry: model plus versioned store."""

    model: NativeContentCoreV1
    store: NativeContentStoreV1
    cursor: dict[str, Any]

    FORMAT = NATIVE_CORE_BUNDLE_FORMAT

    def __post_init__(self) -> None:
        stale = tuple(
            item.content_id
            for item in self.store.records
            if item.encoder_revision != self.model.parameter_revision
        )
        if stale:
            raise ValueError("native content store latents belong to another encoder revision")
        for item in self.store.records:
            if len(item.latent) != self.model.config.content_width:
                raise ValueError("native content store latent width differs from the candidate")

    @classmethod
    def create(
        cls,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v1",
        cursor: Mapping[str, Any] | None = None,
    ) -> NativeContentBundleV1:
        model = NativeContentCoreV1(config, bundle_id=bundle_id)
        return cls(
            model=model,
            store=NativeContentStoreV1(()),
            cursor=copy.deepcopy(dict(cursor or {})),
        )

    # --- ingestion --------------------------------------------------------

    def ingest(
        self,
        *,
        raw_span: bytes,
        source_input_id: str,
        source_sha256: str,
        owner: str,
        scope: str,
        session: str,
        turn_index: int,
        revision: int,
        valid_from: int | None = None,
        entity: str | None = None,
        attribute: str | None = None,
        value: str | None = None,
        status: str = "resolved",
        confidence: float = 1.0,
        source_byte_span: tuple[int, int] | None = None,
    ) -> NativeContentRecordV1:
        _require_bytes(raw_span, label="native content ingest span")
        if not _is_sha256(source_sha256):
            raise ValueError("native content ingest requires a source sha256")
        span = (0, len(raw_span)) if source_byte_span is None else source_byte_span
        with torch.no_grad():
            latent = self.model.encode_content(raw_span).tolist()
        record = NativeContentRecordV1.create(
            entity=entity,
            attribute=attribute,
            value=value,
            raw_span=raw_span,
            source_input_id=source_input_id,
            source_sha256=source_sha256,
            source_byte_span=span,
            owner=owner,
            scope=scope,
            session=session,
            turn_index=turn_index,
            revision=revision,
            valid_from=turn_index if valid_from is None else valid_from,
            encoder_revision=self.model.parameter_revision,
            status=status,
            confidence=confidence,
            latent=latent,
        )
        self.store = self.store.add(record)
        return record

    # --- answering --------------------------------------------------------

    def condition(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        cross_session: bool = True,
        reencode: bool = False,
    ) -> NativeContentConditionV1:
        return self.model.new_condition(
            query=query,
            store=self.store,
            owner=owner,
            scope=scope,
            session=session,
            time_intent=time_intent,
            cross_session=cross_session,
            reencode=reencode,
        )

    def answer(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        max_new_bytes: int | None = None,
    ) -> NativeContentAnswerV1:
        condition = self.condition(
            query=query, time_intent=time_intent, owner=owner, scope=scope, session=session
        )
        return self.model.answer_conditioned(
            condition, scope_id=scope, session_id=session, max_new_bytes=max_new_bytes
        )

    def score_target(
        self,
        *,
        query: bytes,
        target: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        reencode: bool = False,
    ) -> torch.Tensor:
        condition = self.condition(
            query=query,
            time_intent=time_intent,
            owner=owner,
            scope=scope,
            session=session,
            reencode=reencode,
        )
        return self.model.teacher_forced_conditioned(
            condition, target, scope_id=scope, session_id=session
        )

    # --- re-encoding and checkpoint --------------------------------------

    def stale_records(self) -> tuple[NativeContentRecordV1, ...]:
        return tuple(
            item
            for item in self.store.records
            if item.encoder_revision != self.model.parameter_revision
        )

    def reencode_store(self) -> None:
        with torch.no_grad():
            rows: list[NativeContentRecordV1] = []
            for item in self.store.records:
                latent = self.model.encode_content(item.raw_span).tolist()
                rows.append(
                    item.reencode(latent=latent, encoder_revision=self.model.parameter_revision)
                )
        self.store = NativeContentStoreV1(tuple(rows))

    def checkpoint(self, *, state: Any | None = None) -> dict[str, Any]:
        if state is not None:
            self.model.validate(state)
        payload: dict[str, Any] = {
            "format": self.FORMAT,
            "version": 1,
            "bundle_id": self.model.bundle_id,
            "config": asdict(self.model.config),
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "store": self.store.to_payload(),
            "cursor": copy.deepcopy(self.cursor),
            "task_state": None if state is None else state.to_payload(),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path, *, state: Any | None = None) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("native content bundle checkpoint is write-once")
        payload = self.checkpoint(state=state)
        atomic_save(payload, target)
        return target

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeContentBundleV1:
        expected = {
            "format",
            "version",
            "bundle_id",
            "config",
            "parameter_revision",
            "parameters",
            "store",
            "cursor",
            "task_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native content bundle payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native content bundle version")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("native content bundle checkpoint digest differs")
        model = NativeContentCoreV1(
            NativeContentCoreConfigV1(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = int(payload["parameter_revision"])
        store = NativeContentStoreV1.from_payload(payload["store"])
        stamp = content_digest(
            {
                "bundle_id": model.bundle_id,
                "parameter_revision": model.parameter_revision,
                "records": [item.record_sha256 for item in store.records],
            }
        )
        cursor = copy.deepcopy(dict(payload["cursor"]))
        cursor.setdefault("bundle_stamp", stamp)
        return cls(model=model, store=store, cursor=cursor)

    @classmethod
    def load(cls, path: Path) -> NativeContentBundleV1:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls.from_checkpoint(payload)

    def probe_digest(self, *, query: bytes, time_intent: str = "current") -> str:
        condition = self.condition(
            query=query,
            time_intent=time_intent,
            owner="probe",
            scope="probe",
            session="probe",
        )
        with torch.no_grad():
            logits = self.model.teacher_forced_conditioned(
                condition, END_MARKER, scope_id="probe", session_id="probe"
            )
        return content_digest(
            {
                "selected": [item.content_id for item in condition.records],
                "logits": [[round(float(value), 7) for value in row] for row in logits.tolist()],
                "answer": self.answer(
                    query=query,
                    time_intent=time_intent,
                    owner="probe",
                    scope="probe",
                    session="probe",
                ).answer_digest(),
            }
        )


def span_sha256(raw_span: bytes) -> str:
    return hashlib.sha256(raw_span).hexdigest()
