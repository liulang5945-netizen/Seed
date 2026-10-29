"""Opt-in native byte/percept-to-typed-state producer for the N177 slice.

The module owns a trainable span/tagging head over raw UTF-8 bytes and Taiji
percepts. Gold labels are accepted only by ``supervised_loss`` for the train
partition; inference accepts an InputTrace and the authorized JointTaskState.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (
    NativeCheckpoint,
    WorkspaceCandidate,
    WorkspaceSelection,
    WorkspaceState,
    WorldEvent,
    WorldObject,
)
from .input_boundary import InputTrace
from .internalization import content_digest
from .joint_task_state import (
    JOINT_NATIVE_COMPONENT,
    JOINT_NATIVE_COMPONENT_V1,
    JOINT_NATIVE_COMPONENT_V2,
    JointTaskStateV1,
    cognitive_state_digest,
)

JOINT_INPUT_PRODUCER_CHECKPOINT_FORMAT = "taiji-joint-input-producer-v1"
JOINT_INPUT_PRODUCER_COMPONENT_KEY = "joint_input_producer_v1"
JOINT_INPUT_PRODUCER_COMPONENT_FORMAT = "taiji-joint-input-producer-component-v1"

TAG_NAMES = (
    "O",
    "B_ENTITY",
    "I_ENTITY",
    "B_ATTRIBUTE",
    "I_ATTRIBUTE",
    "B_VALUE",
    "I_VALUE",
    "B_NEGATION",
    "I_NEGATION",
)
ACT_NAMES = ("assertion", "correction", "query", "goal", "ambiguous")
_TAG_TO_ROLE = {
    1: ("entity", True),
    2: ("entity", False),
    3: ("attribute", True),
    4: ("attribute", False),
    5: ("value", True),
    6: ("value", False),
    7: ("negation", True),
    8: ("negation", False),
}


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _input_bytes(trace: InputTrace) -> bytes:
    if not isinstance(trace, InputTrace) or not trace.observations:
        raise TypeError("native input producer requires a nonempty InputTrace")
    if len(trace.observations) != len(trace.percepts):
        raise ValueError("native input producer trace lost byte-to-percept alignment")
    raw = bytearray()
    source = trace.observations[0].source
    provenance = trace.observations[0].provenance
    for observation, percept in zip(trace.observations, trace.percepts, strict=True):
        if observation.modality != "text-byte" or type(observation.value) is not int:
            raise ValueError("native input producer accepts only Taiji text-byte observations")
        if not 0 <= observation.value <= 255:
            raise ValueError("native input producer received a byte outside UTF-8 range")
        if (observation.source, observation.provenance) != (source, provenance):
            raise ValueError("one native parse cannot mix input source or provenance")
        if percept.modality != "text-byte":
            raise ValueError("native input producer percept modality differs from its observation")
        raw.append(observation.value)
    bytes(raw).decode("utf-8")
    return bytes(raw)


@dataclass(frozen=True)
class JointSemanticSpanV1:
    """A UTF-8 byte span tied to the exact input and per-byte percept events."""

    role: str
    input_id: str
    source_sha256: str
    start_byte: int
    end_byte: int
    span_bytes: bytes
    span_sha256: str
    percept_event_ids: tuple[str, ...]
    assembly_ids: tuple[str, ...]
    confidence: float

    FORMAT = "taiji-joint-semantic-span-v1"
    ROLES = frozenset({"entity", "attribute", "value", "negation"})

    def __post_init__(self) -> None:
        if self.role not in self.ROLES:
            raise ValueError("native semantic span has an unsupported role")
        _required_text(self.input_id, "input_id")
        _required_text(self.source_sha256, "source_sha256")
        if self.start_byte < 0 or self.end_byte <= self.start_byte:
            raise ValueError("native semantic span offsets must be a nonempty half-open range")
        if not isinstance(self.span_bytes, bytes):
            raise TypeError("native semantic span text must be bytes")
        if len(self.span_bytes) != self.end_byte - self.start_byte:
            raise ValueError("native semantic span length differs from its byte range")
        if hashlib.sha256(self.span_bytes).hexdigest() != self.span_sha256:
            raise ValueError("native semantic span digest differs")
        try:
            self.span_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("native semantic span splits a UTF-8 code point") from error
        if len(self.percept_event_ids) != len(self.span_bytes) or len(self.assembly_ids) != len(
            self.span_bytes
        ):
            raise ValueError("native semantic span lost per-byte percept lineage")
        if not 0.0 <= float(self.confidence) <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("native semantic span confidence must be finite and in [0, 1]")

    @property
    def text(self) -> str:
        return self.span_bytes.decode("utf-8")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "role": self.role,
            "input_id": self.input_id,
            "source_sha256": self.source_sha256,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "span_hex": self.span_bytes.hex(),
            "span_sha256": self.span_sha256,
            "percept_event_ids": list(self.percept_event_ids),
            "assembly_ids": list(self.assembly_ids),
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointSemanticSpanV1:
        required = {
            "format",
            "version",
            "role",
            "input_id",
            "source_sha256",
            "start_byte",
            "end_byte",
            "span_hex",
            "span_sha256",
            "percept_event_ids",
            "assembly_ids",
            "confidence",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("native semantic span schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native semantic span version")
        try:
            span_bytes = bytes.fromhex(str(payload["span_hex"]))
        except ValueError as error:
            raise ValueError("native semantic span byte encoding is invalid") from error
        return cls(
            role=str(payload["role"]),
            input_id=str(payload["input_id"]),
            source_sha256=str(payload["source_sha256"]),
            start_byte=int(payload["start_byte"]),
            end_byte=int(payload["end_byte"]),
            span_bytes=span_bytes,
            span_sha256=str(payload["span_sha256"]),
            percept_event_ids=tuple(str(x) for x in payload["percept_event_ids"]),
            assembly_ids=tuple(str(x) for x in payload["assembly_ids"]),
            confidence=float(payload["confidence"]),
        )


@dataclass(frozen=True)
class JointSemanticParseV1:
    """A model-produced structural hypothesis; not a gold state or an action."""

    joint_bundle_id: str
    owner_id: str
    scope_id: str
    session_id: str
    task_id: str
    source_joint_revision: int
    producer_bundle_id: str
    producer_parameter_revision: int
    input_id: str
    source_sha256: str
    act: str
    status: str
    confidence: float
    ambiguity: float
    spans: tuple[JointSemanticSpanV1, ...]
    semantic_features: tuple[float, ...]
    parse_sha256: str

    FORMAT = "taiji-joint-semantic-parse-v1"
    STATUSES = frozenset({"resolved", "candidate", "ambiguous"})

    def __post_init__(self) -> None:
        for name in (
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "producer_bundle_id",
            "input_id",
            "source_sha256",
            "parse_sha256",
        ):
            _required_text(getattr(self, name), name)
        if self.act not in ACT_NAMES or self.status not in self.STATUSES:
            raise ValueError("native semantic parse has an unsupported act or status")
        if self.source_joint_revision < 0 or self.producer_parameter_revision < 0:
            raise ValueError("native semantic parse revisions cannot be negative")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.ambiguity <= 1.0:
            raise ValueError("native semantic parse confidence/ambiguity must be in [0, 1]")
        if any(not isinstance(span, JointSemanticSpanV1) for span in self.spans):
            raise TypeError("native semantic parse spans must be typed source references")
        if any(
            span.input_id != self.input_id or span.source_sha256 != self.source_sha256
            for span in self.spans
        ):
            raise ValueError("native semantic parse mixes source identities")
        if any(not math.isfinite(float(item)) for item in self.semantic_features):
            raise ValueError("native semantic parse features must be finite")
        if self.parse_sha256 != content_digest(self._payload_without_digest()):
            raise ValueError("native semantic parse digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "source_joint_revision": self.source_joint_revision,
            "producer_bundle_id": self.producer_bundle_id,
            "producer_parameter_revision": self.producer_parameter_revision,
            "input_id": self.input_id,
            "source_sha256": self.source_sha256,
            "act": self.act,
            "status": self.status,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "spans": [span.to_payload() for span in self.spans],
            "semantic_features": list(self.semantic_features),
        }

    @classmethod
    def create(cls, **fields: Any) -> JointSemanticParseV1:
        fields_without_digest = dict(fields)
        fields_without_digest.pop("parse_sha256", None)
        shell = {
            "format": cls.FORMAT,
            "version": 1,
            **{
                name: (
                    [span.to_payload() for span in value]
                    if name == "spans"
                    else list(value) if name == "semantic_features" else value
                )
                for name, value in fields_without_digest.items()
            },
        }
        return cls(**fields_without_digest, parse_sha256=content_digest(shell))

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "parse_sha256": self.parse_sha256}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointSemanticParseV1:
        required = {
            "format",
            "version",
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "source_joint_revision",
            "producer_bundle_id",
            "producer_parameter_revision",
            "input_id",
            "source_sha256",
            "act",
            "status",
            "confidence",
            "ambiguity",
            "spans",
            "semantic_features",
            "parse_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("native semantic parse schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native semantic parse version")
        return cls(
            joint_bundle_id=str(payload["joint_bundle_id"]),
            owner_id=str(payload["owner_id"]),
            scope_id=str(payload["scope_id"]),
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            source_joint_revision=int(payload["source_joint_revision"]),
            producer_bundle_id=str(payload["producer_bundle_id"]),
            producer_parameter_revision=int(payload["producer_parameter_revision"]),
            input_id=str(payload["input_id"]),
            source_sha256=str(payload["source_sha256"]),
            act=str(payload["act"]),
            status=str(payload["status"]),
            confidence=float(payload["confidence"]),
            ambiguity=float(payload["ambiguity"]),
            spans=tuple(JointSemanticSpanV1.from_payload(item) for item in payload["spans"]),
            semantic_features=tuple(float(x) for x in payload["semantic_features"]),
            parse_sha256=str(payload["parse_sha256"]),
        )

    def validate_for(
        self,
        joint_state: JointTaskStateV1,
        *,
        require_current_revision: bool = False,
    ) -> bytes:
        joint_state._validate_live_owner()
        for name in ("joint_bundle_id", "owner_id", "scope_id", "session_id", "task_id"):
            if getattr(self, name) != getattr(joint_state, name):
                raise ValueError(f"native semantic parse {name} differs from its joint owner")
        if require_current_revision and self.source_joint_revision != joint_state.joint_revision:
            raise ValueError("native semantic parse was produced from a stale joint revision")
        if self.source_joint_revision > joint_state.joint_revision:
            raise ValueError("native semantic parse comes from a future joint revision")
        binding = next(
            (item for item in joint_state.input_bindings if item.input_id == self.input_id), None
        )
        if binding is None or binding.payload_sha256 != self.source_sha256:
            raise ValueError("native semantic parse source is outside the joint input ledger")
        observations = dict(joint_state.language_state.observations)
        raw = observations.get(binding.neural_source_id)
        if raw is None or hashlib.sha256(raw).hexdigest() != self.source_sha256:
            raise ValueError("native semantic parse input bytes differ from the joint source")
        for span in self.spans:
            if (
                span.end_byte > len(raw)
                or raw[span.start_byte : span.end_byte] != span.span_bytes
                or span.percept_event_ids
                != binding.percept_event_ids[span.start_byte : span.end_byte]
                or not set(span.assembly_ids) <= set(binding.assembly_ids)
            ):
                raise ValueError("native semantic span is not an exact source/percept slice")
        return raw


@dataclass(frozen=True)
class JointSemanticTrainingTargetV1:
    """Fit-only byte labels; prediction and commit APIs do not accept this object."""

    input_id: str
    source_sha256: str
    family_id: str
    partition: str
    tag_ids: tuple[int, ...]
    act_id: int
    target_sha256: str

    FORMAT = "taiji-joint-semantic-training-target-v1"
    RECEIVER = "input_semantic_producer"

    def __post_init__(self) -> None:
        for name in ("input_id", "source_sha256", "family_id", "target_sha256"):
            _required_text(getattr(self, name), name)
        if self.partition not in {"train", "support", "query", "holdout"}:
            raise ValueError("native semantic target has an unsupported partition")
        if self.act_id < 0 or self.act_id >= len(ACT_NAMES):
            raise ValueError("native semantic target act_id is outside the label schema")
        if any(tag < 0 or tag >= len(TAG_NAMES) for tag in self.tag_ids):
            raise ValueError("native semantic target contains an unsupported byte tag")
        if content_digest(self._payload_without_digest()) != self.target_sha256:
            raise ValueError("native semantic training target digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "input_id": self.input_id,
            "source_sha256": self.source_sha256,
            "family_id": self.family_id,
            "partition": self.partition,
            "tag_ids": list(self.tag_ids),
            "act_id": self.act_id,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "target_sha256": self.target_sha256}

    @classmethod
    def create(
        cls,
        *,
        input_id: str,
        source_sha256: str,
        family_id: str,
        partition: str,
        tag_ids: Sequence[int],
        act_id: int,
    ) -> JointSemanticTrainingTargetV1:
        payload = {
            "format": cls.FORMAT,
            "version": 1,
            "input_id": input_id,
            "source_sha256": source_sha256,
            "family_id": family_id,
            "partition": partition,
            "tag_ids": [int(x) for x in tag_ids],
            "act_id": int(act_id),
        }
        return cls(
            input_id=input_id,
            source_sha256=source_sha256,
            family_id=family_id,
            partition=partition,
            tag_ids=tuple(int(x) for x in tag_ids),
            act_id=int(act_id),
            target_sha256=content_digest(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointSemanticTrainingTargetV1:
        required = {
            "format",
            "version",
            "input_id",
            "source_sha256",
            "family_id",
            "partition",
            "tag_ids",
            "act_id",
            "target_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("native semantic training target schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native semantic training target version")
        return cls(
            input_id=str(payload["input_id"]),
            source_sha256=str(payload["source_sha256"]),
            family_id=str(payload["family_id"]),
            partition=str(payload["partition"]),
            tag_ids=tuple(int(x) for x in payload["tag_ids"]),
            act_id=int(payload["act_id"]),
            target_sha256=str(payload["target_sha256"]),
        )


class JointInputProducerV1(nn.Module):
    """Learnable byte/percept span producer; it never receives gold state at inference."""

    PARAMETER_OWNER = JointSemanticTrainingTargetV1.RECEIVER
    WORKSPACE_FEATURE_EMBEDDING = "prefix_zero_extension_v1"

    def __init__(
        self,
        *,
        percept_feature_dim: int,
        hidden_dim: int = 48,
        bundle_id: str = "joint-input-producer-v1",
        min_confidence: float = 0.6,
        parameter_revision: int = 0,
    ) -> None:
        super().__init__()
        if percept_feature_dim <= 0 or hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("producer feature width must be positive and hidden_dim must be even")
        _required_text(bundle_id, "bundle_id")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("producer min_confidence must be in [0, 1]")
        if parameter_revision < 0:
            raise ValueError("producer parameter revision cannot be negative")
        self.percept_feature_dim = int(percept_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.bundle_id = bundle_id
        self.min_confidence = float(min_confidence)
        self.parameter_revision = int(parameter_revision)
        byte_embedding_dim = max(8, hidden_dim // 2)
        self.byte_embedding = nn.Embedding(257, byte_embedding_dim)
        self.input_projection = nn.Linear(percept_feature_dim + byte_embedding_dim, hidden_dim)
        self.context = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.slot_head = nn.Linear(hidden_dim, len(TAG_NAMES))
        self.act_head = nn.Linear(hidden_dim, len(ACT_NAMES))

    def _validate_trace(
        self, joint_state: JointTaskStateV1, trace: InputTrace
    ) -> tuple[bytes, Any]:
        joint_state._validate_live_owner()
        raw = _input_bytes(trace)
        binding = next(
            (item for item in joint_state.input_bindings if item.input_id == trace.input_id), None
        )
        if binding is None:
            raise ValueError("native input producer trace is not committed to the joint ledger")
        if (
            binding.byte_count != len(raw)
            or binding.payload_sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("native input producer trace differs from the committed source bytes")
        if binding.percept_event_ids != tuple(item.event_id for item in trace.percepts):
            raise ValueError("native input producer percept events differ from the source ledger")
        if binding.assembly_ids != tuple(
            dict.fromkeys(item.assembly_id for item in trace.percepts)
        ):
            raise ValueError("native input producer assemblies differ from the source ledger")
        if any(
            item.features.ndim != 1 or item.features.numel() != self.percept_feature_dim
            for item in trace.percepts
        ):
            raise ValueError("native input producer percept feature width differs")
        if any(not bool(torch.isfinite(item.features).all()) for item in trace.percepts):
            raise ValueError("native input producer percept features must be finite")
        return raw, binding

    def _encode(self, trace: InputTrace) -> torch.Tensor:
        raw = _input_bytes(trace)
        byte_ids = torch.tensor(
            list(raw), dtype=torch.long, device=self.byte_embedding.weight.device
        )
        percept_features = torch.stack(
            [
                item.features.detach().to(self.byte_embedding.weight.device, dtype=torch.float32)
                for item in trace.percepts
            ]
        )
        bytes_embedded = self.byte_embedding(byte_ids)
        joined = torch.cat((percept_features, bytes_embedded), dim=-1)
        encoded, _ = self.context(self.input_projection(joined).unsqueeze(0))
        return cast(torch.Tensor, encoded[0])

    def forward(self, trace: InputTrace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode(trace)
        return self.slot_head(encoded), self.act_head(encoded.mean(dim=0)), encoded

    def supervised_loss(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
        target: JointSemanticTrainingTargetV1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw, _ = self._validate_trace(joint_state, trace)
        if target.partition != "train":
            raise ValueError("semantic targets from support/query/holdout cannot reach the fit API")
        if (
            target.input_id != trace.input_id
            or target.source_sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("semantic fit target differs from the exact input source")
        if len(target.tag_ids) != len(raw):
            raise ValueError("semantic fit target needs one byte tag per source byte")
        slot_logits, act_logits, _ = self(trace)
        slot_target = torch.tensor(target.tag_ids, dtype=torch.long, device=slot_logits.device)
        act_target = torch.tensor([target.act_id], dtype=torch.long, device=act_logits.device)
        slot_loss = F.cross_entropy(slot_logits, slot_target)
        act_loss = F.cross_entropy(act_logits.unsqueeze(0), act_target)
        return slot_loss + act_loss, {"slot": slot_loss, "act": act_loss}

    @staticmethod
    def entity_object_id(joint_state: JointTaskStateV1, entity_text: str) -> str:
        normalized_entity = unicodedata.normalize("NFC", entity_text)
        identity = "\0".join(
            (
                joint_state.joint_bundle_id,
                joint_state.owner_id,
                joint_state.scope_id,
                normalized_entity,
            )
        )
        return "native-entity:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_tag_spans(tag_ids: Sequence[int]) -> tuple[dict[str, list[tuple[int, int]]], bool]:
        spans: dict[str, list[tuple[int, int]]] = {name: [] for name in JointSemanticSpanV1.ROLES}
        active_role: str | None = None
        active_start = 0
        invalid = False
        for index, tag_id in enumerate((*tag_ids, 0)):
            if tag_id == 0:
                if active_role is not None:
                    spans[active_role].append((active_start, index))
                    active_role = None
                continue
            role, is_begin = _TAG_TO_ROLE.get(int(tag_id), ("", False))
            if not role:
                invalid = True
                continue
            if is_begin:
                if active_role is not None:
                    spans[active_role].append((active_start, index))
                active_role = role
                active_start = index
            elif active_role != role:
                invalid = True
                if active_role is not None:
                    spans[active_role].append((active_start, index))
                active_role = role
                active_start = index
        return spans, invalid

    def _parse_logits(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
        slot_logits: torch.Tensor,
        act_logits: torch.Tensor,
        encoded: torch.Tensor,
    ) -> JointSemanticParseV1:
        raw, binding = self._validate_trace(joint_state, trace)
        if slot_logits.shape != (len(raw), len(TAG_NAMES)):
            raise ValueError("native parser slot logits differ from byte/label dimensions")
        if act_logits.shape != (len(ACT_NAMES),):
            raise ValueError("native parser act logits differ from act schema")
        if not bool(torch.isfinite(slot_logits).all()) or not bool(
            torch.isfinite(act_logits).all()
        ):
            raise ValueError("native parser logits must be finite")
        slot_probabilities = torch.softmax(slot_logits.detach(), dim=-1)
        tag_ids = slot_probabilities.argmax(dim=-1).tolist()
        decoded, invalid_transitions = self._decode_tag_spans(tag_ids)
        act_probabilities = torch.softmax(act_logits.detach(), dim=-1)
        act_id = int(act_probabilities.argmax().item())
        act_confidence = float(act_probabilities[act_id].item())
        spans: list[JointSemanticSpanV1] = []
        invalid_span = False
        span_confidences: list[float] = []
        for role, ranges in decoded.items():
            for start, end in ranges:
                if start < 0 or end > len(raw) or start >= end:
                    invalid_span = True
                    continue
                span_bytes = raw[start:end]
                try:
                    span_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    invalid_span = True
                    continue
                confidence = float(slot_probabilities[start:end].max(dim=-1).values.mean().item())
                span_confidences.append(confidence)
                spans.append(
                    JointSemanticSpanV1(
                        role=role,
                        input_id=trace.input_id,
                        source_sha256=binding.payload_sha256,
                        start_byte=start,
                        end_byte=end,
                        span_bytes=span_bytes,
                        span_sha256=hashlib.sha256(span_bytes).hexdigest(),
                        percept_event_ids=binding.percept_event_ids[start:end],
                        assembly_ids=tuple(
                            percept.assembly_id for percept in trace.percepts[start:end]
                        ),
                        confidence=confidence,
                    )
                )
        spans.sort(key=lambda span: (span.start_byte, span.end_byte, span.role))
        confidence = min([act_confidence, *span_confidences]) if span_confidences else 0.0
        act = ACT_NAMES[act_id]
        required_roles = {
            "assertion": {"entity", "attribute", "value"},
            "correction": {"entity", "attribute", "value"},
            "query": {"entity", "attribute"},
            "goal": {"entity", "attribute", "value"},
            "ambiguous": set(),
        }[act]
        role_counts = {role: len(decoded[role]) for role in decoded}
        if any(role_counts[role] != 1 for role in required_roles):
            invalid_span = True
        if any(decoded[role] and role not in required_roles for role in decoded):
            invalid_span = True
        if decoded["negation"]:
            invalid_span = True
        status = (
            "resolved"
            if act != "ambiguous"
            and confidence >= self.min_confidence
            and not invalid_transitions
            and not invalid_span
            else "ambiguous"
        )
        slot_token_indices = sorted(
            {
                index
                for role in ("entity", "attribute", "value")
                for start, end in decoded[role]
                for index in range(start, end)
            }
        )
        if slot_token_indices:
            feature_index = torch.tensor(
                slot_token_indices, dtype=torch.long, device=encoded.device
            )
            semantic_vector = encoded.index_select(0, feature_index).mean(dim=0)
        else:
            semantic_vector = encoded.mean(dim=0)
        features = semantic_vector.detach().to(device="cpu", dtype=torch.float32)
        semantic_features = tuple(float(item) for item in features.tolist())
        return JointSemanticParseV1.create(
            joint_bundle_id=joint_state.joint_bundle_id,
            owner_id=joint_state.owner_id,
            scope_id=joint_state.scope_id,
            session_id=joint_state.session_id,
            task_id=joint_state.task_id,
            source_joint_revision=joint_state.joint_revision,
            producer_bundle_id=self.bundle_id,
            producer_parameter_revision=self.parameter_revision,
            input_id=trace.input_id,
            source_sha256=binding.payload_sha256,
            act=act,
            status=status,
            confidence=confidence,
            ambiguity=1.0 - confidence,
            spans=tuple(spans),
            semantic_features=semantic_features,
        )

    @staticmethod
    def _embed_semantic_features_in_workspace(
        semantic_features: tuple[float, ...], workspace_width: int
    ) -> torch.Tensor:
        """Inject the producer vector losslessly into the wider consumer basis."""

        source = torch.tensor(semantic_features, dtype=torch.float32)
        if workspace_width <= 0 or source.numel() == 0 or source.numel() > workspace_width:
            raise ValueError(
                "semantic producer features cannot be losslessly embedded in workspace"
            )
        if source.numel() == workspace_width:
            return source
        embedded = torch.zeros(workspace_width, dtype=torch.float32)
        embedded[: source.numel()] = source
        return embedded

    @torch.no_grad()
    def predict(self, joint_state: JointTaskStateV1, trace: InputTrace) -> JointSemanticParseV1:
        self._validate_trace(joint_state, trace)
        slot_logits, act_logits, encoded = self(trace)
        return self._parse_logits(joint_state, trace, slot_logits, act_logits, encoded)

    def _predict_with_role_vectors(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
    ) -> tuple[JointSemanticParseV1, dict[str, torch.Tensor]]:
        self._validate_trace(joint_state, trace)
        slot_logits, act_logits, encoded = self(trace)
        parse = self._parse_logits(joint_state, trace, slot_logits, act_logits, encoded)
        vectors: dict[str, torch.Tensor] = {}
        for role in ("entity", "attribute", "value"):
            spans = tuple(item for item in parse.spans if item.role == role)
            if len(spans) != 1:
                continue
            span = spans[0]
            vectors[role] = encoded[span.start_byte : span.end_byte]
        return parse, vectors

    @torch.no_grad()
    def predict_with_role_features(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
    ) -> tuple[JointSemanticParseV1, dict[str, tuple[float, ...]]]:
        """Preserve per-role mean encoder features instead of one all-slot mean."""

        parse, vectors = self._predict_with_role_vectors(joint_state, trace)
        roles: dict[str, tuple[float, ...]] = {}
        for role, vector in vectors.items():
            mean = vector.mean(dim=0)
            roles[role] = tuple(float(item) for item in mean.detach().cpu().tolist())
        return parse, roles

    @torch.no_grad()
    def predict_with_role_byte_features(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
    ) -> tuple[JointSemanticParseV1, dict[str, tuple[tuple[float, ...], ...]]]:
        """Preserve every per-byte encoder row inside each role span."""

        parse, vectors = self._predict_with_role_vectors(joint_state, trace)
        roles: dict[str, tuple[tuple[float, ...], ...]] = {}
        for role, vector in vectors.items():
            roles[role] = tuple(
                tuple(float(item) for item in row.detach().cpu().tolist()) for row in vector
            )
        return parse, roles

    def commit_to_joint_state(
        self,
        joint_state: JointTaskStateV1,
        trace: InputTrace,
        parse: JointSemanticParseV1,
        language_candidate: Any,
    ) -> JointTaskStateV1:
        parse.validate_for(joint_state, require_current_revision=True)
        if getattr(language_candidate, "bundle_id", None) != joint_state.language_state.bundle_id:
            raise ValueError("semantic producer V4 consumer belongs to another candidate bundle")
        language_candidate.validate(joint_state.language_state)
        regenerated = self.predict(joint_state, trace)
        if regenerated.parse_sha256 != parse.parse_sha256:
            raise ValueError("semantic parse differs from this producer's current model output")
        if parse.producer_bundle_id != self.bundle_id or (
            parse.producer_parameter_revision != self.parameter_revision
        ):
            raise ValueError("semantic parse belongs to another producer parameter revision")
        if parse.status != "resolved" or parse.act not in {
            "assertion",
            "correction",
            "query",
        }:
            return joint_state
        spans_by_role = {
            role: [item for item in parse.spans if item.role == role]
            for role in ("entity", "attribute", "value")
        }
        required_roles = (
            ("entity", "attribute")
            if parse.act == "query"
            else (
                "entity",
                "attribute",
                "value",
            )
        )
        if any(len(spans_by_role[role]) != 1 for role in required_roles):
            raise ValueError(
                "resolved semantic parse lacks its required entity/attribute/value spans"
            )
        if parse.act == "query" and spans_by_role["value"]:
            raise ValueError("resolved query parse must not include an asserted value span")
        if len(parse.semantic_features) > language_candidate.config.width:
            raise ValueError("semantic producer feature width exceeds the V4 workspace width")
        entity_text = spans_by_role["entity"][0].text
        attribute_text = spans_by_role["attribute"][0].text
        value_text = "" if parse.act == "query" else spans_by_role["value"][0].text
        if not entity_text or not attribute_text or (parse.act != "query" and not value_text):
            raise ValueError("resolved semantic facts cannot contain empty slots")
        object_id = self.entity_object_id(joint_state, entity_text)
        root = joint_state.cognitive_state
        old_object = next(
            (item for item in root.world.objects if item.object_id == object_id), None
        )
        if parse.act == "query":
            if old_object is None or attribute_text not in dict(old_object.attributes):
                return joint_state
            world = root.world
        else:
            attributes = {} if old_object is None else dict(old_object.attributes)
            attributes[attribute_text] = value_text
            new_object = WorldObject(
                object_id,
                attributes=tuple(sorted(attributes.items())),
                tags=() if old_object is None else old_object.tags,
            )
            world_objects = tuple(
                item for item in root.world.objects if item.object_id != object_id
            ) + (new_object,)
            relations = tuple(
                item
                for item in root.world.relations
                if not (item[0] == object_id and item[1] == attribute_text)
            ) + (
                (object_id, attribute_text, value_text),
            )
            event_id = "native-semantic-event:" + parse.parse_sha256
            if any(item.event_id == event_id for item in root.world.events):
                raise ValueError("native semantic parse event was already committed")
            event = WorldEvent(
                event_id=event_id,
                kind=(
                    "native_fact_correction"
                    if parse.act == "correction"
                    else "native_fact_assertion"
                ),
                tick=root.world.tick,
                subject_id=object_id,
                object_id=object_id,
                attributes=((attribute_text, value_text),),
                provenance=parse.input_id,
            )
            world = replace(
                root.world,
                entities=tuple(item.object_id for item in world_objects),
                relations=relations,
                objects=world_objects,
                events=(*root.world.events, event),
            )
        candidate_id = "native-semantic-candidate:" + parse.parse_sha256
        workspace_feature = self._embed_semantic_features_in_workspace(
            parse.semantic_features,
            language_candidate.config.width,
        )
        candidate = WorkspaceCandidate(
            candidate_id,
            workspace_feature,
            salience=parse.confidence,
            source=parse.input_id,
        )
        capacity = max(1, root.workspace.capacity)
        candidates = (*root.workspace.candidates, candidate)[-capacity:]
        candidate_ids = tuple(item.candidate_id for item in candidates)
        selected_ids = (candidate_id,)
        scores = tuple(1.0 if item.candidate_id == candidate_id else 0.0 for item in candidates)
        broadcasts = candidate.features.detach().clone()
        bindings = dict(root.workspace.bindings)
        bindings[candidate_id] = object_id
        retained_ids = set(candidate_ids)
        percept_event_ids = tuple(
            sorted({event_id for span in parse.spans for event_id in span.percept_event_ids})
        )
        assembly_ids = tuple(
            sorted({assembly_id for span in parse.spans for assembly_id in span.assembly_ids})
        )
        workspace = WorkspaceState(
            tick=root.tick,
            focus=selected_ids,
            bindings=tuple(
                sorted((key, value) for key, value in bindings.items() if key in retained_ids)
            ),
            broadcast=broadcasts,
            capacity=capacity,
            candidates=candidates,
            selection=WorkspaceSelection(
                tick=root.tick,
                mode="learned",
                candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                scores=scores,
                broadcast=broadcasts,
                capacity=capacity,
            ),
            percept_event_id=" ".join(percept_event_ids),
            percept_assembly_id=" ".join(assembly_ids),
        )
        updated = replace(root, world=world, workspace=workspace)
        binding = next(
            item for item in joint_state.input_bindings if item.input_id == parse.input_id
        )
        source_index = next(
            (
                index
                for index, (source_id, _) in enumerate(joint_state.language_state.observations)
                if source_id == binding.neural_source_id
            ),
            None,
        )
        if source_index is None:
            raise ValueError("semantic producer input disappeared from the V4 byte ledger")
        slot_spans = [item for item in parse.spans if item.role in {"entity", "attribute", "value"}]
        span_start = min(item.start_byte for item in slot_spans)
        span_end = max(item.end_byte for item in slot_spans)
        semantic_feature = workspace_feature.to(
            dtype=joint_state.language_state.workspace.dtype,
            device=joint_state.language_state.workspace.device,
        ).reshape(1, -1)
        slot_capacity = int(language_candidate.config.slots)
        neural_state = replace(
            joint_state.language_state,
            workspace=torch.cat((joint_state.language_state.workspace, semantic_feature), dim=0)[
                -slot_capacity:
            ],
            evidence_refs=(
                *joint_state.language_state.evidence_refs,
                (source_index, span_start, span_end),
            )[-slot_capacity:],
        )
        language_candidate.validate(neural_state)
        with_semantic_feature = joint_state.with_language_state(neural_state)
        return with_semantic_feature.advance_cognitive_state(updated)

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": JOINT_INPUT_PRODUCER_CHECKPOINT_FORMAT,
            "version": 1,
            "bundle_id": self.bundle_id,
            "percept_feature_dim": self.percept_feature_dim,
            "hidden_dim": self.hidden_dim,
            "min_confidence": self.min_confidence,
            "parameter_revision": self.parameter_revision,
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
            },
        }
        payload["checkpoint_sha256"] = content_digest(payload)
        return payload

    @staticmethod
    def bind_cognitive_owner_to_native_adapter_checkpoint(
        native_payload: Mapping[str, Any], joint_state: JointTaskStateV1
    ) -> dict[str, Any]:
        """Carry the joint canonical owner into the same TSK-v8 input-kernel checkpoint."""
        joint_state._validate_live_owner()
        envelope = NativeCheckpoint.from_payload(native_payload)
        from .adapter import TSKV8Adapter

        if envelope.adapter != TSKV8Adapter.ADAPTER_NAME:
            raise ValueError("joint state can only bind to its TSK-v8 adapter checkpoint")
        if (
            envelope.cognitive_state.episode_id != joint_state.task_id
            or envelope.cognitive_state.tick != joint_state.cognitive_state.tick
        ):
            raise ValueError("adapter kernel and joint cognitive owner crossed task time/identity")
        return NativeCheckpoint(
            kernel=envelope.kernel,
            cognitive_state=joint_state.cognitive_state,
            components=envelope.components,
            adapter=envelope.adapter,
        ).to_payload()

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> JointInputProducerV1:
        if not isinstance(payload, Mapping):
            raise TypeError("native input producer checkpoint must be a mapping")
        required = {
            "format",
            "version",
            "bundle_id",
            "percept_feature_dim",
            "hidden_dim",
            "min_confidence",
            "parameter_revision",
            "state_dict",
            "checkpoint_sha256",
        }
        if set(payload) != required:
            raise ValueError("native input producer checkpoint schema differs")
        unsigned = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        if payload["checkpoint_sha256"] != content_digest(unsigned):
            raise ValueError("native input producer checkpoint digest differs")
        if (payload["format"], payload["version"]) != (
            JOINT_INPUT_PRODUCER_CHECKPOINT_FORMAT,
            1,
        ):
            raise ValueError("unsupported native input producer checkpoint version")
        producer = cls(
            percept_feature_dim=int(payload["percept_feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            bundle_id=str(payload["bundle_id"]),
            min_confidence=float(payload["min_confidence"]),
            parameter_revision=int(payload["parameter_revision"]),
        )
        producer.load_state_dict(payload["state_dict"])
        return producer

    def attach_to_native_checkpoint(
        self,
        native_payload: Mapping[str, Any],
        joint_state: JointTaskStateV1,
        *,
        parses: Sequence[JointSemanticParseV1] = (),
    ) -> dict[str, Any]:
        joint_state._validate_live_owner()
        envelope = NativeCheckpoint.from_payload(native_payload)
        if cognitive_state_digest(envelope.cognitive_state) != joint_state.cognitive_state_sha256:
            raise ValueError("native checkpoint and semantic producer have different joint owners")
        task_component_keys = {
            JOINT_NATIVE_COMPONENT,
            JOINT_NATIVE_COMPONENT_V1,
            JOINT_NATIVE_COMPONENT_V2,
        }
        if len(task_component_keys.intersection(envelope.components)) != 1:
            raise ValueError(
                "semantic producer checkpoint requires exactly one joint task component"
            )
        if JOINT_INPUT_PRODUCER_COMPONENT_KEY in envelope.components:
            raise ValueError("native checkpoint already contains an input producer component")
        parse_payloads = []
        for parse in parses:
            parse.validate_for(joint_state)
            if parse.producer_bundle_id != self.bundle_id:
                raise ValueError("semantic parse belongs to another input producer bundle")
            parse_payloads.append(parse.to_payload())
        component = {
            "format": JOINT_INPUT_PRODUCER_COMPONENT_FORMAT,
            "version": 1,
            "joint_bundle_id": joint_state.joint_bundle_id,
            "owner_id": joint_state.owner_id,
            "scope_id": joint_state.scope_id,
            "session_id": joint_state.session_id,
            "task_id": joint_state.task_id,
            "cognitive_state_sha256": joint_state.cognitive_state_sha256,
            "producer_checkpoint": self.checkpoint(),
            "parse_history": parse_payloads,
        }
        component["component_digest"] = content_digest(component)
        return NativeCheckpoint(
            kernel=envelope.kernel,
            cognitive_state=envelope.cognitive_state,
            components={**envelope.components, JOINT_INPUT_PRODUCER_COMPONENT_KEY: component},
            adapter=envelope.adapter,
        ).to_payload()

    @classmethod
    def restore_from_native_checkpoint(
        cls,
        native_payload: Mapping[str, Any],
        joint_state: JointTaskStateV1,
    ) -> tuple[JointInputProducerV1, tuple[JointSemanticParseV1, ...]]:
        envelope = NativeCheckpoint.from_payload(native_payload)
        component = envelope.components.get(JOINT_INPUT_PRODUCER_COMPONENT_KEY)
        required = {
            "format",
            "version",
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "cognitive_state_sha256",
            "producer_checkpoint",
            "parse_history",
            "component_digest",
        }
        if not isinstance(component, Mapping) or set(component) != required:
            raise ValueError("native checkpoint lacks the semantic input producer component")
        unsigned = {key: value for key, value in component.items() if key != "component_digest"}
        if component["component_digest"] != content_digest(unsigned):
            raise ValueError("semantic input producer component digest differs")
        if (component["format"], component["version"]) != (
            JOINT_INPUT_PRODUCER_COMPONENT_FORMAT,
            1,
        ):
            raise ValueError("unsupported semantic input producer component version")
        if cognitive_state_digest(envelope.cognitive_state) != joint_state.cognitive_state_sha256:
            raise ValueError("producer restore and joint checkpoint cognitive owners differ")
        for name in ("joint_bundle_id", "owner_id", "scope_id", "session_id", "task_id"):
            if component[name] != getattr(joint_state, name):
                raise ValueError(f"semantic producer restored {name} differs from its joint owner")
        if component["cognitive_state_sha256"] != joint_state.cognitive_state_sha256:
            raise ValueError("semantic producer restored cognitive owner digest differs")
        producer = cls.from_checkpoint(component["producer_checkpoint"])
        parses = tuple(
            JointSemanticParseV1.from_payload(item) for item in component["parse_history"]
        )
        for parse in parses:
            parse.validate_for(joint_state)
            if parse.producer_bundle_id != producer.bundle_id:
                raise ValueError("restored semantic parse belongs to another producer bundle")
        return producer, parses
