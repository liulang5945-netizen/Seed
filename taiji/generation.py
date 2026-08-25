"""Native content, expression, and structured tool-call generation.

Generation is an effector boundary, not a second decision-maker.  The
controller preserves the semantic fields already present in an ``ActionIntent``
and turns them into an organ-specific expression.  The byte codec is only the
lossless transport boundary for a structured call.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import ActionIntent, WorldAction

GENERATION_CHECKPOINT_FORMAT = "taiji-generation-v1"
TOOL_CALL_CODEC_FORMAT = "taiji-tool-call-codec-v1"
TEXT_EXPRESSION_CODEC_FORMAT = "taiji-text-expression-codec-v1"


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _json_value(value: Any, path: str = "value") -> Any:
    """Validate and detach the JSON subset accepted by an external tool."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} is not JSON-compatible tool data: {type(value).__name__}")


@dataclass(frozen=True)
class ContentPlan:
    """Semantic content selected before choosing an output organ."""

    content_id: str
    intent_id: str
    intent_kind: str
    semantic_slots: Mapping[str, Any] = field(default_factory=dict)
    source_goal_id: str | None = None
    expected_outcome: str = ""
    confidence: float = 0.0
    provenance: str = "planned"
    tick: int = 0

    def __post_init__(self) -> None:
        _text(self.content_id, "content_id")
        _text(self.intent_id, "intent_id")
        _text(self.intent_kind, "intent_kind")
        if self.source_goal_id is not None:
            _text(self.source_goal_id, "source_goal_id")
        _text(self.provenance, "provenance")
        if int(self.tick) < 0:
            raise ValueError("content tick cannot be negative")
        _unit(self.confidence, "content confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "intent_id": self.intent_id,
            "intent_kind": self.intent_kind,
            "semantic_slots": dict(self.semantic_slots),
            "source_goal_id": self.source_goal_id,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContentPlan:
        slots = payload.get("semantic_slots", {})
        if not isinstance(slots, Mapping):
            raise ValueError("content semantic_slots must be a mapping")
        return cls(
            content_id=str(payload["content_id"]),
            intent_id=str(payload["intent_id"]),
            intent_kind=str(payload["intent_kind"]),
            semantic_slots=dict(slots),
            source_goal_id=payload.get("source_goal_id"),
            expected_outcome=str(payload.get("expected_outcome", "")),
            confidence=float(payload.get("confidence", 0.0)),
            provenance=str(payload.get("provenance", "planned")),
            tick=int(payload.get("tick", 0)),
        )


@dataclass(frozen=True)
class ExpressionPlan:
    """An organ-specific rendering plan derived from semantic content."""

    expression_id: str
    content_id: str
    modality: str
    channel: str
    source_goal_id: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: str = "planned"
    tick: int = 0

    def __post_init__(self) -> None:
        _text(self.expression_id, "expression_id")
        _text(self.content_id, "content_id")
        _text(self.modality, "expression modality")
        _text(self.channel, "expression channel")
        if self.source_goal_id is not None:
            _text(self.source_goal_id, "expression source_goal_id")
        _text(self.provenance, "expression provenance")
        if int(self.tick) < 0:
            raise ValueError("expression tick cannot be negative")
        _unit(self.confidence, "expression confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "expression_id": self.expression_id,
            "content_id": self.content_id,
            "modality": self.modality,
            "channel": self.channel,
            "source_goal_id": self.source_goal_id,
            "fields": dict(self.fields),
            "confidence": self.confidence,
            "provenance": self.provenance,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExpressionPlan:
        fields = payload.get("fields", {})
        if not isinstance(fields, Mapping):
            raise ValueError("expression fields must be a mapping")
        return cls(
            expression_id=str(payload["expression_id"]),
            content_id=str(payload["content_id"]),
            modality=str(payload["modality"]),
            channel=str(payload["channel"]),
            source_goal_id=payload.get("source_goal_id"),
            fields=dict(fields),
            confidence=float(payload.get("confidence", 0.0)),
            provenance=str(payload.get("provenance", "planned")),
            tick=int(payload.get("tick", 0)),
        )


@dataclass(frozen=True)
class ToolCall:
    """A structured tool action emitted by an effector organ."""

    call_id: str
    intent_id: str
    expression_id: str
    tool_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_goal_id: str | None = None
    confidence: float = 0.0
    provenance: str = "planned"
    tick: int = 0

    def __post_init__(self) -> None:
        _text(self.call_id, "call_id")
        _text(self.intent_id, "intent_id")
        _text(self.expression_id, "expression_id")
        _text(self.tool_name, "tool_name")
        if self.source_goal_id is not None:
            _text(self.source_goal_id, "source_goal_id")
        _text(self.provenance, "tool call provenance")
        if int(self.tick) < 0:
            raise ValueError("tool call tick cannot be negative")
        _unit(self.confidence, "tool call confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "intent_id": self.intent_id,
            "expression_id": self.expression_id,
            "tool_name": self.tool_name,
            "parameters": dict(self.parameters),
            "source_goal_id": self.source_goal_id,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ToolCall:
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("tool call parameters must be a mapping")
        return cls(
            call_id=str(payload["call_id"]),
            intent_id=str(payload["intent_id"]),
            expression_id=str(payload["expression_id"]),
            tool_name=str(payload["tool_name"]),
            parameters=dict(parameters),
            source_goal_id=payload.get("source_goal_id"),
            confidence=float(payload.get("confidence", 0.0)),
            provenance=str(payload.get("provenance", "planned")),
            tick=int(payload.get("tick", 0)),
        )

    def to_world_action(self, *, tick: int | None = None) -> WorldAction:
        """Expose the same intent to the causal world/action contract."""

        return WorldAction(
            action_id=self.intent_id,
            kind=self.tool_name,
            tick=self.tick if tick is None else int(tick),
            parameters=tuple((str(key), value) for key, value in self.parameters.items()),
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class GenerationTrace:
    """The inspectable stages of one content-to-tool emission."""

    content: ContentPlan
    expression: ExpressionPlan
    tool_call: ToolCall
    encoded: bytes

    def to_payload(self) -> dict[str, Any]:
        return {
            "content": self.content.to_payload(),
            "expression": self.expression.to_payload(),
            "tool_call": self.tool_call.to_payload(),
            "encoded": bytes(self.encoded),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GenerationTrace:
        encoded = payload.get("encoded", b"")
        if not isinstance(encoded, bytes):
            encoded = bytes(encoded)
        return cls(
            content=ContentPlan.from_payload(payload["content"]),
            expression=ExpressionPlan.from_payload(payload["expression"]),
            tool_call=ToolCall.from_payload(payload["tool_call"]),
            encoded=encoded,
        )


class StructuredToolCallCodec:
    """Lossless UTF-8 transport for a JSON-compatible structured tool call."""

    @staticmethod
    def encode(call: ToolCall) -> bytes:
        payload = {"format": TOOL_CALL_CODEC_FORMAT, "call": _json_value(call.to_payload())}
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def decode(data: bytes | bytearray | memoryview) -> ToolCall:
        try:
            payload = json.loads(bytes(data).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid structured tool-call bytes") from exc
        if not isinstance(payload, Mapping) or payload.get("format") != TOOL_CALL_CODEC_FORMAT:
            raise ValueError("unsupported structured tool-call codec format")
        call = payload.get("call")
        if not isinstance(call, Mapping):
            raise ValueError("structured tool-call payload is missing call")
        return ToolCall.from_payload(call)


class TextExpressionCodec:
    """Lossless UTF-8 transport for a structured text-organ expression."""

    @staticmethod
    def encode(expression: ExpressionPlan) -> bytes:
        payload = {
            "format": TEXT_EXPRESSION_CODEC_FORMAT,
            "expression": _json_value(expression.to_payload()),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def decode(data: bytes | bytearray | memoryview) -> ExpressionPlan:
        try:
            payload = json.loads(bytes(data).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid structured text-expression bytes") from exc
        if not isinstance(payload, Mapping) or payload.get("format") != TEXT_EXPRESSION_CODEC_FORMAT:
            raise ValueError("unsupported structured text-expression codec format")
        expression = payload.get("expression")
        if not isinstance(expression, Mapping):
            raise ValueError("structured text-expression payload is missing expression")
        return ExpressionPlan.from_payload(expression)


class GenerationController:
    """Turn a Taiji action intent into an organ-specific tool emission."""

    def __init__(self, *, codec: StructuredToolCallCodec | None = None) -> None:
        self.codec = codec or StructuredToolCallCodec()

    def plan_content(
        self,
        intent: ActionIntent,
        *,
        source_goal_id: str | None = None,
        provenance: str = "planned",
    ) -> ContentPlan:
        if not isinstance(intent, ActionIntent):
            raise TypeError("intent must be an ActionIntent")
        return ContentPlan(
            content_id=f"{intent.intent_id}:content",
            intent_id=intent.intent_id,
            intent_kind=intent.kind,
            semantic_slots=dict(intent.parameters),
            source_goal_id=(source_goal_id if source_goal_id is not None else intent.source_goal_id),
            expected_outcome=intent.expected_outcome,
            confidence=intent.confidence,
            provenance=provenance,
            tick=intent.tick,
        )

    def plan_expression(
        self,
        content: ContentPlan,
        *,
        modality: str = "tool",
        channel: str | None = None,
    ) -> ExpressionPlan:
        if not isinstance(content, ContentPlan):
            raise TypeError("content must be a ContentPlan")
        selected_channel = content.intent_kind if channel is None else str(channel)
        return ExpressionPlan(
            expression_id=f"{content.content_id}:expression:{modality}",
            content_id=content.content_id,
            modality=modality,
            channel=selected_channel,
            source_goal_id=content.source_goal_id,
            fields={
                "intent_kind": content.intent_kind,
                "semantic_slots": dict(content.semantic_slots),
                "expected_outcome": content.expected_outcome,
            },
            confidence=content.confidence,
            provenance=content.provenance,
            tick=content.tick,
        )

    def generate_tool_call(
        self,
        intent: ActionIntent,
        *,
        tool_name: str | None = None,
        source_goal_id: str | None = None,
        channel: str | None = None,
        provenance: str = "planned",
    ) -> GenerationTrace:
        content = self.plan_content(
            intent,
            source_goal_id=source_goal_id,
            provenance=provenance,
        )
        expression = self.plan_expression(content, modality="tool", channel=channel)
        call = ToolCall(
            call_id=f"{intent.intent_id}:tool-call",
            intent_id=intent.intent_id,
            expression_id=expression.expression_id,
            tool_name=intent.kind if tool_name is None else str(tool_name),
            parameters=dict(content.semantic_slots),
            source_goal_id=content.source_goal_id,
            confidence=expression.confidence,
            provenance=expression.provenance,
            tick=expression.tick,
        )
        return GenerationTrace(
            content=content,
            expression=expression,
            tool_call=call,
            encoded=self.codec.encode(call),
        )

    def checkpoint(self) -> dict[str, str]:
        return {
            "format": GENERATION_CHECKPOINT_FORMAT,
            "codec_format": TOOL_CALL_CODEC_FORMAT,
            "text_codec_format": TEXT_EXPRESSION_CODEC_FORMAT,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> GenerationController:
        if payload.get("format") != GENERATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported generation checkpoint format")
        if payload.get("codec_format") != TOOL_CALL_CODEC_FORMAT:
            raise ValueError("unsupported generation codec format")
        if payload.get("text_codec_format", TEXT_EXPRESSION_CODEC_FORMAT) != TEXT_EXPRESSION_CODEC_FORMAT:
            raise ValueError("unsupported text expression codec format")
        return cls()
