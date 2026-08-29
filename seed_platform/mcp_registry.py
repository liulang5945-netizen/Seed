"""Seed-owned MCP-shaped capability registry.

This module intentionally stops at the protocol boundary: it describes and
validates local tool contracts, while Workbench owns execution and policy.
Legacy MCP installation, networking and external tool selection remain out
of this registry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MCP_REGISTRY_FORMAT = "seed-mcp-registry-v1"
MCP_REGISTRY_VERSION = 1


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class McpToolDescriptor:
    """A versioned MCP-shaped tool contract, not an executor."""

    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]
    executor_id: str
    source: str = "seed.mcp.local"
    risk: str = "read_only"
    timeout_seconds: float = 5.0
    output_limit: int = 65_536
    enabled: bool = True
    version: int = MCP_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if self.version != MCP_REGISTRY_VERSION:
            raise ValueError("unsupported MCP tool version")
        if not self.tool_id.strip() or not self.name.strip():
            raise ValueError("MCP tool id and name cannot be empty")
        if not self.description.strip() or not self.executor_id.strip():
            raise ValueError("MCP tool description and executor cannot be empty")
        if self.risk not in {"read_only", "reversible_ui", "file_write", "terminal"}:
            raise ValueError("unsupported MCP tool risk")
        if self.timeout_seconds <= 0 or self.output_limit < 1:
            raise ValueError("MCP tool budget must be positive")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("MCP tool input_schema must be a mapping")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "executor_id": self.executor_id,
            "source": self.source,
            "risk": self.risk,
            "timeout_seconds": self.timeout_seconds,
            "output_limit": self.output_limit,
            "enabled": self.enabled,
        }


class McpToolRegistry:
    """Content-addressed registry shared by Workbench and API projections."""

    def __init__(
        self,
        tools: Sequence[McpToolDescriptor] = (),
        *,
        revision: int = 1,
    ) -> None:
        if revision < 1:
            raise ValueError("MCP registry revision must be positive")
        self.revision = int(revision)
        self._tools: dict[str, McpToolDescriptor] = {}
        for descriptor in tools:
            self.register(descriptor)

    @classmethod
    def default(cls) -> McpToolRegistry:
        return cls(
            (
                McpToolDescriptor(
                    tool_id="mcp.local.workspace_summary",
                    name="Workspace summary",
                    description="Read a bounded directory summary from the active workspace.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "default": "."},
                        },
                        "additionalProperties": False,
                    },
                    executor_id="workspace.list",
                ),
            )
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpToolRegistry:
        """Restore a registry only when its content-addressed identity matches."""

        if payload.get("format") != MCP_REGISTRY_FORMAT:
            raise ValueError("unsupported MCP registry format")
        if int(payload.get("version", 0)) != MCP_REGISTRY_VERSION:
            raise ValueError("unsupported MCP registry version")
        raw_tools = payload.get("tools", ())
        if isinstance(raw_tools, (str, bytes)) or not isinstance(raw_tools, Sequence):
            raise ValueError("MCP registry tools must be a list")
        descriptors = tuple(
            McpToolDescriptor(
                tool_id=str(item["tool_id"]),
                name=str(item["name"]),
                description=str(item["description"]),
                input_schema=dict(item.get("input_schema") or {}),
                executor_id=str(item["executor_id"]),
                source=str(item.get("source", "seed.mcp.local")),
                risk=str(item.get("risk", "read_only")),
                timeout_seconds=float(item.get("timeout_seconds", 5.0)),
                output_limit=int(item.get("output_limit", 65_536)),
                enabled=bool(item.get("enabled", True)),
                version=int(item.get("version", MCP_REGISTRY_VERSION)),
            )
            for item in raw_tools
        )
        registry = cls(descriptors, revision=int(payload.get("revision", 0)))
        if str(payload.get("snapshot_id", "")) != registry.snapshot_id:
            raise ValueError("MCP registry snapshot digest mismatch")
        return registry

    @property
    def snapshot_id(self) -> str:
        return _digest(
            {
                "format": MCP_REGISTRY_FORMAT,
                "version": MCP_REGISTRY_VERSION,
                "revision": self.revision,
                "tools": [item.to_payload() for item in self.list_tools()],
            }
        )

    def register(self, descriptor: McpToolDescriptor) -> None:
        if descriptor.tool_id in self._tools:
            raise ValueError(f"duplicate MCP tool: {descriptor.tool_id}")
        self._tools[descriptor.tool_id] = descriptor

    def get(self, tool_id: str) -> McpToolDescriptor | None:
        return self._tools.get(str(tool_id).strip())

    def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        return tuple(self._tools[key] for key in sorted(self._tools))

    def validate_call(
        self,
        tool_id: Any,
        arguments: Any,
        *,
        registry_revision: Any = None,
    ) -> McpToolDescriptor:
        descriptor = self.get(str(tool_id or ""))
        if descriptor is None:
            raise KeyError("unknown MCP tool")
        if not descriptor.enabled:
            raise PermissionError("MCP tool is disabled")
        if registry_revision not in (None, "") and int(registry_revision) != self.revision:
            raise ValueError("MCP registry revision drifted")
        if not isinstance(arguments, Mapping):
            raise TypeError("MCP tool arguments must be an object")
        properties = descriptor.input_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError("MCP input schema properties must be an object")
        required = descriptor.input_schema.get("required", ())
        if isinstance(required, (str, bytes)) or not isinstance(required, Sequence):
            raise ValueError("MCP input schema required must be a list")
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"MCP tool arguments are missing: {', '.join(missing)}")
        if descriptor.input_schema.get("additionalProperties") is False:
            unknown = set(arguments) - set(properties)
            if unknown:
                raise ValueError("MCP tool arguments contain unknown fields")
        for name, value in arguments.items():
            schema = properties.get(name)
            if not isinstance(schema, Mapping) or "type" not in schema:
                continue
            expected = schema["type"]
            valid = {
                "object": isinstance(value, Mapping),
                "array": isinstance(value, (list, tuple)),
                "string": isinstance(value, str),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "null": value is None,
            }.get(expected)
            if valid is False:
                raise ValueError(f"MCP argument {name} must be {expected}")
        return descriptor

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": MCP_REGISTRY_FORMAT,
            "version": MCP_REGISTRY_VERSION,
            "revision": self.revision,
            "snapshot_id": self.snapshot_id,
            "tools": [item.to_payload() for item in self.list_tools()],
        }
