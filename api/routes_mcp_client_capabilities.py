"""Seed-owned MCP client-capability shadow routes.

These routes expose only candidate, policy and digest-only shadow lifecycle
records.  They never connect to an MCP server, dispatch a tool, load an
executor, or activate a client extension.  The current Workbench MCP snapshot
is rebound before each operation so stale candidates fail closed.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.seed_runtime import get_seed_runtime
from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_client_capability_registry import (
    McpClientCapabilityShadowRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment, default_workspace_root

router = APIRouter(
    prefix="/api/mcp-client-capabilities",
    tags=["mcp-client-capabilities"],
)
_registry: McpClientCapabilityShadowRegistry | None = None


def _environment() -> WorkbenchEnvironment:
    runtime = get_seed_runtime()
    if runtime is not None:
        return runtime.workbench_environment
    return WorkbenchEnvironment(default_workspace_root())


def _shadow_registry() -> tuple[McpClientCapabilityShadowRegistry, str]:
    global _registry
    environment = _environment()
    current_snapshot_id = environment.mcp_registry.snapshot_id
    if _registry is None:
        _registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
            environment.mcp_registry,
            parent_checkpoint_id=f"checkpoint:mcp-client-capability:{current_snapshot_id}",
        )
    elif _registry.current_mcp_registry_snapshot_id != current_snapshot_id:
        _registry.bind_mcp_snapshot(
            current_snapshot_id,
            expected_current_snapshot_id=_registry.current_mcp_registry_snapshot_id,
        )
    return _registry, current_snapshot_id


def _error(exc: Exception) -> None:
    message = str(exc)
    if isinstance(exc, (KeyError, TypeError)):
        raise HTTPException(status_code=400, detail=message) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=409, detail=message) from exc
    if isinstance(exc, ValueError):
        status = 409 if "stale" in message or "already" in message else 400
        raise HTTPException(status_code=status, detail=message) from exc
    raise HTTPException(status_code=500, detail="MCP client capability shadow operation failed") from exc


@router.get("")
def mcp_client_capability_status() -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    return {
        "status": "ok",
        "format": "seed-mcp-client-capability-shadow-registry-v1",
        "snapshot_id": registry.snapshot_id,
        "mcp_registry_snapshot_id": current_snapshot_id,
        "parent_checkpoint_id": registry.parent_checkpoint_id,
        "records": [item.to_payload() for item in registry.records],
        "shadow_validated": [item.to_payload() for item in registry.shadow_validated],
        "client_activation": "not_available_in_e6_1",
    }


@router.post("/proposals")
def propose_mcp_client_capability(request: dict[str, Any]) -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    try:
        candidate = McpCapabilityInheritanceCandidate.from_payload(
            request.get("candidate") or {}
        )
        policy = McpCapabilityInheritancePolicy.from_payload(request.get("policy") or {})
        record = registry.propose(
            candidate,
            policy,
            expected_current_snapshot_id=current_snapshot_id,
        )
    except Exception as exc:
        _error(exc)
    return {"status": record.state, "registry": registry.snapshot_id, "record": record.to_payload()}


@router.post("/{candidate_digest}/shadow")
def record_mcp_client_capability_shadow(
    candidate_digest: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    try:
        observation = McpCapabilityShadowObservation.from_payload(
            request.get("observation") or {}
        )
        record = registry.record_shadow(
            candidate_digest,
            observation,
            expected_current_snapshot_id=current_snapshot_id,
            approval_id=str(request.get("approval_id", "")),
        )
    except Exception as exc:
        _error(exc)
    return {"status": record.state, "registry": registry.snapshot_id, "record": record.to_payload()}


@router.post("/{candidate_digest}/rollback")
def rollback_mcp_client_capability(candidate_digest: str) -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    try:
        record = registry.rollback(
            candidate_digest,
            expected_current_snapshot_id=current_snapshot_id,
        )
    except Exception as exc:
        _error(exc)
    return {"status": record.state, "registry": registry.snapshot_id, "record": record.to_payload()}


__all__ = ["router"]
