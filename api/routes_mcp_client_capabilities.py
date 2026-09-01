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
from seed_platform.client_extension_host import ClientExtensionHost
from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_client_activation_dry_run import run_client_activation_dry_run
from seed_platform.mcp_client_capability_registry import (
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_client_connection_authorization import (
    McpClientConnectionAuthorizationStore,
    authorize_mcp_client_connection,
)
from seed_platform.workbench import WorkbenchEnvironment, default_workspace_root

router = APIRouter(
    prefix="/api/mcp-client-capabilities",
    tags=["mcp-client-capabilities"],
)
_registry: McpClientCapabilityShadowRegistry | None = None
_authorization_store: McpClientConnectionAuthorizationStore | None = None


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


def _authorization_store_for_current() -> McpClientConnectionAuthorizationStore:
    global _authorization_store
    environment = _environment()
    mcp_snapshot_id = environment.mcp_registry.snapshot_id
    client_snapshot_id = environment.capability_snapshot.snapshot_id
    if (
        _authorization_store is None
        or _authorization_store.mcp_registry_snapshot_id != mcp_snapshot_id
        or _authorization_store.client_capability_snapshot_id != client_snapshot_id
    ):
        _authorization_store = McpClientConnectionAuthorizationStore(
            mcp_registry_snapshot_id=mcp_snapshot_id,
            client_capability_snapshot_id=client_snapshot_id,
        )
    return _authorization_store


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
    client_capability_snapshot_id = _environment().capability_snapshot.snapshot_id
    authorization_store = _authorization_store_for_current()
    return {
        "status": "ok",
        "format": "seed-mcp-client-capability-shadow-registry-v1",
        "snapshot_id": registry.snapshot_id,
        "mcp_registry_snapshot_id": current_snapshot_id,
        "parent_checkpoint_id": registry.parent_checkpoint_id,
        "records": [item.to_payload() for item in registry.records],
        "shadow_validated": [item.to_payload() for item in registry.shadow_validated],
        "activation_proposals": [item.to_payload() for item in registry.activation_proposals],
        "client_capability_snapshot_id": client_capability_snapshot_id,
        "connection_authorizations": [
            item.to_payload() for item in authorization_store.records
        ],
        "client_activation": "authorization_only_in_e6_4",
        "connection": "not_attempted",
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


@router.post("/{candidate_digest}/activation-proposals")
def propose_mcp_client_capability_activation(
    candidate_digest: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    client_snapshot_id = _environment().capability_snapshot.snapshot_id
    requested_client_snapshot_id = str(
        request.get("client_capability_snapshot_id") or client_snapshot_id
    )
    if requested_client_snapshot_id != client_snapshot_id:
        raise HTTPException(status_code=409, detail="client capability snapshot is stale")
    try:
        proposal = registry.propose_activation(
            candidate_digest,
            client_capability_snapshot_id=client_snapshot_id,
            expected_current_snapshot_id=current_snapshot_id,
        )
    except Exception as exc:
        _error(exc)
    return {
        "status": proposal.state,
        "registry": registry.snapshot_id,
        "activation": "proposal_only",
        "proposal": proposal.to_payload(),
    }


@router.post("/{candidate_digest}/activation-dry-run")
def dry_run_mcp_client_capability_activation(
    candidate_digest: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    registry, current_snapshot_id = _shadow_registry()
    client_snapshot_id = _environment().capability_snapshot.snapshot_id
    requested_client_snapshot_id = str(
        request.get("client_capability_snapshot_id") or client_snapshot_id
    )
    if requested_client_snapshot_id != client_snapshot_id:
        raise HTTPException(status_code=409, detail="client capability snapshot is stale")
    record = registry.get(candidate_digest)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown MCP client capability candidate")
    proposals = tuple(
        item
        for item in registry.activation_proposals
        if item.candidate_digest == record.candidate_digest and item.state == "proposed"
    )
    if not proposals:
        raise HTTPException(status_code=409, detail="activation proposal is required")
    try:
        dry_run = run_client_activation_dry_run(
            ClientExtensionHost(
                capability_snapshot_id=client_snapshot_id,
                parent_checkpoint_id=f"checkpoint:mcp-client-dry-run:{current_snapshot_id}",
            ),
            record.candidate,
            proposals[-1],
            client_capability_snapshot_id=client_snapshot_id,
            available_capabilities=tuple(
                item.tool_id for item in record.candidate.tool_contracts
            ),
        )
    except Exception as exc:
        _error(exc)
    return {
        "status": "dry_run",
        "activation": "not_committed",
        "dry_run": dry_run.to_payload(),
    }


@router.post("/{candidate_digest}/connection-authorization")
def authorize_mcp_client_capability_connection(
    candidate_digest: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Record explicit authorization without connecting or activating."""

    registry, current_snapshot_id = _shadow_registry()
    environment = _environment()
    client_snapshot_id = environment.capability_snapshot.snapshot_id
    requested_client_snapshot_id = str(request.get("client_capability_snapshot_id", "")).strip()
    if requested_client_snapshot_id != client_snapshot_id:
        raise HTTPException(status_code=409, detail="client capability snapshot is stale")
    record = registry.get(candidate_digest)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown MCP client capability candidate")
    proposal_id = str(request.get("proposal_id", "")).strip()
    if not proposal_id:
        raise HTTPException(status_code=400, detail="explicit activation proposal_id is required")
    proposal = next(
        (
            item
            for item in registry.activation_proposals
            if item.proposal_id == proposal_id
            and item.candidate_digest == record.candidate_digest
        ),
        None,
    )
    if proposal is None:
        raise HTTPException(status_code=404, detail="unknown activation proposal")
    dry_run_digest = str(request.get("dry_run_digest", "")).strip()
    if not dry_run_digest:
        raise HTTPException(status_code=400, detail="activation dry-run digest is required")
    try:
        dry_run = run_client_activation_dry_run(
            ClientExtensionHost(
                capability_snapshot_id=client_snapshot_id,
                parent_checkpoint_id=f"checkpoint:mcp-client-dry-run:{current_snapshot_id}",
            ),
            record.candidate,
            proposal,
            client_capability_snapshot_id=client_snapshot_id,
            available_capabilities=tuple(
                item.tool_id for item in record.candidate.tool_contracts
            ),
        )
        if dry_run.dry_run_digest != dry_run_digest:
            raise ValueError("activation dry-run digest mismatch")
        decision = authorize_mcp_client_connection(
            record.candidate,
            proposal,
            record.policy,
            current_mcp_registry_snapshot_id=current_snapshot_id,
            current_client_capability_snapshot_id=client_snapshot_id,
            network_scopes=request.get("network_scopes", ()),
            credential_refs=request.get("credential_refs", ()),
            approval_id=str(request.get("approval_id", "")),
            issuer=str(request.get("issuer", "")),
            issued_at_epoch=request.get("issued_at_epoch"),
            expires_at_epoch=request.get("expires_at_epoch"),
            max_lifetime_seconds=int(request.get("max_lifetime_seconds", 3_600)),
        )
    except Exception as exc:
        _error(exc)
    if not decision.passed or decision.authorization is None:
        raise HTTPException(status_code=409, detail=decision.to_payload())
    authorization_store = _authorization_store_for_current()
    issued = authorization_store.issue(decision.authorization)
    return {
        "status": "authorized",
        "connection": "not_attempted",
        "activation": "not_committed",
        "dry_run_digest": dry_run.dry_run_digest,
        "authorization": issued.to_payload(),
        "authorization_store_snapshot_id": authorization_store.snapshot_id,
    }


@router.post("/connection-authorizations/{authorization_id}/revoke")
def revoke_mcp_client_connection_authorization(
    authorization_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Revoke metadata only; no client or MCP operation is performed."""

    authorization_store = _authorization_store_for_current()
    try:
        revoked = authorization_store.revoke(
            authorization_id,
            reason=str(request.get("reason", "")),
        )
    except Exception as exc:
        _error(exc)
    return {
        "status": "revoked",
        "connection": "not_attempted",
        "authorization": revoked.to_payload(),
        "authorization_store_snapshot_id": authorization_store.snapshot_id,
    }


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
