"""Seed-owned client extension lifecycle routes.

The API exposes the declarative client-body snapshot only.  It does not accept
source paths, import paths, commands or executable plugin payloads.  Actual
desktop/Vue slot rendering remains a client concern and receives only the
content-addressed snapshot produced here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.seed_runtime import get_seed_runtime
from seed_platform.client_extension_host import (
    ClientExtensionHost,
    ClientPluginManifest,
    ExtensionHostError,
)
from seed_platform.workbench import WorkbenchEnvironment, default_workspace_root

router = APIRouter(prefix="/api/client-extensions", tags=["client-extensions"])
_host: ClientExtensionHost | None = None
_prepared: dict[str, Any] = {}


def _environment() -> WorkbenchEnvironment:
    runtime = get_seed_runtime()
    if runtime is not None:
        return runtime.workbench_environment
    return WorkbenchEnvironment(default_workspace_root())


def _client_host() -> ClientExtensionHost:
    global _host
    if _host is None:
        capability_snapshot = _environment().capability_snapshot
        _host = ClientExtensionHost(
            capability_snapshot_id=capability_snapshot.snapshot_id,
            parent_checkpoint_id=f"checkpoint:client-extension:{capability_snapshot.snapshot_id}",
        )
    return _host


def _raise_extension_error(exc: Exception) -> None:
    if isinstance(exc, ExtensionHostError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail="client extension operation failed") from exc


def _current_capability_context() -> tuple[str, tuple[str, ...]]:
    snapshot = _environment().capability_snapshot
    return snapshot.snapshot_id, tuple(item.capability_id for item in snapshot.capabilities)


@router.get("")
def client_extensions() -> dict[str, Any]:
    host = _client_host()
    capability_snapshot_id, capability_ids = _current_capability_context()
    return {
        "status": "ok",
        "snapshot": host.snapshot.to_payload(),
        "policy": {**host.policy.to_payload(), "policy_digest": host.policy.policy_digest},
        "active": [item.to_payload() for item in host.active_manifests],
        "dependency_health": host.dependency_health,
        "lifecycle": [item.to_payload() for item in host.lifecycle_records],
        "capability_snapshot_id": capability_snapshot_id,
        "available_capabilities": list(capability_ids),
    }


@router.post("/prepare")
def prepare_client_extensions(request: dict[str, Any]) -> dict[str, Any]:
    host = _client_host()
    current_capability_snapshot_id, capability_ids = _current_capability_context()
    requested_snapshot_id = str(
        request.get("capability_snapshot_id") or current_capability_snapshot_id
    )
    if requested_snapshot_id != current_capability_snapshot_id:
        raise HTTPException(status_code=409, detail="capability snapshot is stale")
    raw_manifests = request.get("manifests", ())
    if isinstance(raw_manifests, (str, bytes)) or not isinstance(raw_manifests, list):
        raise HTTPException(status_code=400, detail="manifests must be a JSON array")
    try:
        manifests = tuple(ClientPluginManifest.from_payload(item) for item in raw_manifests)
        prepared = host.prepare(
            manifests,
            capability_snapshot_id=current_capability_snapshot_id,
            available_capabilities=capability_ids,
            dependency_health=request.get("dependency_health"),
            states=request.get("states"),
        )
    except Exception as exc:  # normalize host contract failures at API boundary
        _raise_extension_error(exc)
    _prepared[prepared.prepared_digest] = prepared
    return {
        "status": "prepared",
        "prepared_id": prepared.prepared_digest,
        "snapshot": prepared.target_snapshot.to_payload(),
    }


@router.post("/commit")
def commit_client_extensions(request: dict[str, Any]) -> dict[str, Any]:
    prepared_id = str(request.get("prepared_id", "")).strip()
    if not prepared_id:
        raise HTTPException(status_code=400, detail="prepared_id cannot be empty")
    prepared = _prepared.get(prepared_id)
    if prepared is None:
        raise HTTPException(status_code=404, detail="prepared client extension snapshot not found")
    try:
        snapshot = _client_host().commit(prepared)
    except Exception as exc:
        _raise_extension_error(exc)
    _prepared.pop(prepared_id, None)
    return {"status": "committed", "snapshot": snapshot.to_payload()}


@router.post("/dependency")
def report_client_dependency(request: dict[str, Any]) -> dict[str, Any]:
    service = str(request.get("service", "")).strip()
    if not service or not isinstance(request.get("healthy"), bool):
        raise HTTPException(status_code=400, detail="service and boolean healthy are required")
    try:
        affected = _client_host().report_dependency(service, request["healthy"])
    except Exception as exc:
        _raise_extension_error(exc)
    return {
        "status": "ok",
        "affected": list(affected),
        "snapshot": _client_host().snapshot.to_payload(),
    }


@router.post("/rollback")
def rollback_client_extensions(request: dict[str, Any]) -> dict[str, Any]:
    snapshot_id = str(request.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise HTTPException(status_code=400, detail="snapshot_id cannot be empty")
    try:
        snapshot = _client_host().rollback(snapshot_id)
    except Exception as exc:
        _raise_extension_error(exc)
    return {"status": "rolled_back", "snapshot": snapshot.to_payload()}


@router.post("/{plugin_id}/call/begin")
def begin_client_extension_call(plugin_id: str) -> dict[str, Any]:
    try:
        _client_host().begin_call(plugin_id)
    except Exception as exc:
        _raise_extension_error(exc)
    return {"status": "in_flight", "plugin_id": plugin_id, "count": _client_host().inflight(plugin_id)}


@router.post("/{plugin_id}/call/end")
def end_client_extension_call(plugin_id: str) -> dict[str, Any]:
    try:
        _client_host().end_call(plugin_id)
    except Exception as exc:
        _raise_extension_error(exc)
    return {"status": "settled", "plugin_id": plugin_id, "count": _client_host().inflight(plugin_id)}


@router.post("/{plugin_id}/retire")
def retire_client_extension(plugin_id: str) -> dict[str, Any]:
    try:
        snapshot = _client_host().retire(plugin_id)
    except Exception as exc:
        _raise_extension_error(exc)
    return {"status": "retired", "snapshot": snapshot.to_payload()}


@router.post("/{plugin_id}/quarantine")
def quarantine_client_extension(plugin_id: str, request: dict[str, Any]) -> dict[str, Any]:
    reason = str(request.get("reason", "")).strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason cannot be empty")
    try:
        snapshot = _client_host().quarantine(plugin_id, reason=reason)
    except Exception as exc:
        _raise_extension_error(exc)
    return {"status": "quarantined", "snapshot": snapshot.to_payload()}
