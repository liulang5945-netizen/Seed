"""Taiji-native read-only workbench routes.

This router is always available. It is deliberately separate from the
Legacy-dependent agent/workspace router, so opening and reading the IDE does
not depend on ``SEED_ENABLE_LEGACY``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.models import WorkbenchIntentRequest
from api.seed_runtime import get_seed_runtime
from seed_platform.workbench import (
    WorkbenchActionRequest,
    WorkbenchEnvironment,
    default_workspace_root,
)

router = APIRouter(prefix="/api/workbench", tags=["workbench"])


def _environment() -> WorkbenchEnvironment:
    runtime = get_seed_runtime()
    if runtime is not None:
        return runtime.workbench_environment
    return WorkbenchEnvironment(default_workspace_root())


def _read_only_result(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    environment = _environment()
    request = WorkbenchActionRequest(
        request_id=f"api:{tool_name}",
        intent_id=f"api:{tool_name}",
        capability_id=tool_name,
        parameters=parameters,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        source="seed.api.read_only",
    )
    policy = environment.policy_for(request)
    if policy.decision != "allow":
        raise HTTPException(status_code=409, detail=policy.to_payload())
    result = environment.execute_tool(tool_name, parameters)
    payload = environment.last_result
    if not result.success:
        raise HTTPException(status_code=400, detail=payload)
    return payload


@router.get("/capabilities")
def workbench_capabilities() -> dict[str, Any]:
    return _environment().capability_snapshot.to_payload()


@router.get("/files")
def workbench_files(path: str = ".") -> dict[str, Any]:
    return _read_only_result("workspace.list", {"path": path})


@router.get("/file")
def workbench_file(path: str) -> dict[str, Any]:
    return _read_only_result("workspace.read", {"path": path})


@router.get("/stat")
def workbench_stat(path: str = ".") -> dict[str, Any]:
    return _read_only_result("workspace.stat", {"path": path})


@router.get("/search")
def workbench_search(query: str, path: str = ".") -> dict[str, Any]:
    return _read_only_result("workspace.search", {"query": query, "path": path})


@router.get("/events")
def workbench_events() -> dict[str, Any]:
    runtime = get_seed_runtime()
    if runtime is None:
        return {"events": []}
    return {
        "events": [event.to_payload() for event in runtime.workbench_audit.events],
    }


@router.post("/execute")
def execute_workbench_intent(request: WorkbenchIntentRequest) -> dict[str, Any]:
    runtime = get_seed_runtime()
    if runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    from taiji import ActionIntent

    try:
        intent = ActionIntent(
            intent_id=request.intent_id,
            kind=request.kind,
            parameters=request.parameters,
            source_goal_id=request.source_goal_id,
            expected_outcome=request.expected_outcome,
            confidence=request.confidence,
            tick=request.tick,
        )
        return runtime.execute_workbench_intent(
            intent,
            snapshot_id=request.snapshot_id,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
