"""Taiji-native workbench routes.

This router is always available. It is deliberately separate from the
Legacy-dependent agent/workspace router, so native IDE reads and controlled
workspace mutations do not depend on ``SEED_ENABLE_LEGACY``.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from api.models import (
    WorkbenchIntentRequest,
    WorkbenchLoopExecuteRequest,
    WorkbenchLoopPreflightRequest,
)
from api.seed_runtime import get_seed_runtime
from seed_platform.settings import update_settings
from seed_platform.workbench import (
    WorkbenchActionRequest,
    WorkbenchEnvironment,
    default_workspace_root,
    validate_workspace_root,
)

router = APIRouter(prefix="/api/workbench", tags=["workbench"])


def _environment() -> WorkbenchEnvironment:
    runtime = get_seed_runtime()
    if runtime is not None:
        return runtime.workbench_environment
    return WorkbenchEnvironment(default_workspace_root())


def _read_only_result(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    environment = _environment()
    runtime = get_seed_runtime()
    if runtime is not None:
        from taiji import ActionIntent

        request_id = f"api:{tool_name}:{uuid4().hex}"
        result = runtime.execute_workbench_intent(
            ActionIntent(
                intent_id=request_id,
                kind=tool_name,
                parameters=parameters,
                confidence=1.0,
                tick=runtime.model.tick,
            ),
            snapshot_id=environment.capability_snapshot.snapshot_id,
        )
        outcome = result["outcome"]
        if outcome["status"] != "success":
            raise HTTPException(status_code=400, detail=outcome)
        return dict(outcome.get("result") or {})

    request_id = f"api:{tool_name}:{uuid4().hex}"
    request = WorkbenchActionRequest(
        request_id=request_id,
        intent_id=request_id,
        capability_id=tool_name,
        parameters=parameters,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        source="seed.api.read_only",
        mcp_registry_snapshot_id=(
            environment.mcp_registry.snapshot_id if tool_name in {"mcp.list", "mcp.invoke"} else ""
        ),
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
    environment = _environment()
    payload = environment.capability_snapshot.to_payload()
    payload["workspace_root"] = str(environment.root)
    payload["programming_languages"] = (
        environment.programming_language_registry.public_descriptors()
    )
    payload["programming_language_registry_revision"] = (
        environment.programming_language_registry.revision
    )
    payload["mcp_registry"] = environment.mcp_registry.to_payload()
    return payload


@router.get("/workspace")
def workbench_workspace() -> dict[str, Any]:
    """Return the active workspace root from the native environment."""

    return {"status": "ok", "path": str(_environment().root)}


@router.post("/workspace")
def set_workbench_workspace(request: dict[str, Any]) -> dict[str, Any]:
    """Persist a validated workspace root for the native Workbench."""

    try:
        path = validate_workspace_root(request.get("path"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    update_settings({"workspace_path": str(path)})
    return {"status": "ok", "path": str(path)}


@router.get("/programming-languages")
def workbench_programming_languages() -> dict[str, Any]:
    """Return the backend-owned programming-language registry."""

    environment = _environment()
    return {
        "format": "seed-programming-language-v1",
        "version": 1,
        "revision": environment.programming_language_registry.revision,
        "languages": environment.programming_language_registry.public_descriptors(),
    }


@router.get("/mcp")
def workbench_mcp_tools() -> dict[str, Any]:
    """Return the native MCP-shaped registry through the workbench boundary."""

    return _read_only_result("mcp.list", {})


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


@router.get("/programming-language")
def workbench_programming_language(path: str, lsp_language_id: str | None = None) -> dict[str, Any]:
    return _read_only_result(
        "workspace.programming_language.resolve",
        {"path": path, "lsp_language_id": lsp_language_id},
    )


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
            approval_token=request.approval_token,
            mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/preview")
def preview_workbench_intent(request: WorkbenchIntentRequest) -> dict[str, Any]:
    """Validate a mutating action and return a short-lived approval token."""

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
        return runtime.preview_workbench_intent(
            intent,
            snapshot_id=request.snapshot_id,
            mcp_registry_snapshot_id=request.mcp_registry_snapshot_id,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/loop/preflight")
def preflight_workbench_loop(
    request: WorkbenchLoopPreflightRequest,
) -> dict[str, Any]:
    """Validate a bounded native loop without executing any step."""

    runtime = get_seed_runtime()
    if runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    from taiji import ActionIntent

    try:
        environment = runtime.workbench_environment
        requests: list[WorkbenchActionRequest] = []
        for item in request.intents:
            intent = ActionIntent(
                intent_id=item.intent_id,
                kind=item.kind,
                parameters=item.parameters,
                source_goal_id=item.source_goal_id,
                expected_outcome=item.expected_outcome,
                confidence=item.confidence,
                tick=item.tick,
            )
            requests.append(
                WorkbenchActionRequest.from_action_intent(
                    intent,
                    snapshot_id=item.snapshot_id,
                    approval_token=item.approval_token,
                    mcp_registry_snapshot_id=(
                        item.mcp_registry_snapshot_id
                        or (
                            environment.mcp_registry.snapshot_id
                            if item.kind.startswith("mcp.")
                            else ""
                        )
                    ),
                )
            )
        return runtime.preflight_workbench_loop(
            requests,
            loop_id=request.loop_id,
            max_steps=request.max_steps,
            max_budget_units=request.max_budget_units,
            on_failure=request.on_failure,
            checkpoint_boundary=request.checkpoint_boundary,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/loop/execute")
def execute_workbench_loop(
    request: WorkbenchLoopExecuteRequest,
) -> dict[str, Any]:
    """Execute an accepted native loop and checkpoint after every attempted step."""

    runtime = get_seed_runtime()
    if runtime is None:
        raise HTTPException(status_code=409, detail="Seed runtime is not active")
    from taiji import ActionIntent

    try:
        environment = runtime.workbench_environment
        intents: list[ActionIntent] = []
        requests: list[WorkbenchActionRequest] = []
        for item in request.intents:
            intent = ActionIntent(
                intent_id=item.intent_id,
                kind=item.kind,
                parameters=item.parameters,
                source_goal_id=item.source_goal_id,
                expected_outcome=item.expected_outcome,
                confidence=item.confidence,
                tick=item.tick,
            )
            intents.append(intent)
            requests.append(
                WorkbenchActionRequest.from_action_intent(
                    intent,
                    snapshot_id=item.snapshot_id,
                    approval_token=item.approval_token,
                    mcp_registry_snapshot_id=(
                        item.mcp_registry_snapshot_id
                        or (
                            environment.mcp_registry.snapshot_id
                            if item.kind.startswith("mcp.")
                            else ""
                        )
                    ),
                )
            )
        return runtime.execute_preflighted_workbench_loop(
            intents,
            requests,
            loop_id=request.loop_id,
            preflight_id=request.preflight_id,
            max_steps=request.max_steps,
            max_budget_units=request.max_budget_units,
            on_failure=request.on_failure,
            checkpoint_boundary=request.checkpoint_boundary,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
