from __future__ import annotations

import hashlib
import sys
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.mcp_registry import McpToolDescriptor, McpToolRegistry
from seed_platform.programming_languages import ProgrammingLanguageRegistry
from seed_platform.workbench import (
    CapabilitySnapshot,
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import (
    ActionIntent,
    ContentPlan,
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveDecision,
    TSKV8Adapter,
)


def _grounded_read_candidate(*, tick: int = 0, kind: str = "workspace.read", parameters=None):
    intent = ActionIntent(
        intent_id="candidate-read:intent",
        kind=kind,
        parameters={"path": "README.md"} if parameters is None else parameters,
        confidence=1.0,
        tick=tick,
    )
    content = ContentPlan(
        content_id="candidate-read:content",
        intent_id=intent.intent_id,
        intent_kind=intent.kind,
        semantic_slots={"parameters": dict(intent.parameters)},
        confidence=1.0,
        provenance="affordance-derived",
        tick=tick,
    )
    return ExecutiveCandidate(
        candidate_id="candidate-read",
        action_intent=intent,
        content_plan=content,
        features=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        provenance="affordance-derived/learned",
        source_percept_id="percept-read",
        source_affordance_id="affordance-read",
    )


def test_workbench_snapshot_is_content_addressed() -> None:
    snapshot = CapabilitySnapshot.default()
    restored = CapabilitySnapshot.from_payload(snapshot.to_payload())

    assert restored == snapshot
    assert snapshot.get("workspace.list") is not None
    assert snapshot.get("editor.diagnostics.read").enabled is False  # type: ignore[union-attr]
    assert snapshot.get("workspace.programming_language.resolve") is not None
    assert snapshot.get("editor.set_language") is not None
    assert snapshot.get("workspace.apply_patch") is not None
    assert snapshot.get("terminal.run") is not None
    assert snapshot.get("mcp.list") is not None
    assert snapshot.get("mcp.invoke") is not None

    tampered = snapshot.to_payload()
    tampered["capabilities"][0]["description"] = "tampered"
    try:
        CapabilitySnapshot.from_payload(tampered)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:  # pragma: no cover - protects the red-path contract
        raise AssertionError("tampered workbench snapshot was accepted")


def test_workbench_snapshot_projects_explicit_read_only_affordances() -> None:
    snapshot = CapabilitySnapshot.default()
    bindings = {
        "workspace.list": {"path": "."},
        "workspace.read": {"path": "README.md"},
    }

    affordances = snapshot.to_taiji_affordances(bindings)

    assert [item.action_kind for item in affordances] == [
        "workspace.list",
        "workspace.read",
    ]
    assert affordances[0].parameters == (("path", "."),)
    assert f"workbench-snapshot:{snapshot.snapshot_id}" in affordances[1].grounding_lineage
    assert f"workbench-capability-revision:{snapshot.revision}" in affordances[1].grounding_lineage
    assert snapshot.to_taiji_affordances(bindings)[1].affordance_id == affordances[1].affordance_id

    with pytest.raises(ValueError, match="read-only"):
        snapshot.to_taiji_affordances({"workspace.apply_patch": {"path": "README.md"}})
    with pytest.raises(ValueError, match="undeclared parameters"):
        snapshot.to_taiji_affordances(
            {"workspace.read": {"path": "README.md", "tool": "workspace.read"}}
        )


def test_adapter_accepts_workbench_projection_without_owning_capabilities() -> None:
    adapter = TSKV8Adapter()
    affordances = CapabilitySnapshot.default().to_taiji_affordances(
        {"workspace.list": {"path": "."}}
    )

    world = adapter.set_world_affordances(affordances)

    assert world.affordances[0].action_kind == "workspace.list"
    assert world.affordances[0].parameters == (("path", "."),)
    assert any(
        item.startswith("workbench-snapshot:") for item in world.affordances[0].grounding_lineage
    )


def test_runtime_projects_workbench_evidence_into_current_taiji_world(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="taiji-projection"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id

    result = runtime.project_workbench_affordances(
        snapshot_id=snapshot_id,
        parameter_bindings={"workspace.list": {"path": "."}},
    )

    assert result["snapshot_id"] == snapshot_id
    assert result["revision"] == runtime.workbench_environment.capability_snapshot.revision
    assert result["affordances"][0]["action_kind"] == "workspace.list"
    assert (
        runtime.model.architecture.cognitive_snapshot().world.affordances[0].action_kind
        == "workspace.list"
    )


def test_workspace_evidence_becomes_taiji_world_event_and_invalidates_affordances(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "README.md").write_bytes(b"fresh workspace evidence\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-world-evidence"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id
    runtime.project_workbench_affordances(
        snapshot_id=snapshot_id,
        parameter_bindings={"workspace.read": {"path": "README.md"}},
    )
    intent = ActionIntent(
        intent_id="intent-evidence-read",
        kind="workspace.read",
        parameters={"path": "README.md"},
        confidence=1.0,
        tick=runtime.model.tick,
    )

    result = runtime.execute_workbench_intent(intent, snapshot_id=snapshot_id, learn=False)

    event = result["taiji_world_event"]
    assert event["kind"] == "workbench.evidence"
    assert event["tick"] == runtime.model.architecture.cognitive_snapshot().world.tick
    world = runtime.model.architecture.cognitive_snapshot().world
    assert world.events[-1].event_id == event["event_id"]
    assert world.events[-1].kind == "workbench.evidence"
    attributes = dict(world.events[-1].attributes)
    assert attributes["result"]["content"] == "fresh workspace evidence\n"
    assert attributes["after_state_digest"]
    assert world.affordances == ()

    restored = TSKV8Adapter.from_native_checkpoint(runtime.model.architecture.native_checkpoint())
    restored_world = restored.cognitive_snapshot().world
    assert restored_world.events[-1].event_id == event["event_id"]
    assert restored_world.events[-1].to_payload() == world.events[-1].to_payload()
    assert restored_world.affordances == ()


def test_latest_workspace_evidence_reprojects_and_rejects_old_candidate(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "README.md").write_bytes(b"reproject me\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-reproject"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id
    initial = runtime.project_workbench_affordances(
        snapshot_id=snapshot_id,
        parameter_bindings={"workspace.read": {"path": "README.md"}},
    )
    old_affordance_id = initial["affordances"][0]["affordance_id"]
    old_candidate = replace(
        _grounded_read_candidate(tick=runtime.model.tick),
        source_affordance_id=old_affordance_id,
    )
    intent = ActionIntent(
        intent_id="intent-reproject-read",
        kind="workspace.read",
        parameters={"path": "README.md"},
        confidence=1.0,
        tick=runtime.model.tick,
    )

    runtime.execute_workbench_intent(intent, snapshot_id=snapshot_id, learn=False)
    world_after_evidence = runtime.model.architecture.cognitive_snapshot().world
    stale_candidate = replace(
        old_candidate,
        action_intent=replace(old_candidate.action_intent, tick=world_after_evidence.tick),
    )
    stale = environment.admit_taiji_candidate(
        stale_candidate,
        snapshot_id=snapshot_id,
        current_tick=world_after_evidence.tick,
        current_affordance_ids=tuple(
            item.affordance_id for item in world_after_evidence.affordances
        ),
    )
    assert stale.accepted is False
    assert stale.reason_code == "stale_taiji_affordance"

    reprojected = runtime.reproject_workbench_from_latest_evidence(snapshot_id=snapshot_id)
    new_affordance_id = reprojected["affordances"][0]["affordance_id"]
    assert new_affordance_id != old_affordance_id
    assert reprojected["evidence"]["event_id"] == world_after_evidence.events[-1].event_id
    reprojected_world = runtime.model.architecture.cognitive_snapshot().world
    assert any(
        item == "workbench-evidence:" + reprojected["evidence"]["event_id"]
        for item in reprojected_world.affordances[0].grounding_lineage
    )
    fresh_candidate = replace(
        _grounded_read_candidate(
            tick=runtime.model.architecture.cognitive_snapshot().tick,
        ),
        source_affordance_id=new_affordance_id,
    )
    fresh_world = runtime.model.architecture.cognitive_snapshot().world
    admitted = environment.admit_taiji_candidate(
        fresh_candidate,
        snapshot_id=snapshot_id,
        current_tick=fresh_world.tick,
        current_affordance_ids=tuple(item.affordance_id for item in fresh_world.affordances),
    )
    assert admitted.accepted is True
    assert "workbench-evidence:" + reprojected["evidence"]["event_id"] in {
        item for item in fresh_world.affordances[0].grounding_lineage
    }


def test_workspace_evidence_event_order_survives_checkpoint_continuation(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "README.md").write_bytes(b"searchable workspace evidence\n")
    (tmp_path / "notes.txt").write_bytes(b"another searchable entry\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-evidence-order"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    calls = (
        ("workspace.list", {"path": "."}),
        ("workspace.read", {"path": "README.md"}),
        ("workspace.stat", {"path": "README.md"}),
        ("workspace.search", {"query": "searchable", "path": "."}),
    )

    for index, (kind, parameters) in enumerate(calls):
        runtime.execute_workbench_intent(
            ActionIntent(
                intent_id=f"intent-evidence-order-{index}",
                kind=kind,
                parameters=parameters,
                confidence=1.0,
                tick=runtime.model.tick,
            ),
            snapshot_id=snapshot_id,
            learn=False,
        )

    world = runtime.model.architecture.cognitive_snapshot().world
    evidence_events = [item for item in world.events if item.kind == "workbench.evidence"]
    assert [dict(item.attributes)["capability_id"] for item in evidence_events] == [
        item[0] for item in calls
    ]
    assert [item.tick for item in evidence_events] == sorted(item.tick for item in evidence_events)
    checkpoint = tmp_path / "evidence-order.pt"
    runtime.save(checkpoint)
    restored = SeedRuntime.load(checkpoint)
    restored_events = [
        item
        for item in restored.model.architecture.cognitive_snapshot().world.events
        if item.kind == "workbench.evidence"
    ]
    assert [item.event_id for item in restored_events] == [
        item.event_id for item in evidence_events
    ]
    assert [dict(item.attributes)["after_state_digest"] for item in restored_events] == [
        dict(item.attributes)["after_state_digest"] for item in evidence_events
    ]


def test_runtime_native_executive_canary_needs_no_manual_candidate(tmp_path, monkeypatch) -> None:
    (tmp_path / "README.md").write_bytes(b"native executive canary\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-native-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    runtime.model.architecture.observe(65, learn=False)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    runtime.project_workbench_affordances(
        snapshot_id=snapshot_id,
        parameter_bindings={"workspace.list": {"path": "."}},
    )

    admission = runtime.admit_taiji_workbench_task(snapshot_id=snapshot_id)
    execution = runtime.execute_taiji_workbench_task(snapshot_id=snapshot_id, learn=False)

    assert admission["admission"]["accepted"] is True
    assert execution["admission"]["accepted"] is True
    assert execution["execution"]["outcome"]["result"]["entries"][0]["path"] == "README.md"
    assert runtime.model.architecture.cognitive_snapshot().world.affordances == ()

    checkpoint = tmp_path / "native-canary.pt"
    runtime.save(checkpoint)
    runtime = SeedRuntime.load(checkpoint)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    assert runtime.model.architecture.cognitive_snapshot().world.affordances == ()

    reprojected = runtime.reproject_workbench_from_latest_evidence(snapshot_id=snapshot_id)
    assert {item["action_kind"] for item in reprojected["affordances"]} == {
        "workspace.read",
        "workspace.stat",
    }
    resumed = runtime.execute_taiji_workbench_task(snapshot_id=snapshot_id, learn=False)
    assert resumed["admission"]["accepted"] is True
    assert resumed["execution"]["outcome"]["status"] == "success"
    evidence_capabilities = [
        dict(item.attributes)["capability_id"]
        for item in runtime.model.architecture.cognitive_snapshot().world.events
        if item.kind == "workbench.evidence"
    ]
    assert evidence_capabilities == ["workspace.list", "workspace.read"]


def test_read_only_environment_reads_and_rejects_escape(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    with (tmp_path / "src" / "main.py").open("w", encoding="utf-8", newline="") as handle:
        handle.write("print('seed')\n")
    environment = WorkbenchEnvironment(tmp_path)

    listed = environment.execute_tool("workspace.list", {"path": "src"})
    assert listed.success is True
    assert environment.last_result["entries"][0]["path"] == "src/main.py"

    read = environment.execute_tool("workspace.read", {"path": "src/main.py"})
    assert read.success is True
    assert environment.last_result["content"] == "print('seed')\n"
    assert environment.last_result["digest"]

    escaped = environment.execute_tool("workspace.read", {"path": "../outside.txt"})
    assert escaped.success is False
    assert environment.last_result["error_code"] == "unsafe_path"


def test_runtime_status_exposes_seed_capabilities_without_legacy() -> None:
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        response = client.get("/api/runtime/status")
        assert response.status_code == 200
        tools = response.json()["tools"]
        names = {item["name"] for item in tools["tools"]}
        assert "workspace.list" in names
        assert tools["count"] == len(tools["tools"])

        capabilities = client.get("/api/workbench/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["snapshot_id"]
        assert capabilities.json()["mcp_registry"]["format"] == "seed-mcp-registry-v1"

        runtime_status = response.json()
        tools = runtime_status["tools"]
        assert tools["snapshot_id"] == capabilities.json()["snapshot_id"]
        assert tools["revision"] == capabilities.json()["revision"]
        assert tools["source"] == "seed_platform.workbench.CapabilitySnapshot"
        assert tools["owner"] == "Taiji native Workbench"
        assert tools["observed_at"] > 0
        mcp = client.get("/api/workbench/mcp")
        assert mcp.status_code == 200
        assert mcp.json()["tools"][0]["tool_id"] == "mcp.local.workspace_summary"
        languages = client.get("/api/workbench/programming-languages")
        assert languages.status_code == 200
        assert languages.json()["languages"]


def test_taiji_intent_reaches_workbench_and_audit(tmp_path, monkeypatch) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("Taiji workbench\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime._workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id
    intent = ActionIntent(
        intent_id="intent-read-readme",
        kind="workspace.read",
        parameters={"path": "README.md"},
        expected_outcome="read the workspace file",
        confidence=1.0,
        tick=runtime.model.tick,
    )

    result = runtime.execute_workbench_intent(intent, snapshot_id=snapshot_id, learn=False)

    assert result["outcome"]["status"] == "success"
    assert result["outcome"]["result"]["content"] == "Taiji workbench\n"
    assert result["tool_call"]["intent_id"] == intent.intent_id
    assert [event["phase"] for event in result["events"]] == [
        "planned",
        "policy",
        "executing",
        "outcome",
    ]


def test_taiji_tool_intent_bridge_does_not_select_the_intent(tmp_path) -> None:
    adapter = TSKV8Adapter()
    environment = WorkbenchEnvironment(tmp_path)
    intent = ActionIntent(
        intent_id="intent-list-root",
        kind="workspace.list",
        parameters={"path": "."},
        tick=adapter.tick,
    )

    call, outcome = adapter.execute_tool_intent(intent, environment, learn=False)

    assert call.intent_id == intent.intent_id
    assert call.tool_name == intent.kind
    assert outcome.intent_id == intent.intent_id


def test_taiji_candidate_admission_is_read_only_and_snapshot_bound(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    candidate = _grounded_read_candidate()

    admitted = environment.admit_taiji_candidate(
        candidate,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        current_tick=0,
    )

    assert admitted.accepted is True
    assert admitted.reason_code == "taiji_read_only_admitted"
    assert admitted.request is not None
    assert admitted.request.capability_id == "workspace.read"
    assert admitted.policy is not None
    assert admitted.policy.decision == "allow"
    assert admitted.to_payload()["candidate"]["source_affordance_id"] == "affordance-read"

    stale = environment.admit_taiji_candidate(
        candidate,
        snapshot_id="stale-snapshot",
        current_tick=0,
    )
    assert stale.accepted is False
    assert stale.reason_code == "stale_capability_snapshot"


def test_taiji_candidate_admission_rejects_untrusted_risk_and_parameter_drift(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)

    mutating = environment.admit_taiji_candidate(
        _grounded_read_candidate(kind="workspace.apply_patch"),
        snapshot_id=environment.capability_snapshot.snapshot_id,
        current_tick=0,
    )
    assert mutating.accepted is False
    assert mutating.reason_code == "taiji_read_only_gate_rejects_risk"

    drifted = environment.admit_taiji_candidate(
        _grounded_read_candidate(parameters={"path": "README.md", "tool": "workspace.read"}),
        snapshot_id=environment.capability_snapshot.snapshot_id,
        current_tick=0,
    )
    assert drifted.accepted is False
    assert drifted.reason_code == "capability_parameter_drift"

    ungrounded = replace(
        _grounded_read_candidate(),
        provenance="external",
        source_affordance_id=None,
    )
    rejected = environment.admit_taiji_candidate(
        ungrounded,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        current_tick=0,
    )
    assert rejected.accepted is False
    assert rejected.reason_code == "taiji_candidate_not_grounded"


def test_runtime_taiji_task_selects_then_executes_the_admitted_candidate(
    tmp_path, monkeypatch
) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("Taiji admission\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="taiji-admission"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    projected = runtime.project_workbench_affordances(
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        parameter_bindings={"workspace.read": {"path": "README.md"}},
    )
    candidate = replace(
        _grounded_read_candidate(tick=runtime.model.architecture.cognitive_snapshot().tick),
        source_affordance_id=projected["affordances"][0]["affordance_id"],
    )
    context = ExecutiveContext.from_state(runtime.model.architecture.cognitive_snapshot())
    decision = ExecutiveDecision(
        selected=candidate,
        scores={candidate.candidate_id: 1.0},
        context=context,
    )
    monkeypatch.setattr(runtime, "_select_taiji_workbench_candidate", lambda **_: decision)

    result = runtime.execute_taiji_workbench_task(
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        learn=False,
    )

    assert result["admission"]["accepted"] is True
    assert result["admission"]["reason_code"] == "taiji_read_only_admitted"
    assert result["execution"]["outcome"]["status"] == "success"
    assert result["execution"]["outcome"]["result"]["content"] == "Taiji admission\n"


def test_taiji_task_routes_expose_the_same_read_only_gate(tmp_path, monkeypatch) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("Taiji route admission\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="taiji-route-admission"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    projected_payload = runtime.project_workbench_affordances(
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        parameter_bindings={"workspace.read": {"path": "README.md"}},
    )
    candidate = replace(
        _grounded_read_candidate(tick=runtime.model.architecture.cognitive_snapshot().tick),
        source_affordance_id=projected_payload["affordances"][0]["affordance_id"],
    )
    context = ExecutiveContext.from_state(runtime.model.architecture.cognitive_snapshot())
    decision = ExecutiveDecision(
        selected=candidate,
        scores={candidate.candidate_id: 1.0},
        context=context,
    )
    monkeypatch.setattr(runtime, "_select_taiji_workbench_candidate", lambda **_: decision)

    import api.seed_runtime as seed_runtime_module

    monkeypatch.setattr(seed_runtime_module, "_runtime", runtime)
    with TestClient(create_app(startup_tasks=False)) as client:
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        projected = client.post(
            "/api/workbench/taiji/project",
            json={
                "snapshot_id": snapshot_id,
                "parameter_bindings": {"workspace.read": {"path": "README.md"}},
            },
        )
        admitted = client.post(
            "/api/workbench/taiji/admit",
            json={"snapshot_id": snapshot_id},
        )
        executed = client.post(
            "/api/workbench/taiji/execute",
            json={"snapshot_id": snapshot_id},
        )
        reprojected = client.post(
            "/api/workbench/taiji/reproject",
            json={"snapshot_id": snapshot_id},
        )

    assert projected.status_code == 200
    assert projected.json()["affordances"][0]["action_kind"] == "workspace.read"
    assert admitted.status_code == 200
    assert admitted.json()["admission"]["accepted"] is True
    assert executed.status_code == 200
    assert executed.json()["admission"]["accepted"] is True
    assert executed.json()["execution"]["outcome"]["status"] == "success"
    assert reprojected.status_code == 200
    assert reprojected.json()["affordances"][0]["action_kind"] == "workspace.stat"


def test_native_mcp_registry_lists_and_invokes_local_read_only_canary(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('seed')\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)

    listed = environment.execute_tool("mcp.list", {})

    assert listed.success is True
    assert environment.last_result["registry"]["format"] == "seed-mcp-registry-v1"
    assert environment.last_result["tools"][0]["tool_id"] == "mcp.local.workspace_summary"
    tool_id = environment.last_result["tools"][0]["tool_id"]

    request = WorkbenchActionRequest(
        request_id="request-mcp-local",
        intent_id="intent-mcp-local",
        capability_id="mcp.invoke",
        parameters={"tool_id": tool_id, "arguments": {"path": "src"}},
        snapshot_id=environment.capability_snapshot.snapshot_id,
        confidence=1.0,
        mcp_registry_snapshot_id=environment.mcp_registry.snapshot_id,
    )
    assert environment.policy_for(request).reason_code == "mcp_tool_read_only"
    invoked = environment.execute_tool("mcp.invoke", request.parameters)

    assert invoked.success is True
    assert environment.last_result["provenance"]["kind"] == "mcp"
    assert environment.last_result["result"]["entries"][0]["path"] == "src/main.py"

    drifted = environment.execute_tool(
        "mcp.invoke",
        {
            "tool_id": tool_id,
            "arguments": {"path": "src"},
            "registry_revision": 999,
        },
    )
    assert drifted.success is False
    assert environment.last_result["error_code"] == "invalid_parameters"

    unknown = environment.execute_tool("mcp.invoke", {"tool_id": "mcp.unknown", "arguments": {}})
    assert unknown.success is False
    assert environment.last_result["error_code"] == "unknown_mcp_tool"


def test_mcp_risk_is_dynamic_and_empty_registry_fails_closed(tmp_path) -> None:
    descriptor = McpToolDescriptor(
        tool_id="mcp.local.high-risk",
        name="High-risk canary",
        description="Synthetic high-risk contract for policy testing.",
        input_schema={"type": "object", "additionalProperties": False},
        executor_id="workspace.list",
        risk="file_write",
    )
    environment = WorkbenchEnvironment(tmp_path, mcp_registry=McpToolRegistry((descriptor,)))
    request = WorkbenchActionRequest(
        request_id="request-mcp-risk",
        intent_id="intent-mcp-risk",
        capability_id="mcp.invoke",
        parameters={"tool_id": descriptor.tool_id, "arguments": {}},
        snapshot_id=environment.capability_snapshot.snapshot_id,
        confidence=1.0,
        mcp_registry_snapshot_id=environment.mcp_registry.snapshot_id,
    )
    assert environment.policy_for(request).reason_code == "capability_requires_approval"
    approval = environment.issue_approval(request)
    assert approval["preview"]["mutation"]["risk"] == "file_write"

    empty = WorkbenchEnvironment(tmp_path, mcp_registry=McpToolRegistry())
    rejected = empty.execute_tool("mcp.invoke", {"tool_id": descriptor.tool_id, "arguments": {}})
    assert rejected.success is False
    assert empty.last_result["error_code"] == "unknown_mcp_tool"


def test_mcp_schema_and_output_budget_fail_closed(tmp_path) -> None:
    descriptor = McpToolDescriptor(
        tool_id="mcp.local.bounded",
        name="Bounded canary",
        description="Synthetic bounded canary for schema and output checks.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        executor_id="workspace.list",
        output_limit=1,
    )
    (tmp_path / "main.py").write_text("print('seed')\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path, mcp_registry=McpToolRegistry((descriptor,)))

    invalid = environment.execute_tool(
        "mcp.invoke", {"tool_id": descriptor.tool_id, "arguments": {}}
    )
    assert invalid.success is False
    assert environment.last_result["error_code"] == "invalid_parameters"

    oversized = environment.execute_tool(
        "mcp.invoke",
        {"tool_id": descriptor.tool_id, "arguments": {"path": "."}},
    )
    assert oversized.success is False
    assert environment.last_result["error_code"] == "mcp_output_limit"


def test_mcp_binding_loop_preflight_and_runtime_tool_call_are_versioned(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("native workbench\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)
    request = WorkbenchActionRequest(
        request_id="request-loop-read",
        intent_id="intent-loop-read",
        capability_id="mcp.invoke",
        parameters={
            "tool_id": "mcp.local.workspace_summary",
            "arguments": {"path": "."},
        },
        snapshot_id=environment.capability_snapshot.snapshot_id,
        confidence=1.0,
        mcp_registry_snapshot_id=environment.mcp_registry.snapshot_id,
    )

    preflight = environment.preflight_loop((request,), loop_id="loop-mcp-read", max_steps=2)

    assert preflight["accepted"] is True
    assert preflight["step_count"] == 1
    assert preflight["checkpoint"]["boundary"] == "after_each_step"
    assert preflight["mcp_registry_snapshot_id"] == environment.mcp_registry.snapshot_id
    assert preflight["preflight_id"]

    repeated = environment.preflight_loop(
        (
            request,
            replace_request_mcp_registry(
                request,
                environment.mcp_registry.snapshot_id,
                request_id="request-loop-read-2",
            ),
        ),
        loop_id="loop-repeat",
        max_steps=2,
    )
    assert repeated["accepted"] is False
    assert repeated["error_code"] == "repeated_call"

    stale = replace_request_mcp_registry(request, "stale-registry")
    assert environment.policy_for(stale).reason_code == "stale_mcp_registry"

    runtime = SeedRuntime(Seed(episode_id="workbench-mcp-binding"))
    runtime._workbench_environment = environment
    runtime_preflight = runtime.preflight_workbench_loop((request,), loop_id="runtime-loop-mcp")
    assert runtime_preflight["accepted"] is True
    assert runtime_preflight["runtime"]["checkpoint_boundary"] == "after_each_step"
    result = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-runtime-mcp",
            kind="mcp.invoke",
            parameters={
                "tool_id": "mcp.local.workspace_summary",
                "arguments": {"path": "."},
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=environment.capability_snapshot.snapshot_id,
        learn=False,
    )
    assert result["outcome"]["status"] == "success"
    assert result["request"]["mcp_registry_snapshot_id"] == environment.mcp_registry.snapshot_id
    assert result["outcome"]["mcp_registry_snapshot_id"] == environment.mcp_registry.snapshot_id
    assert (
        result["tool_call"]["workbench_binding"]["mcp_registry_snapshot_id"]
        == environment.mcp_registry.snapshot_id
    )


def replace_request_mcp_registry(
    request: WorkbenchActionRequest,
    snapshot_id: str,
    request_id: str | None = None,
) -> WorkbenchActionRequest:
    return WorkbenchActionRequest(
        request_id=request_id or request.request_id,
        intent_id=request.intent_id,
        capability_id=request.capability_id,
        parameters=request.parameters,
        snapshot_id=request.snapshot_id,
        confidence=request.confidence,
        tick=request.tick,
        source=request.source,
        version=request.version,
        approval_token=request.approval_token,
        mcp_registry_snapshot_id=snapshot_id,
    )


def test_preflighted_loop_checkpoints_each_step_and_rejects_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    checkpoint = tmp_path / "loop.pt"
    runtime = SeedRuntime(Seed(episode_id="workbench-loop-checkpoint"), checkpoint_path=checkpoint)
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    intents = (
        ActionIntent(
            intent_id="intent-loop-mcp",
            kind="mcp.invoke",
            parameters={
                "tool_id": "mcp.local.workspace_summary",
                "arguments": {"path": "."},
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="intent-loop-read",
            kind="workspace.read",
            parameters={"path": "README.md"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
    )
    (tmp_path / "README.md").write_text("loop\n", encoding="utf-8")
    requests = tuple(
        WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=environment.capability_snapshot.snapshot_id,
            mcp_registry_snapshot_id=(
                environment.mcp_registry.snapshot_id if intent.kind.startswith("mcp.") else ""
            ),
        )
        for intent in intents
    )
    preflight = runtime.preflight_workbench_loop(requests, loop_id="loop-checkpoint", max_steps=2)
    result = runtime.execute_preflighted_workbench_loop(
        intents,
        requests,
        loop_id="loop-checkpoint",
        preflight_id=preflight["preflight_id"],
        max_steps=2,
        learn=False,
    )

    assert result["status"] == "completed"
    assert result["completed_prefix"] == 2
    assert len(result["steps"]) == 2
    assert all(step["checkpoint"]["committed"] for step in result["steps"])
    assert checkpoint.exists()

    restored = SeedRuntime.load(checkpoint)
    replay = restored.execute_preflighted_workbench_loop(
        intents,
        requests,
        loop_id="loop-checkpoint",
        preflight_id=preflight["preflight_id"],
        max_steps=2,
        learn=False,
    )
    assert replay["status"] == "rejected"
    assert replay["error_code"] == "loop_request_already_committed"


def test_preflighted_loop_stops_after_first_failed_step(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    checkpoint = tmp_path / "failed-loop.pt"
    runtime = SeedRuntime(Seed(episode_id="workbench-loop-failure"), checkpoint_path=checkpoint)
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    intents = (
        ActionIntent(
            intent_id="intent-loop-list",
            kind="workspace.list",
            parameters={"path": "."},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="intent-loop-missing",
            kind="workspace.read",
            parameters={"path": "missing.txt"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="intent-loop-never",
            kind="workspace.stat",
            parameters={"path": "."},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
    )
    requests = tuple(
        WorkbenchActionRequest.from_action_intent(
            intent, snapshot_id=environment.capability_snapshot.snapshot_id
        )
        for intent in intents
    )
    preflight = runtime.preflight_workbench_loop(requests, loop_id="loop-failure", max_steps=3)
    result = runtime.execute_preflighted_workbench_loop(
        intents,
        requests,
        loop_id="loop-failure",
        preflight_id=preflight["preflight_id"],
        max_steps=3,
        learn=False,
    )

    assert result["status"] == "failed"
    assert result["stopped_at"] == 1
    assert result["completed_prefix"] == 1
    assert len(result["steps"]) == 2
    assert result["steps"][1]["success"] is False
    assert checkpoint.exists()


def test_w3_cross_file_task_gate_replans_after_diagnostics_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "build").mkdir()
    source = tmp_path / "src" / "main.py"
    original = b'def answer():\n    return "seed"\n'
    updated = b'def answer():\n    return "taiji"\n'
    source.write_bytes(original)
    (tmp_path / "src" / "test_main.py").write_text(
        "from main import answer\n\n" "def test_answer():\n" "    assert answer() == 'taiji'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    checkpoint = tmp_path / "w3-loop.pt"
    runtime = SeedRuntime(Seed(episode_id="workbench-w3-cross-file"), checkpoint_path=checkpoint)
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id

    before_digest = hashlib.sha256(original).hexdigest()
    after_digest = hashlib.sha256(updated).hexdigest()
    diagnostic_code = (
        "from pathlib import Path; "
        "print('src/main.py:1:1: error: missing diagnostics marker') "
        "if not Path('build/diagnostic.ok').exists() else "
        "print('src/main.py:1:1: info: diagnostics clear')"
    )
    intents = (
        ActionIntent(
            intent_id="w3-language",
            kind="workspace.programming_language.resolve",
            parameters={"path": "src/main.py"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="w3-patch",
            kind="workspace.apply_patch",
            parameters={
                "path": "src/main.py",
                "before_digest": before_digest,
                "patch": {
                    "kind": "text_replace",
                    "operations": [
                        {
                            "start": original.decode().index("seed"),
                            "end": original.decode().index("seed") + len("seed"),
                            "text": "taiji",
                        }
                    ],
                },
                "expected_after_digest": after_digest,
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="w3-test",
            kind="terminal.run",
            parameters={
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "assert 'taiji' in Path('src/main.py').read_text(); "
                    "Path('build/result.txt').write_text('ok')",
                ],
                "execution_kind": "test",
                "expected_artifacts": ["build/result.txt"],
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        ActionIntent(
            intent_id="w3-diagnostics-fail",
            kind="terminal.run",
            parameters={
                "argv": [sys.executable, "-c", diagnostic_code],
                "execution_kind": "diagnostics",
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
    )

    def approved_request(intent: ActionIntent) -> WorkbenchActionRequest:
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=snapshot_id,
        )
        if environment.policy_for(request).reason_code == "capability_requires_approval":
            approval = environment.issue_approval(request)
            request = replace(request, approval_token=approval["approval_token"])
        return request

    requests = tuple(approved_request(intent) for intent in intents)
    preflight = runtime.preflight_workbench_loop(requests, loop_id="w3-cross-file", max_steps=4)
    assert preflight["accepted"] is True
    first_run = runtime.execute_preflighted_workbench_loop(
        intents,
        requests,
        loop_id="w3-cross-file",
        preflight_id=preflight["preflight_id"],
        max_steps=4,
        learn=False,
    )

    assert first_run["status"] == "failed"
    assert first_run["stopped_at"] == 3
    assert first_run["completed_prefix"] == 3
    assert len(first_run["steps"]) == 4
    assert first_run["steps"][0]["outcome"]["result"]["programming_language_id"] == "python"
    assert first_run["steps"][1]["success"] is True
    assert first_run["steps"][2]["success"] is True
    assert first_run["steps"][3]["success"] is False
    assert first_run["steps"][3]["outcome"]["result"]["diagnostics"][0]["severity"] == "error"
    assert (tmp_path / "build" / "result.txt").read_text(encoding="utf-8") == "ok"
    assert checkpoint.exists()

    restored = SeedRuntime.load(checkpoint)
    recovery_create = ActionIntent(
        intent_id="w3-recovery-marker",
        kind="workspace.create",
        parameters={"path": "build/diagnostic.ok", "content": "ok\n"},
        confidence=1.0,
        tick=restored.model.tick,
    )
    recovery_diagnostics = ActionIntent(
        intent_id="w3-recovery-diagnostics",
        kind="terminal.run",
        parameters={
            "argv": [sys.executable, "-c", diagnostic_code],
            "execution_kind": "diagnostics",
        },
        confidence=1.0,
        tick=restored.model.tick,
    )
    recovery_intents = (recovery_create, recovery_diagnostics)
    recovery_environment = restored.workbench_environment
    recovery_snapshot_id = recovery_environment.capability_snapshot.snapshot_id
    recovery_requests = []
    for intent in recovery_intents:
        request = WorkbenchActionRequest.from_action_intent(
            intent, snapshot_id=recovery_snapshot_id
        )
        if recovery_environment.policy_for(request).reason_code == "capability_requires_approval":
            approval = recovery_environment.issue_approval(request)
            request = replace(request, approval_token=approval["approval_token"])
        recovery_requests.append(request)
    recovery_requests = tuple(recovery_requests)
    recovery_preflight = restored.preflight_workbench_loop(
        recovery_requests, loop_id="w3-recovery", max_steps=2
    )
    recovery = restored.execute_preflighted_workbench_loop(
        recovery_intents,
        recovery_requests,
        loop_id="w3-recovery",
        preflight_id=recovery_preflight["preflight_id"],
        max_steps=2,
        learn=False,
    )

    assert recovery["status"] == "completed"
    assert recovery["completed_prefix"] == 2
    assert (tmp_path / "build" / "diagnostic.ok").exists()
    assert recovery["steps"][1]["success"] is True
    assert recovery["steps"][1]["outcome"]["result"]["diagnostics"][0]["severity"] == "info"

    replay = restored.execute_preflighted_workbench_loop(
        intents,
        requests,
        loop_id="w3-cross-file",
        preflight_id=preflight["preflight_id"],
        max_steps=4,
        learn=False,
    )
    assert replay["status"] == "rejected"
    assert replay["error_code"] == "loop_request_already_committed"


def test_programming_language_evidence_uses_content_manifest_and_ambiguity(
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='seed'\n", encoding="utf-8")
    source = tmp_path / "main.py"
    source.write_text("def answer(value: int):\n    return value\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)

    resolved = environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )

    assert resolved.success is True
    assert environment.last_result["programming_language_id"] == "python"
    assert environment.last_result["editor_language_id"] == "python"
    assert {item["source"] for item in environment.last_result["provenance"]} >= {
        "extension",
        "content",
        "manifest",
    }
    assert environment.last_result["execution_snapshot"]["runner_id"] == "python"
    assert environment.last_result["execution_snapshot"]["lsp_id"] == "pyright"
    assert environment.last_result["explanation"]["selected_language"] == "python"
    assert environment.last_result["explanation"]["evidence"]

    header = tmp_path / "shared.h"
    header.write_text("#include <stdio.h>\n", encoding="utf-8")
    environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "shared.h"},
    )
    assert environment.last_result["selection_state"] == "ambiguous"

    registry = ProgrammingLanguageRegistry.default()
    no_extension = registry.resolve(
        path="run",
        content="#!/usr/bin/env node\nconsole.log('seed')\n",
        file_digest="node-script",
    )
    assert no_extension.programming_language_id == "javascript"
    assert any(item.source == "shebang" for item in no_extension.provenance)

    wrong_extension = registry.resolve(
        path="notes.txt",
        content="def answer(value):\n    return value\n",
        file_digest="python-content",
    )
    assert wrong_extension.programming_language_id == "python"
    filename_only = registry.resolve(
        path="unknown.py",
        content="42\n",
        file_digest="filename-only",
    )
    assert filename_only.selection_state == "ambiguous"

    vue = registry.resolve(
        path="App.vue",
        content='<template><main /></template>\n<script lang="ts">const x: number = 1</script>\n',
        file_digest="vue-ts",
        manifest_names={"package.json", "tsconfig.json"},
    )
    assert vue.programming_language_id == "vue"
    assert vue.editor_language_id == "html"

    lsp = registry.resolve(
        path="wrong-name.data",
        content="",
        file_digest="lsp-rust",
        lsp_language_id="rust",
    )
    assert lsp.programming_language_id == "rust"
    assert any(item.source == "lsp" for item in lsp.provenance)

    monorepo_typescript = registry.resolve(
        path="packages/ui/index.ts",
        content="interface Props { title: string }\n",
        file_digest="monorepo-typescript",
        manifest_names={"package.json", "pyproject.toml", "tsconfig.json"},
    )
    assert monorepo_typescript.programming_language_id == "typescript"
    assert all(
        item.language_id != "python"
        for item in monorepo_typescript.provenance
        if item.source == "manifest"
    )

    markdown = registry.resolve(
        path="guide.md",
        content="# Guide\n\n```python\nprint('seed')\n```\n",
        file_digest="markdown-code-block",
    )
    assert markdown.programming_language_id == "markdown"

    notebook = registry.resolve(
        path="analysis.ipynb",
        content='{"cells": [], "nbformat": 4, "nbformat_minor": 5}',
        file_digest="notebook",
    )
    assert notebook.programming_language_id == "notebook"
    assert notebook.editor_language_id == "json"


def test_programming_language_override_is_content_bound_and_checkpointable(
    tmp_path,
) -> None:
    source = tmp_path / "main.py"
    source.write_text("print('seed')\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)
    override = environment.execute_tool(
        "editor.set_language",
        {
            "path": "main.py",
            "programming_language_id": "javascript",
            "user_override": True,
        },
    )
    assert override.success is True
    assert environment.last_result["selection_state"] == "user_override"

    environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert environment.last_result["programming_language_id"] == "javascript"

    restored = WorkbenchEnvironment(
        tmp_path,
        snapshot=environment.capability_snapshot,
        programming_language_registry=ProgrammingLanguageRegistry.default(),
    )
    restored.restore_language_state(environment.language_state_checkpoint())
    restored.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert restored.last_result["programming_language_id"] == "javascript"

    source.write_text("#!/usr/bin/env node\nconst answer = 1;\n", encoding="utf-8")
    restored.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert restored.last_result["programming_language_id"] == "javascript"
    assert restored.last_result["selection_state"] != "user_override"


def test_taiji_language_selection_requires_evidence_or_explicit_override(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "shared.h").write_text("#include <stdio.h>\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="language-policy-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime._workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id

    ambiguous = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-ambiguous-language",
            kind="editor.set_language",
            parameters={
                "path": "shared.h",
                "programming_language_id": "c",
                "user_override": False,
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert ambiguous["policy"]["decision"] == "ask_user"
    assert ambiguous["policy"]["reason_code"] == "language_evidence_ambiguous"
    assert ambiguous["outcome"]["status"] == "rejected"

    autonomous = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-python-language",
            kind="editor.set_language",
            parameters={
                "path": "main.py",
                "programming_language_id": "python",
                "user_override": False,
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert autonomous["policy"]["decision"] == "allow"
    assert autonomous["outcome"]["result"]["programming_language_id"] == "python"
    assert autonomous["outcome"]["result"]["execution_snapshot"]["runner_id"] == "python"

    explicit = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-explicit-language",
            kind="editor.set_language",
            parameters={
                "path": "shared.h",
                "programming_language_id": "cpp",
                "user_override": True,
            },
            confidence=0.1,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert explicit["policy"]["decision"] == "allow"
    assert explicit["outcome"]["result"]["selection_state"] == "user_override"


def test_file_transactions_are_digest_checked_and_undoable(tmp_path) -> None:
    source = tmp_path / "main.py"
    original = b"print('seed')\n"
    updated = b"print('taiji')\n"
    source.write_bytes(original)
    environment = WorkbenchEnvironment(tmp_path)

    start = original.decode("utf-8").index("seed")
    end = start + len("seed")
    applied = environment.execute_tool(
        "workspace.apply_patch",
        {
            "path": "main.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
            "patch": {
                "kind": "text_replace",
                "operations": [{"start": start, "end": end, "text": "taiji"}],
            },
            "expected_after_digest": hashlib.sha256(updated).hexdigest(),
        },
    )
    assert applied.success is True
    patch_transaction = environment.last_result["transaction"]
    assert patch_transaction["undo_token"]
    assert source.read_bytes() == updated

    stale = environment.execute_tool(
        "workspace.apply_patch",
        {
            "path": "main.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
            "patch": {
                "kind": "text_replace",
                "operations": [{"start": start, "end": end, "text": "seed"}],
            },
            "expected_after_digest": hashlib.sha256(original).hexdigest(),
        },
    )
    assert stale.success is False
    assert environment.last_result["error_code"] == "transaction_conflict"

    undone = environment.execute_tool(
        "workspace.undo", {"undo_token": patch_transaction["undo_token"]}
    )
    assert undone.success is True
    assert source.read_bytes() == original

    created = environment.execute_tool(
        "workspace.create", {"path": "created.txt", "content": "created\n"}
    )
    assert created.success is True
    create_token = environment.last_result["transaction"]["undo_token"]
    assert (tmp_path / "created.txt").is_file()
    assert environment.execute_tool("workspace.undo", {"undo_token": create_token}).success
    assert not (tmp_path / "created.txt").exists()

    renamed = environment.execute_tool(
        "workspace.rename",
        {
            "path": "main.py",
            "new_path": "renamed.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
        },
    )
    assert renamed.success is True
    rename_token = environment.last_result["transaction"]["undo_token"]
    assert (tmp_path / "renamed.py").is_file()
    assert environment.execute_tool("workspace.undo", {"undo_token": rename_token}).success
    assert source.is_file()

    deleted = environment.execute_tool(
        "workspace.delete",
        {"path": "main.py", "before_digest": hashlib.sha256(original).hexdigest()},
    )
    assert deleted.success is True
    delete_token = environment.last_result["transaction"]["undo_token"]
    assert not source.exists()
    assert environment.execute_tool("workspace.undo", {"undo_token": delete_token}).success
    assert source.read_bytes() == original


def test_terminal_run_is_bounded_and_shell_free(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    completed = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "print('seed')"],
            "cwd": ".",
            "timeout_seconds": 5,
            "output_limit": 1024,
            "env": {},
            "env_allowlist": [],
            "expected_artifacts": [],
        },
    )
    assert completed.success is True
    assert environment.last_result["shell"] is False
    assert environment.last_result["exit_code"] == 0
    assert "seed" in environment.last_result["stdout"]

    invalid_env = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "pass"],
            "env": {"SEED_TEST": "1"},
            "env_allowlist": [],
        },
    )
    assert invalid_env.success is False
    assert environment.last_result["error_code"] == "invalid_parameters"

    timed_out = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(1)"],
            "timeout_seconds": 0.05,
        },
    )
    assert timed_out.success is False
    assert environment.last_result["timed_out"] is True

    artifact = environment.execute_tool(
        "terminal.run",
        {
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('artifact.txt').write_text('ok')",
            ],
            "execution_kind": "build",
            "expected_artifacts": ["artifact.txt"],
        },
    )
    assert artifact.success is True
    assert environment.last_result["after_state"]["artifacts"][0]["exists"] is True

    diagnostic = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "print('main.py:2:3: error: broken')"],
            "execution_kind": "diagnostics",
        },
    )
    assert diagnostic.success is False
    assert environment.last_result["exit_code"] == 0
    assert environment.last_result["diagnostics"][0]["severity"] == "error"

    flooded = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "print('x' * 5000)"],
            "output_limit": 100,
        },
    )
    assert flooded.success is True
    assert environment.last_result["stdout_truncated"] is True

    cwd_drift = environment.execute_tool(
        "terminal.run",
        {"argv": [sys.executable, "-c", "pass"], "cwd": ".."},
    )
    assert cwd_drift.success is False
    assert environment.last_result["error_code"] == "unsafe_path"


def test_write_and_terminal_capabilities_require_approval(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = environment.capability_snapshot.snapshot_id
    for capability_id in ("workspace.apply_patch", "terminal.run"):
        request = WorkbenchActionRequest(
            request_id=f"request-{capability_id}",
            intent_id=f"intent-{capability_id}",
            capability_id=capability_id,
            parameters={},
            snapshot_id=snapshot_id,
            confidence=1.0,
        )
        decision = environment.policy_for(request)
        assert decision.decision == "ask_user"
        assert decision.reason_code == "capability_requires_approval"


def test_preview_issues_single_use_approval_and_runtime_projects_transaction(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-approval-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    intent = ActionIntent(
        intent_id="intent-create-approved",
        kind="workspace.create",
        parameters={"path": "approved.txt", "content": "approved\n"},
        expected_outcome="create one file",
        confidence=1.0,
        tick=runtime.model.tick,
    )

    preview = runtime.preview_workbench_intent(intent, snapshot_id=snapshot_id)

    assert preview["policy"]["decision"] == "ask_user"
    assert preview["preview"]["validated"] is True
    assert preview["preview"]["mutation"]["after_digest"]
    approval_token = preview["approval"]["approval_token"]
    assert approval_token
    assert not (tmp_path / "approved.txt").exists()

    executed = runtime.execute_workbench_intent(
        intent,
        snapshot_id=snapshot_id,
        approval_token=approval_token,
        learn=False,
    )
    assert executed["policy"]["reason_code"] == "explicit_approval"
    assert executed["outcome"]["success"] is True
    assert executed["outcome"]["transaction"]["undo_token"]
    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "approved\n"

    replayed = runtime.execute_workbench_intent(
        intent,
        snapshot_id=snapshot_id,
        approval_token=approval_token,
        learn=False,
    )
    assert replayed["policy"]["decision"] == "ask_user"
    assert replayed["policy"]["reason_code"] == "approval_invalid"
    assert replayed["outcome"]["status"] == "rejected"


def test_transaction_checkpoint_restores_undo_but_not_approval(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    created = environment.execute_tool(
        "workspace.create", {"path": "checkpoint.txt", "content": "keep\n"}
    )
    assert created.success is True
    undo_token = environment.last_result["transaction"]["undo_token"]
    transaction_state = environment.transaction_state_checkpoint()

    restored = WorkbenchEnvironment(tmp_path)
    restored.restore_transaction_state(transaction_state)
    assert restored.status()["undoable_transactions"] == 1
    undo_request = WorkbenchActionRequest(
        request_id="request-checkpoint-undo",
        intent_id="intent-checkpoint-undo",
        capability_id="workspace.undo",
        parameters={"undo_token": undo_token},
        snapshot_id=restored.capability_snapshot.snapshot_id,
        confidence=1.0,
    )
    undo_approval = restored.issue_approval(undo_request)
    assert undo_approval["preview"]["mutation"]["undo_of"] == "workspace.create"
    approved_undo_request = WorkbenchActionRequest(
        request_id=undo_request.request_id,
        intent_id=undo_request.intent_id,
        capability_id=undo_request.capability_id,
        parameters=undo_request.parameters,
        snapshot_id=undo_request.snapshot_id,
        confidence=undo_request.confidence,
        approval_token=undo_approval["approval_token"],
    )
    assert restored.policy_for(approved_undo_request).reason_code == "explicit_approval"
    restored.consume_approval(approved_undo_request)
    assert restored.execute_tool("workspace.undo", {"undo_token": undo_token}).success
    assert not (tmp_path / "checkpoint.txt").exists()

    request = WorkbenchActionRequest(
        request_id="request-checkpoint-approval",
        intent_id="intent-checkpoint-approval",
        capability_id="terminal.run",
        parameters={"argv": [sys.executable, "-c", "pass"]},
        snapshot_id=environment.capability_snapshot.snapshot_id,
        confidence=1.0,
    )
    approval_token = environment.issue_approval(request)["approval_token"]
    restored_request = WorkbenchActionRequest(
        request_id=request.request_id,
        intent_id=request.intent_id,
        capability_id=request.capability_id,
        parameters=request.parameters,
        snapshot_id=request.snapshot_id,
        confidence=request.confidence,
        approval_token=approval_token,
    )
    assert restored.policy_for(restored_request).reason_code == "approval_invalid"


def test_runtime_checkpoint_restores_transaction_lineage_without_approval_revival(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    checkpoint_descriptor = McpToolDescriptor(
        tool_id="mcp.local.checkpoint",
        name="Checkpoint canary",
        description="A custom registry entry used to verify checkpoint identity.",
        input_schema={"type": "object", "additionalProperties": False},
        executor_id="workspace.list",
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-checkpoint-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(
        tmp_path,
        mcp_registry=McpToolRegistry((checkpoint_descriptor,)),
    )
    environment = runtime.workbench_environment
    created = environment.execute_tool(
        "workspace.create", {"path": "runtime.txt", "content": "resume\n"}
    )
    assert created.success is True
    undo_token = environment.last_result["transaction"]["undo_token"]
    old_undo_intent = ActionIntent(
        intent_id="intent-old-undo",
        kind="workspace.undo",
        parameters={"undo_token": undo_token},
        confidence=1.0,
        tick=runtime.model.tick,
    )
    old_preview = runtime.preview_workbench_intent(
        old_undo_intent,
        snapshot_id=environment.capability_snapshot.snapshot_id,
    )
    old_approval = old_preview["approval"]["approval_token"]
    checkpoint = tmp_path / "runtime.pt"
    runtime.save(checkpoint)

    restored = SeedRuntime.load(checkpoint)
    assert restored.workbench_environment.status()["undoable_transactions"] == 1
    assert (
        restored.workbench_environment.mcp_registry.get(checkpoint_descriptor.tool_id)
        == checkpoint_descriptor
    )
    old_request = WorkbenchActionRequest(
        request_id="workbench:intent-old-undo",
        intent_id=old_undo_intent.intent_id,
        capability_id=old_undo_intent.kind,
        parameters=old_undo_intent.parameters,
        snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
        confidence=1.0,
        tick=restored.model.tick,
        approval_token=old_approval,
    )
    assert restored.workbench_environment.policy_for(old_request).reason_code == "approval_invalid"

    new_undo_intent = ActionIntent(
        intent_id="intent-new-undo",
        kind="workspace.undo",
        parameters={"undo_token": undo_token},
        confidence=1.0,
        tick=restored.model.tick,
    )
    new_preview = restored.preview_workbench_intent(
        new_undo_intent,
        snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
    )
    resumed = restored.execute_workbench_intent(
        new_undo_intent,
        snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
        approval_token=new_preview["approval"]["approval_token"],
        learn=False,
    )
    assert resumed["outcome"]["success"] is True
    assert not (tmp_path / "runtime.txt").exists()


def test_w2_temporary_project_gate_covers_language_patch_test_and_diagnostics(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "build").mkdir()
    source = tmp_path / "src" / "main.py"
    original = b'def answer():\n    return "seed"\n'
    updated = b'def answer():\n    return "taiji"\n'
    source.write_bytes(original)
    (tmp_path / "src" / "test_main.py").write_text(
        "from main import answer\n\ndef test_answer():\n    assert answer() == 'taiji'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-w2-project-gate"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime.workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id

    language = environment.execute_tool(
        "workspace.programming_language.resolve", {"path": "src/main.py"}
    )
    assert language.success is True
    assert environment.last_result["programming_language_id"] == "python"

    before_digest = hashlib.sha256(original).hexdigest()
    after_digest = hashlib.sha256(updated).hexdigest()
    patch_intent = ActionIntent(
        intent_id="w2-project-patch",
        kind="workspace.apply_patch",
        parameters={
            "path": "src/main.py",
            "before_digest": before_digest,
            "patch": {
                "kind": "text_replace",
                "operations": [
                    {
                        "start": original.decode().index("seed"),
                        "end": original.decode().index("seed") + len("seed"),
                        "text": "taiji",
                    }
                ],
            },
            "expected_after_digest": after_digest,
        },
        confidence=1.0,
        tick=runtime.model.tick,
    )
    patch_preview = runtime.preview_workbench_intent(patch_intent, snapshot_id=snapshot_id)
    assert patch_preview["preview"]["mutation"]["after_digest"] == after_digest
    assert source.read_bytes() == original
    patch_result = runtime.execute_workbench_intent(
        patch_intent,
        snapshot_id=snapshot_id,
        approval_token=patch_preview["approval"]["approval_token"],
        learn=False,
    )
    assert patch_result["outcome"]["success"] is True
    assert source.read_bytes() == updated

    test_intent = ActionIntent(
        intent_id="w2-project-test",
        kind="terminal.run",
        parameters={
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('build/result.txt').write_text('ok')",
            ],
            "execution_kind": "test",
            "expected_artifacts": ["build/result.txt"],
        },
        confidence=1.0,
        tick=runtime.model.tick,
    )
    test_preview = runtime.preview_workbench_intent(test_intent, snapshot_id=snapshot_id)
    test_result = runtime.execute_workbench_intent(
        test_intent,
        snapshot_id=snapshot_id,
        approval_token=test_preview["approval"]["approval_token"],
        learn=False,
    )
    assert test_result["outcome"]["success"] is True
    assert test_result["outcome"]["result"]["after_state"]["artifacts"][0]["exists"]

    diagnostic_intent = ActionIntent(
        intent_id="w2-project-diagnostic",
        kind="terminal.run",
        parameters={
            "argv": [
                sys.executable,
                "-c",
                "print('src/main.py:1:1: error: test regression')",
            ],
            "execution_kind": "diagnostics",
        },
        confidence=1.0,
        tick=runtime.model.tick,
    )
    diagnostic_preview = runtime.preview_workbench_intent(
        diagnostic_intent, snapshot_id=snapshot_id
    )
    diagnostic_result = runtime.execute_workbench_intent(
        diagnostic_intent,
        snapshot_id=snapshot_id,
        approval_token=diagnostic_preview["approval"]["approval_token"],
        learn=False,
    )
    assert diagnostic_result["outcome"]["success"] is False
    assert diagnostic_result["outcome"]["result"]["diagnostics"][0]["severity"] == "error"
    assert (tmp_path / "build" / "result.txt").read_text(encoding="utf-8") == "ok"
