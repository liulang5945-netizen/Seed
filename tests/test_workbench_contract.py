from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app
from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.workbench import CapabilitySnapshot, WorkbenchEnvironment
from taiji import ActionIntent, TSKV8Adapter


def test_workbench_snapshot_is_content_addressed() -> None:
    snapshot = CapabilitySnapshot.default()
    restored = CapabilitySnapshot.from_payload(snapshot.to_payload())

    assert restored == snapshot
    assert snapshot.get("workspace.list") is not None
    assert snapshot.get("editor.diagnostics.read").enabled is False  # type: ignore[union-attr]

    tampered = snapshot.to_payload()
    tampered["capabilities"][0]["description"] = "tampered"
    try:
        CapabilitySnapshot.from_payload(tampered)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:  # pragma: no cover - protects the red-path contract
        raise AssertionError("tampered workbench snapshot was accepted")


def test_read_only_environment_reads_and_rejects_escape(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('seed')\n", encoding="utf-8")
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


def test_taiji_intent_reaches_workbench_and_audit(tmp_path) -> None:
    (tmp_path / "README.md").write_text("Taiji workbench\n", encoding="utf-8")
    runtime = SeedRuntime(Seed(episode_id="workbench-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
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
