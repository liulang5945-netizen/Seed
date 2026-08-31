"""P6-1b provider-evidence-backed chat Workbench journey tests."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.workbench import WorkbenchEnvironment
from taiji import SemanticEvidenceProposal, SemanticProviderRequest


class _ChatJourneyProvider:
    provider_id = "test.p6-1b.chat-journey"

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        return SemanticEvidenceProposal.from_frame(
            request.frame,
            provider_id=self.provider_id,
            goal_description="读取 README.md 并准备工作台",
            constraints=request.constraints,
            context_digest=request.context_digest,
            semantic_steps=(
                {
                    "description": "读取目标文件并确认当前内容",
                    "semantic_slots": {"operation": "read", "path": "README.md"},
                },
            ),
            confidence=0.9,
            ambiguity=0.1,
            provenance="tests.p6-1b.chat-journey",
            tick=request.frame.timestamp,
        )

    def checkpoint(self) -> Mapping[str, object]:
        return {
            "format": "test-p6-1b-semantic-provider-descriptor-v1",
            "provider_id": self.provider_id,
        }


@pytest.fixture()
def chat_journey_client(monkeypatch):
    import api.seed_runtime as seed_runtime
    import seed_platform.workbench as workbench_module
    from seed_platform.app_state import app_state

    workspace_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(workbench_module, "default_workspace_root", lambda: workspace_root)
    runtime = SeedRuntime(
        Seed(episode_id="p6-1b-chat-workbench-journey"),
        semantic_provider=_ChatJourneyProvider(),
    )
    runtime._workbench_environment = WorkbenchEnvironment(workspace_root)
    monkeypatch.setattr(seed_runtime, "_runtime", runtime)
    monkeypatch.setattr(app_state, "startup_complete", True)
    monkeypatch.setattr(app_state, "startup_error", None)
    client = TestClient(create_app(startup_tasks=False))
    yield client
    monkeypatch.setattr(seed_runtime, "_runtime", None)


def test_provider_evidence_drives_chat_plan_approval_execute_transport(chat_journey_client):
    client = chat_journey_client
    prompt = "读取 README.md 并准备工作台"

    interpreted = client.post(
        "/api/chat/workbench/interpret",
        json={"prompt": prompt, "history": [], "constraints": ["只读"]},
    )
    assert interpreted.status_code == 200
    interpretation = interpreted.json()
    assert interpretation["semantic_provider"]["state"] == "attached"
    assert interpretation["interpretation"]["status"] == "resolved"
    assert interpretation["decomposition"]["steps"]
    assert interpretation["execution"]["action_intent"] is None

    capabilities = client.get("/api/workbench/capabilities")
    assert capabilities.status_code == 200
    snapshot_id = capabilities.json()["snapshot_id"]
    planned = client.post(
        "/api/chat/workbench/natural-language/plan",
        json={
            "prompt": prompt,
            "semantic_evidence": interpretation["provider_evidence"],
            "snapshot_id": snapshot_id,
            "loop_id": "p6-1b-chat-journey",
            "max_steps": 1,
            "max_budget_units": 1.0,
        },
    )
    assert planned.status_code == 200, planned.text
    plan = planned.json()
    assert plan["status"] == "planned", (plan.get("reason_code"), plan.get("planning"))
    assert plan["plan_id"]
    assert plan["approval_requirements"] == []
    assert plan["execution"]["side_effects"] is False

    executed = client.post(
        "/api/chat/workbench/natural-language/execute",
        json={"plan_id": plan["plan_id"], "approval_tokens": {}},
    )
    assert executed.status_code == 200, executed.text
    outcome = executed.json()
    assert outcome["execution"]["status"] == "completed"
    assert outcome["execution"]["side_effects"] is False
