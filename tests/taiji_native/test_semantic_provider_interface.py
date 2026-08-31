"""P6-1a independent semantic-provider interface tests."""

from collections.abc import Mapping

from api.seed_runtime import SeedRuntime
from seed import Seed
from taiji import (
    SemanticEvidenceProposal,
    SemanticEvidenceProvider,
    SemanticProviderRequest,
)


class _EvidenceProvider:
    provider_id = "test.semantic.interface"

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        return SemanticEvidenceProposal.from_frame(
            request.frame,
            provider_id=self.provider_id,
            goal_description="检查目标文件并准备工作台",
            constraints=request.constraints,
            context_digest=request.context_digest,
            semantic_steps=(
                {
                    "description": "读取目标文件并确认当前内容",
                    "semantic_slots": {"operation": "inspect", "path": "README.md"},
                },
            ),
            confidence=0.9,
            ambiguity=0.1,
            provenance="tests.semantic.interface",
            tick=request.frame.timestamp,
        )

    def checkpoint(self) -> Mapping[str, object]:
        return {
            "format": "test-semantic-provider-descriptor-v1",
            "provider_id": self.provider_id,
        }


def test_provider_request_is_content_addressed_and_does_not_carry_execution_authority() -> None:
    runtime = SeedRuntime(Seed(episode_id="semantic-provider-interface-request"))
    _, frame = runtime._task_frame("读取 README.md 并准备工作台")

    first = SemanticProviderRequest.from_frame(frame, constraints=("只读",))
    second = SemanticProviderRequest.from_frame(frame, constraints=("只读",))

    assert first.request_id == second.request_id
    assert first.to_payload()["input_digest"] == first.input_digest
    assert "action_intent" not in first.to_payload()
    assert "capability_id" not in first.to_payload()
    assert "parameter_bindings" not in first.to_payload()


def test_runtime_accepts_only_independent_semantic_evidence_provider() -> None:
    provider = _EvidenceProvider()
    assert isinstance(provider, SemanticEvidenceProvider)
    runtime = SeedRuntime(
        Seed(episode_id="semantic-provider-interface-runtime"),
        semantic_provider=provider,
    )

    result = runtime.interpret_workbench_task(
        "读取 README.md 并准备工作台",
        constraints=("只读",),
    )

    assert result["semantic_provider"]["state"] == "attached"
    assert result["semantic_provider"]["provider_id"] == provider.provider_id
    assert result["interpretation"]["status"] == "resolved"
    assert result["decomposition"]["steps"]
    assert result["execution"] == {
        "status": "not_planned",
        "action_intent": None,
        "tool_call": None,
        "side_effects": False,
        "next": "taiji_workbench_grounding",
    }
    assert runtime.workbench_audit.events == ()


def test_no_provider_remains_an_honest_goal_only_boundary() -> None:
    runtime = SeedRuntime(Seed(episode_id="semantic-provider-interface-unavailable"))

    result = runtime.interpret_workbench_task("读取 README.md")

    assert result["semantic_provider"]["state"] == "unavailable"
    assert result["semantic_provider"]["reason_code"] == "semantic_provider_not_attached"
    assert result["interpretation"]["status"] == "candidate"
    assert "decomposition" not in result
    assert runtime.workbench_audit.events == ()
