"""P3-1 provider semantic evidence boundary tests."""

import pytest

from api.seed_runtime import SeedRuntime
from seed import Seed
from taiji import InputFrame, SemanticEvidenceProposal


def _frame(model: Seed, payload: bytes = "读取 api/app.py 并检查语言".encode()) -> InputFrame:
    return InputFrame(
        input_id=f"provider-task:{model.architecture.tick}",
        modality="text",
        payload=payload,
        source="tests.semantic_provider",
        timestamp=model.architecture.tick,
        provenance="tests.semantic_provider",
        confidence=1.0,
    )


def _proposal(frame: InputFrame, *, confidence: float = 0.9, ambiguity: float = 0.1):
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="test.semantic",
        goal_description="检查目标文件并准备编辑环境",
        constraints=("保持可恢复",),
        semantic_steps=(
            {
                "description": "读取目标文件并确认当前内容",
                "semantic_slots": {"operation": "inspect", "path": "api/app.py"},
            },
            {
                "description": "根据文件证据准备语言环境",
                "semantic_slots": {"operation": "resolve-language", "path": "api/app.py"},
            },
        ),
        confidence=confidence,
        ambiguity=ambiguity,
        provenance="tests.provider",
        tick=frame.timestamp,
    )


def test_provider_evidence_is_resolved_by_taiji_and_checkpointed() -> None:
    model = Seed(episode_id="semantic-provider")
    frame = _frame(model)
    proposal = _proposal(frame)

    interpretation, decomposition = model.architecture.admit_semantic_provider_evidence(
        frame, proposal
    )

    assert interpretation.status == "resolved"
    assert interpretation.provenance == "tests.provider:test.semantic"
    assert decomposition is not None
    assert model.architecture.last_semantic_provider_evidence == proposal
    assert model.architecture.cognitive_snapshot().action_intent is None
    assert Seed.from_checkpoint(model.checkpoint()).architecture.last_semantic_provider_evidence == (
        proposal
    )


def test_provider_evidence_mismatch_fails_before_mutation() -> None:
    model = Seed(episode_id="semantic-provider-mismatch")
    live_frame = _frame(model)
    other_model = Seed(episode_id="semantic-provider-other")
    stale_proposal = _proposal(_frame(other_model, "另一个任务".encode()))

    with pytest.raises(ValueError, match="input digest"):
        model.architecture.admit_semantic_provider_evidence(live_frame, stale_proposal)

    assert model.architecture.last_task_interpretation is None
    assert model.architecture.last_semantic_provider_evidence is None


def test_provider_evidence_rejects_execution_fields() -> None:
    model = Seed(episode_id="semantic-provider-fields")
    frame = _frame(model)

    with pytest.raises(ValueError, match="execution field"):
        SemanticEvidenceProposal.from_frame(
            frame,
            provider_id="test.semantic",
            goal_description="检查文件",
            semantic_steps=(
                {
                    "description": "越权步骤",
                    "semantic_slots": {"tool": "workspace.read"},
                },
            ),
            confidence=0.9,
            ambiguity=0.1,
        )


def test_low_confidence_provider_evidence_stays_out_of_decomposition() -> None:
    model = Seed(episode_id="semantic-provider-uncertain")
    frame = _frame(model)
    interpretation, decomposition = model.architecture.admit_semantic_provider_evidence(
        frame, _proposal(frame, confidence=0.4, ambiguity=0.2)
    )

    assert interpretation.status == "candidate"
    assert decomposition is None
    assert model.architecture.cognitive_snapshot().action_intent is None


def test_runtime_provider_boundary_has_no_workbench_side_effect() -> None:
    runtime = SeedRuntime(Seed(episode_id="semantic-provider-runtime"))
    prompt, frame = runtime._task_frame("读取 api/app.py 并检查语言")
    proposal = _proposal(frame)

    result = runtime.admit_semantic_provider_evidence(prompt, proposal)

    assert result["status"] == "resolved"
    assert result["decomposition"] is not None
    assert result["execution"]["action_intent"] is None
    assert result["execution"]["tool_call"] is None
    assert result["execution"]["side_effects"] is False
    assert runtime.workbench_audit.events == ()
