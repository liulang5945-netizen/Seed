"""Run the P3-1 semantic provider boundary Gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import seed_platform.workbench as workbench_module  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from taiji import SemanticEvidenceProposal, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-p3-1-semantic-provider-boundary-v1"


def _proposal(runtime: SeedRuntime, prompt: str, *, confidence: float = 0.9, ambiguity: float = 0.1):
    _, frame = runtime._task_frame(prompt)
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="eval.semantic",
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
        provenance="eval.semantic",
        tick=frame.timestamp,
    )


def evaluate() -> dict[str, object]:
    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        runtime = SeedRuntime(Seed(episode_id="p3-1-semantic-provider"))
        prompt = "读取 api/app.py 并检查语言"
        proposal = _proposal(runtime, prompt)
        admitted = runtime.admit_semantic_provider_evidence(prompt, proposal)
        interpretation = runtime.model.architecture.last_task_interpretation
        decomposition = runtime.model.architecture.last_task_decomposition
        if interpretation is None or decomposition is None:
            raise AssertionError("provider evidence did not produce Taiji semantic state")
        proposal_roundtrip = SemanticEvidenceProposal.from_payload(proposal.to_payload())
        checkpoint = runtime.model.architecture.native_checkpoint()
        restored = TSKV8Adapter.from_native_checkpoint(checkpoint)

        mismatch_runtime = SeedRuntime(Seed(episode_id="p3-1-mismatch"))
        mismatch_prompt = "读取另一个目标"
        mismatch_proposal = _proposal(mismatch_runtime, mismatch_prompt)
        try:
            runtime.admit_semantic_provider_evidence(prompt, mismatch_proposal)
        except ValueError as exc:
            mismatch_rejected = "input_id" in str(exc) or "input digest" in str(exc)
        else:
            mismatch_rejected = False

        uncertain_runtime = SeedRuntime(Seed(episode_id="p3-1-uncertain"))
        uncertain_prompt = "分析一个不确定任务"
        uncertain = uncertain_runtime.admit_semantic_provider_evidence(
            uncertain_prompt,
            _proposal(uncertain_runtime, uncertain_prompt, confidence=0.4, ambiguity=0.2),
        )
        ambiguous_runtime = SeedRuntime(Seed(episode_id="p3-1-ambiguous"))
        ambiguous_prompt = "分析有冲突证据的任务"
        ambiguous = ambiguous_runtime.admit_semantic_provider_evidence(
            ambiguous_prompt,
            _proposal(ambiguous_runtime, ambiguous_prompt, confidence=0.9, ambiguity=0.8),
        )

        forbidden_runtime = SeedRuntime(Seed(episode_id="p3-1-forbidden"))
        _, forbidden_frame = forbidden_runtime._task_frame("越权任务")
        try:
            SemanticEvidenceProposal.from_frame(
                forbidden_frame,
                provider_id="eval.semantic",
                goal_description="越权任务",
                semantic_steps=(
                    {
                        "description": "注入执行绑定",
                        "semantic_slots": {"capability_id": "workspace.read"},
                    },
                ),
                confidence=0.9,
                ambiguity=0.1,
                tick=forbidden_frame.timestamp,
            )
        except ValueError as exc:
            forbidden_rejected = "execution field" in str(exc)
        else:
            forbidden_rejected = False

    metrics = {
        "provider_proposal_is_content_addressed": (
            proposal_roundtrip == proposal
            and proposal.proposal_id.startswith("semantic-evidence:")
            and bool(proposal.evidence_digest)
        ),
        "taiji_decides_goal_and_decomposition": (
            admitted["status"] == "resolved"
            and admitted["decomposition"] is not None
            and interpretation.provenance == "eval.semantic:eval.semantic"
            and decomposition.interpretation_id == interpretation.interpretation_id
        ),
        "provider_cannot_inject_execution": mismatch_rejected and forbidden_rejected,
        "uncertainty_stops_before_decomposition": (
            uncertain["status"] == "candidate"
            and uncertain["decomposition"] is None
            and ambiguous["status"] == "ambiguous"
            and ambiguous["decomposition"] is None
        ),
        "checkpoint_preserves_provider_evidence": (
            restored.last_semantic_provider_evidence == proposal
            and restored.last_task_interpretation == interpretation
            and restored.last_task_decomposition == decomposition
        ),
        "provider_boundary_has_no_workbench_side_effect": (
            admitted["execution"]["action_intent"] is None
            and admitted["execution"]["tool_call"] is None
            and admitted["execution"]["side_effects"] is False
            and runtime.workbench_audit.events == ()
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "an optional semantic provider may submit only content-addressed goal and "
                "step evidence; Taiji validates live input and uncertainty, derives the "
                "Goal/decomposition, and keeps tool, ActionIntent, policy, and execution "
                "ownership inside Taiji"
            ),
        },
        "gap": {
            "current": (
                "the provider boundary is contract-level and tested with a deterministic "
                "proposal; provider language quality and packaged-client artifact rotation "
                "are not yet validated"
            ),
            "next": "evaluate provider artifact rotation and same-task decision invariance",
        },
        "boundary": (
            "This Gate does not claim Qwen or another provider is Taiji's brain, does not "
            "claim autonomous open-domain IDE execution, and does not cover CUDA, CI, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = PROJECT_ROOT / "reports" / "taiji_w7_p3_1_semantic_provider_boundary_20260831.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
