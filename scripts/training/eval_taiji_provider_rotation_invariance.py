"""Run the P3-2 provider rotation and same-task decision invariance Gate."""

from __future__ import annotations

import hashlib
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
from taiji import (  # noqa: E402
    LanguageProviderArtifact,
    LanguageProviderArtifactRegistry,
    SemanticEvidenceProposal,
)

REPORT_FORMAT = "taiji-w7-p3-2-provider-rotation-invariance-v1"


def _artifact(artifact_id: str) -> LanguageProviderArtifact:
    digest = hashlib.sha256(artifact_id.encode()).hexdigest()
    return LanguageProviderArtifact(
        artifact_id=artifact_id,
        backend_id=f"semantic-surface-{artifact_id}",
        mode="raw",
        base_model=f"synthetic/{artifact_id}",
        provenance="eval.provider-rotation",
        content_digests=(("base_model", digest),),
    )


def _proposal(runtime: SeedRuntime, prompt: str, provider_id: str) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id=provider_id,
        goal_description="读取目标文件并准备编辑环境",
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
        confidence=0.9,
        ambiguity=0.1,
        provenance="eval.provider-rotation",
        tick=frame.timestamp,
    )


def _decision_projection(result: dict[str, object]) -> dict[str, object]:
    interpretation = result["interpretation"]
    decomposition = result["decomposition"]
    steps = []
    if isinstance(decomposition, dict):
        steps = [
            {
                "description": item["description"],
                "semantic_slots": item["semantic_slots"],
                "expected_outcome": item["expected_outcome"],
            }
            for item in decomposition["steps"]
        ]
    return {
        "goal_description": interpretation["goal_description"],
        "constraints": interpretation["constraints"],
        "status": interpretation["status"],
        "confidence": interpretation["confidence"],
        "ambiguity": interpretation["ambiguity"],
        "steps": steps,
    }


def _grounding_projection(planned: dict[str, object]) -> list[dict[str, object]]:
    projection = []
    for index, item in enumerate(planned["steps"]):
        planner = item["planner"]
        decision = planner["decision"]
        action = decision["selected"]["action_intent"] if decision else None
        projection.append(
            {
                "step_index": index,
                "kind": None if action is None else action["kind"],
                "parameters": None if action is None else action["parameters"],
                "planner_status": planner["status"],
            }
        )
    return projection


def evaluate() -> dict[str, object]:
    first = _artifact("provider-artifact-a")
    second = _artifact("provider-artifact-b")
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(first, allow=True)
        .with_artifact(second, allow=True)
        .activate(first.artifact_id)
    )
    rotated_registry = registry.activate(second.artifact_id)
    prompt = "读取 api/app.py 并检查语言"

    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        first_runtime = SeedRuntime(Seed(episode_id="p3-2-first"))
        _, first_frame = first_runtime._task_frame(prompt)
        first_runtime.model.architecture.ingest_input(first_frame, learn=False)
        first_admission = first_runtime.admit_semantic_provider_evidence(
            prompt, _proposal(first_runtime, prompt, first.artifact_id)
        )
        first_plan = first_runtime.plan_task_sequence(
            snapshot_id=first_runtime.workbench_environment.capability_snapshot.snapshot_id,
            parameter_bindings=(
                {"workspace.read": {"path": "api/app.py"}},
                {"workspace.programming_language.resolve": {"path": "api/app.py"}},
            ),
            resource_budget=0.8,
        )

        second_runtime = SeedRuntime(Seed(episode_id="p3-2-second"))
        _, second_frame = second_runtime._task_frame(prompt)
        second_runtime.model.architecture.ingest_input(second_frame, learn=False)
        second_admission = second_runtime.admit_semantic_provider_evidence(
            prompt, _proposal(second_runtime, prompt, second.artifact_id)
        )
        second_plan = second_runtime.plan_task_sequence(
            snapshot_id=second_runtime.workbench_environment.capability_snapshot.snapshot_id,
            parameter_bindings=(
                {"workspace.read": {"path": "api/app.py"}},
                {"workspace.programming_language.resolve": {"path": "api/app.py"}},
            ),
            resource_budget=0.8,
        )

    first_semantics = _decision_projection(first_admission)
    second_semantics = _decision_projection(second_admission)
    first_grounding = _grounding_projection(first_plan)
    second_grounding = _grounding_projection(second_plan)
    metrics = {
        "artifact_registry_rotates_with_previous_pointer": (
            registry.active_artifact_id == first.artifact_id
            and rotated_registry.active_artifact_id == second.artifact_id
            and rotated_registry.previous_artifact_id == first.artifact_id
            and rotated_registry.get(first.artifact_id) == first
            and rotated_registry.get(second.artifact_id) == second
        ),
        "provider_identity_is_auditable_but_not_cognitive_owner": (
            first_admission["provider_evidence"]["provider_id"] == first.artifact_id
            and second_admission["provider_evidence"]["provider_id"] == second.artifact_id
            and first_admission["interpretation"]["provenance"]
            != second_admission["interpretation"]["provenance"]
        ),
        "same_task_semantic_decision_is_invariant": first_semantics == second_semantics,
        "same_task_grounding_is_invariant": first_grounding == second_grounding,
        "rotation_does_not_execute_workbench": (
            first_admission["execution"]["action_intent"] is None
            and second_admission["execution"]["action_intent"] is None
            and first_plan["execution"]["side_effects"] is False
            and second_plan["execution"]["side_effects"] is False
            and first_runtime.workbench_audit.events == ()
            and second_runtime.workbench_audit.events == ()
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "artifacts": {
            "before": first.to_payload(),
            "after": second.to_payload(),
            "active_before": registry.active_artifact_id,
            "active_after": rotated_registry.active_artifact_id,
            "previous_after": rotated_registry.previous_artifact_id,
        },
        "decision_projection": {"first": first_semantics, "second": second_semantics},
        "grounding_projection": {"first": first_grounding, "second": second_grounding},
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "provider artifact rotation may change auditable provider provenance, but "
                "the same live task must yield the same Taiji semantic and Workbench grounding "
                "decision without provider execution authority"
            ),
        },
        "gap": {
            "current": (
                "deterministic metadata rotation and same-task invariance are proven; this "
                "does not yet load a real packaged Qwen artifact or prove provider language quality"
            ),
            "next": "run the packaged-client artifact rotation, watchdog fallback, and restart rebinding Gate",
        },
        "boundary": (
            "This Gate does not claim that provider output is Taiji cognition, does not claim "
            "open-domain autonomous IDE execution, and does not cover CUDA, CI, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = PROJECT_ROOT / "reports" / "taiji_w7_p3_2_provider_rotation_invariance_20260831.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
