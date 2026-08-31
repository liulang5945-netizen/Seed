"""Run the P2-7 controlled semantic task-decomposition Gate."""

from __future__ import annotations

import argparse
import copy
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
from taiji import InputFrame, TaskDecomposition, TaskInterpretation, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-task-decomposition-v1"


def _frame() -> InputFrame:
    return InputFrame(
        input_id="p2-7-task",
        modality="text",
        payload="更新 api/app.py 并验证结果".encode(),
        source="scripts.p2_7",
        provenance="scripts.p2_7",
    )


def _semantic_steps() -> tuple[dict[str, object], ...]:
    return (
        {
            "description": "读取目标文件并确认当前内容",
            "semantic_slots": {"operation": "inspect", "path": "api/app.py"},
            "expected_outcome": "获得当前文件事实",
        },
        {
            "description": "根据文件证据解析适用的语言环境",
            "semantic_slots": {"operation": "resolve-language", "path": "api/app.py"},
            "expected_outcome": "获得语言选择证据",
        },
    )


def evaluate() -> dict[str, object]:
    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        runtime = SeedRuntime(Seed(episode_id="p2-7-task-decomposition"))
        architecture = runtime.model.architecture
        frame = _frame()
        architecture.ingest_input(frame, learn=False)
        interpretation = architecture.interpret_task_input(
            frame,
            goal_description="更新 api/app.py 并验证结果",
            status="resolved",
            confidence=0.9,
            ambiguity=0.1,
        )
        admitted = runtime.decompose_task(_semantic_steps())
        decomposition = architecture.last_task_decomposition
        if decomposition is None:
            raise AssertionError("task decomposition was not admitted")
        checkpoint = architecture.native_checkpoint()
        restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
        planned = runtime.plan_task_sequence(
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            parameter_bindings=(
                {"workspace.read": {"path": "api/app.py"}},
                {"workspace.programming_language.resolve": {"path": "api/app.py"}},
            ),
            resource_budget=0.8,
        )
        tampered = copy.deepcopy(admitted["decomposition"])
        tampered["steps"][0]["semantic_slots"]["tool"] = "workspace.read"
        try:
            TaskDecomposition.from_payload(tampered)
        except ValueError as exc:
            tamper_rejected = "execution field" in str(exc)
        else:
            tamper_rejected = False

    step_kinds = []
    for item in planned["steps"]:
        decision = item["planner"]["decision"] or {}
        selected = decision.get("selected", {}) if isinstance(decision, dict) else {}
        action_intent = selected.get("action_intent", {})
        if isinstance(action_intent, dict):
            step_kinds.append(action_intent.get("kind"))

    semantic_payload = admitted["decomposition"]
    semantic_step_keys = {
        key
        for step in semantic_payload["steps"]
        for key in step.get("semantic_slots", {})
    }
    metrics = {
        "semantic_decomposition_is_bound_to_goal": (
            decomposition.interpretation_id == interpretation.interpretation_id
            and decomposition.goal_id == interpretation.goal_id
            and decomposition.status == "resolved"
            and len(decomposition.steps) == 2
        ),
        "semantic_evidence_contains_no_execution_binding": (
            not semantic_step_keys.intersection(
                {"action", "action_kind", "capability", "capability_id", "intent", "tool", "tool_id"}
            )
            and all("action_intent" not in step for step in semantic_payload["steps"])
            and tamper_rejected
        ),
        "taiji_grounds_each_step_without_execution": (
            planned["status"] == "planned"
            and len(planned["steps"]) == 2
            and step_kinds == ["workspace.read", "workspace.programming_language.resolve"]
            and planned["execution"]["status"] == "not_executed"
            and planned["execution"]["side_effects"] is False
            and runtime.workbench_audit.events == ()
        ),
        "decomposition_checkpoint_roundtrip_is_stable": (
            restored.last_task_decomposition == decomposition
            and restored.last_task_interpretation == interpretation
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "semantic evidence may describe a bounded multi-step task and survive a "
                "checkpoint, but it cannot inject tools or intents; Taiji grounds each step "
                "against live Workbench affordances without executing it"
            ),
        },
        "gap": {
            "current": (
                "multi-step grounding is proven only when semantic evidence is already resolved "
                "and bindings are supplied by the Workbench boundary"
            ),
            "next": (
                "connect the language provider to this tool-free semantic evidence contract, "
                "then evaluate provider rotation and same-task decision invariance"
            ),
        },
        "boundary": (
            "This Gate does not claim ordinary prose is autonomously resolved, provider output "
            "quality, unrestricted execution, CUDA, CI, open-domain gains, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_task_decomposition_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
