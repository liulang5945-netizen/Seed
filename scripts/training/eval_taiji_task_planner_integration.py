"""Run the P2-3 Taiji planner integration Gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from taiji import InputFrame, TaskInterpretation  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-task-planner-integration-v1"


def _frame() -> InputFrame:
    return InputFrame(
        input_id="p2-3-task",
        modality="text",
        payload="读取 README.md".encode(),
        source="scripts.p2_3",
        provenance="scripts.p2_3",
    )


def evaluate() -> dict[str, object]:
    runtime = SeedRuntime(Seed(episode_id="p2-3-resolved"))
    architecture = runtime.model.architecture
    frame = _frame()
    architecture.ingest_input(frame, learn=False)
    interpretation = TaskInterpretation.from_input(
        frame,
        status="resolved",
        confidence=0.9,
        ambiguity=0.1,
        tick=architecture.tick,
    )
    architecture.admit_task_interpretation(interpretation)
    snapshot = runtime.workbench_environment.capability_snapshot
    affordances = snapshot.to_taiji_affordances(
        {"workspace.read": {"path": "README.md"}}
    )
    architecture.set_world_affordances(affordances)
    planned = architecture.plan_task_from_current_state(resource_budget=0.8)
    decision = planned["decision"]

    unresolved_runtime = SeedRuntime(Seed(episode_id="p2-3-unresolved"))
    unresolved_snapshot = unresolved_runtime.workbench_environment.capability_snapshot.snapshot_id
    unresolved = unresolved_runtime.plan_task(
        "请读取 README.md",
        snapshot_id=unresolved_snapshot,
        parameter_bindings={"workspace.read": {"path": "README.md"}},
    )
    metrics = {
        "resolved_goal_reaches_taiji_planner": (
            planned["status"] == "planned"
            and decision is not None
            and decision.action_intent.kind == "workspace.read"
            and decision.action_intent.source_goal_id == interpretation.goal_id
        ),
        "confidence_and_resource_state_reach_decision": (
            decision is not None
            and math.isclose(decision.action_intent.confidence, 0.9, abs_tol=1e-6)
            and math.isclose(float(decision.context.features[-1]), 0.8, abs_tol=1e-6)
        ),
        "unresolved_goal_stops_before_action_intent": (
            unresolved["planner"]["status"] == "needs_clarification"
            and unresolved["execution"]["action_intent"] is None
        ),
        "planning_has_no_workbench_side_effect": (
            runtime.workbench_audit.events == ()
            and unresolved_runtime.workbench_audit.events == ()
            and decision is not None
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "resolved Taiji Goal evidence is combined with live structured Workbench "
                "affordances and resource/confidence state to form a non-executing ActionIntent; "
                "unresolved prose stops at clarification"
            ),
        },
        "gap": {
            "current": "planner integration is non-executing and requires resolved semantic evidence plus structured bindings",
            "next": "connect semantic resolution and programming-language evidence, then pass the selected intent through preview/policy",
        },
        "boundary": (
            "This Gate does not claim natural-language semantic resolution, IDE execution, "
            "provider quality, CUDA, CI, open-domain gains, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_task_planner_integration_20260831.json",
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
