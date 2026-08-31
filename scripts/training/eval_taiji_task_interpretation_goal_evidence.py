"""Run the P2-2 Taiji-owned task interpretation/Goal evidence Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.seed_runtime as seed_runtime_module  # noqa: E402
from api.app import create_app  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-task-interpretation-goal-evidence-v1"


def evaluate() -> dict[str, object]:
    previous_runtime = seed_runtime_module._runtime
    seed_runtime_module._runtime = SeedRuntime(Seed(episode_id="p2-2-canary"))
    try:
        app = create_app(startup_tasks=False)
        with TestClient(app) as client:
            response = client.post(
                "/api/chat/workbench/interpret",
                json={
                    "prompt": "请读取 README.md 并告诉我项目如何启动",
                    "history": [],
                    "constraints": ["只读", "可恢复"],
                },
            )
            payload = response.json()
            events = client.get("/api/workbench/events").json().get("events", [])
    finally:
        seed_runtime_module._runtime = previous_runtime

    interpretation = payload.get("interpretation", {}) if isinstance(payload, dict) else {}
    goal = payload.get("goal", {}) if isinstance(payload, dict) else {}
    execution = payload.get("execution", {}) if isinstance(payload, dict) else {}
    metrics = {
        "prose_is_admitted_as_taiji_evidence": (
            response.status_code == 200
            and interpretation.get("format") == "taiji-task-interpretation-v1"
            and interpretation.get("status") == "candidate"
        ),
        "goal_is_projected_from_evidence": (
            bool(interpretation.get("goal_id"))
            and interpretation.get("goal_id") == goal.get("goal_id")
            and interpretation.get("goal_description") == goal.get("description")
        ),
        "evidence_is_content_addressed_and_uncertain": (
            len(str(interpretation.get("input_digest", ""))) == 64
            and len(str(interpretation.get("evidence_digest", ""))) == 64
            and interpretation.get("confidence") == 0.0
            and interpretation.get("ambiguity") == 1.0
        ),
        "interpretation_cannot_execute_or_select_tool": (
            execution.get("status") == "not_planned"
            and execution.get("action_intent") is None
            and execution.get("tool_call") is None
            and execution.get("side_effects") is False
            and events == []
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "endpoint": "/api/chat/workbench/interpret",
        "status_code": response.status_code,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "natural language becomes content-addressed Taiji Goal evidence with visible "
                "uncertainty, while planner/action/tool execution remains a later owned stage"
            ),
        },
        "gap": {
            "current": "TaskInterpretation preserves a goal candidate but does not resolve semantics",
            "next": "feed goal evidence, Workbench affordances and resource/risk state into planner",
        },
        "boundary": (
            "This Gate proves the Taiji-owned interpretation boundary only; it does not claim "
            "autonomous tool planning, IDE execution, provider quality, CUDA, CI, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p2_task_interpretation_goal_evidence_20260831.json",
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
