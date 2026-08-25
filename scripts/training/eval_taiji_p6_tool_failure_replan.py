"""Evaluate tool failure, prediction error, replanning, and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    EnvironmentOutcome,
    EpisodicMemoryStore,
    GenerationController,
    Goal,
    GoalPlanner,
    ImaginedRollout,
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p6-tool-failure-replan-manifest-v1"
REPORT_FORMAT = "taiji-p6-tool-failure-replan-v1"


class FailingThenRecoveringEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_tool(self, tool_name: str, parameters: dict[str, object]) -> EnvironmentOutcome:
        del parameters
        self.calls.append(tool_name)
        recovered = tool_name == "weather.recover.v1"
        return EnvironmentOutcome(
            sensation=98 if recovered else 99,
            reward=0.4 if recovered else -1.0,
            success=recovered,
            terminal=recovered,
        )


def _candidate(rollout_id: str, kind: str, reward: float, progress: float) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=f"{rollout_id}-step-0",
        action=WorldAction(
            action_id=f"{rollout_id}-action-0",
            kind=kind,
            tick=1,
            provenance="imagined",
        ),
        predicted_reward=reward,
        success_probability=0.9,
        expected_progress=progress,
        uncertainty=0.1,
    )


def evaluate() -> dict[str, object]:
    environment = FailingThenRecoveringEnvironment()
    adapter = TSKV8Adapter()
    adapter.attach_goal_planner(GoalPlanner())
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.observe(97, learn=False)
    adapter.plan_rollouts(
        (
            ImaginedRollout(
                rollout_id="lookup-first",
                goal_id="stay-informed",
                confidence=0.9,
                steps=(_candidate("lookup-first", "lookup_weather", 1.0, 0.8),),
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_weather", "idle"),
        use_plan=True,
    )
    first_call = adapter.generate_tool_call(tool_name="weather.lookup.v1")
    first_outcome = adapter.execute_tool_call(environment, call=first_call, learn=False)
    first_replan = adapter.replan_required
    first_error = adapter.last_rollout_prediction_error

    adapter.plan_rollouts(
        (
            ImaginedRollout(
                rollout_id="recover-second",
                goal_id="stay-informed",
                confidence=0.95,
                steps=(_candidate("recover-second", "recover_weather", 0.2, 0.7),),
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("recover_weather", "idle"),
        use_plan=True,
    )
    second_call = adapter.generate_tool_call(tool_name="weather.recover.v1")
    second_outcome = adapter.execute_tool_call(environment, call=second_call, learn=False)
    store = adapter._episodic_memory
    gate_passed = bool(
        first_outcome.success is False
        and first_replan
        and first_error is not None
        and first_error > 0.25
        and second_outcome.success is True
        and not adapter.replan_required
        and environment.calls == ["weather.lookup.v1", "weather.recover.v1"]
        and store is not None
        and store.count == 2
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "first_tool": first_call.tool_name,
            "first_success": first_outcome.success,
            "first_prediction_error": first_error,
            "replan_after_failure": first_replan,
            "recovery_tool": second_call.tool_name,
            "recovery_success": second_outcome.success,
            "final_replan_required": adapter.replan_required,
            "episodic_memory_count": 0 if store is None else store.count,
            "tool_sequence": environment.calls,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "tool failure creates prediction error and replan, alternative tool succeeds, and both outcomes remain in episodic memory",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "fail the first structured tool call, replan to a recovery tool, and retain both outcomes",
        "lesions": ["no_prediction_error_replan", "no_recovery_tool", "no_episodic_write"],
        "signals": ["tool_success", "prediction_error", "replan_required", "tool_sequence", "episodic_memory"],
        "boundary": "simulated failure/replan Gate only; no broad reliability or general planning claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_tool_failure_replan_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_tool_failure_replan_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

