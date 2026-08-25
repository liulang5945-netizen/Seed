"""Evaluate structured tool execution, outcome feedback, and byte lesion."""

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
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p6-tool-execution-manifest-v1"
REPORT_FORMAT = "taiji-p6-tool-execution-v1"


class WeatherToolEnvironment:
    def execute_tool(self, tool_name: str, parameters: dict[str, object]) -> EnvironmentOutcome:
        valid = tool_name == "weather.lookup.v1" and parameters["action_symbol"] == 10
        return EnvironmentOutcome(
            sensation=98 if valid else 99,
            reward=1.0 if valid else -1.0,
            success=valid,
            terminal=True,
        )


def _adapter(*, generation: bool) -> TSKV8Adapter:
    adapter = TSKV8Adapter()
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    if generation:
        adapter.attach_generation_controller(GenerationController())
    adapter.observe(97, learn=False)
    adapter.plan_actions(
        (
            PlanningCandidate(
                candidate_id="lookup",
                action=WorldAction(action_id="lookup", kind="lookup_weather", tick=1),
                predicted_reward=0.8,
                success_probability=0.9,
                expected_progress=0.8,
            ),
            PlanningCandidate(
                candidate_id="idle",
                action=WorldAction(action_id="idle", kind="idle", tick=1),
                predicted_reward=0.0,
                success_probability=0.4,
                expected_progress=0.0,
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_weather", "idle"),
        use_plan=True,
    )
    return adapter


def evaluate() -> dict[str, object]:
    environment = WeatherToolEnvironment()
    adapter = _adapter(generation=True)
    call = adapter.generate_tool_call(tool_name="weather.lookup.v1")
    outcome = adapter.execute_tool_call(environment, call=call, learn=False)

    direct_byte = _adapter(generation=False)
    direct_byte_failure = False
    try:
        direct_byte.execute_tool_call(environment, learn=False)
    except RuntimeError:
        direct_byte_failure = True

    records = adapter._episodic_memory
    memory_outcome = None if records is None or not records.records else records.records[0].outcome
    gate_passed = bool(
        outcome.intent_id == call.intent_id
        and outcome.reward > 0.0
        and outcome.success is True
        and outcome.terminal is True
        and memory_outcome == outcome
        and direct_byte_failure
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "tool_name": call.tool_name,
            "tool_success": outcome.success,
            "tool_reward": outcome.reward,
            "tool_terminal": outcome.terminal,
            "outcome_intent_match": outcome.intent_id == call.intent_id,
            "episodic_memory_count": 0 if records is None else records.count,
            "outcome_recalled_from_memory": memory_outcome == outcome,
            "direct_byte_lesion_blocked": direct_byte_failure,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "structured tool execution succeeds, outcome returns to episodic memory, and direct-byte-only runtime cannot replace the tool organ",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "execute a generated structured tool call and close the outcome feedback loop",
        "lesions": ["direct_byte_only", "generation_organ", "episodic_outcome_write"],
        "signals": ["tool_name", "parameters", "success", "reward", "terminal", "intent_id"],
        "boundary": "simulated structured tool execution Gate only; no external service reliability or general language claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_tool_execution_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_tool_execution_baseline_20260825.json",
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

