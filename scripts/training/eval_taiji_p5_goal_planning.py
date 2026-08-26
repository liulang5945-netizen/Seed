"""Evaluate goal-directed candidate planning and outcome progress updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    Goal,
    GoalPlanner,
    GoalState,
    PlanningCandidate,
    PlanningConfig,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p5-goal-planning-manifest-v1"
REPORT_FORMAT = "taiji-p5-goal-planning-v1"


def _candidates() -> tuple[PlanningCandidate, ...]:
    return (
        PlanningCandidate(
            candidate_id="safe-route",
            action=WorldAction("safe-action", "safe", tick=0),
            predicted_reward=0.4,
            success_probability=0.95,
            expected_progress=0.6,
            uncertainty=0.1,
            resource_cost=0.1,
            conflict=0.0,
        ),
        PlanningCandidate(
            candidate_id="risky-route",
            action=WorldAction("risky-action", "risky", tick=0),
            predicted_reward=1.2,
            success_probability=0.45,
            expected_progress=1.0,
            uncertainty=0.8,
            resource_cost=0.1,
            conflict=0.3,
        ),
    )


def evaluate() -> dict[str, object]:
    goals = GoalState(
        tick=0,
        goals=(Goal("reach-target", "reach the target state", priority=1.0),),
    )
    candidates = _candidates()
    planner = GoalPlanner()
    decision = planner.plan(goals, candidates, tick=0)
    lesion_planner = GoalPlanner(
        PlanningConfig(
            reward_weight=1.0,
            progress_weight=0.0,
            success_weight=0.0,
            uncertainty_weight=0.0,
            resource_weight=0.0,
            conflict_weight=0.0,
        )
    )
    lesion_decision = lesion_planner.plan(goals, candidates, tick=0)

    runtime = TSKV8Adapter()
    runtime.attach_goal_planner(planner)
    runtime.set_goals(goals.goals)
    runtime_decision = runtime.plan_actions(candidates)
    runtime.observe(97, learn=False)
    selected = runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("risky", "safe"),
        use_plan=True,
    )
    runtime.settle_action(1.0, success=True, learn=False)
    after_outcome = runtime.cognitive_snapshot()
    restored = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    restored_state = restored.cognitive_snapshot()
    selected_plan = next(
        candidate
        for candidate in decision.plan.candidates
        if candidate.plan_id == decision.plan.selected_plan_id
    )
    gate_passed = bool(
        decision.selected.candidate_id == "safe-route"
        and lesion_decision.selected.candidate_id == "risky-route"
        and runtime_decision.selected.candidate_id == "safe-route"
        and selected.action_symbol == 11
        and after_outcome.action_intent is not None
        and after_outcome.action_intent.kind == "safe"
        and after_outcome.goals.goals[0].progress > 0.0
        and restored_state.goals == after_outcome.goals
        and restored_state.plan == after_outcome.plan
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "selected_candidate": decision.selected.candidate_id,
            "selected_expected_value": selected_plan.expected_value,
            "uncertainty_aware_lesion_candidate": lesion_decision.selected.candidate_id,
            "runtime_selected_action_kind": (
                after_outcome.action_intent.kind
                if after_outcome.action_intent is not None
                else None
            ),
            "runtime_goal_progress": after_outcome.goals.goals[0].progress,
            "checkpoint_goal_progress": restored_state.goals.goals[0].progress,
            "checkpoint_plan_id": restored_state.plan.selected_plan_id,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "planner selects an executable low-risk candidate, runtime executes it, and experienced outcome advances the goal through checkpoint",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "choose between safe and risky executable world actions for a registered goal",
        "signals": [
            "predicted_reward",
            "success_probability",
            "expected_progress",
            "uncertainty",
            "resource_cost",
            "conflict",
        ],
        "lesions": ["reward_only", "runtime_plan_bypass", "checkpoint_continuation"],
        "boundary": "goal-directed candidate ranking and progress update only; not long-horizon planning or language generation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_goal_planning_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_goal_planning_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
