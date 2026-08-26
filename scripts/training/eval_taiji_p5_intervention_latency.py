"""Evaluate delayed-reward planning, intervention recovery, and planning lesions."""

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
    ImaginedRollout,
    PlanningCandidate,
    PlanningConfig,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p5-intervention-latency-manifest-v1"
REPORT_FORMAT = "taiji-p5-intervention-latency-v1"


def _candidate(
    rollout_id: str,
    index: int,
    kind: str,
    reward: float,
    success: float,
    progress: float,
    uncertainty: float,
    resource_cost: float,
    conflict: float,
) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=f"{rollout_id}-step-{index}",
        action=WorldAction(
            action_id=f"{rollout_id}-action-{index}",
            kind=kind,
            tick=index,
            provenance="imagined",
        ),
        predicted_reward=reward,
        success_probability=success,
        expected_progress=progress,
        uncertainty=uncertainty,
        resource_cost=resource_cost,
        conflict=conflict,
    )


def _rollouts() -> tuple[ImaginedRollout, ...]:
    return (
        ImaginedRollout(
            rollout_id="delayed-safe",
            goal_id="finish-task",
            confidence=0.9,
            steps=(
                _candidate("delayed-safe", 0, "prepare", 0.0, 0.95, 0.2, 0.1, 0.1, 0.0),
                _candidate("delayed-safe", 1, "finish", 1.0, 0.9, 1.0, 0.1, 0.2, 0.0),
            ),
        ),
        ImaginedRollout(
            rollout_id="immediate-risky",
            goal_id="finish-task",
            confidence=0.6,
            steps=(
                _candidate("immediate-risky", 0, "shortcut", 0.7, 0.55, 0.3, 0.7, 0.1, 0.2),
                _candidate("immediate-risky", 1, "crash", -0.5, 0.4, 0.2, 0.9, 0.3, 0.5),
            ),
        ),
    )


def evaluate() -> dict[str, object]:
    goals = (Goal("finish-task", "finish a delayed task", priority=1.0),)
    rollouts = _rollouts()
    planner = GoalPlanner()
    goal_state = GoalState(tick=0, goals=goals)
    decision = planner.plan_rollouts(goal_state, rollouts, tick=0)
    reactive = max(rollouts, key=lambda rollout: rollout.steps[0].predicted_reward)
    value_lesion = GoalPlanner(
        PlanningConfig(
            progress_weight=0.0,
            uncertainty_weight=0.0,
            resource_weight=0.0,
            conflict_weight=0.0,
            discount=0.0,
        )
    ).plan_rollouts(goal_state, rollouts, tick=0)

    runtime = TSKV8Adapter()
    runtime.attach_goal_planner(planner)
    runtime.set_goals(goals)
    runtime.plan_rollouts(rollouts)
    runtime.observe(97, learn=False)
    runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("shortcut", "prepare"),
        use_plan=True,
    )
    runtime.settle_action(-0.6, success=False, learn=False)
    intervention_replan = runtime.replan_required
    runtime.observe(98, learn=False)
    recovery = ImaginedRollout(
        rollout_id="recovery",
        goal_id="finish-task",
        confidence=0.95,
        steps=(
            _candidate("recovery", 0, "recover", 0.4, 0.9, 0.5, 0.1, 0.1, 0.0),
            _candidate("recovery", 1, "finish", 0.8, 0.9, 1.0, 0.1, 0.2, 0.0),
        ),
    )
    recovery_decision = runtime.plan_rollouts((recovery,))
    runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("recover", "prepare"),
        use_plan=True,
    )
    runtime.settle_action(0.4, success=True, learn=False)
    final_state = runtime.cognitive_snapshot()
    planner_gain = (
        decision.selected.steps[0].success_probability - reactive.steps[0].success_probability
    )
    gate_passed = bool(
        decision.selected.rollout_id == "delayed-safe"
        and reactive.rollout_id == "immediate-risky"
        and value_lesion.selected.rollout_id == "immediate-risky"
        and planner_gain > 0.3
        and intervention_replan
        and recovery_decision.selected.rollout_id == "recovery"
        and not runtime.replan_required
        and final_state.goals.goals[0].progress > 0.0
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "planner_rollout": decision.selected.rollout_id,
            "reactive_rollout": reactive.rollout_id,
            "value_world_model_lesion_rollout": value_lesion.selected.rollout_id,
            "delayed_first_reward": decision.selected.steps[0].predicted_reward,
            "delayed_final_reward": decision.selected.steps[-1].predicted_reward,
            "planner_vs_reactive_success_gain": planner_gain,
            "intervention_replan_required": intervention_replan,
            "recovery_rollout": recovery_decision.selected.rollout_id,
            "final_replan_required": runtime.replan_required,
            "final_goal_progress": final_state.goals.goals[0].progress,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "planning beats immediate-reward reactive selection on delayed reward, lesions prefer risk, and intervention recovery executes",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "compare delayed safe rollout with immediate risky shortcut and recover after intervention",
        "lesions": ["reactive_immediate_reward", "value_world_model", "no_replan_recovery"],
        "signals": [
            "delayed_reward",
            "success_probability",
            "uncertainty",
            "resource_cost",
            "conflict",
        ],
        "boundary": "short delayed-reward intervention Gate only; not general long-horizon planning",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_intervention_latency_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_intervention_latency_baseline_20260825.json",
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
