"""Evaluate multi-step imagined rollout selection and replanning triggers."""

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
    ImaginedRollout,
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p5-imagined-rollout-manifest-v1"
REPORT_FORMAT = "taiji-p5-imagined-rollout-v1"


def _step(
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
            rollout_id="safe-rollout",
            goal_id="reach-target",
            confidence=1.0,
            steps=(
                _step("safe-rollout", 0, "scout", 0.1, 0.95, 0.2, 0.1, 0.1, 0.0),
                _step("safe-rollout", 1, "finish", 0.6, 0.9, 1.0, 0.1, 0.2, 0.0),
            ),
        ),
        ImaginedRollout(
            rollout_id="risky-rollout",
            goal_id="reach-target",
            confidence=1.0,
            steps=(
                _step("risky-rollout", 0, "gamble", 0.9, 0.55, 0.3, 0.7, 0.1, 0.2),
                _step("risky-rollout", 1, "commit", 0.9, 0.5, 1.0, 0.8, 0.2, 0.4),
            ),
        ),
    )


def evaluate() -> dict[str, object]:
    runtime = TSKV8Adapter()
    runtime.attach_goal_planner(GoalPlanner())
    runtime.set_goals((Goal("reach-target", "reach the target state", priority=1.0),))
    rollouts = _rollouts()
    decision = runtime.plan_rollouts(rollouts)
    runtime.observe(97, learn=False)
    runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("gamble", "scout"),
        use_plan=True,
    )
    runtime.settle_action(-0.5, success=False, learn=False)
    after_error = runtime.cognitive_snapshot()
    restored = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    gate_passed = bool(
        decision.selected.rollout_id == "safe-rollout"
        and decision.selected.provenance == "imagined"
        and decision.selected.steps[-1].expected_progress == 1.0
        and runtime.replan_required
        and runtime.last_rollout_prediction_error is not None
        and runtime.last_rollout_prediction_error > 0.25
        and restored.replan_required
        and restored.last_rollout_prediction_error == runtime.last_rollout_prediction_error
        and after_error.plan.selected_plan_id == "safe-rollout"
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "selected_rollout": decision.selected.rollout_id,
            "selected_rollout_steps": len(decision.selected.steps),
            "selected_provenance": decision.selected.provenance,
            "selected_confidence": decision.selected.confidence,
            "replan_required_after_error": runtime.replan_required,
            "rollout_prediction_error": runtime.last_rollout_prediction_error,
            "checkpoint_replan_required": restored.replan_required,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "multi-step imagined rollout is selected by value/risk, carries provenance, and real outcome error triggers a checkpointed replan flag",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "choose a safe two-step imagined rollout over a high-reward risky rollout",
        "rollouts": ["safe-rollout", "risky-rollout"],
        "provenance": "imagined",
        "lesions": ["rollout_bypass", "uncertainty_weight", "prediction_error_replan"],
        "boundary": "short-horizon rollout and replan trigger only; not general planning or language generation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_imagined_rollout_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_imagined_rollout_baseline_20260825.json",
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
