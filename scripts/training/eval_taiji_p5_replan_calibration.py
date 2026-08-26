"""Evaluate actual alternative-rollout execution and confidence calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p5_imagined_rollout import _rollouts  # noqa: E402
from taiji import Goal, GoalPlanner, TSKV8Adapter  # noqa: E402

MANIFEST_FORMAT = "taiji-p5-replan-calibration-manifest-v1"
REPORT_FORMAT = "taiji-p5-replan-calibration-v1"


def evaluate() -> dict[str, object]:
    runtime = TSKV8Adapter()
    planner = GoalPlanner()
    runtime.attach_goal_planner(planner)
    runtime.set_goals((Goal("reach-target", "reach the target state", priority=1.0),))
    safe_rollout, risky_rollout = _rollouts()
    first_decision = runtime.plan_rollouts((safe_rollout, risky_rollout))
    runtime.observe(97, learn=False)
    runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("gamble", "scout"),
        use_plan=True,
    )
    runtime.settle_action(-0.5, success=False, learn=False)
    first_replan_required = runtime.replan_required
    first_confidence = runtime.last_rollout_calibrated_confidence
    runtime.observe(98, learn=False)
    second_decision = runtime.plan_rollouts((risky_rollout,))
    runtime.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("gamble", "scout"),
        use_plan=True,
    )
    runtime.settle_action(0.9, success=True, learn=False)
    restored = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    safe_calibration = (
        None
        if restored._goal_planner is None
        else restored._goal_planner.calibrated_confidence("safe-rollout")
    )
    risky_calibration = (
        None
        if restored._goal_planner is None
        else restored._goal_planner.calibrated_confidence("risky-rollout")
    )
    gate_passed = bool(
        first_decision.selected.rollout_id == "safe-rollout"
        and first_replan_required
        and first_confidence == 0.0
        and second_decision.selected.rollout_id == "risky-rollout"
        and not runtime.replan_required
        and runtime.last_rollout_prediction_error == 0.0
        and runtime.last_rollout_calibrated_confidence == 1.0
        and safe_calibration == 0.0
        and risky_calibration == 1.0
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "first_selected_rollout": first_decision.selected.rollout_id,
            "first_replan_required": first_replan_required,
            "first_calibrated_confidence": first_confidence,
            "second_selected_rollout": second_decision.selected.rollout_id,
            "second_replan_required": runtime.replan_required,
            "second_prediction_error": runtime.last_rollout_prediction_error,
            "second_calibrated_confidence": runtime.last_rollout_calibrated_confidence,
            "checkpoint_safe_confidence": safe_calibration,
            "checkpoint_risky_confidence": risky_calibration,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "prediction error causes an alternative rollout to execute and empirical success confidence is restored through checkpoint",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "fail the first safe rollout, replan to an alternative, execute it, and calibrate confidence",
        "phases": [
            "safe_rollout_failure",
            "alternative_rollout_execution",
            "confidence_calibration",
        ],
        "controls": ["replan_flag", "success_rate", "native_checkpoint"],
        "boundary": "two-rollout replan and empirical confidence only; not long-horizon policy evaluation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_replan_calibration_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_replan_calibration_baseline_20260825.json",
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
