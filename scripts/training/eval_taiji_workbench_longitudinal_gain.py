"""Measure bounded interaction-group gain on real Seed Workbench traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_interaction_groups import evaluate_workbench  # noqa: E402

REPORT_FORMAT = "taiji-w7-p4-3-workbench-longitudinal-gain-v1"
MIN_GAIN = 0.2


def _cell_rewards(
    records: list[dict[str, object]],
    *,
    split: str,
    context: str,
) -> dict[str, float]:
    prefix = f"r2-s2-{split}-{context}-"
    selected = {
        str(record["episode_id"])[len(prefix) :]: float(
            record["workbench_outcome"]["workflow_reward"]
        )
        for record in records
        if str(record["episode_id"]).startswith(prefix)
    }
    expected = {"none", "first", "second", "pair"}
    if set(selected) != expected:
        raise AssertionError(
            f"missing Workbench factorial cells for {split}/{context}: {selected}"
        )
    return selected


def _family_gain(records: list[dict[str, object]], *, split: str, context: str) -> dict[str, object]:
    rewards = _cell_rewards(records, split=split, context=context)
    strongest_single = max(rewards["first"], rewards["second"])
    dense_average = (rewards["first"] + rewards["second"]) / 2.0
    random_single_expectation = dense_average
    grouped = rewards["pair"]
    return {
        "single_first_reward": rewards["first"],
        "single_second_reward": rewards["second"],
        "grouped_pair_reward": grouped,
        "strongest_single_reward": strongest_single,
        "dense_average_reward": dense_average,
        "random_single_expectation_reward": random_single_expectation,
        "grouped_gain_vs_strongest_single": grouped - strongest_single,
        "grouped_gain_vs_dense_average": grouped - dense_average,
        "grouped_gain_vs_random_expectation": grouped - random_single_expectation,
    }


def evaluate() -> dict[str, object]:
    workbench = evaluate_workbench()
    records = list(workbench["input"]["workbench_records"])
    train = _family_gain(records, split="train", context="complementary")
    holdout = _family_gain(records, split="holdout", context="complementary")
    conflict = _family_gain(records, split="train", context="conflicting")

    neutral_capability_retained = all(
        any(
            action["capability_id"] == "workspace.stat" and action["success"]
            for action in record["workbench_outcome"]["raw_actions"]
        )
        for record in records
    )
    replay_preserved = all(bool(record["replay_equal"]) for record in records)
    recovery_preserved = any(
        record["workbench_outcome"]["recovery"] is not None
        and bool(record["workbench_outcome"]["recovery"]["success"])
        for record in records
    )
    metrics = {
        "real_workbench_factorial_cells": len(records) == 16,
        "train_group_gain_beats_strongest_single": train["grouped_gain_vs_strongest_single"]
        >= MIN_GAIN,
        "train_group_gain_beats_dense_average": train["grouped_gain_vs_dense_average"] >= MIN_GAIN,
        "train_group_gain_beats_random_expectation": train["grouped_gain_vs_random_expectation"]
        >= MIN_GAIN,
        "holdout_group_gain_beats_strongest_single": holdout["grouped_gain_vs_strongest_single"]
        >= MIN_GAIN,
        "holdout_group_gain_beats_dense_average": holdout["grouped_gain_vs_dense_average"] >= MIN_GAIN,
        "holdout_direction_preserved": bool(workbench["metrics"]["holdout_direction_preserved"]),
        "conflicting_group_is_negative_control": conflict["grouped_pair_reward"]
        < conflict["strongest_single_reward"],
        "old_workbench_capability_retained": neutral_capability_retained,
        "checkpoint_replay_preserved": replay_preserved
        and bool(workbench["metrics"]["workbench_checkpoint_replay"]),
        "recovery_trace_preserved": recovery_preserved
        and bool(workbench["metrics"]["workbench_recovery_trace"]),
        "lesion_effect_preserved": bool(workbench["metrics"]["lesion_effects_observed"]),
        "resource_bound_preserved": all(
            float(group["resource_cost"]) <= 10.0
            for group in workbench["checkpoint"]["groups"]
        ),
        "no_policy_tool_or_provider_mutation": all(
            not bool(workbench["boundary"][name])
            for name in ("policy_mutation", "tool_selection", "provider_selection")
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "real Workbench complementary interaction-group longitudinal gain",
        "controls": [
            "strongest_single",
            "dense_average",
            "random_single_expectation",
            "conflicting_group_negative_control",
        ],
        "train_complementary": train,
        "holdout_complementary": holdout,
        "train_conflicting_negative_control": conflict,
        "source": {
            "workbench_contract": workbench["input"]["workbench_contract"],
            "train_episode_count": workbench["input"]["train_episode_count"],
            "holdout_episode_count": workbench["input"]["holdout_episode_count"],
            "semantic_role_labels": workbench["input"]["semantic_role_labels"],
            "source_trace_digest": workbench["source_trace_digest"],
            "holdout_trace_digest": workbench["holdout_trace_digest"],
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "on real Workbench factorial tasks, an admitted complementary group must beat "
                "the strongest single, dense average, and random single expectation on train "
                "and holdout; a conflicting group must remain a negative control, while old "
                "capabilities, resources, lesion causality, recovery, and checkpoint replay hold"
            ),
        },
        "boundary": (
            "This gate proves bounded causal gain for the registered Workbench task family. "
            "It does not claim open-domain transfer, unrestricted self-evolution, provider "
            "quality, CUDA advantage, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_3_workbench_longitudinal_gain_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
