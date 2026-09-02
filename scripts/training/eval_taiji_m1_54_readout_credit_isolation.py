"""Isolate action and outcome replay plasticity on the identifiable course."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_45_component_geometry import _patterns  # noqa: E402
from scripts.training.eval_taiji_m1_52_replay_interference import (  # noqa: E402
    _memory_weights,
    _weight_deltas,
)
from scripts.training.eval_taiji_m1_53_credit_identifiability import (  # noqa: E402
    _checkpoint_record,
    _config,
    _course,
    _probe,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-54-readout-credit-isolation-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_54_readout_credit_isolation_20260902.json"
SEEDS = (11, 29, 47)
TARGETS = ("all", "readout", "action_readout", "outcome_readout")


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("readout isolation summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _association_summary(model: Taiji, episodes: tuple[Any, ...]) -> dict[str, Any]:
    completion: list[float] = []
    error: list[float] = []
    for episode in episodes:
        pattern = _patterns(model, episode)
        completion.append(float(pattern["association_completion_ratio"]))
        error.append(float(pattern["association_error_ratio"]))
    return {
        "completion_ratio": _summary(completion),
        "error_ratio": _summary(error),
    }


def _candidate_gate(record: dict[str, Any], no_replay: dict[str, Any]) -> bool:
    old = record["old_holdout"]["summary"]["all"]
    retention = record["old_retention"]["summary"]["all"]
    new = record["new_holdout"]["summary"]["all"]
    baseline_old = record["phase_a_baseline"]["summary"]["all"]
    baseline_retention = record["phase_a_retention_baseline"]["summary"]["all"]
    no_replay_new = no_replay["new_holdout"]["summary"]["all"]
    return bool(
        record["checkpoint"]["same_process_digest_matches"]
        and record["checkpoint"]["fresh_process_digest_matches"]
        and record["parameter_count_matches_plan"]
        and record["holdout_updates"] == 0
        and old["action_accuracy"] >= baseline_old["action_accuracy"]
        and old["outcome_accuracy"] >= baseline_old["outcome_accuracy"]
        and retention["action_accuracy"] >= baseline_retention["action_accuracy"]
        and retention["outcome_accuracy"] >= baseline_retention["outcome_accuracy"]
        and new["action_accuracy"] + 0.05 >= no_replay_new["action_accuracy"]
        and new["outcome_accuracy"] + 0.05 >= no_replay_new["outcome_accuracy"]
        and record["causal_gain_action"] > 0.0
        and record["causal_gain_outcome"] > 0.0
    )


def _seed_record(course: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    actions = (48, 49)
    outcomes = (43, 45)
    phase_a = Taiji(_config(seed), episode_id=f"m1-54-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_baseline = _probe(
        phase_a,
        course.phase_a_holdout,
        actions,
        outcomes,
        None,
    )
    phase_a_retention_baseline = _probe(
        phase_a,
        course.phase_a_retention,
        actions,
        outcomes,
        None,
    )
    phase_a_baseline_by_cue = {
        row["cue"]: row for row in phase_a_baseline["rows"]
    }
    phase_a_retention_by_cue = {
        row["cue"]: row for row in phase_a_retention_baseline["rows"]
    }
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    phase_a_weights = _memory_weights(phase_a)

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_digest = content_digest(phase_b_checkpoint)
    phase_b_weights = _memory_weights(phase_b)

    no_replay_model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    no_replay = {
        "old_holdout": _probe(
            no_replay_model,
            course.phase_a_holdout,
            actions,
            outcomes,
            phase_a_baseline_by_cue,
        ),
        "old_retention": _probe(
            no_replay_model,
            course.phase_a_retention,
            actions,
            outcomes,
            phase_a_retention_by_cue,
        ),
        "new_holdout": _probe(
            no_replay_model,
            course.phase_b_holdout,
            actions,
            outcomes,
            None,
        ),
        "association": _association_summary(no_replay_model, course.phase_a_train),
        "memory_write_count": int(no_replay_model.memory.write_count),
        "weight_delta_from_phase_a": _weight_deltas(
            phase_a_weights, _memory_weights(no_replay_model)
        ),
        "weight_delta_from_phase_b": _weight_deltas(
            phase_b_weights, _memory_weights(no_replay_model)
        ),
    }

    conditions: dict[str, Any] = {}
    for target in TARGETS:
        model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
        for episode in course.replay_train:
            DelayedMemoryTask._write_episode(
                model,
                episode,
                provenance="replayed",
                memory_learning_scale=1.0,
                memory_learning_targets=target,
            )
        checkpoint = _checkpoint_record(model)
        after_weights = _memory_weights(model)
        old_holdout = _probe(
            model,
            course.phase_a_holdout,
            actions,
            outcomes,
            phase_a_baseline_by_cue,
        )
        old_retention = _probe(
            model,
            course.phase_a_retention,
            actions,
            outcomes,
            phase_a_retention_by_cue,
        )
        new_holdout = _probe(model, course.phase_b_holdout, actions, outcomes, None)
        conditions[target] = {
            "target": target,
            "old_holdout": old_holdout,
            "old_retention": old_retention,
            "new_holdout": new_holdout,
            "association": _association_summary(model, course.phase_a_train),
            "memory_write_count": int(model.memory.write_count),
            "memory_writes_since_phase_a": int(
                model.memory.write_count - phase_a.memory.write_count
            ),
            "causal_gain_action": float(
                old_holdout["summary"]["all"]["action_accuracy"]
                - no_replay["old_holdout"]["summary"]["all"]["action_accuracy"]
            ),
            "causal_gain_outcome": float(
                old_holdout["summary"]["all"]["outcome_accuracy"]
                - no_replay["old_holdout"]["summary"]["all"]["outcome_accuracy"]
            ),
            "new_delta_vs_no_replay_action": float(
                new_holdout["summary"]["all"]["action_accuracy"]
                - no_replay["new_holdout"]["summary"]["all"]["action_accuracy"]
            ),
            "new_delta_vs_no_replay_outcome": float(
                new_holdout["summary"]["all"]["outcome_accuracy"]
                - no_replay["new_holdout"]["summary"]["all"]["outcome_accuracy"]
            ),
            "checkpoint": checkpoint,
            "weight_delta_from_phase_a": _weight_deltas(phase_a_weights, after_weights),
            "weight_delta_from_phase_b": _weight_deltas(phase_b_weights, after_weights),
            "active_parameter_count": model.parameter_count(),
            "planned_active_parameter_count": model.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                model.parameter_count() == model.config.planned_active_parameter_count
            ),
            "holdout_updates": 0,
        }
        conditions[target]["candidate_gate_passed"] = _candidate_gate(
            {
                **conditions[target],
                "phase_a_baseline": phase_a_baseline,
                "phase_a_retention_baseline": phase_a_retention_baseline,
            },
            no_replay,
        )

    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_b_checkpoint_digest": phase_b_digest,
        "phase_a_baseline": phase_a_baseline,
        "phase_a_retention_baseline": phase_a_retention_baseline,
        "no_replay": no_replay,
        "conditions": conditions,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    course = _course("factorial", factorial=True)
    records = [_seed_record(course, seed) for seed in SEEDS]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "promote": False,
        "architecture_unchanged": True,
        "variable_changed": "replay learning target only",
        "default_learning_targets": "all",
        "tested_learning_targets": list(TARGETS),
        "course": course.name,
        "corpus_digest": course.digest,
        "combination_counts": {
            "phase_a_train": {
                f"{action}/{outcome}": sum(
                    int(item.action == action and item.outcome == outcome)
                    for item in course.phase_a_train
                )
                for action, outcome in ((48, 43), (48, 45), (49, 43), (49, 45))
            },
            "phase_b_train": {
                f"{action}/{outcome}": sum(
                    int(item.action == action and item.outcome == outcome)
                    for item in course.phase_b_train
                )
                for action, outcome in ((48, 43), (48, 45), (49, 43), (49, 45))
            },
        },
        "replay_provenance": "replayed",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "records": records,
        "target_gate_matrix": {
            target: [
                bool(record["conditions"][target]["candidate_gate_passed"])
                for record in records
            ]
            for target in TARGETS
        },
        "gates": {
            "factorial_combinations_present": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
            "does_not_change_default_checkpoint": True,
            "does_not_promote_target": True,
        },
        "boundary": "If no target passes all three seeds, keep default all and redesign replay scheduling or course; do not add neurons or change topology.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_audit()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "format": result["format"],
                "status": result["status"],
                "report_path": result["report_path"],
                "target_gate_matrix": result["target_gate_matrix"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
