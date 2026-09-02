"""Audit replay timing under an identifiable action/outcome memory course."""

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
from scripts.training.eval_taiji_m1_54_readout_credit_isolation import (  # noqa: E402
    _association_summary,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-55-replay-schedule-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_55_replay_schedule_20260902.json"
SEEDS = (11, 29, 47)
SCHEDULES = ("no_replay", "posthoc_replay", "interleaved_replay")


def _schedule_order(course: Any, schedule: str) -> tuple[tuple[str, str, str], ...]:
    if schedule == "no_replay":
        return tuple(
            ("phase_b", episode.memory_id, "experienced")
            for episode in course.phase_b_train
        )
    if schedule == "posthoc_replay":
        return (
            *tuple(
                ("phase_b", episode.memory_id, "experienced")
                for episode in course.phase_b_train
            ),
            *tuple(
                ("replay", episode.memory_id, "replayed")
                for episode in course.replay_train
            ),
        )
    if schedule == "interleaved_replay":
        return tuple(
            item
            for phase_b, replay in zip(
                course.phase_b_train, course.replay_train, strict=True
            )
            for item in (
                ("phase_b", phase_b.memory_id, "experienced"),
                ("replay", replay.memory_id, "replayed"),
            )
        )
    raise ValueError(f"unsupported replay schedule: {schedule}")


def _apply_schedule(model: Taiji, course: Any, schedule: str) -> None:
    if schedule == "no_replay":
        for episode in course.phase_b_train:
            DelayedMemoryTask._write_episode(model, episode)
        return
    if schedule == "posthoc_replay":
        for episode in course.phase_b_train:
            DelayedMemoryTask._write_episode(model, episode)
        for episode in course.replay_train:
            DelayedMemoryTask._write_episode(model, episode, provenance="replayed")
        return
    if schedule == "interleaved_replay":
        for phase_b, replay in zip(
            course.phase_b_train, course.replay_train, strict=True
        ):
            DelayedMemoryTask._write_episode(model, phase_b)
            DelayedMemoryTask._write_episode(model, replay, provenance="replayed")
        return
    raise ValueError(f"unsupported replay schedule: {schedule}")


def _schedule_gate(
    record: dict[str, Any],
    no_replay: dict[str, Any],
    phase_a_baseline: dict[str, Any],
    phase_a_retention: dict[str, Any],
) -> bool:
    old = record["old_holdout"]["summary"]["all"]
    retention = record["old_retention"]["summary"]["all"]
    new = record["new_holdout"]["summary"]["all"]
    baseline_old = phase_a_baseline["summary"]["all"]
    baseline_retention = phase_a_retention["summary"]["all"]
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
    phase_a = Taiji(_config(seed), episode_id=f"m1-55-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_baseline = _probe(phase_a, course.phase_a_holdout, actions, outcomes, None)
    phase_a_retention = _probe(
        phase_a, course.phase_a_retention, actions, outcomes, None
    )
    phase_a_baseline_by_cue = {
        row["cue"]: row for row in phase_a_baseline["rows"]
    }
    phase_a_retention_by_cue = {
        row["cue"]: row for row in phase_a_retention["rows"]
    }
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    phase_a_weights = _memory_weights(phase_a)
    records: dict[str, Any] = {}
    no_replay_record: dict[str, Any] | None = None
    for schedule in SCHEDULES:
        model = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        order = _schedule_order(course, schedule)
        _apply_schedule(model, course, schedule)
        checkpoint = _checkpoint_record(model)
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
        after_weights = _memory_weights(model)
        record = {
            "schedule": schedule,
            "order_digest": content_digest(order),
            "order_count": len(order),
            "phase_b_write_count": len(course.phase_b_train),
            "replay_write_count": (
                0 if schedule == "no_replay" else len(course.replay_train)
            ),
            "old_holdout": old_holdout,
            "old_retention": old_retention,
            "new_holdout": new_holdout,
            "association": _association_summary(model, course.phase_a_train),
            "memory_write_count": int(model.memory.write_count),
            "memory_writes_since_phase_a": int(
                model.memory.write_count - phase_a.memory.write_count
            ),
            "causal_gain_action": 0.0,
            "causal_gain_outcome": 0.0,
            "new_delta_vs_no_replay_action": 0.0,
            "new_delta_vs_no_replay_outcome": 0.0,
            "checkpoint": checkpoint,
            "weight_delta_from_phase_a": _weight_deltas(phase_a_weights, after_weights),
            "active_parameter_count": model.parameter_count(),
            "planned_active_parameter_count": model.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                model.parameter_count() == model.config.planned_active_parameter_count
            ),
            "holdout_updates": 0,
        }
        records[schedule] = record
        if schedule == "no_replay":
            no_replay_record = record
    if no_replay_record is None:
        raise RuntimeError("replay schedule audit did not build no-replay baseline")
    for schedule, record in records.items():
        record["causal_gain_action"] = float(
            record["old_holdout"]["summary"]["all"]["action_accuracy"]
            - no_replay_record["old_holdout"]["summary"]["all"]["action_accuracy"]
        )
        record["causal_gain_outcome"] = float(
            record["old_holdout"]["summary"]["all"]["outcome_accuracy"]
            - no_replay_record["old_holdout"]["summary"]["all"]["outcome_accuracy"]
        )
        record["new_delta_vs_no_replay_action"] = float(
            record["new_holdout"]["summary"]["all"]["action_accuracy"]
            - no_replay_record["new_holdout"]["summary"]["all"]["action_accuracy"]
        )
        record["new_delta_vs_no_replay_outcome"] = float(
            record["new_holdout"]["summary"]["all"]["outcome_accuracy"]
            - no_replay_record["new_holdout"]["summary"]["all"]["outcome_accuracy"]
        )
        record["candidate_gate_passed"] = _schedule_gate(
            record,
            no_replay_record,
            phase_a_baseline,
            phase_a_retention,
        )
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_a_baseline": phase_a_baseline,
        "phase_a_retention_baseline": phase_a_retention,
        "schedules": records,
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
        "variable_changed": "replay temporal schedule only",
        "schedules": list(SCHEDULES),
        "course": course.name,
        "corpus_digest": course.digest,
        "phase_a_train_count": len(course.phase_a_train),
        "phase_b_train_count": len(course.phase_b_train),
        "replay_train_count": len(course.replay_train),
        "replay_provenance": "replayed",
        "memory_learning_targets": "all",
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "records": records,
        "schedule_gate_matrix": {
            schedule: [
                bool(record["schedules"][schedule]["candidate_gate_passed"])
                for record in records
            ]
            for schedule in SCHEDULES
        },
        "gates": {
            "same_total_phase_b_and_replay_writes_for_replay_schedules": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
            "does_not_change_default_checkpoint": True,
            "does_not_promote_schedule": True,
        },
        "boundary": "If no schedule passes all three seeds, freeze current timing and redesign the data/replay objective before structural changes.",
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
                "schedule_gate_matrix": result["schedule_gate_matrix"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
