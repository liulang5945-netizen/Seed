"""Trace a protected-replay signal without changing the replay write path."""

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
    _topology_digest,
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

FORMAT = "taiji-native-m1-57-protected-replay-trace-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_57_protected_replay_trace_20260902.json"
SEEDS = (11, 29, 47)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("protected replay trace summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _seed_record(course: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    phase_a = Taiji(_config(seed), episode_id=f"m1-57-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    phase_a_topology = _topology_digest(phase_a)

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_digest = content_digest(phase_b_checkpoint)
    phase_b_baseline = _probe(
        phase_b,
        course.phase_b_train,
        (48, 49),
        (43, 45),
        None,
    )
    phase_b_baseline_by_cue = {
        row["cue"]: row for row in phase_b_baseline["rows"]
    }
    baseline_summary = phase_b_baseline["summary"]["all"]

    no_replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    no_replay_checkpoint = _checkpoint_record(no_replay)
    no_replay_memory_digest = content_digest(no_replay.memory.to_payload())

    replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    trace_rows: list[dict[str, Any]] = []
    previous_summary = baseline_summary
    previous_weights = _memory_weights(replay)
    previous_memory_digest = content_digest(replay.memory.to_payload())
    order: list[tuple[int, str, str]] = []
    for step_index, episode in enumerate(course.replay_train, start=1):
        DelayedMemoryTask._write_episode(replay, episode, provenance="replayed")
        order.append((step_index, episode.memory_id, "replayed"))
        current_memory_digest = content_digest(replay.memory.to_payload())
        current_weights = _memory_weights(replay)
        current_probe = _probe(
            replay,
            course.phase_b_train,
            (48, 49),
            (43, 45),
            phase_b_baseline_by_cue,
        )
        current_summary = current_probe["summary"]["all"]
        action_delta = float(
            current_summary["action_accuracy"] - baseline_summary["action_accuracy"]
        )
        outcome_delta = float(
            current_summary["outcome_accuracy"] - baseline_summary["outcome_accuracy"]
        )
        step_action_delta = float(
            current_summary["action_accuracy"] - previous_summary["action_accuracy"]
        )
        step_outcome_delta = float(
            current_summary["outcome_accuracy"] - previous_summary["outcome_accuracy"]
        )
        trace_rows.append(
            {
                "step": step_index,
                "replay_memory_id": episode.memory_id,
                "action": int(episode.action),
                "outcome": int(episode.outcome),
                "combination": f"{episode.action}/{episode.outcome}",
                "phase_b_train_action_delta_vs_no_replay": action_delta,
                "phase_b_train_outcome_delta_vs_no_replay": outcome_delta,
                "phase_b_train_action_step_delta": step_action_delta,
                "phase_b_train_outcome_step_delta": step_outcome_delta,
                "phase_b_train_action_outcome_margin_gap": float(
                    current_summary["delta_action_outcome_margin_gap"]["mean"]
                ),
                "association_error_ratio": float(
                    _association_summary(replay, course.phase_b_train)["error_ratio"][
                        "mean"
                    ]
                ),
                "memory_digest": current_memory_digest,
                "memory_digest_changed": current_memory_digest != previous_memory_digest,
                "weight_delta_from_previous": _weight_deltas(
                    previous_weights, current_weights
                ),
                "memory_write_count": int(replay.memory.write_count),
                "trace_only_would_flag_both_train_deltas": bool(
                    action_delta < 0.0 and outcome_delta < 0.0
                ),
                "trace_only_would_flag_either_train_delta": bool(
                    action_delta < 0.0 or outcome_delta < 0.0
                ),
            }
        )
        previous_summary = current_summary
        previous_weights = current_weights
        previous_memory_digest = current_memory_digest

    replay_checkpoint = _checkpoint_record(replay)
    final_summary = trace_rows[-1]
    conflict_steps = [
        int(row["step"])
        for row in trace_rows
        if row["trace_only_would_flag_both_train_deltas"]
    ]
    either_steps = [
        int(row["step"])
        for row in trace_rows
        if row["trace_only_would_flag_either_train_delta"]
    ]
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_b_checkpoint_digest": phase_b_digest,
        "phase_b_train_baseline": phase_b_baseline,
        "no_replay": {
            "checkpoint": no_replay_checkpoint,
            "memory_write_count": int(no_replay.memory.write_count),
            "memory_digest": no_replay_memory_digest,
            "holdout_updates": 0,
        },
        "protected_objective_trace_only": {
            "writes": 0,
            "memory_digest_unchanged": no_replay_memory_digest
            == content_digest(no_replay.memory.to_payload()),
            "uses_phase_b_holdout": False,
            "uses_phase_b_retention": False,
        },
        "all_replay_trace": {
            "rows": trace_rows,
            "order_digest": content_digest(order),
            "order_count": len(order),
            "replay_write_count": len(course.replay_train),
            "final_phase_b_train_action_delta": float(
                final_summary["phase_b_train_action_delta_vs_no_replay"]
            ),
            "final_phase_b_train_outcome_delta": float(
                final_summary["phase_b_train_outcome_delta_vs_no_replay"]
            ),
            "both_delta_conflict_steps": conflict_steps,
            "either_delta_conflict_steps": either_steps,
            "first_both_delta_conflict_step": (
                conflict_steps[0] if conflict_steps else None
            ),
            "last_both_delta_conflict_step": (
                conflict_steps[-1] if conflict_steps else None
            ),
            "checkpoint": replay_checkpoint,
            "holdout_updates": 0,
            "topology_digest": _topology_digest(replay),
            "topology_matches_phase_a": _topology_digest(replay) == phase_a_topology,
            "active_parameter_count": replay.parameter_count(),
            "planned_active_parameter_count": replay.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                replay.parameter_count() == replay.config.planned_active_parameter_count
            ),
        },
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    course = _course("factorial", factorial=True)
    records = [_seed_record(course, seed) for seed in SEEDS]
    conflict_steps = [
        record["all_replay_trace"]["first_both_delta_conflict_step"]
        for record in records
    ]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "promote": False,
        "architecture_unchanged": True,
        "variable_changed": "none; protected objective trace only",
        "course": course.name,
        "corpus_digest": course.digest,
        "replay_provenance": "replayed",
        "replay_learning_targets": "all",
        "phase_b_holdout_read": False,
        "phase_b_retention_read": False,
        "records": records,
        "cross_seed_trace": {
            "first_both_delta_conflict_step_by_seed": conflict_steps,
            "conflict_started_in_all_seeds": all(step is not None for step in conflict_steps),
            "same_conflict_step_across_seeds": len(
                {step for step in conflict_steps if step is not None}
            )
            == 1,
            "interpretation": "the trace is descriptive only; no rollback or admission decision is applied",
        },
        "gates": {
            "trace_only_does_not_write": True,
            "phase_b_holdout_not_used": True,
            "phase_b_retention_not_used": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
            "does_not_change_default_checkpoint": True,
            "does_not_promote_protected_objective": True,
        },
        "boundary": "Only a stable cumulative conflict pattern may authorize one rollback candidate; otherwise freeze protected replay and redesign the memory objective.",
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
                "cross_seed_trace": result["cross_seed_trace"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
