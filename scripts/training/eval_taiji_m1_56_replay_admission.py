"""Attribute replay admission conflicts using phase-B train evidence only."""

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

from scripts.training.eval_taiji_m1_52_replay_interference import (  # noqa: E402  # noqa: E402
    _memory_weights,
    _static_similarity_features,
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

FORMAT = "taiji-native-m1-56-replay-admission-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_56_replay_admission_20260902.json"
SEEDS = (11, 29, 47)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("replay admission summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _single_replay_effect(
    phase_b_checkpoint: dict[str, Any],
    replay_episode: Any,
    course: Any,
    phase_b_baseline: dict[str, Any],
    phase_b_baseline_by_cue: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    DelayedMemoryTask._write_episode(model, replay_episode, provenance="replayed")
    after = _probe(
        model,
        course.phase_b_train,
        (48, 49),
        (43, 45),
        phase_b_baseline_by_cue,
    )
    baseline = phase_b_baseline["summary"]["all"]
    summary = after["summary"]["all"]
    return {
        "replay_memory_id": replay_episode.memory_id,
        "phase_b_train_action_delta": float(
            summary["action_accuracy"] - baseline["action_accuracy"]
        ),
        "phase_b_train_outcome_delta": float(
            summary["outcome_accuracy"] - baseline["outcome_accuracy"]
        ),
        "phase_b_train_action_margin_delta": float(
            summary["delta_action_margin_vs_phase_a"]["mean"]
        ),
        "phase_b_train_outcome_margin_delta": float(
            summary["delta_outcome_margin_vs_phase_a"]["mean"]
        ),
        "phase_b_train_action_outcome_margin_gap": float(
            summary["delta_action_outcome_margin_gap"]["mean"]
        ),
        "memory_writes_since_phase_b": int(
            model.memory.write_count
            - int(phase_b_checkpoint["memory"]["write_count"])
        ),
        "association": _association_summary(model, course.phase_b_train),
    }


def _seed_record(course: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    phase_a = Taiji(_config(seed), episode_id=f"m1-56-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    static_features = _static_similarity_features(course, phase_a)

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_digest = content_digest(phase_b_checkpoint)
    phase_b_weights = _memory_weights(phase_b)
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

    no_replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    no_replay_checkpoint = _checkpoint_record(no_replay)
    trace_digest_before = content_digest(no_replay.memory.to_payload())
    admission_rows: list[dict[str, Any]] = []
    for episode in course.replay_train:
        effect = _single_replay_effect(
            phase_b_checkpoint,
            episode,
            course,
            phase_b_baseline,
            phase_b_baseline_by_cue,
        )
        admission_rows.append(
            {
                "replay_memory_id": episode.memory_id,
                "action": int(episode.action),
                "outcome": int(episode.outcome),
                "combination": f"{episode.action}/{episode.outcome}",
                **static_features[episode.memory_id],
                **effect,
                "conflict_evidence": bool(
                    effect["phase_b_train_action_delta"] < 0.0
                    and effect["phase_b_train_outcome_delta"] < 0.0
                ),
            }
        )
    trace_digest_after = content_digest(no_replay.memory.to_payload())

    all_replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    for episode in course.replay_train:
        DelayedMemoryTask._write_episode(all_replay, episode, provenance="replayed")
    all_replay_checkpoint = _checkpoint_record(all_replay)
    all_replay_train = _probe(
        all_replay,
        course.phase_b_train,
        (48, 49),
        (43, 45),
        phase_b_baseline_by_cue,
    )
    baseline_summary = phase_b_baseline["summary"]["all"]
    replay_summary = all_replay_train["summary"]["all"]
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_b_checkpoint_digest": phase_b_digest,
        "phase_b_train_baseline": phase_b_baseline,
        "no_replay": {
            "memory_write_count": int(no_replay.memory.write_count),
            "checkpoint": no_replay_checkpoint,
            "trace_only_memory_unchanged": trace_digest_before == trace_digest_after,
            "trace_digest": content_digest(admission_rows),
            "holdout_updates": 0,
        },
        "admission_trace": {
            "sample_count": len(admission_rows),
            "rows": admission_rows,
            "conflict_count": sum(int(row["conflict_evidence"]) for row in admission_rows),
            "conflict_memory_ids": [
                row["replay_memory_id"]
                for row in admission_rows
                if row["conflict_evidence"]
            ],
            "phase_b_train_action_delta": _summary(
                [float(row["phase_b_train_action_delta"]) for row in admission_rows]
            ),
            "phase_b_train_outcome_delta": _summary(
                [float(row["phase_b_train_outcome_delta"]) for row in admission_rows]
            ),
            "phase_b_train_action_outcome_gap": _summary(
                [
                    float(row["phase_b_train_action_outcome_margin_gap"])
                    for row in admission_rows
                ]
            ),
        },
        "all_replay": {
            "phase_b_train": all_replay_train,
            "phase_b_train_action_delta": float(
                replay_summary["action_accuracy"] - baseline_summary["action_accuracy"]
            ),
            "phase_b_train_outcome_delta": float(
                replay_summary["outcome_accuracy"] - baseline_summary["outcome_accuracy"]
            ),
            "memory_write_count": int(all_replay.memory.write_count),
            "memory_writes_since_phase_b": int(
                all_replay.memory.write_count - phase_b.memory.write_count
            ),
            "checkpoint": all_replay_checkpoint,
            "weight_delta_from_phase_b": _weight_deltas(
                phase_b_weights, _memory_weights(all_replay)
            ),
            "association": _association_summary(all_replay, course.phase_b_train),
            "holdout_updates": 0,
        },
        "active_parameter_count": all_replay.parameter_count(),
        "planned_active_parameter_count": all_replay.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            all_replay.parameter_count() == all_replay.config.planned_active_parameter_count
        ),
        "topology_digest": _topology_digest(all_replay),
        "phase_a_topology_digest": _topology_digest(phase_a),
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    course = _course("factorial", factorial=True)
    records = [_seed_record(course, seed) for seed in SEEDS]
    conflict_sets = [
        set(record["admission_trace"]["conflict_memory_ids"]) for record in records
    ]
    stable_conflict_ids = sorted(set.intersection(*conflict_sets))
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "promote": False,
        "architecture_unchanged": True,
        "variable_changed": "none; phase-B-train-only replay admission observation",
        "course": course.name,
        "corpus_digest": course.digest,
        "replay_provenance": "replayed",
        "replay_learning_targets": "all",
        "phase_b_holdout_read": False,
        "phase_b_retention_read": False,
        "records": records,
        "cross_seed_admission": {
            "conflict_count_by_seed": [
                record["admission_trace"]["conflict_count"] for record in records
            ],
            "stable_conflict_memory_ids": stable_conflict_ids,
            "stable_conflict_count": len(stable_conflict_ids),
            "interpretation": "conflict evidence is defined only as a negative action and outcome delta on phase-B train after one replay write; it is a diagnostic trace, not an admission filter",
        },
        "gates": {
            "admission_trace_only_does_not_write": True,
            "phase_b_holdout_not_used_for_admission": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
            "does_not_change_default_checkpoint": True,
            "does_not_promote_admission_rule": True,
        },
        "boundary": "Only a stable conflict subset may authorize an equal-budget admission candidate; otherwise freeze replay admission and redesign memory data/objective.",
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
                "cross_seed_admission": result["cross_seed_admission"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
