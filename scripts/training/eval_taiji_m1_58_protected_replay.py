"""Evaluate one dynamic rollback candidate for replay interference."""

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

FORMAT = "taiji-native-m1-58-protected-replay-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_58_protected_replay_20260902.json"
SEEDS = (11, 29, 47)


def _candidate_gate(
    candidate: dict[str, Any],
    no_replay: dict[str, Any],
    phase_a_baseline: dict[str, Any],
    phase_a_retention: dict[str, Any],
) -> bool:
    old = candidate["old_holdout"]["summary"]["all"]
    retention = candidate["old_retention"]["summary"]["all"]
    new = candidate["new_holdout"]["summary"]["all"]
    baseline_old = phase_a_baseline["summary"]["all"]
    baseline_retention = phase_a_retention["summary"]["all"]
    no_replay_new = no_replay["new_holdout"]["summary"]["all"]
    return bool(
        candidate["checkpoint"]["same_process_digest_matches"]
        and candidate["checkpoint"]["fresh_process_digest_matches"]
        and candidate["parameter_count_matches_plan"]
        and candidate["holdout_updates"] == 0
        and old["action_accuracy"] >= baseline_old["action_accuracy"]
        and old["outcome_accuracy"] >= baseline_old["outcome_accuracy"]
        and retention["action_accuracy"] >= baseline_retention["action_accuracy"]
        and retention["outcome_accuracy"] >= baseline_retention["outcome_accuracy"]
        and new["action_accuracy"] + 0.05 >= no_replay_new["action_accuracy"]
        and new["outcome_accuracy"] + 0.05 >= no_replay_new["outcome_accuracy"]
        and candidate["causal_gain_action"] > 0.0
        and candidate["causal_gain_outcome"] > 0.0
    )


def _protected_replay(
    phase_b_checkpoint: dict[str, Any],
    course: Any,
    phase_b_baseline: dict[str, Any],
    phase_b_baseline_by_cue: dict[int, dict[str, Any]],
) -> tuple[Taiji, list[dict[str, Any]]]:
    model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    rows: list[dict[str, Any]] = []
    for step, episode in enumerate(course.replay_train, start=1):
        before_checkpoint = deepcopy(model.checkpoint())
        before_checkpoint_digest = content_digest(before_checkpoint)
        before_memory_digest = content_digest(model.memory.to_payload())
        before_probe = _probe(
            model,
            course.phase_b_train,
            (48, 49),
            (43, 45),
            phase_b_baseline_by_cue,
        )
        before_summary = before_probe["summary"]["all"]
        before_weights = _memory_weights(model)
        DelayedMemoryTask._write_episode(model, episode, provenance="replayed")
        after_probe = _probe(
            model,
            course.phase_b_train,
            (48, 49),
            (43, 45),
            phase_b_baseline_by_cue,
        )
        after_summary = after_probe["summary"]["all"]
        action_margin_delta = float(
            after_summary["action_margin"]["mean"]
            - before_summary["action_margin"]["mean"]
        )
        outcome_margin_delta = float(
            after_summary["outcome_margin"]["mean"]
            - before_summary["outcome_margin"]["mean"]
        )
        reject = action_margin_delta < 0.0 and outcome_margin_delta < 0.0
        after_memory_digest = content_digest(model.memory.to_payload())
        after_weights = _memory_weights(model)
        after_checkpoint_digest = content_digest(model.checkpoint())
        rollback_exact = False
        if reject:
            model = Taiji.from_checkpoint(deepcopy(before_checkpoint))
            rollback_exact = content_digest(model.checkpoint()) == before_checkpoint_digest
        rows.append(
            {
                "step": step,
                "replay_memory_id": episode.memory_id,
                "action": int(episode.action),
                "outcome": int(episode.outcome),
                "combination": f"{episode.action}/{episode.outcome}",
                "before_action_margin": float(before_summary["action_margin"]["mean"]),
                "after_action_margin": float(after_summary["action_margin"]["mean"]),
                "before_outcome_margin": float(before_summary["outcome_margin"]["mean"]),
                "after_outcome_margin": float(after_summary["outcome_margin"]["mean"]),
                "action_margin_delta": action_margin_delta,
                "outcome_margin_delta": outcome_margin_delta,
                "rejected": reject,
                "reject_reason": "both_phase_b_train_margins_decreased" if reject else None,
                "before_checkpoint_digest": before_checkpoint_digest,
                "after_checkpoint_digest": after_checkpoint_digest,
                "rollback_checkpoint_exact": rollback_exact,
                "before_memory_digest": before_memory_digest,
                "after_memory_digest": after_memory_digest,
                "memory_changed_before_rollback": after_memory_digest != before_memory_digest,
                "weight_delta_from_before": _weight_deltas(before_weights, after_weights),
                "accepted_write_count": int(model.memory.write_count),
                "memory_write_count": int(model.memory.write_count),
            }
        )
    return model, rows


def _seed_record(course: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    phase_a = Taiji(_config(seed), episode_id=f"m1-58-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    phase_a_baseline = _probe(phase_a, course.phase_a_holdout, (48, 49), (43, 45), None)
    phase_a_retention = _probe(
        phase_a, course.phase_a_retention, (48, 49), (43, 45), None
    )
    phase_a_baseline_by_cue = {
        row["cue"]: row for row in phase_a_baseline["rows"]
    }
    phase_a_retention_by_cue = {
        row["cue"]: row for row in phase_a_retention["rows"]
    }

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_digest = content_digest(phase_b_checkpoint)
    phase_b_baseline = _probe(phase_b, course.phase_b_train, (48, 49), (43, 45), None)
    phase_b_baseline_by_cue = {
        row["cue"]: row for row in phase_b_baseline["rows"]
    }

    no_replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    no_replay_checkpoint = _checkpoint_record(no_replay)
    no_replay_old = _probe(
        no_replay,
        course.phase_a_holdout,
        (48, 49),
        (43, 45),
        phase_a_baseline_by_cue,
    )
    no_replay_retention = _probe(
        no_replay,
        course.phase_a_retention,
        (48, 49),
        (43, 45),
        phase_a_retention_by_cue,
    )
    no_replay_new = _probe(no_replay, course.phase_b_holdout, (48, 49), (43, 45), None)
    no_replay_record = {
        "old_holdout": no_replay_old,
        "old_retention": no_replay_retention,
        "new_holdout": no_replay_new,
        "checkpoint": no_replay_checkpoint,
        "memory_write_count": int(no_replay.memory.write_count),
        "association": _association_summary(no_replay, course.phase_a_train),
        "holdout_updates": 0,
    }

    all_replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    for episode in course.replay_train:
        DelayedMemoryTask._write_episode(all_replay, episode, provenance="replayed")
    all_replay_checkpoint = _checkpoint_record(all_replay)
    all_replay_old = _probe(
        all_replay,
        course.phase_a_holdout,
        (48, 49),
        (43, 45),
        phase_a_baseline_by_cue,
    )
    all_replay_retention = _probe(
        all_replay,
        course.phase_a_retention,
        (48, 49),
        (43, 45),
        phase_a_retention_by_cue,
    )
    all_replay_new = _probe(
        all_replay,
        course.phase_b_holdout,
        (48, 49),
        (43, 45),
        None,
    )
    all_replay_record = {
        "old_holdout": all_replay_old,
        "old_retention": all_replay_retention,
        "new_holdout": all_replay_new,
        "checkpoint": all_replay_checkpoint,
        "memory_write_count": int(all_replay.memory.write_count),
        "physical_replay_write_count": int(
            all_replay.memory.write_count - phase_b.memory.write_count
        ),
        "active_parameter_count": all_replay.parameter_count(),
        "planned_active_parameter_count": all_replay.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            all_replay.parameter_count() == all_replay.config.planned_active_parameter_count
        ),
        "holdout_updates": 0,
    }

    protected, trace_rows = _protected_replay(
        phase_b_checkpoint,
        course,
        phase_b_baseline,
        phase_b_baseline_by_cue,
    )
    protected_checkpoint = _checkpoint_record(protected)
    protected_old = _probe(
        protected,
        course.phase_a_holdout,
        (48, 49),
        (43, 45),
        phase_a_baseline_by_cue,
    )
    protected_retention = _probe(
        protected,
        course.phase_a_retention,
        (48, 49),
        (43, 45),
        phase_a_retention_by_cue,
    )
    protected_new = _probe(protected, course.phase_b_holdout, (48, 49), (43, 45), None)
    protected_record = {
        "old_holdout": protected_old,
        "old_retention": protected_retention,
        "new_holdout": protected_new,
        "checkpoint": protected_checkpoint,
        "memory_write_count": int(protected.memory.write_count),
        "attempt_count": len(trace_rows),
        "accepted_count": sum(int(not row["rejected"]) for row in trace_rows),
        "rejected_count": sum(int(row["rejected"]) for row in trace_rows),
        "physical_replay_write_count": int(
            protected.memory.write_count - phase_b.memory.write_count
        ),
        "association": _association_summary(protected, course.phase_a_train),
        "weight_delta_from_phase_b": _weight_deltas(
            _memory_weights(phase_b), _memory_weights(protected)
        ),
        "holdout_updates": 0,
    }
    protected_record["active_parameter_count"] = protected.parameter_count()
    protected_record["planned_active_parameter_count"] = (
        protected.config.planned_active_parameter_count
    )
    protected_record["parameter_count_matches_plan"] = (
        protected.parameter_count() == protected.config.planned_active_parameter_count
    )
    protected_record["causal_gain_action"] = float(
        protected_old["summary"]["all"]["action_accuracy"]
        - no_replay_old["summary"]["all"]["action_accuracy"]
    )
    protected_record["causal_gain_outcome"] = float(
        protected_old["summary"]["all"]["outcome_accuracy"]
        - no_replay_old["summary"]["all"]["outcome_accuracy"]
    )
    protected_record["new_delta_vs_no_replay_action"] = float(
        protected_new["summary"]["all"]["action_accuracy"]
        - no_replay_new["summary"]["all"]["action_accuracy"]
    )
    protected_record["new_delta_vs_no_replay_outcome"] = float(
        protected_new["summary"]["all"]["outcome_accuracy"]
        - no_replay_new["summary"]["all"]["outcome_accuracy"]
    )
    protected_record["candidate_gate_passed"] = _candidate_gate(
        {
            **protected_record,
            "phase_a_baseline": phase_a_baseline,
            "phase_a_retention_baseline": phase_a_retention,
        },
        no_replay_record,
        phase_a_baseline,
        phase_a_retention,
    )
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_b_checkpoint_digest": phase_b_digest,
        "phase_a_baseline": phase_a_baseline,
        "phase_a_retention_baseline": phase_a_retention,
        "phase_b_train_baseline": phase_b_baseline,
        "no_replay": no_replay_record,
        "all_replay": all_replay_record,
        "protected": protected_record,
        "protected_trace": {
            "rows": trace_rows,
            "attempt_count": len(trace_rows),
            "accepted_count": protected_record["accepted_count"],
            "rejected_count": protected_record["rejected_count"],
            "phase_b_holdout_read_during_decision": False,
            "phase_b_retention_read_during_decision": False,
            "order_digest": content_digest(
                [
                    (row["step"], row["replay_memory_id"], "replayed")
                    for row in trace_rows
                ]
            ),
        },
        "topology_digest": _topology_digest(protected),
        "phase_a_topology_digest": _topology_digest(phase_a),
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
        "variable_changed": "dynamic replay rollback decision only",
        "course": course.name,
        "corpus_digest": course.digest,
        "replay_provenance": "replayed",
        "replay_learning_targets": "all",
        "protection_signal": "reject only when phase-B train action and outcome margins both decrease versus the committed pre-write state",
        "phase_b_holdout_read_during_decision": False,
        "phase_b_retention_read_during_decision": False,
        "records": records,
        "candidate_gate_matrix": {
            "protected": [
                bool(record["protected"]["candidate_gate_passed"]) for record in records
            ]
        },
        "gates": {
            "fixed_replay_attempt_count": True,
            "decision_uses_train_only": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
            "does_not_change_default_checkpoint": True,
            "does_not_promote_candidate": True,
        },
        "boundary": "The protected path remains an experiment until all three seeds pass old/new retention and checkpoint gates.",
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
                "candidate_gate_matrix": result["candidate_gate_matrix"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
