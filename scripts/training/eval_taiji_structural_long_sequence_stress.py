"""Run the R5C-S21 long-sequence compaction and ledger stress canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_measured_multi_round import (  # noqa: E402
    _record_round_two_evidence,
    _round_two_schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s21-structural-long-sequence-stress-v1"
COMPACTED_WINDOW_CAP = 16


def _record_round_three_evidence(runtime) -> tuple[dict[str, object], ...]:
    """Record a third fresh six-read round with new task-slice identities."""

    return (
        _execute_observation(
            runtime,
            ordinal=13,
            region_id="workbench.code",
            task_slice_id="code-adapter-round-3",
            partition="train",
            path="taiji/adapter.py",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=14,
            region_id="workbench.code",
            task_slice_id="code-growth-round-3",
            partition="train",
            path="taiji/structural_growth.py",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=15,
            region_id="workbench.code",
            task_slice_id="code-holdout-round-3",
            partition="holdout",
            path="plans/active/roadmap/04_EXECUTION_PLAN.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
        _execute_observation(
            runtime,
            ordinal=16,
            region_id="workbench.docs",
            task_slice_id="docs-plan-round-3",
            partition="train",
            path="plans/active/roadmap/03_CURRENT_EXECUTION.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=17,
            region_id="workbench.docs",
            task_slice_id="docs-status-round-3",
            partition="train",
            path="plans/reference/IMPLEMENTATION_STATUS_2026_08.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=18,
            region_id="workbench.docs",
            task_slice_id="docs-holdout-round-3",
            partition="holdout",
            path="README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
    )


def _round_three_schedule_requests() -> tuple[dict[str, object], ...]:
    return (
        {
            "network_id": "workbench",
            "region_id": "workbench.code",
            "controller_region_id": "adaptive.cortex",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.cortex",),
            "specification": {"region_id": "adaptive.cortex", "unit_id": "u4"},
        },
        {
            "network_id": "workbench",
            "region_id": "workbench.docs",
            "controller_region_id": "adaptive.memory",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.memory",),
            "specification": {"region_id": "adaptive.memory", "unit_id": "m4"},
        },
    )


def _batch(model: TSKV8Adapter, batch_id: str):
    return next(item for item in model.structural_candidate_batches if item.batch_id == batch_id)


def _admit_batch(
    model: TSKV8Adapter,
    batch_id: str,
    evidence: tuple[dict[str, object], ...],
) -> dict[str, object]:
    batch = _batch(model, batch_id)
    artifacts: dict[str, object] = {}
    replays: dict[str, object] = {}
    measurements: dict[str, object] = {}
    results: dict[str, object] = {}
    for candidate_id in batch.selected_candidate_ids:
        artifact, replay, measured = _build_artifact(model, candidate_id, evidence, capacity_limit=8)
        artifacts[candidate_id] = artifact
        replays[candidate_id] = replay
        measurements[candidate_id] = measured
        result = model.continue_structural_candidate_batch_from_validation_artifacts(
            batch_id,
            artifacts_by_candidate={candidate_id: artifact},
            replays_by_candidate={candidate_id: replay},
        )
        results[candidate_id] = result
        if result["results"][candidate_id]["status"] != "admitted":
            raise AssertionError(f"candidate did not admit in {batch_id}: {result}")
    return {
        "artifacts": artifacts,
        "replays": replays,
        "measurements": measurements,
        "results": results,
    }


def _ledger_snapshot(model: TSKV8Adapter) -> dict[str, object]:
    ledger = model.structural_evidence_ledger
    audit = model.structural_evidence_consumption_audit
    return {
        "digest": ledger.digest,
        "active_window_digests": tuple(item.window_digest for item in ledger.sealed_summaries),
        "compacted_window_digests": tuple(item.window_digest for item in ledger.compacted_windows),
        "pressure_snapshot_digests": tuple(
            item.snapshot_digest for item in ledger.pressure_snapshots
        ),
        "active_count": ledger.active_observed_count,
        "compacted_count": len(ledger.compacted_windows),
        "max_compacted_windows": ledger.max_compacted_windows,
        "audit": audit,
    }


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    model = runtime.model.architecture
    model.structural_evidence_ledger.max_compacted_windows = COMPACTED_WINDOW_CAP

    round_one_evidence = _record_real_evidence(runtime)
    round_one_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if round_one_schedule.get("status") != "batch_created":
        raise AssertionError(f"round one batch was not created: {round_one_schedule}")
    round_one_batch_id = str(round_one_schedule["batch_id"])
    round_one_batch = _batch(model, round_one_batch_id)
    round_one_admission = _admit_batch(model, round_one_batch_id, round_one_evidence)
    round_one_checkpoint = model.native_checkpoint()

    round_one_compaction = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    round_one_ledger = _ledger_snapshot(model)
    round_one_compacted_checkpoint = model.native_checkpoint()

    runtime.model.substrate = TSKV8Adapter.from_native_checkpoint(round_one_compacted_checkpoint)
    model = runtime.model.architecture
    round_two_evidence = _record_round_two_evidence(runtime)
    round_two_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _round_two_schedule_requests()
    )
    if round_two_schedule.get("status") != "batch_created":
        raise AssertionError(f"round two batch was not created: {round_two_schedule}")
    round_two_batch_id = str(round_two_schedule["batch_id"])
    round_two_batch = _batch(model, round_two_batch_id)
    round_two_admission = _admit_batch(model, round_two_batch_id, round_two_evidence)
    round_two_success_budget = int(model.cognitive_snapshot().development.structural_budget)
    round_two_rollback_candidate = round_two_batch.selected_candidate_ids[-1]
    round_two_rollback = model.rollback_structural_candidate_batch(
        round_two_batch_id,
        round_two_rollback_candidate,
    )
    round_two_after_rollback_budget = int(
        model.cognitive_snapshot().development.structural_budget
    )
    round_two_ledger_before_compaction = _ledger_snapshot(model)
    round_two_compaction = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    round_two_ledger = _ledger_snapshot(model)
    round_two_compacted_checkpoint = model.native_checkpoint()

    runtime.model.substrate = TSKV8Adapter.from_native_checkpoint(round_two_compacted_checkpoint)
    model = runtime.model.architecture
    round_three_evidence = _record_round_three_evidence(runtime)
    pre_schedule_ledger = _ledger_snapshot(model)
    pre_schedule_checkpoint = model.native_checkpoint()
    pre_schedule_compaction = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    post_noop_compaction_ledger = _ledger_snapshot(model)
    round_three_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _round_three_schedule_requests()
    )
    if round_three_schedule.get("status") != "batch_created":
        raise AssertionError(f"round three batch was not created: {round_three_schedule}")
    round_three_batch_id = str(round_three_schedule["batch_id"])
    round_three_batch = _batch(model, round_three_batch_id)
    round_three_admission = _admit_batch(model, round_three_batch_id, round_three_evidence)
    round_three_rollback_candidate = round_three_batch.selected_candidate_ids[-1]
    round_three_rollback = model.rollback_structural_candidate_batch(
        round_three_batch_id,
        round_three_rollback_candidate,
    )
    round_three_batch_after_rollback = _batch(model, round_three_batch_id)

    atomic_before = _ledger_snapshot(model)
    model.structural_evidence_ledger.max_compacted_windows = COMPACTED_WINDOW_CAP - 1
    try:
        model.compact_structural_evidence_history(keep_latest_per_stream=1)
    except OverflowError:
        atomic_overflow = True
    else:
        atomic_overflow = False
    model.structural_evidence_ledger.max_compacted_windows = COMPACTED_WINDOW_CAP
    atomic_after = _ledger_snapshot(model)
    round_three_compaction = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    final_ledger = _ledger_snapshot(model)
    final_checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(final_checkpoint)
    restored_ledger = _ledger_snapshot(restored)

    pre_schedule_unconsumed = set(pre_schedule_ledger["audit"].unconsumed_window_digests)
    post_noop_unconsumed = set(post_noop_compaction_ledger["audit"].unconsumed_window_digests)
    metrics = {
        "three_real_rounds_create_fresh_windows_and_batches": (
            all(item["outcome"]["status"] == "success" for item in round_one_evidence)
            and all(item["outcome"]["status"] == "success" for item in round_two_evidence)
            and all(item["outcome"]["status"] == "success" for item in round_three_evidence)
            and set(round_one_schedule["source_window_digests"]).isdisjoint(
                set(round_two_schedule["source_window_digests"])
            )
            and set(round_two_schedule["source_window_digests"]).isdisjoint(
                set(round_three_schedule["source_window_digests"])
            )
            and len({round_one_batch_id, round_two_batch_id, round_three_batch_id}) == 3
        ),
        "per_stream_retention_is_bounded_after_each_compaction": (
            round_one_compaction.status == "compacted"
            and round_two_compaction.status == "compacted"
            and round_one_ledger["active_count"] == 2
            and round_two_ledger["active_count"] == 2
            and round_one_ledger["compacted_count"] == 4
            and round_two_ledger["compacted_count"] == 10
            and round_one_ledger["compacted_count"] <= COMPACTED_WINDOW_CAP
            and round_two_ledger["compacted_count"] <= COMPACTED_WINDOW_CAP
        ),
        "unconsumed_round_survives_pre_schedule_compaction": (
            pre_schedule_compaction.status == "nothing_to_compact"
            and len(pre_schedule_unconsumed) == 6
            and post_noop_unconsumed == pre_schedule_unconsumed
            and set(pre_schedule_ledger["active_window_digests"]).issubset(
                set(post_noop_compaction_ledger["active_window_digests"])
            )
        ),
        "rollback_reopens_budget_and_does_not_contaminate_other_candidate": (
            round_two_rollback["status"] == "rolled_back"
            and round_two_after_rollback_budget == round_two_success_budget + 1
            and round_three_rollback["status"] == "rolled_back"
            and round_three_batch_after_rollback.state_by_candidate[round_three_batch.selected_candidate_ids[0]]
            == "admitted"
            and round_three_batch_after_rollback.state_by_candidate[round_three_rollback_candidate]
            == "rolled_back"
        ),
        "compaction_overflow_is_atomic": (
            atomic_overflow and atomic_before["digest"] == atomic_after["digest"]
            and atomic_before["active_window_digests"] == atomic_after["active_window_digests"]
            and atomic_before["compacted_window_digests"]
            == atomic_after["compacted_window_digests"]
        ),
        "final_long_sequence_stays_within_compacted_cap": (
            round_three_compaction.status == "compacted"
            and final_ledger["compacted_count"] == COMPACTED_WINDOW_CAP
            and final_ledger["active_count"] == 2
            and final_ledger["compacted_count"] <= final_ledger["max_compacted_windows"]
        ),
        "checkpoint_restores_long_sequence_audit_and_lineage": (
            restored_ledger["digest"] == final_ledger["digest"]
            and restored_ledger["active_window_digests"] == final_ledger["active_window_digests"]
            and restored_ledger["compacted_window_digests"]
            == final_ledger["compacted_window_digests"]
            and restored_ledger["pressure_snapshot_digests"]
            == final_ledger["pressure_snapshot_digests"]
            and len(restored.structural_candidate_batches) >= 3
            and len(restored.structural_admission_results) >= 6
        ),
        "round_one_parent_checkpoint_remains_recoverable": (
            bool(round_one_checkpoint)
            and TSKV8Adapter.from_native_checkpoint(round_one_checkpoint).structural_evidence_ledger.digest
            != final_ledger["digest"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "cap": COMPACTED_WINDOW_CAP,
        "rounds": {
            "round_one": {
                "evidence_count": len(round_one_evidence),
                "schedule": round_one_schedule,
                "batch": round_one_batch.to_payload(),
                "admission": round_one_admission["results"],
                "compaction": round_one_compaction.to_payload(),
                "ledger": round_one_ledger,
            },
            "round_two": {
                "evidence_count": len(round_two_evidence),
                "schedule": round_two_schedule,
                "batch": round_two_batch.to_payload(),
                "admission": round_two_admission["results"],
                "rollback": round_two_rollback,
                "ledger_before_compaction": round_two_ledger_before_compaction,
                "compaction": round_two_compaction.to_payload(),
                "ledger": round_two_ledger,
            },
            "round_three": {
                "evidence_count": len(round_three_evidence),
                "pre_schedule_ledger": pre_schedule_ledger,
                "pre_schedule_checkpoint_present": bool(pre_schedule_checkpoint),
                "pre_schedule_compaction": pre_schedule_compaction.to_payload(),
                "schedule": round_three_schedule,
                "batch": round_three_batch.to_payload(),
                "admission": round_three_admission["results"],
                "rollback": round_three_rollback,
                "compaction": round_three_compaction.to_payload(),
                "ledger": final_ledger,
            },
        },
        "atomic_overflow": {
            "before": atomic_before,
            "after": atomic_after,
            "max_during_attempt": COMPACTED_WINDOW_CAP - 1,
        },
        "restored_ledger": restored_ledger,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "at least three fresh real Workbench rounds must keep active retention bounded, "
                "preserve unconsumed windows, isolate rollback/resource pressure, fail atomically "
                "at the compaction cap, and restore the complete lineage ledger"
            ),
        },
        "boundary": (
            "This canary proves bounded native evidence/lineage durability under a finite CPU cap; "
            "it does not claim unlimited growth, open-domain quality, CUDA, CI, or physical deletion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s21_structural_long_sequence_stress_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
