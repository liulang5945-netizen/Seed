"""Run the R5C-S23 coordinated structural-lineage retention canary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import (  # noqa: E402
    StructuralAdmissionResult,
    StructuralCandidateBatch,
    StructuralCandidateRollback,
    StructuralCandidateValidation,
    StructuralLineageRetentionResult,
    StructuralRuntimeObservation,
    StructuralValidationArtifactBatch,
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
    evaluate_structural_candidate_validation,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s23-structural-lineage-compaction-v1"
RETENTION_LIMIT = 1
TERMINAL_BATCH_ID = "batch:terminal-lineage"
TERMINAL_CANDIDATE_ID = "candidate:terminal-lineage"


def _terminal_batch(source: StructuralCandidateBatch) -> StructuralCandidateBatch:
    return replace(
        source,
        batch_id=TERMINAL_BATCH_ID,
        candidate_ids=(TERMINAL_CANDIDATE_ID,),
        selected_candidate_ids=(),
        deferred_candidate_ids=(),
        rejected_candidate_ids=(TERMINAL_CANDIDATE_ID,),
        candidate_states=((TERMINAL_CANDIDATE_ID, "rejected"),),
        reasons=((TERMINAL_CANDIDATE_ID, "terminal"),),
        reserved_resource_cost=0,
        reservation_remaining=0,
        arbitration_digest=f"terminal:{TERMINAL_BATCH_ID}",
        revision=source.revision + 1,
        status="completed",
    )


def _record_terminal_subgraph(model: TSKV8Adapter, source: StructuralCandidateBatch) -> None:
    model._record_structural_candidate_batch(_terminal_batch(source))
    artifact = WorkbenchStructuralValidationArtifact.from_measurements(
        candidate_id=TERMINAL_CANDIDATE_ID,
        network_id="workbench",
        region_id="workbench.code",
        task_slice_id="terminal-lineage",
        outcome_digests=("terminal:outcome",),
        parent_checkpoint_digest="terminal:parent",
        trial_checkpoint_digest="terminal:trial",
        holdout_inputs=(),
        holdout_outputs=(),
        retention_baseline=(),
        retention_candidate=(),
        lesion_baseline=(),
        lesion_candidate=(),
        resource_measurement=(),
        holdout_gain=0.0,
        retention_regression=0.0,
        lesion_effect=0.0,
        resource_state=0.0,
        resource_cost=1,
        evidence_ids=("terminal:evidence",),
    )
    model._record_structural_validation_artifact(artifact)
    model._record_structural_validation_artifact_batch(
        StructuralValidationArtifactBatch.from_digest_map(
            batch_id=TERMINAL_BATCH_ID,
            expected_candidate_ids=(TERMINAL_CANDIDATE_ID,),
            artifact_digests_by_candidate={
                TERMINAL_CANDIDATE_ID: artifact.artifact_digest,
            },
        )
    )
    model._record_structural_candidate_validation(
        StructuralCandidateValidation(
            candidate_id=TERMINAL_CANDIDATE_ID,
            proposal_id=None,
            status="rejected",
            validation_score=0.0,
            parent_checkpoint_digest="terminal:parent",
            validation_checkpoint_digest="terminal:validation",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            evidence_ids=("terminal:evidence",),
            error="terminal validation",
        )
    )
    decision = evaluate_structural_candidate_validation(
        TERMINAL_CANDIDATE_ID,
        holdout_gain=0.0,
        retention_regression=0.0,
        lesion_effect=0.0,
        resource_state=0.0,
        resource_cost=1,
        structural_budget=1,
        evidence_ids=("terminal:evidence",),
    )
    model._record_structural_validation_gate_decision(decision)
    model._record_structural_admission_result(
        StructuralAdmissionResult(
            candidate_id=TERMINAL_CANDIDATE_ID,
            proposal_id="terminal:proposal",
            status="rejected",
            decision_digest=decision.decision_digest,
            parent_checkpoint_digest="terminal:parent",
            child_checkpoint_digest="terminal:child",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            error="terminal admission",
        )
    )
    model._record_structural_candidate_rollback(
        StructuralCandidateRollback(
            batch_id=TERMINAL_BATCH_ID,
            candidate_id=TERMINAL_CANDIDATE_ID,
            proposal_id="terminal:proposal",
            status="not_rollbackable",
            admission_parent_checkpoint_digest="terminal:parent",
            admission_child_checkpoint_digest="terminal:child",
            rollback_checkpoint_digest="terminal:rollback",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            resource_cost=1,
            reason="terminal canary record",
        )
    )
    schedule = model.structural_workbench_batch_schedule_results[-1]
    model._record_structural_workbench_batch_schedule_result(
        replace(
            schedule,
            batch_id=TERMINAL_BATCH_ID,
            candidate_ids=(TERMINAL_CANDIDATE_ID,),
        )
    )


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S23 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active_batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    _record_terminal_subgraph(model, active)
    terminal_artifact = next(
        item
        for item in model.structural_validation_artifacts
        if item.candidate_id == TERMINAL_CANDIDATE_ID
    )

    result = model.compact_structural_lineage_history(max_batches=RETENTION_LIMIT)
    result_roundtrip = StructuralLineageRetentionResult.from_payload(result.to_payload())
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    stale = restored.continue_structural_candidate_from_validation_artifact(
        terminal_artifact,
        holdout_inputs=(),
        expected_activities=(),
    )
    try:
        restored.rollback_structural_candidate_batch(
            TERMINAL_BATCH_ID,
            TERMINAL_CANDIDATE_ID,
        )
    except ValueError as exc:
        rollback_failed_closed = "unknown structural candidate batch" in str(exc)
    else:
        rollback_failed_closed = False

    active_restored = next(
        item for item in restored.structural_candidate_batches if item.batch_id == active_batch_id
    )
    restored_checkpoint_digest = _checkpoint_digest(restored.native_checkpoint())
    fresh_observation = StructuralRuntimeObservation(
        network_id="workbench",
        region_id="workbench.code",
        tick=model.structural_runtime_tick + 1,
        usage=0.8,
        resource_pressure=0.2,
        prediction_error=0.2,
        learning_gain=0.1,
        holdout_transfer=0.0,
        evidence_id="s23:fresh",
        task_slice_id="s23-fresh",
        partition="train",
    )
    for current in (model, restored):
        current.record_structural_runtime_observation(fresh_observation)
        current.seal_structural_evidence_window(
            "workbench",
            "workbench.code",
            task_slice_id="s23-fresh",
            partition="train",
        )
    fresh_schedule = model.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    restored_fresh_schedule = restored.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )

    pressure_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    pressure_active = next(
        item for item in pressure_model.structural_candidate_batches if item.batch_id == active_batch_id
    )
    pressure_model._record_structural_candidate_batch(
        replace(active_restored, batch_id="batch:protected-copy", revision=active_restored.revision + 1)
    )
    pressure = pressure_model.compact_structural_lineage_history(max_batches=RETENTION_LIMIT)

    record_counts = dict(result.removed_record_counts)
    metrics = {
        "real_workbench_batch_is_active": (
            all(item["outcome"]["status"] == "success" for item in executions)
            and active.active_reservation
        ),
        "terminal_subgraph_is_removed_as_one_unit": (
            result.status == "compacted"
            and result.removed_batch_ids == (TERMINAL_BATCH_ID,)
            and TERMINAL_BATCH_ID not in {item.batch_id for item in model.structural_candidate_batches}
            and TERMINAL_BATCH_ID
            not in {item.batch_id for item in model.structural_validation_artifact_batches}
            and record_counts.get("candidate_rollbacks", 0) == 1
            and record_counts.get("validation_artifact_batches", 0) == 1
            and record_counts.get("candidate_validations", 0) == 1
            and record_counts.get("validation_artifacts", 0) == 1
            and record_counts.get("validation_gate_decisions", 0) == 1
            and record_counts.get("admission_results", 0) == 1
        ),
        "active_lineage_is_retained": (
            active_batch_id in result.retained_batch_ids
            and active_restored.active_reservation
            and set(active_restored.candidate_ids).issubset(set(result.retained_candidate_ids))
        ),
        "checkpoint_restore_is_deterministic": (
            restored_checkpoint_digest == _checkpoint_digest(checkpoint)
            and active_batch_id in {item.batch_id for item in restored.structural_candidate_batches}
            and TERMINAL_BATCH_ID
            not in {item.batch_id for item in restored.structural_candidate_batches}
        ),
        "stale_replay_and_rollback_fail_closed": (
            restored.materialize_structural_candidate(TERMINAL_CANDIDATE_ID) is None
            and stale["status"] == "failed_closed"
            and rollback_failed_closed
        ),
        "retention_result_is_content_addressed": (
            result_roundtrip == result and bool(result.result_digest)
        ),
        "checkpoint_restore_continues_new_evidence_deterministically": (
            fresh_schedule == restored_fresh_schedule
            and fresh_schedule.status == "batch_created"
            and fresh_schedule.trigger_tick == fresh_observation.tick
            and fresh_schedule.batch_id != active_batch_id
        ),
        "protected_pressure_is_observable_without_deletion": (
            pressure.status == "nothing_to_compact"
            and pressure.retention_pressure
            and pressure.removed_batch_ids == ()
            and pressure_active.active_reservation
            and {item.batch_id for item in pressure_model.structural_candidate_batches}
            == {active_batch_id, "batch:protected-copy"}
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "schedule": schedule,
        "compaction": result.to_payload(),
        "pressure": pressure.to_payload(),
        "stale_replay": stale,
        "fresh_schedule": fresh_schedule.to_payload(),
        "restored_fresh_schedule": restored_fresh_schedule.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "only a terminal structural lineage subgraph without active reservation, pending "
                "continuation, pending topology, or rollback dependency may be compacted; all "
                "dependent ledgers are removed atomically, stale replay fails closed, and protected "
                "lineage pressure remains observable"
            ),
        },
        "boundary": (
            "This canary covers native CPU lineage-graph retention and checkpoint determinism. "
            "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s23_structural_lineage_compaction_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
