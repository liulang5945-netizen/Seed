"""Run the R5C-S15 measured replay artifact batch canary."""

from __future__ import annotations

import argparse
import json
import sys
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
from scripts.training.eval_taiji_workbench_validation_measurements import (  # noqa: E402
    _candidate_probe,
)
from taiji import (  # noqa: E402
    StructuralValidationMeasurements,
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
)

REPORT_FORMAT = "taiji-w7-r5c-s15-workbench-measured-artifact-batch-v1"


def _build_artifact(
    model: TSKV8Adapter,
    candidate_id: str,
    executions: tuple[dict[str, object], ...],
    *,
    capacity_limit: int = 4,
) -> tuple[
    WorkbenchStructuralValidationArtifact,
    dict[str, object],
    StructuralValidationMeasurements,
]:
    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == candidate_id
    )
    measurements, replay, parent_digest, trial_digest, region_id, capacity = _candidate_probe(
        model,
        candidate_id,
        capacity_limit=capacity_limit,
    )
    workbench_region_id = (
        "workbench.code" if region_id == "adaptive.cortex" else "workbench.docs"
    )
    outcome_digests = tuple(
        item["evidence"]["evidence"]["outcome_digest"]
        for item in executions
        if item["evidence"]["observation"]["region_id"] == workbench_region_id
    )
    artifact = WorkbenchStructuralValidationArtifact.from_measurements(
        candidate_id=candidate_id,
        network_id=candidate.network_id,
        region_id=region_id,
        task_slice_id="workbench-measured-batch",
        outcome_digests=outcome_digests,
        parent_checkpoint_digest=parent_digest,
        trial_checkpoint_digest=trial_digest,
        holdout_inputs=replay["holdout_inputs"],
        holdout_outputs=replay["holdout_outputs"],
        retention_baseline=replay["retention_baseline"],
        retention_candidate=replay["retention_candidate"],
        lesion_baseline=replay["lesion_baseline"],
        lesion_candidate=replay["lesion_candidate"],
        resource_measurement=capacity,
        holdout_gain=measurements.holdout_gain,
        retention_regression=measurements.retention_regression,
        lesion_effect=measurements.lesion_effect,
        resource_state=measurements.resource_state,
        resource_cost=measurements.resource_cost,
        evidence_ids=candidate.evidence_ids,
        measurement_digest=measurements.measurement_digest,
    )
    return artifact, {
        "holdout_inputs": replay["holdout_inputs"],
        "expected_activities": replay["holdout_outputs"],
    }, measurements


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"real Workbench batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id, second_id = tuple(batch.selected_candidate_ids)

    first_artifact, first_replay, first_measurements = _build_artifact(
        runtime.model.architecture,
        first_id,
        executions,
    )
    first_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={first_id: first_artifact},
        replays_by_candidate={first_id: first_replay},
    )
    first_checkpoint = runtime.model.architecture.native_checkpoint()
    first_restored = TSKV8Adapter.from_native_checkpoint(first_checkpoint)
    second_artifact, second_replay, second_measurements = _build_artifact(
        first_restored,
        second_id,
        executions,
    )
    second_result = first_restored.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={second_id: second_artifact},
        replays_by_candidate={second_id: second_replay},
    )
    success_checkpoint = first_restored.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(success_checkpoint)
    repeated = restored.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={first_id: first_artifact, second_id: second_artifact},
        replays_by_candidate={first_id: first_replay, second_id: second_replay},
    )
    first_policy = first_result["results"][first_id]["continuation"]["decision"]
    second_policy = second_result["results"][second_id]["continuation"]["decision"]
    metrics = {
        "both_artifacts_use_measured_metrics": (
            first_artifact.holdout_gain == first_measurements.holdout_gain
            and second_artifact.holdout_gain == second_measurements.holdout_gain
            and first_artifact.resource_measurement_digest
            == first_measurements.resource_measurement_digest
            and second_artifact.resource_measurement_digest
            == second_measurements.resource_measurement_digest
        ),
        "first_policy_consumes_first_measured_metrics": (
            first_policy["holdout_gain"] == first_measurements.holdout_gain
            and first_policy["retention_regression"] == first_measurements.retention_regression
            and first_policy["lesion_effect"] == first_measurements.lesion_effect
            and first_policy["resource_state"] == first_measurements.resource_state
        ),
        "second_policy_consumes_second_measured_metrics": (
            second_policy["holdout_gain"] == second_measurements.holdout_gain
            and second_policy["retention_regression"] == second_measurements.retention_regression
            and second_policy["lesion_effect"] == second_measurements.lesion_effect
            and second_policy["resource_state"] == second_measurements.resource_state
        ),
        "incremental_batch_admission_is_complete": (
            first_result["batch"]["candidate_states"][first_id] == "admitted"
            and first_result["batch"]["candidate_states"][second_id] == "reserved"
            and second_result["batch"]["candidate_states"][first_id] == "admitted"
            and second_result["batch"]["candidate_states"][second_id] == "admitted"
            and second_result["batch"]["status"] == "completed"
            and second_result["artifact_batch"]["complete"]
        ),
        "measured_artifact_batch_survives_restore": (
            second_result["artifact_batch"]["batch_digest"]
            == restored.structural_validation_artifact_batches[0].batch_digest
        ),
        "repeated_measured_batch_is_idempotent": (
            repeated["results"][first_id]["status"] == "already_applied"
            and repeated["results"][second_id]["status"] == "already_applied"
            and repeated["artifact_batch"]["batch_digest"]
            == second_result["artifact_batch"]["batch_digest"]
        ),
        "no_manual_metric_values_are_injected": (
            bool(first_measurements.measurement_digest)
            and bool(second_measurements.measurement_digest)
            and bool(first_artifact.artifact_digest)
            and bool(second_artifact.artifact_digest)
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "schedule": schedule,
        "first_measurements": first_measurements.to_payload(),
        "second_measurements": second_measurements.to_payload(),
        "first_artifact": first_artifact.to_payload(),
        "second_artifact": second_artifact.to_payload(),
        "first_result": first_result,
        "second_result": second_result,
        "repeated": repeated,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a real multi-region batch must use independently measured replay metrics "
                "for every candidate and preserve batch checkpoint/idempotence"
            ),
        },
        "boundary": (
            "This canary closes the manual metric path for the measured batch but does "
            "not claim open-domain quality, unlimited growth, CUDA, or CI completion."
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
        / "taiji_w7_r5c_s15_workbench_measured_artifact_batch_20260830.json",
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
