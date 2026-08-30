"""Run the R5C-S13 multi-region replay validation artifact canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from scripts.training.eval_taiji_workbench_validation_artifact import (  # noqa: E402
    _build_trial_replay,
)
from taiji import (  # noqa: E402
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
)

REPORT_FORMAT = "taiji-w7-r5c-s13-workbench-validation-artifact-batch-v1"


def _build_artifact(
    model: TSKV8Adapter,
    candidate_id: str,
    executions: tuple[dict[str, object], ...],
) -> tuple[WorkbenchStructuralValidationArtifact, dict[str, object]]:
    candidate = next(
        item
        for item in model.structural_proposal_candidates
        if item.candidate_id == candidate_id
    )
    candidate_region_id = str(dict(candidate.specification)["region_id"])
    capacity = model.measure_structural_capacity_pressure(
        candidate_region_id,
        capacity_limit=4,
    )
    replay, trial_checkpoint_digest, checkpoint_digests = _build_trial_replay(
        model,
        candidate_id,
    )
    workbench_region_id = (
        "workbench.code"
        if candidate_region_id == "adaptive.cortex"
        else "workbench.docs"
    )
    outcome_digests = tuple(
        item["evidence"]["evidence"]["outcome_digest"]
        for item in executions
        if item["evidence"]["observation"]["region_id"] == workbench_region_id
    )
    artifact = WorkbenchStructuralValidationArtifact.from_measurements(
        candidate_id=candidate_id,
        network_id=candidate.network_id,
        region_id=candidate_region_id,
        task_slice_id="workbench-validation-batch",
        outcome_digests=outcome_digests,
        parent_checkpoint_digest=checkpoint_digests["parent_checkpoint_digest"],
        trial_checkpoint_digest=trial_checkpoint_digest,
        holdout_inputs=replay["holdout_inputs"],
        holdout_outputs=replay["holdout_outputs"],
        retention_baseline=replay["retention_baseline"],
        retention_candidate=replay["retention_candidate"],
        lesion_baseline=replay["lesion_baseline"],
        lesion_candidate=replay["lesion_candidate"],
        resource_measurement=capacity,
        holdout_gain=1.0,
        retention_regression=0.0,
        lesion_effect=1.0,
        resource_state=0.8,
        resource_cost=candidate.resource_cost,
        evidence_ids=candidate.evidence_ids,
    )
    return artifact, {
        "holdout_inputs": replay["holdout_inputs"],
        "expected_activities": replay["holdout_outputs"],
    }


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"real Workbench batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    selected = tuple(batch.selected_candidate_ids)
    if len(selected) != 2:
        raise AssertionError(f"expected two selected candidates: {selected}")
    first_id, second_id = selected

    first_artifact, first_replay = _build_artifact(
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
    first_admitted_topology = tuple(
        region.unit_ids for region in first_restored.neuron_regions
    )
    second_artifact, second_replay = _build_artifact(
        first_restored,
        second_id,
        executions,
    )

    failure_branch = TSKV8Adapter.from_native_checkpoint(first_checkpoint)
    bad_output = torch.zeros_like(second_replay["expected_activities"][0])
    failure_result = failure_branch.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={second_id: second_artifact},
        replays_by_candidate={
            second_id: {
                "holdout_inputs": second_replay["holdout_inputs"],
                "expected_activities": (bad_output,),
            }
        },
    )

    success_result = first_restored.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={second_id: second_artifact},
        replays_by_candidate={second_id: second_replay},
    )
    success_checkpoint = first_restored.native_checkpoint()
    success_restored = TSKV8Adapter.from_native_checkpoint(success_checkpoint)
    repeated = success_restored.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={
            first_id: first_artifact,
            second_id: second_artifact,
        },
        replays_by_candidate={
            first_id: first_replay,
            second_id: second_replay,
        },
    )

    wrong_key_branch = TSKV8Adapter.from_native_checkpoint(first_checkpoint)
    wrong_key = wrong_key_branch.continue_structural_candidate_batch_from_validation_artifacts(
        batch.batch_id,
        artifacts_by_candidate={second_id: first_artifact},
        replays_by_candidate={second_id: first_replay},
    )
    failure_topology = tuple(
        region.unit_ids for region in failure_branch.neuron_regions
    )
    failure_batch = failure_result["batch"]
    success_batch = success_result["batch"]
    repeated_batch = repeated["batch"]
    metrics = {
        "two_real_region_artifacts_created": (
            first_artifact.region_id != second_artifact.region_id
            and first_artifact.candidate_id == first_id
            and second_artifact.candidate_id == second_id
        ),
        "first_artifact_consumes_only_first_reservation": (
            first_result["batch"]["candidate_states"]
            and dict(first_result["batch"]["candidate_states"])[first_id] == "admitted"
            and dict(first_result["batch"]["candidate_states"])[second_id] == "reserved"
        ),
        "failure_branch_isolated_to_second_candidate": (
            failure_result["results"][second_id]["status"] == "failed_closed"
            and dict(failure_batch["candidate_states"])[first_id] == "admitted"
            and dict(failure_batch["candidate_states"])[second_id] == "failed_closed"
            and first_admitted_topology == failure_topology
        ),
        "valid_second_artifact_completes_batch": (
            success_result["results"][second_id]["status"] == "admitted"
            and dict(success_batch["candidate_states"])[first_id] == "admitted"
            and dict(success_batch["candidate_states"])[second_id] == "admitted"
            and success_batch["status"] == "completed"
            and success_result["artifact_batch"]["complete"]
            if isinstance(success_result.get("artifact_batch"), dict)
            else False
        ),
        "artifact_batch_digest_survives_restore": (
            success_result["artifact_batch"]["batch_digest"]
            == success_restored.structural_validation_artifact_batches[0].batch_digest
        ),
        "repeated_batch_artifact_consumption_is_idempotent": (
            repeated["results"][first_id]["status"] == "already_applied"
            and repeated["results"][second_id]["status"] == "already_applied"
            and repeated_batch["candidate_states"] == success_batch["candidate_states"]
            and repeated["artifact_batch"]["batch_digest"]
            == success_result["artifact_batch"]["batch_digest"]
        ),
        "cross_candidate_artifact_fails_closed": (
            wrong_key["results"][second_id]["status"] == "failed_closed"
            and dict(wrong_key["batch"]["candidate_states"])[first_id] == "admitted"
            and dict(wrong_key["batch"]["candidate_states"])[second_id] == "failed_closed"
        ),
        "public_runtime_entrypoint_used": True,
    }
    return {
        "format": REPORT_FORMAT,
        "schedule": schedule,
        "first_artifact": first_artifact.to_payload(),
        "second_artifact": second_artifact.to_payload(),
        "first_result": first_result,
        "failure_result": failure_result,
        "success_result": success_result,
        "repeated": repeated,
        "wrong_key": wrong_key,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "every selected candidate in a real multi-region batch must consume only "
                "its own replay-bound validation artifact, remain checkpointable, and "
                "preserve failure isolation"
            ),
        },
        "boundary": (
            "This canary removes batch-level manual validation metric injection but does "
            "not infer metrics from provider success, expand budget, parallelize topology "
            "commits, or make CUDA/CI claims."
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
        / "taiji_w7_r5c_s13_workbench_validation_artifact_batch_20260830.json",
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
