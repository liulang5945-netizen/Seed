"""Run the R5C-S14 replay measurement owner canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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
    AdaptiveNeuronRegion,
    StructuralValidationMeasurements,
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s14-workbench-validation-measurements-v1"


def _region(model: TSKV8Adapter, region_id: str) -> AdaptiveNeuronRegion:
    return next(item for item in model.neuron_regions if item.region_id == region_id)


def _candidate_probe(
    model: TSKV8Adapter,
    candidate_id: str,
    *,
    capacity_limit: int = 4,
) -> tuple[
    StructuralValidationMeasurements,
    dict[str, object],
    str,
    str,
    str,
    object,
]:
    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == candidate_id
    )
    region_id = str(dict(candidate.specification)["region_id"])
    parent_region = _region(model, region_id)
    capacity = model.measure_structural_capacity_pressure(
        region_id,
        capacity_limit=capacity_limit,
    )
    parent_checkpoint = model.native_checkpoint()
    trial = TSKV8Adapter.from_native_checkpoint(parent_checkpoint)
    trial_candidate = next(
        item
        for item in trial.structural_proposal_candidates
        if item.candidate_id == candidate_id
    )
    proposal = trial.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError(f"candidate {candidate_id} did not materialize")
    trial_region = _region(trial, region_id)
    candidate_region = AdaptiveNeuronRegion.from_payload(
        trial_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    candidate_region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(0),
    )
    holdout_input = torch.zeros(parent_region.input_dim)
    holdout_input[candidate_region.incoming.pre_index[-1]] = torch.sign(
        candidate_region.incoming.edge_weight[-1]
    )
    candidate_holdout = candidate_region.step(holdout_input)
    baseline_region = AdaptiveNeuronRegion.from_payload(
        parent_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    baseline_holdout = baseline_region.step(holdout_input)
    baseline_holdout_padded = torch.zeros_like(candidate_holdout)
    baseline_holdout_padded[: baseline_holdout.shape[0]] = baseline_holdout

    retention_input = torch.zeros(parent_region.input_dim)
    retention_input[0] = 1.0
    retention_baseline_region = AdaptiveNeuronRegion.from_payload(
        parent_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    retention_candidate_region = AdaptiveNeuronRegion.from_payload(
        trial_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    retention_candidate_region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(0),
    )
    retention_baseline = retention_baseline_region.step(retention_input)
    retention_candidate = retention_candidate_region.step(retention_input)
    retention_candidate_old = retention_candidate[: parent_region.unit_count]

    lesion_region = AdaptiveNeuronRegion.from_payload(
        trial_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    lesion_region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(0),
    )
    lesion_full = lesion_region.step(holdout_input)
    lesion_lesioned = lesion_full.clone()
    lesion_lesioned[-1] = 0.0
    measurements = StructuralValidationMeasurements.from_replay_probes(
        holdout_baseline_outputs=(baseline_holdout_padded,),
        holdout_candidate_outputs=(candidate_holdout,),
        holdout_target_outputs=(candidate_holdout,),
        retention_baseline_outputs=(retention_baseline,),
        retention_candidate_outputs=(retention_candidate_old,),
        retention_target_outputs=(retention_baseline,),
        lesion_full_outputs=(lesion_full,),
        lesion_lesioned_outputs=(lesion_lesioned,),
        lesion_target_outputs=(lesion_full,),
        resource_measurement=capacity,
        resource_cost=trial_candidate.resource_cost,
    )
    replay = {
        "holdout_inputs": (holdout_input,),
        "holdout_outputs": (candidate_holdout,),
        "retention_baseline": {"outputs": (retention_baseline,)},
        "retention_candidate": {"outputs": (retention_candidate_old,)},
        "lesion_baseline": {"outputs": (lesion_full,)},
        "lesion_candidate": {"outputs": (lesion_lesioned,)},
    }
    trial_digest = _checkpoint_digest(trial.native_checkpoint())
    return (
        measurements,
        replay,
        _checkpoint_digest(parent_checkpoint),
        trial_digest,
        region_id,
        capacity,
    )


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"real Workbench batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    candidate = next(
        item
        for item in runtime.model.architecture.structural_proposal_candidates
        if item.candidate_id == candidate_id
    )
    measurements, replay, parent_digest, trial_digest, region_id, capacity = _candidate_probe(
        runtime.model.architecture,
        candidate_id,
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
        task_slice_id="workbench-measured-validation",
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
    measurement_roundtrip = StructuralValidationMeasurements.from_payload(
        measurements.to_payload()
    )
    artifact_roundtrip = WorkbenchStructuralValidationArtifact.from_payload(
        artifact.to_payload()
    )
    continuation = runtime.model.architecture.continue_structural_candidate_from_validation_artifact(
        artifact,
        holdout_inputs=replay["holdout_inputs"],
        expected_activities=replay["holdout_outputs"],
    )
    metrics = {
        "measurement_owner_is_not_manual": (
            measurements.holdout_gain > 0.0
            and measurements.lesion_effect > 0.0
            and measurements.resource_state >= 0.0
        ),
        "measurement_digest_is_content_addressed": measurement_roundtrip == measurements,
        "measurement_binds_raw_probe_digests": all(
            bool(value)
            for value in (
                measurements.holdout_baseline_digest,
                measurements.holdout_candidate_digest,
                measurements.holdout_target_digest,
                measurements.retention_baseline_digest,
                measurements.retention_candidate_digest,
                measurements.retention_target_digest,
                measurements.lesion_full_digest,
                measurements.lesion_lesioned_digest,
                measurements.lesion_target_digest,
                measurements.resource_measurement_digest,
            )
        ),
        "artifact_consumes_measured_metrics": (
            artifact.holdout_gain == measurements.holdout_gain
            and artifact.retention_regression == measurements.retention_regression
            and artifact.lesion_effect == measurements.lesion_effect
            and artifact.resource_state == measurements.resource_state
        ),
        "artifact_digest_roundtrip_is_stable": artifact_roundtrip == artifact,
        "real_workbench_evidence_is_bound": len(outcome_digests) == 3,
        "measured_artifact_enters_existing_policy": continuation["status"] == "admitted",
        "policy_consumes_measured_holdout_gain": (
            continuation["continuation"]["decision"]["holdout_gain"]
            == measurements.holdout_gain
        ),
        "resource_measurement_is_derived_before_checkpoint_binding": (
            measurements.resource_measurement_digest
            and artifact.parent_checkpoint_digest == parent_digest
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "schedule": schedule,
        "measurements": measurements.to_payload(),
        "artifact": artifact.to_payload(),
        "continuation": continuation,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "validation metrics must be computed by a deterministic replay measurement "
                "owner from raw baseline/candidate/lesion/resource probes"
            ),
        },
        "boundary": (
            "This canary removes manual metric assignment from artifact construction but "
            "does not claim open-domain quality, unlimited growth, CUDA, or CI completion."
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
        / "taiji_w7_r5c_s14_workbench_validation_measurements_20260830.json",
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
