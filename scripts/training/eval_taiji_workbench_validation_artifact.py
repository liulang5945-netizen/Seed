"""Run the R5C-S12 replay-backed Workbench validation artifact canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import (  # noqa: E402
    AdaptiveNeuronRegion,
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s12-workbench-validation-artifact-v1"


def _build_trial_replay(
    model: TSKV8Adapter,
    candidate_id: str,
) -> tuple[dict[str, object], str, dict[str, object]]:
    parent_checkpoint = model.native_checkpoint()
    trial = TSKV8Adapter.from_native_checkpoint(parent_checkpoint)
    candidate = next(
        item for item in trial.structural_proposal_candidates if item.candidate_id == candidate_id
    )
    proposal = trial.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError(f"candidate {candidate_id} was not materialized in trial")
    trial_region_id = str(dict(candidate.specification)["region_id"])
    trial_region = next(item for item in trial.neuron_regions if item.region_id == trial_region_id)
    trial_checkpoint_digest = _checkpoint_digest(trial.native_checkpoint())
    shadow_region = AdaptiveNeuronRegion.from_payload(
        trial_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    shadow_region.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(trial_region.input_dim)
    holdout_input[shadow_region.incoming.pre_index[-1]] = torch.sign(
        shadow_region.incoming.edge_weight[-1]
    )
    holdout_output = shadow_region.step(holdout_input)
    replay = {
        "holdout_inputs": (holdout_input,),
        "holdout_outputs": (holdout_output,),
        "retention_baseline": {"unit_ids": list(trial_region.unit_ids)},
        "retention_candidate": {"unit_ids": list(shadow_region.unit_ids)},
        "lesion_baseline": {"candidate_active": False, "region_id": trial_region_id},
        "lesion_candidate": {"candidate_active": True, "region_id": trial_region_id},
    }
    return replay, trial_checkpoint_digest, {
        "parent_checkpoint_digest": _checkpoint_digest(parent_checkpoint),
        "trial_checkpoint_digest": trial_checkpoint_digest,
        "candidate_region_id": trial_region_id,
    }


def evaluate() -> dict[str, object]:
    from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: PLC0415
        _build_runtime,
        _schedule_requests,
    )

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
    candidate_region_id = str(dict(candidate.specification)["region_id"])
    capacity = runtime.model.architecture.measure_structural_capacity_pressure(
        candidate_region_id,
        capacity_limit=4,
    )
    replay, trial_checkpoint_digest, checkpoint_digests = _build_trial_replay(
        runtime.model.architecture,
        candidate_id,
    )
    outcome_digests = tuple(
        item["evidence"]["evidence"]["outcome_digest"]
        for item in executions
        if item["evidence"]["observation"]["region_id"]
        == ("workbench.code" if candidate_region_id == "adaptive.cortex" else "workbench.docs")
    )
    artifact = WorkbenchStructuralValidationArtifact.from_measurements(
        candidate_id=candidate_id,
        network_id=candidate.network_id,
        region_id=candidate_region_id,
        task_slice_id="workbench-validation",
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
    mismatch_model = TSKV8Adapter.from_native_checkpoint(runtime.model.architecture.native_checkpoint())
    mismatch_admission_count = len(mismatch_model.structural_admission_results)
    mismatch_topology_before = tuple(
        region.unit_ids for region in mismatch_model.neuron_regions
    )
    mismatch_output = torch.zeros_like(replay["holdout_outputs"][0])
    mismatch = mismatch_model.continue_structural_candidate_from_validation_artifact(
        artifact,
        holdout_inputs=replay["holdout_inputs"],
        expected_activities=(mismatch_output,),
    )
    mismatch_topology_after = tuple(
        region.unit_ids for region in mismatch_model.neuron_regions
    )
    continuation = runtime.model.architecture.continue_structural_candidate_from_validation_artifact(
        artifact,
        holdout_inputs=replay["holdout_inputs"],
        expected_activities=replay["holdout_outputs"],
    )
    checkpoint_after = runtime.model.architecture.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint_after)
    repeated = restored.continue_structural_candidate_from_validation_artifact(
        artifact,
        holdout_inputs=replay["holdout_inputs"],
        expected_activities=replay["holdout_outputs"],
    )
    tamper_detected = False
    tampered = artifact.to_payload()
    tampered["holdout_gain"] = 0.9
    try:
        WorkbenchStructuralValidationArtifact.from_payload(tampered)
    except ValueError:
        tamper_detected = True
    metrics = {
        "real_workbench_outcomes_are_bound": (
            len(outcome_digests) == 3 and all(bool(item) for item in outcome_digests)
        ),
        "artifact_digest_is_content_addressed": (
            WorkbenchStructuralValidationArtifact.from_payload(artifact.to_payload()) == artifact
        ),
        "artifact_binds_parent_and_trial_checkpoint": (
            bool(artifact.parent_checkpoint_digest)
            and bool(artifact.trial_checkpoint_digest)
            and artifact.trial_checkpoint_digest == checkpoint_digests["trial_checkpoint_digest"]
        ),
        "holdout_replay_digest_matches": artifact.matches_holdout_replay(
            replay["holdout_inputs"], replay["holdout_outputs"]
        ),
        "replay_mismatch_fails_closed_without_admission": (
            mismatch["status"] == "failed_closed"
            and len(mismatch_model.structural_admission_results) == mismatch_admission_count
            and mismatch_topology_after == mismatch_topology_before
        ),
        "artifact_drives_existing_policy_lifecycle": (
            continuation["status"] == "admitted"
            and continuation["continuation"]["decision"]["evidence_ids"]
            == list(artifact.evidence_ids)
        ),
        "artifact_survives_checkpoint_restore": (
            len(restored.structural_validation_artifacts) == 1
            and restored.structural_validation_artifacts[0] == artifact
        ),
        "repeated_artifact_continuation_is_idempotent": (
            repeated["status"] == "already_applied"
            and repeated["continuation"]["status"] == "admitted"
        ),
        "tampered_artifact_is_rejected": tamper_detected,
    }
    return {
        "format": REPORT_FORMAT,
        "schedule": schedule,
        "capacity": capacity.to_payload(),
        "artifact": artifact.to_payload(),
        "mismatch": mismatch,
        "continuation": continuation,
        "repeated": repeated,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "real Workbench replay facts must be content-addressed, checkpoint-bound, "
                "replay-verifiable, and consumed by the existing validation policy"
            ),
        },
        "boundary": (
            "This canary does not infer metrics from provider success, bypass policy, "
            "parallelize topology commits, expand budget, or make CUDA/CI claims."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s12_workbench_validation_artifact_20260830.json",
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
