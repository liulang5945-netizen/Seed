"""Run the R5C-S37 SeedRuntime artifact failure/concurrency canary."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s37-runtime-artifact-failure-concurrency-v1"


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S37 batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        executions,
    )

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    checkpoint_path = checkpoint_root / f"s37-runtime-artifact-{suffix}.pt"
    after_path = checkpoint_root / f"s37-runtime-artifact-after-{suffix}.pt"
    try:
        runtime.save(checkpoint_path)

        unknown_branch = SeedRuntime.load(checkpoint_path)
        before_unknown = _checkpoint_digest(unknown_branch.model.architecture.native_checkpoint())
        try:
            unknown_branch.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={"candidate:foreign:runtime": artifact},
                replays_by_candidate={},
            )
        except ValueError as exc:
            unknown_key_rejected = "outside the selected batch" in str(exc)
        else:
            unknown_key_rejected = False
        unknown_key_atomic = (
            _checkpoint_digest(unknown_branch.model.architecture.native_checkpoint())
            == before_unknown
        )

        stale_branch = SeedRuntime.load(checkpoint_path)
        _execute_observation(
            stale_branch,
            ordinal=99,
            region_id="workbench.code",
            task_slice_id="runtime-stale-parent",
            partition="train",
            path="README.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        )
        stale_before_topology = tuple(
            (region.region_id, region.unit_ids)
            for region in stale_branch.model.architecture.neuron_regions
        )
        stale_before_budget = (
            stale_branch.model.architecture.cognitive_snapshot().development.structural_budget
        )
        stale = stale_branch.continue_structural_candidate_from_validation_artifact(
            artifact,
            holdout_inputs=replay["holdout_inputs"],
            expected_activities=replay["expected_activities"],
        )
        stale_parent_rejected = (
            stale["status"] == "failed_closed"
            and "parent checkpoint" in stale["reason"]
            and tuple(
                (region.region_id, region.unit_ids)
                for region in stale_branch.model.architecture.neuron_regions
            )
            == stale_before_topology
            and stale_branch.model.architecture.cognitive_snapshot().development.structural_budget
            == stale_before_budget
        )

        tamper_branch = SeedRuntime.load(checkpoint_path)
        malformed = copy.deepcopy(artifact.to_payload())
        malformed["measurement_digest"] = "0" * 64
        tampered = tamper_branch.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: malformed},
            replays_by_candidate={candidate_id: replay},
        )
        tamper_batch = next(
            item
            for item in tamper_branch.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        )
        tamper_isolated = (
            tampered["results"][candidate_id]["status"] == "failed_closed"
            and tamper_batch.state_by_candidate[candidate_id] == "failed_closed"
            and tuple(
                (region.region_id, region.unit_ids)
                for region in tamper_branch.model.architecture.neuron_regions
            )
            == tuple(
                (region.region_id, region.unit_ids)
                for region in unknown_branch.model.architecture.neuron_regions
            )
        )

        concurrent_branch = SeedRuntime.load(checkpoint_path)
        before_concurrent_topology = tuple(
            (region.region_id, region.unit_ids)
            for region in concurrent_branch.model.architecture.neuron_regions
        )
        before_concurrent_budget = (
            concurrent_branch.model.architecture.cognitive_snapshot().development.structural_budget
        )

        def submit(_: int) -> str:
            result = concurrent_branch.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={candidate_id: artifact},
                replays_by_candidate={candidate_id: replay},
            )
            return result["results"][candidate_id]["status"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            statuses = tuple(executor.map(submit, (1, 2)))
        concurrent_topology = tuple(
            (region.region_id, region.unit_ids)
            for region in concurrent_branch.model.architecture.neuron_regions
        )
        concurrent_budget = (
            concurrent_branch.model.architecture.cognitive_snapshot().development.structural_budget
        )
        concurrent_idempotent = (
            sorted(statuses) == ["admitted", "already_applied"]
            and concurrent_topology != before_concurrent_topology
            and concurrent_budget == before_concurrent_budget - artifact.resource_cost
        )
        concurrent_branch.save(after_path)
        resumed = SeedRuntime.load(after_path)
        provenance_persisted = any(
            item.artifact_digest == artifact.artifact_digest
            and item.measurement_digest == measurements.measurement_digest
            for item in resumed.model.architecture.structural_validation_artifacts
        )
        metrics = {
            "cross_batch_input_rejected_atomically": unknown_key_rejected and unknown_key_atomic,
            "stale_parent_fails_closed_without_structure_change": stale_parent_rejected,
            "tampered_artifact_isolated_to_candidate": tamper_isolated,
            "concurrent_submit_has_one_real_admission": concurrent_idempotent,
            "concurrent_artifact_provenance_survives_restart": provenance_persisted,
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_id": candidate_id,
            "concurrent_statuses": list(statuses),
            "artifact_digest": artifact.artifact_digest,
            "measurement_digest": measurements.measurement_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "SeedRuntime must fail closed on stale/tampered/cross-batch artifact inputs "
                    "and serialize concurrent valid submissions so that only one admission occurs "
                    "while provenance remains checkpointable"
                ),
            },
            "boundary": (
                "This canary covers SeedRuntime/native CPU artifact failure isolation and concurrency. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        after_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s37_runtime_artifact_failure_concurrency_20260831.json",
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
