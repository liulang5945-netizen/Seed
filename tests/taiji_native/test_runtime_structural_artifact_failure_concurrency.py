from __future__ import annotations

import concurrent.futures
import copy
import os
from pathlib import Path

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji.adapter import _checkpoint_digest


def test_runtime_artifact_failures_are_isolated_and_concurrent_submit_is_idempotent() -> None:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, _ = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        executions,
    )
    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
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
            assert "outside the selected batch" in str(exc)
        else:
            raise AssertionError("runtime accepted a cross-batch artifact key")
        assert _checkpoint_digest(unknown_branch.model.architecture.native_checkpoint()) == before_unknown

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
            (region.region_id, region.unit_ids) for region in stale_branch.model.architecture.neuron_regions
        )
        stale_before_budget = stale_branch.model.architecture.cognitive_snapshot().development.structural_budget
        stale = stale_branch.continue_structural_candidate_from_validation_artifact(
            artifact,
            holdout_inputs=replay["holdout_inputs"],
            expected_activities=replay["expected_activities"],
        )
        assert stale["status"] == "failed_closed"
        assert "parent checkpoint" in stale["reason"]
        assert tuple(
            (region.region_id, region.unit_ids) for region in stale_branch.model.architecture.neuron_regions
        ) == stale_before_topology
        assert stale_branch.model.architecture.cognitive_snapshot().development.structural_budget == stale_before_budget

        tamper_branch = SeedRuntime.load(checkpoint_path)
        malformed = copy.deepcopy(artifact.to_payload())
        malformed["measurement_digest"] = "0" * 64
        tampered = tamper_branch.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: malformed},
            replays_by_candidate={candidate_id: replay},
        )
        assert tampered["results"][candidate_id]["status"] == "failed_closed"
        tamper_batch = next(
            item for item in tamper_branch.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        )
        assert tamper_batch.state_by_candidate[candidate_id] == "failed_closed"

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
        assert sorted(statuses) == ["admitted", "already_applied"]
        assert tuple(
            (region.region_id, region.unit_ids)
            for region in concurrent_branch.model.architecture.neuron_regions
        ) != before_concurrent_topology
        assert (
            concurrent_branch.model.architecture.cognitive_snapshot().development.structural_budget
            == before_concurrent_budget - artifact.resource_cost
        )
        concurrent_branch.save(after_path)
        resumed = SeedRuntime.load(after_path)
        assert any(
            item.artifact_digest == artifact.artifact_digest
            for item in resumed.model.architecture.structural_validation_artifacts
        )
    finally:
        checkpoint_path.unlink(missing_ok=True)
        after_path.unlink(missing_ok=True)
