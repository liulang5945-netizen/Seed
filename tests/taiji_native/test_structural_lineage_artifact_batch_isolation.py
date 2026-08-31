from __future__ import annotations

import os
from pathlib import Path

import torch

from scripts.training.eval_taiji_structural_lineage_restart_continuation import (
    _build_migrated_runtime,
    _continuation_requests,
    _record_continuation_evidence,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from taiji import TSKV8Adapter
from taiji.adapter import _checkpoint_digest


def _save_native_checkpoint(model: TSKV8Adapter, path: Path) -> None:
    torch.save(model.native_checkpoint(), path)


def _load_native_checkpoint(path: Path) -> TSKV8Adapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TSKV8Adapter.from_native_checkpoint(checkpoint)


def test_artifact_batch_rejects_unknown_keys_and_isolates_partial_failure() -> None:
    runtime = _build_migrated_runtime()
    continuation_evidence = _record_continuation_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids
    unknown_candidate = "candidate:foreign:unselected"

    before_unknown = _checkpoint_digest(runtime.model.architecture.native_checkpoint())
    before_unknown_batch = batch.to_payload()
    for artifacts, replays in (
        ({unknown_candidate: {}}, {}),
        ({}, {unknown_candidate: {}}),
    ):
        try:
            runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate=artifacts,
                replays_by_candidate=replays,
            )
        except ValueError as exc:
            assert "outside the selected batch" in str(exc)
        else:
            raise AssertionError("unknown artifact/replay key unexpectedly accepted")
        assert _checkpoint_digest(runtime.model.architecture.native_checkpoint()) == before_unknown
        assert next(
            item.to_payload()
            for item in runtime.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        ) == before_unknown_batch

    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_failure_path = checkpoint_root / f"s35-before-failure-{suffix}.pt"
    after_failure_path = checkpoint_root / f"s35-after-failure-{suffix}.pt"
    after_success_path = checkpoint_root / f"s35-after-success-{suffix}.pt"
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_candidate,
            continuation_evidence,
        )
        _save_native_checkpoint(runtime.model.architecture, before_failure_path)
        restored = _load_native_checkpoint(before_failure_path)
        malformed_payload = dict(first_artifact.to_payload())
        malformed_payload["artifact_digest"] = "0" * 64
        failed = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: malformed_payload},
            replays_by_candidate={first_candidate: first_replay},
        )
        assert failed["results"][first_candidate]["status"] == "failed_closed"
        failed_batch = next(
            item for item in restored.structural_candidate_batches if item.batch_id == batch.batch_id
        )
        assert failed_batch.state_by_candidate[first_candidate] == "failed_closed"
        assert failed_batch.state_by_candidate[second_candidate] == "reserved"
        _save_native_checkpoint(restored, after_failure_path)

        resumed = _load_native_checkpoint(after_failure_path)
        second_artifact, second_replay, second_measurements = _build_artifact(
            resumed,
            second_candidate,
            continuation_evidence,
        )
        succeeded = resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        assert succeeded["results"][second_candidate]["status"] == "admitted"
        _save_native_checkpoint(resumed, after_success_path)
        final = _load_native_checkpoint(after_success_path)
        final_batch = next(
            item for item in final.structural_candidate_batches if item.batch_id == batch.batch_id
        )
        assert final_batch.state_by_candidate[first_candidate] == "failed_closed"
        assert final_batch.state_by_candidate[second_candidate] == "admitted"
        assert second_measurements.measurement_digest == second_artifact.measurement_digest

        before_repeat_topology = tuple(
            (region.region_id, region.unit_ids) for region in final.neuron_regions
        )
        before_repeat_budget = final.cognitive_snapshot().development.structural_budget
        repeated = final.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        assert repeated["results"][second_candidate]["status"] == "already_applied"
        assert tuple(
            (region.region_id, region.unit_ids) for region in final.neuron_regions
        ) == before_repeat_topology
        assert final.cognitive_snapshot().development.structural_budget == before_repeat_budget
        repeated_batch = next(
            item for item in final.structural_candidate_batches if item.batch_id == batch.batch_id
        )
        assert repeated_batch.state_by_candidate[first_candidate] == "failed_closed"
        assert repeated_batch.state_by_candidate[second_candidate] == "admitted"
    finally:
        before_failure_path.unlink(missing_ok=True)
        after_failure_path.unlink(missing_ok=True)
        after_success_path.unlink(missing_ok=True)
