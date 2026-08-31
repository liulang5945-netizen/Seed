from __future__ import annotations

import json
import os
from pathlib import Path

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji.adapter import _checkpoint_digest


def test_seed_runtime_restarts_and_consumes_measured_artifact_batch() -> None:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    first_artifact, first_replay, first_measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        executions,
    )

    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    checkpoint_path = checkpoint_root / f"s36-runtime-artifact-{suffix}.pt"
    artifact_path = checkpoint_root / f"s36-runtime-artifact-{suffix}.json"
    try:
        artifact_path.write_text(
            json.dumps(first_artifact.to_payload(), sort_keys=True),
            encoding="utf-8",
        )
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        parent_matches = (
            _checkpoint_digest(restored.model.architecture.native_checkpoint())
            == first_artifact.parent_checkpoint_digest
        )
        before_unknown = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={"candidate:foreign:runtime": artifact_payload},
                replays_by_candidate={},
            )
        except ValueError as exc:
            unknown_key_rejected = "outside the selected batch" in str(exc)
        else:
            unknown_key_rejected = False
        unknown_key_atomic = (
            _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before_unknown
        )
        result = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: artifact_payload},
            replays_by_candidate={candidate_id: first_replay},
        )
        assert result["results"][candidate_id]["status"] == "admitted"
        restored.save(checkpoint_path)
        resumed = SeedRuntime.load(checkpoint_path)
        repeated = resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: artifact_payload},
            replays_by_candidate={candidate_id: first_replay},
        )
        assert repeated["results"][candidate_id]["status"] == "already_applied"
        persisted = next(
            item
            for item in resumed.model.architecture.structural_validation_artifacts
            if item.artifact_digest == first_artifact.artifact_digest
        )
        assert persisted.measurement_digest == first_measurements.measurement_digest
        assert parent_matches
        assert unknown_key_rejected and unknown_key_atomic
    finally:
        checkpoint_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)
