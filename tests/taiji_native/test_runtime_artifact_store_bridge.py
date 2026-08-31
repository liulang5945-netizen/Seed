from __future__ import annotations

import os
from pathlib import Path

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import _record_round
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from taiji import ArtifactConsumptionPolicy, StructuralValidationArtifactStore
from taiji.adapter import _checkpoint_digest


def test_runtime_artifact_store_bridge_validates_before_native_mutation() -> None:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="bridge-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, _ = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )
    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s42-bridge-{os.getpid()}"
    )
    checkpoint_path = store_root.parent / f"s42-bridge-runtime-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s42-bridge-test"
    )
    try:
        store.put(artifact)
        runtime.save(checkpoint_path)

        unknown = SeedRuntime.load(checkpoint_path)
        before_unknown = _checkpoint_digest(unknown.model.architecture.native_checkpoint())
        try:
            unknown.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={"candidate:foreign": artifact.artifact_digest},
                replays_by_candidate={},
                artifact_consumption_policy=legacy_policy,
            )
        except ValueError as exc:
            assert "outside the selected batch" in str(exc)
        else:
            raise AssertionError("bridge accepted an unknown candidate key")
        assert _checkpoint_digest(unknown.model.architecture.native_checkpoint()) == before_unknown

        invalid_digest = SeedRuntime.load(checkpoint_path)
        before_invalid = _checkpoint_digest(invalid_digest.model.architecture.native_checkpoint())
        try:
            invalid_digest.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={candidate_id: "0" * 64},
                replays_by_candidate={candidate_id: replay},
                artifact_consumption_policy=legacy_policy,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("bridge accepted a missing artifact digest")
        assert _checkpoint_digest(invalid_digest.model.architecture.native_checkpoint()) == before_invalid

        restored = SeedRuntime.load(checkpoint_path)
        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=StructuralValidationArtifactStore(store_root),
            artifact_digests_by_candidate={candidate_id: artifact.artifact_digest},
            replays_by_candidate={candidate_id: replay},
            artifact_consumption_policy=legacy_policy,
        )
        repeated = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={candidate_id: artifact.artifact_digest},
            replays_by_candidate={candidate_id: replay},
            artifact_consumption_policy=legacy_policy,
        )
        assert result["results"][candidate_id]["status"] == "admitted"
        assert repeated["results"][candidate_id]["status"] == "already_applied"
    finally:
        checkpoint_path.unlink(missing_ok=True)
        if store_root.exists():
            for child in store_root.iterdir():
                child.unlink(missing_ok=True)
            store_root.rmdir()
