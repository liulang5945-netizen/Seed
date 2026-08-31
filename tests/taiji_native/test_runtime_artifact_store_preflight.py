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


def test_runtime_artifact_store_preflights_all_candidates_before_mutation() -> None:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="preflight-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id, second_id = batch.selected_candidate_ids
    artifact, replay, _ = _build_artifact(runtime.model.architecture, first_id, evidence)
    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s43-preflight-{os.getpid()}"
    )
    checkpoint_path = store_root.parent / f"s43-preflight-runtime-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s43-preflight-test"
    )
    try:
        store.put(artifact)
        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        before = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={first_id: artifact.artifact_digest, second_id: "0" * 64},
                replays_by_candidate={first_id: replay},
                artifact_consumption_policy=legacy_policy,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("bridge partially consumed a multi-candidate artifact set")
        assert _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before
        unchanged_batch = next(
            item
            for item in restored.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        )
        assert all(state == "reserved" for _, state in unchanged_batch.candidate_states)

        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=StructuralValidationArtifactStore(store_root),
            artifact_digests_by_candidate={first_id: artifact.artifact_digest},
            replays_by_candidate={first_id: replay},
            artifact_consumption_policy=legacy_policy,
        )
        repeat = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: artifact.artifact_digest},
            replays_by_candidate={first_id: replay},
            artifact_consumption_policy=legacy_policy,
        )
        assert result["results"][first_id]["status"] == "admitted"
        assert repeat["results"][first_id]["status"] == "already_applied"
    finally:
        checkpoint_path.unlink(missing_ok=True)
        if store_root.exists():
            for child in store_root.iterdir():
                child.unlink(missing_ok=True)
            store_root.rmdir()
