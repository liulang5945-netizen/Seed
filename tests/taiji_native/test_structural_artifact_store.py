from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import _record_round
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from taiji import StructuralValidationArtifactStore
from taiji.adapter import _checkpoint_digest


def test_external_artifact_store_is_immutable_and_runtime_consumable() -> None:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="store-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )

    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s41-store-{os.getpid()}"
    )
    concurrent_root = store_root.parent / f"s41-concurrent-{os.getpid()}"
    checkpoint_path = store_root.parent / f"s41-runtime-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    try:
        stored = store.put(artifact)
        artifact_path = store.path_for(stored.artifact_digest)
        original_bytes = artifact_path.read_bytes()
        assert artifact_path.name == f"{artifact.artifact_digest}.json"
        assert store.contains(artifact.artifact_digest)
        assert store.load(artifact.artifact_digest) == artifact
        assert store.put(artifact.to_payload()) == artifact

        concurrent_store = StructuralValidationArtifactStore(concurrent_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_digests = tuple(
                item.artifact_digest
                for item in executor.map(lambda _: concurrent_store.put(artifact), range(4))
            )
        assert concurrent_digests == (artifact.artifact_digest,) * 4
        assert concurrent_store.load(artifact.artifact_digest) == artifact
        assert artifact_path.read_bytes() == original_bytes

        runtime.save(checkpoint_path)
        handed_off = StructuralValidationArtifactStore(store_root).load(artifact.artifact_digest)
        restored = SeedRuntime.load(checkpoint_path)
        before_budget = restored.model.architecture.cognitive_snapshot().development.structural_budget
        result = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: handed_off.to_payload()},
            replays_by_candidate={candidate_id: replay},
        )
        repeat = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: handed_off.to_payload()},
            replays_by_candidate={candidate_id: replay},
        )
        assert result["results"][candidate_id]["status"] == "admitted"
        assert repeat["results"][candidate_id]["status"] == "already_applied"
        assert (
            restored.model.architecture.cognitive_snapshot().development.structural_budget
            == before_budget - artifact.resource_cost
        )
        assert measurements.measurement_digest == handed_off.measurement_digest

        before_tamper = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        artifact_path.write_bytes(b"{}")
        try:
            store.load(artifact.artifact_digest)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError("tampered external artifact unexpectedly loaded")
        assert _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before_tamper
        try:
            store.put(artifact)
        except ValueError as exc:
            assert "content collision" in str(exc)
        else:
            raise AssertionError("tampered external artifact unexpectedly overwrote bytes")
        artifact_path.write_bytes(original_bytes)
    finally:
        checkpoint_path.unlink(missing_ok=True)
        for root in (store_root, concurrent_root):
            if root.exists():
                for child in root.iterdir():
                    child.unlink(missing_ok=True)
                root.rmdir()
