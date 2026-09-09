from __future__ import annotations

import copy

import pytest

from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (
    run_preflight,
)
from taiji import (
    K_WORKER_IDS,
    KContinualAdapter,
    KWorkerManifest,
    KWorkerManifestBundle,
    content_digest,
)


def _bundle() -> KWorkerManifestBundle:
    parent_digest = "1" * 64
    workers = []
    for index, worker_id in enumerate(K_WORKER_IDS):
        workers.append(
            KWorkerManifest.create(
                worker_id=worker_id,
                checkpoint_format=f"taiji-{worker_id}-checkpoint-v1",
                checkpoint_version=1,
                worker_checkpoint_digest=content_digest(
                    {"worker_id": worker_id, "checkpoint": index}
                ),
                owner_digests=(
                    (
                        f"{worker_id}.owner",
                        content_digest({"owner": worker_id}),
                    ),
                ),
                source_digest=content_digest({"source": worker_id}),
                input_contract_digest=content_digest({"input": worker_id}),
                output_contract_digest=content_digest({"output": worker_id}),
                parent_checkpoint_digest=parent_digest,
                candidate_namespace="taiji:k:candidate",
                training_steps=index,
                optimizer_state_present=False,
            )
        )
    return KWorkerManifestBundle.create(
        parent_checkpoint_digest=parent_digest,
        source_manifest_digest=content_digest({"source_manifest": "r6"}),
        resource_manifest_digest=content_digest({"resource_manifest": "r6"}),
        candidate_namespace="taiji:k:candidate",
        workers=workers,
    )


def test_worker_bundle_is_content_addressed_and_roundtrips() -> None:
    bundle = _bundle()

    assert tuple(item.worker_id for item in bundle.workers) == K_WORKER_IDS
    assert KWorkerManifestBundle.from_payload(bundle.to_payload()) == bundle

    tampered = copy.deepcopy(bundle.to_payload())
    tampered["workers"][0]["training_steps"] = 99
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        KWorkerManifestBundle.from_payload(tampered)


def test_worker_bundle_requires_all_workers_on_one_parent() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="exactly K1/K2/K3"):
        KWorkerManifestBundle.create(
            parent_checkpoint_digest=bundle.parent_checkpoint_digest,
            source_manifest_digest=bundle.source_manifest_digest,
            resource_manifest_digest=bundle.resource_manifest_digest,
            candidate_namespace=bundle.candidate_namespace,
            workers=bundle.workers[:2],
        )

    changed_parent = copy.deepcopy(bundle.workers[0].to_payload())
    changed_parent["parent_checkpoint_digest"] = "2" * 64
    changed_parent.pop("manifest_digest")
    changed_parent["manifest_digest"] = content_digest(changed_parent)
    changed_workers = (
        KWorkerManifest.from_payload(changed_parent),
        *bundle.workers[1:],
    )
    with pytest.raises(ValueError, match="cross-parent"):
        KWorkerManifestBundle.create(
            parent_checkpoint_digest=bundle.parent_checkpoint_digest,
            source_manifest_digest=bundle.source_manifest_digest,
            resource_manifest_digest=bundle.resource_manifest_digest,
            candidate_namespace=bundle.candidate_namespace,
            workers=changed_workers,
        )


def test_adapter_attaches_worker_bundle_and_preserves_it_on_restore() -> None:
    bundle = _bundle()
    adapter = KContinualAdapter(
        parent_checkpoint_digest=bundle.parent_checkpoint_digest,
        owner_graph_digest=bundle.owner_graph_digest,
        source_manifest_digest=bundle.source_manifest_digest,
        resource_manifest_digest=bundle.resource_manifest_digest,
        dependency_scope_id="r6-worker-scope",
    )

    adapter.bind_worker_bundle(bundle)
    checkpoint = adapter.checkpoint()
    restored = KContinualAdapter.from_checkpoint(checkpoint)

    assert adapter.worker_bundle == bundle
    assert restored.worker_bundle == bundle
    assert restored.checkpoint() == checkpoint


def test_attachment_preflight_stops_when_real_worker_artifacts_are_missing() -> None:
    report = run_preflight(
        semantic_checkpoint="checkpoints/missing-k1.pt",
        transition_checkpoint="checkpoints/missing-k2.pt",
        projection_checkpoint="checkpoints/missing-k3.pt",
    )

    assert report["status"] == "artifact_missing"
    assert report["artifact_missing"] == list(K_WORKER_IDS)
    assert report["training_performed"] is False
    assert report["can_start_r6_formal"] is False
    assert report["can_promote"] is False
