from __future__ import annotations

from scripts.training.eval_taiji_p3_schema_registry import build_manifest, evaluate


def test_schema_registry_lifecycle_gate_is_fail_closed_and_checkpointable() -> None:
    report = evaluate(seeds=(11,))

    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["alias_stable_min"] == 1.0
    assert report["aggregate"]["old_weights_preserved_min"] == 1.0
    assert report["aggregate"]["mixed_schema_min"] == 1.0
    assert report["aggregate"]["conflict_stable_min"] == 1.0
    assert report["aggregate"]["budget_blocked_min"] == 1.0
    assert report["aggregate"]["prune_tombstone_min"] == 1.0
    assert report["aggregate"]["rollback_restored_min"] == 1.0
    assert report["aggregate"]["checkpoint_min"] == 1.0
    assert report["aggregate"]["checkpoint_rollback_min"] == 1.0


def test_schema_registry_manifest_declares_lifecycle_controls() -> None:
    manifest = build_manifest()

    assert manifest["format"] == "taiji-p3-schema-registry-manifest-v1"
    assert "canonical object alias merge" in manifest["controls"]
    assert "contradictory outcome feedback fail-closed" in manifest["controls"]
    assert "network schema rollback" in manifest["controls"]
