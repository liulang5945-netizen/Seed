from __future__ import annotations

import json
from pathlib import Path

SEALED_TEST = (
    Path(__file__).resolve().parents[2]
    / "plans"
    / "manifests"
    / "taiji_m4v2_b3_k_c_sealed_test_v1.json"
)


def test_c_entry_sealed_test_is_materialized_without_scores_or_targets() -> None:
    artifact = json.loads(SEALED_TEST.read_text(encoding="utf-8"))

    assert artifact["format"] == "taiji-m4v2-b3-k-c-sealed-test-v1"
    assert artifact["status"] == "materialized-unread"
    assert artifact["episode_count"] == 3
    assert artifact["step_count"] == 9
    assert artifact["record_disjoint_from_existing_fixture"] is True
    assert artifact["raw_source_content_embedded"] is False
    assert artifact["scores_or_targets_embedded"] is False
    assert artifact["anchor_payload"]["path"] == "missing_00.txt"
    assert len(artifact["episodes"]) == 3
    assert len(artifact["observation_payloads"]) == 9
    assert len(artifact["artifact_digest"]) == 64
    assert len({item["observation_digest"] for item in artifact["observation_payloads"]}) == 9


def test_c_entry_sealed_test_paths_are_not_validation_paths() -> None:
    artifact = json.loads(SEALED_TEST.read_text(encoding="utf-8"))
    paths = [path for episode in artifact["episodes"] for path in episode["paths"]]

    assert len(paths) == len(set(paths))
    assert all(path.startswith("sealed_") for path in paths)
    assert all(
        digest for episode in artifact["episodes"] for digest in episode["observation_digests"]
    )
