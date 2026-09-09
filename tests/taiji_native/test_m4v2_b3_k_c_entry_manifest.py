from __future__ import annotations

import json
from pathlib import Path

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "plans"
    / "manifests"
    / "taiji_m4v2_b3_k_c_entry_evaluation_v1.json"
)


def test_b3_k_c_entry_manifest_freezes_target_tensor_semantics() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["format"] == "taiji-m4v2-b3-k-c-entry-evaluation-manifest-v1"
    assert manifest["status"] == "frozen-pre-formal"
    course = manifest["course_contract"]
    assert course["target_digest_semantics"].startswith("combined K1")
    assert course["id_bearing_experience_digest_is_valid_target_evidence"] is False
    assert [item["course_seed"] for item in course["variants"]] == [0, 1, 2]
    assert [item["target_multiset"] for item in course["variants"]] == [
        {"A": 1, "B": 1, "C": 1},
        {"A": 2, "B": 1},
        {"A": 1, "B": 2},
    ]


def test_b3_k_c_entry_manifest_blocks_formal_until_sealed_and_control_ready() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    sealed = manifest["sealed_test_split"]
    assert sealed["status"] == "must_materialize-before-formal"
    assert sealed["artifact_path"] is None
    assert sealed["artifact_digest"] is None
    assert sealed["formal_start_blocked_until_materialized"] is True

    promotion = manifest["promotion"]
    assert promotion["can_start_formal"] is False
    assert promotion["can_promote"] is False
    assert manifest["resource_contract"]["planned_cell_count"] == 27
    assert manifest["gates"]["formal_candidate"]["fixed_large_comparison_required"] is True
