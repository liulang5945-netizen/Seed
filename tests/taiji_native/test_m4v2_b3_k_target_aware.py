from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_target_aware_v2_20260910.json"
)


def test_b3_k_target_aware_gate_uses_distinct_target_compositions() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["report_format"] == "taiji-m4v2-b3-k-target-aware-v2"
    assert report["version"] == 2
    assert report["run_kind"] == "target-aware-course-diversity-diagnostic"
    assert report["train_variant_strategy"] == "target_aware"
    assert report["train_episode_count"] == 3
    assert report["require_target_diversity"] is True
    assert report["target_course_variants_distinct"] is True
    assert report["technical_gate_passed"] is True
    assert report["candidate_updates_distinct"] is True
    assert report["all_course_seeds_improved"] is True
    assert report["performance_gate_passed"] is True
    assert report["stability_gate_passed"] is True
    assert report["can_promote"] is False


def test_b3_k_target_aware_keeps_learning_boundaries_explicit() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert len(report["target_course_digests"]) == 3
    assert all(
        len(cell["report"]["train_target_digests"]) == 3
        and cell["report"]["target_digest_semantics"] == "combined K1/K2 target tensor digests"
        and len(cell["report"]["train_target_multiplicity"]) in {2, 3}
        and sum(cell["report"]["train_target_multiplicity"].values()) == 3
        and cell["report"]["checks"]["train_holdout_disjoint"]
        and cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        and cell["report"]["checks"]["k3_owner_unchanged"]
        for cell in report["cells"]
    )
