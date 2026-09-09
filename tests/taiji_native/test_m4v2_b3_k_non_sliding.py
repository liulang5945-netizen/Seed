from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_non_sliding_20260910.json"
)


def test_b3_k_non_sliding_uses_the_preregistered_course_combinations() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "non-sliding-course-combination-diagnostic"
    assert report["train_episode_count"] == 3
    assert report["train_variant_strategy"] == "non_sliding"
    assert report["train_episode_indexes"] == [[0, 2, 4], [1, 3, 5], [0, 3, 5]]
    assert report["same_parent"] is True
    assert report["distinct_course_variants"] is True
    assert report["all_course_seeds_improved"] is True
    assert report["candidate_updates_distinct"] is False
    assert report["technical_gate_passed"] is True
    assert report["performance_gate_passed"] is False
    assert report["stability_gate_passed"] is False
    assert report["can_promote"] is False


def test_b3_k_non_sliding_preserves_checkpoint_and_rollback_boundaries() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["no_update_control_passed"] is True
    assert all(
        cell["report"]["train_variant_strategy"] == "non_sliding"
        and cell["report"]["training_update_steps"] == 6
        and cell["report"]["checks"]["train_holdout_disjoint"]
        and cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        and cell["report"]["checks"]["k3_owner_unchanged"]
        for cell in report["cells"]
    )
