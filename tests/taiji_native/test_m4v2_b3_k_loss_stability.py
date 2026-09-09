from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_loss_stability_20260910.json"
)


def test_b3_k_loss_stability_uses_three_same_parent_cells() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "learning-stability-diagnostic"
    assert report["cell_count"] == 3
    assert report["course_seeds"] == [0, 1, 2]
    assert report["same_parent"] is True
    assert report["distinct_course_variants"] is True
    assert report["train_episode_indexes"] == [0, 1, 2]
    assert report["technical_gate_passed"] is True
    assert report["can_promote"] is False


def test_b3_k_loss_stability_exposes_course_sensitivity_without_formal_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["all_course_seeds_improved"] is False
    assert report["performance_gate_passed"] is False
    assert report["stability_gate_passed"] is False
    assert report["mean_combined_loss_delta"] < 0.0
    assert report["worst_combined_loss_delta"] > 0.0
    assert report["combined_loss_delta_by_course_seed"]["0"] < 0.0
    assert report["combined_loss_delta_by_course_seed"]["1"] > 0.0
    assert report["combined_loss_delta_by_course_seed"]["2"] > 0.0
    assert len(set(report["train_experience_digests"])) == 3
    assert all(
        cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        for cell in report["cells"]
    )
    assert all(cell["report"]["can_promote"] is False for cell in report["cells"])
