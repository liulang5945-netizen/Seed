from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2] / "reports" / "taiji_m4v2_b3_k_bounded_batch_20260910.json"
)


def test_b3_k_bounded_batch_has_three_two_example_course_cells() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "bounded-multi-example-diagnostic"
    assert report["bounded_batch_count"] == 2
    assert report["train_episode_count"] == 2
    assert report["cell_count"] == 3
    assert report["same_parent"] is True
    assert report["distinct_course_variants"] is True
    assert report["technical_gate_passed"] is True
    assert report["all_course_seeds_improved"] is False
    assert report["performance_gate_passed"] is False
    assert report["stability_gate_passed"] is False
    assert report["can_promote"] is False
    assert report["can_start_r6_formal"] is False


def test_b3_k_bounded_batch_preserves_learning_boundaries() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["no_update_control_passed"] is True
    assert report["combined_loss_delta_by_course_seed"]["0"] < 0.0
    assert report["combined_loss_delta_by_course_seed"]["1"] > 0.0
    assert report["combined_loss_delta_by_course_seed"]["2"] < 0.0
    assert all(
        cell["report"]["train_episode_count"] == 2
        and len(cell["report"]["train_experience_digests"]) == 2
        and cell["report"]["training_update_steps"] == 4
        and cell["report"]["checks"]["train_holdout_disjoint"]
        and cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        and cell["report"]["checks"]["k3_owner_unchanged"]
        and cell["report"]["parameter_delta_norm"]["k1.semantic"] > 0.0
        and cell["report"]["parameter_delta_norm"]["k2.transition"] > 0.0
        for cell in report["cells"]
    )
