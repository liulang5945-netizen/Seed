from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_target_aware_model_seeds_20260910.json"
)


def test_b3_k_target_aware_model_seed_stability_uses_real_parents() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "target-aware-model-seed-stability-diagnostic"
    assert report["model_seeds"] == [17, 23, 31]
    assert report["model_count"] == 3
    assert report["cell_count"] == 9
    assert report["independent_model_parents"] is True
    assert report["technical_gate_passed"] is True
    assert report["can_promote"] is False


def test_b3_k_target_aware_model_seed_stability_has_no_negative_boundary_breaks() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["model_technical_gates_passed"] is True
    assert report["model_candidate_updates_distinct"] is True
    assert report["performance_gate_passed"] is True
    assert report["stability_gate_passed"] is True
    assert report["worst_combined_loss_delta"] < 0.0
    assert all(
        model["report"]["target_course_variants_distinct"]
        and model["report"]["performance_gate_passed"]
        and model["report"]["stability_gate_passed"]
        and all(
            cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
            and cell["report"]["checks"]["adapter_rollback_restored"]
            and cell["report"]["checks"]["k3_owner_unchanged"]
            for cell in model["report"]["cells"]
        )
        for model in report["model_reports"]
    )
