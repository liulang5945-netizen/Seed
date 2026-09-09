from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_feature_target_audit_20260910.json"
)


def test_b3_k_feature_target_audit_covers_real_fit_tensors() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "feature-target-collision-audit"
    assert report["cell_count"] == 6
    assert report["technical_gate_passed"] is True
    assert report["fit_tensor_signatures_distinct"]["k1.semantic.input_tensor_digest"] is True
    assert report["fit_tensor_signatures_distinct"]["k2.transition.input_tensor_digest"] is True
    assert report["fit_tensor_signatures_distinct"]["k1.semantic.target_tensor_digest"] is False
    assert report["fit_tensor_signatures_distinct"]["k2.transition.target_tensor_digest"] is False
    assert report["all_fit_tensor_signatures_distinct"] is False
    assert report["can_promote"] is False


def test_b3_k_feature_target_audit_links_distinct_tensors_to_worker_collisions() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["fit_tensor_signatures_distinct"]["k1.semantic.input_tensor_digest"]
    assert report["fit_tensor_signatures_distinct"]["k2.transition.input_tensor_digest"]
    assert report["fit_tensor_collision_groups"]["k1.semantic.target_tensor_digest"]
    assert report["fit_tensor_collision_groups"]["k2.transition.target_tensor_digest"]
    assert report["worker_parameter_delta_collision_groups"]["k1.semantic"]
    assert report["worker_parameter_delta_collision_groups"]["k2.transition"]
    assert all(
        cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        and cell["report"]["checks"]["k3_owner_unchanged"]
        for cell in report["cells"]
    )
