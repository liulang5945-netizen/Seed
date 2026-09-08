from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r4_shadow_formal import run_formal


def test_r4_formal_matrix_smoke_keeps_diagnostic_only_boundary() -> None:
    report = run_formal(model_seeds=(71,), course_seeds=(101,))

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert report["matrix"]["cell_count"] == 1
    assert all(report["technical_gates"].values())
    assert report["cells"][0]["candidate_only_smoke"]["parent_substrate_unchanged"] is True
