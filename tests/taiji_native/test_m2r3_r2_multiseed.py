from __future__ import annotations

from scripts.training.eval_taiji_m2r3_r2_multiseed import evaluate


def test_m2r3_r2_multiseed_structured_semantic_gate_passes() -> None:
    report = evaluate((11, 29, 47))

    assert report["format"] == "taiji-m2r3-r2-multiseed-structured-semantics-v1"
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["all_seeds_passed"] is True
    assert report["aggregate"]["mean_test_fact_f1"] == 1.0
    assert report["aggregate"]["mean_test_content_accuracy"] == 1.0
    assert all(run["gate"]["passed"] is True for run in report["runs"])
