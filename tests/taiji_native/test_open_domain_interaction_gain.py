from __future__ import annotations

from scripts.training.eval_taiji_open_domain_interaction_gain import evaluate


def test_open_domain_interaction_gain_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-7-open-domain-interaction-gain-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
    assert (
        report["method_mean_scores"]["relation_transfer"]
        > report["method_mean_scores"]["weight_only"]
    )
