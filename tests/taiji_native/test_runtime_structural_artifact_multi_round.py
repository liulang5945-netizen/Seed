from __future__ import annotations

from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import evaluate


def test_runtime_artifact_multi_round_lifecycle_and_retention() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
