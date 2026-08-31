from __future__ import annotations

from scripts.training.eval_taiji_runtime_structural_artifact_repeated_retention import evaluate


def test_runtime_artifact_repeated_retention_stays_bounded() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
