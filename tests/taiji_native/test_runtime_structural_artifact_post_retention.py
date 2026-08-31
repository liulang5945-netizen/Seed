from __future__ import annotations

from scripts.training.eval_taiji_runtime_structural_artifact_post_retention import evaluate


def test_runtime_artifact_continues_after_retention_and_restart() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
