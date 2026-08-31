from __future__ import annotations

from scripts.training.eval_taiji_runtime_retention_store_separation import evaluate


def test_runtime_retention_does_not_delete_or_resurrect_external_artifacts() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
