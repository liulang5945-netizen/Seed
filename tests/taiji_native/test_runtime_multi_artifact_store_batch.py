from __future__ import annotations

from scripts.training.eval_taiji_runtime_artifact_store_batch import evaluate


def test_runtime_multi_artifact_store_batch_preserves_parent_order() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
