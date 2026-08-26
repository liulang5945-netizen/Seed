from __future__ import annotations

from scripts.training.eval_taiji_concept_transfer import evaluate


def test_concept_transfer_schema_scale_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-concept-transfer-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["unseen_task_transfer"]["rate"] == 1.0
    assert report["metrics"]["capacity_interference"]["interference_detected"] is True
    assert all(report["metrics"]["signal_lesions"].values())
    assert report["metrics"]["native_runtime_checkpoint_recovery"] is True
