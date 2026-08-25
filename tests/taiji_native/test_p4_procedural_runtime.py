from __future__ import annotations

from scripts.training.eval_taiji_p4_procedural_runtime import evaluate


def test_p4_procedural_runtime_ownership_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p4-procedural-runtime-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["procedural_runtime_accuracy"] == 1.0
    assert report["metrics"]["episode_id_lesion_accuracy"] == 1.0
    assert report["metrics"]["checkpoint_continuation_accuracy"] == 1.0
    assert report["metrics"]["procedural_runtime_accuracy"] > report["metrics"]["runtime_lesion_accuracy"]
