"""P2-9 Taiji-owned semantic-slot grounding Gate."""

from scripts.training.eval_taiji_semantic_grounding import evaluate


def test_semantic_grounding_gate_passes() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p2-9-semantic-grounding-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
