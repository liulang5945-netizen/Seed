from __future__ import annotations

from scripts.training.eval_taiji_concept_branch_attribution import evaluate
from taiji import TaijiConfig


def test_concept_branch_attribution_gate() -> None:
    config = TaijiConfig(concept_branch_owner_weights=(0.50, 0.35, 0.15))
    assert TaijiConfig.from_dict(config.to_dict()) == config
    report = evaluate()

    assert report["gate"]["passed"] is True
    metrics = report["metrics"]
    assert metrics["low_confidence_owner"] is None
    assert metrics["interference_owner"] is None
    assert metrics["owner_after_lesion"] is None
    assert metrics["buffer_owner_count_before_checkpoint"] == 1
    assert metrics["owner_trace_count_after_birth"] == (
        metrics["owner_trace_count_before_birth"] + 1
    )
    assert metrics["other_trace_count_after_birth"] == 1
    assert metrics["checkpoint_recovery"] is True
