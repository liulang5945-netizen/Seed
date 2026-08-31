"""P5-2 semantic Workbench grounding modularization Gate."""

from scripts.training.eval_taiji_natural_language_workbench_grounding_modularization import (
    evaluate,
)


def test_natural_language_workbench_grounding_modularization_gate_passes() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
