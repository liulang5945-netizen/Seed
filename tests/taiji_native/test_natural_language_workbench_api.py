from scripts.training.eval_taiji_natural_language_workbench_api import evaluate


def test_natural_language_workbench_api_gate_passes() -> None:
    report = evaluate()
    assert report["gate"]["passed"] is True
