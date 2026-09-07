"""M3 first project-disjoint Workbench read-only Gate."""

from scripts.training.eval_taiji_m3_workbench_readonly import evaluate


def test_m3_workbench_readonly_gate() -> None:
    report = evaluate()

    assert report["status"] == "passed", report
    assert report["gate"]["passed"] is True
    assert report["gate"]["can_promote"] is False
    assert all(report["metrics"].values())
