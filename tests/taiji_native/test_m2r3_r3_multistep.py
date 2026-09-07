"""M2.R3.R3 gate for persistent native semantic world transitions."""

from scripts.training.eval_taiji_m2r3_r3_multistep import evaluate


def test_m2r3_r3_multistep_transition_gate() -> None:
    report = evaluate((11, 29, 47))

    assert report["status"] == "passed", report
    assert report["gate"]["passed"] is True
    assert all(run["gate"]["passed"] for run in report["runs"])
