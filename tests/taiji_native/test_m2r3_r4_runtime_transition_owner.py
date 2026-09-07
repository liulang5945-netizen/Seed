"""M2.R3.R4 gate for runtime-owned semantic transition state."""

from scripts.training.eval_taiji_m2r3_r4_runtime_transition_owner import evaluate


def test_m2r3_r4_runtime_transition_owner_gate() -> None:
    report = evaluate((11, 29, 47))

    assert report["status"] == "passed", report
    assert report["gate"]["passed"] is True
    assert all(run["gate"]["passed"] for run in report["runs"])
