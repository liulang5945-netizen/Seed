"""M2.R4 gate for joint native semantic training and retention."""

from scripts.training.eval_taiji_m2r4_joint_course import evaluate


def test_m2r4_joint_course_gate() -> None:
    report = evaluate((11, 29, 47))

    assert report["status"] == "passed", report
    assert report["gate"]["passed"] is True
    assert all(run["gate"]["passed"] for run in report["runs"])
