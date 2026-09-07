from __future__ import annotations

from scripts.training.eval_taiji_m3r2_task_state_sequence import (
    build_transition_course,
    run_gate,
)


def test_m3r2_transition_course_is_record_disjoint() -> None:
    corpus, _ = build_transition_course()
    assert corpus.manifest()["record_disjoint"] is True
    assert len(corpus.train) == len(corpus.dev) == len(corpus.test) == 5


def test_m3r2_native_task_state_gate(tmp_path) -> None:
    report = run_gate(tmp_path / "m3r2.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
