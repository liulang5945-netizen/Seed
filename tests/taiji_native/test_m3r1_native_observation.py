from __future__ import annotations

import torch

from scripts.training.eval_taiji_m3r1_native_observation import (
    _schema,
    build_course,
    run_gate,
)
from taiji import WorkbenchObservation


def test_workbench_observation_keeps_truncated_file_as_file() -> None:
    schema = _schema()
    observation = WorkbenchObservation.from_workbench_evidence(
        observation_id="truncated",
        project_id="project",
        task_id="task",
        path="large.py",
        capability_snapshot_id="snapshot",
        capability_revision=1,
        read_result={
            "success": True,
            "digest": "a" * 64,
            "byte_length": 2_000_000,
            "truncated": True,
        },
        language_result={
            "programming_language_id": "python",
            "selection_state": "resolved",
            "confidence": 0.9,
            "file_digest": "a" * 64,
            "execution_snapshot": {"available_for_language": ["python"]},
        },
        task_kind="inspect-language",
        schema=schema,
    )
    assert observation.file_is_file is True
    assert observation.feature_vector.numel() == schema.feature_dim
    assert torch.equal(
        observation.to_percept_event(tick=1).features,
        WorkbenchObservation.from_payload(observation.to_payload())
        .to_percept_event(tick=1)
        .features,
    )


def test_m3r1_native_observation_gate(tmp_path) -> None:
    corpus, observations = build_course()
    assert corpus.manifest()["record_disjoint"] is True
    assert {item.project_id for values in observations.values() for item in values} == {
        "m3r1-project-train",
        "m3r1-project-dev",
        "m3r1-project-test",
    }
    report = run_gate(tmp_path / "m3r1.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
