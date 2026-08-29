from __future__ import annotations

from scripts.training.eval_taiji_interaction_groups import (
    build_workbench_corpus,
    evaluate_workbench,
)


def test_workbench_corpus_uses_native_evidence_and_replays_exactly() -> None:
    corpus, records = build_workbench_corpus()

    assert len(corpus.train) == 8
    assert len(corpus.holdout) == 8
    assert all(record["replay_equal"] for record in records)
    assert all(record["native_checkpoint_format"] == "taiji-native-v1" for record in records)
    assert all(record["native_world_event_count"] > 0 for record in records)
    assert all(
        action["selected_candidate_id"]
        for record in records
        for action in record["workbench_outcome"]["raw_actions"]
    )
    assert any(
        record["workbench_outcome"]["recovery"]
        and record["workbench_outcome"]["recovery"]["success"]
        for record in records
    )


def test_workbench_gate_requires_world_selection_recovery_and_replay() -> None:
    report = evaluate_workbench()

    assert report["input"]["workbench_contract"] == "seed-workbench-contract-v1"
    assert report["input"]["semantic_role_labels"] == 0
    assert report["metrics"]["workbench_checkpoint_replay"] is True
    assert report["metrics"]["workbench_world_evidence"] is True
    assert report["metrics"]["workbench_executive_selection"] is True
    assert report["metrics"]["workbench_recovery_trace"] is True
    assert report["metrics"]["gate_passed"] is True
