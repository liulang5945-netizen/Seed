from __future__ import annotations

from scripts.training.eval_taiji_p4_semantic_consolidation import evaluate_consolidation


def test_p4_consolidation_beats_nearest_episode_on_new_composition() -> None:
    report = evaluate_consolidation()

    assert report["format"] == "taiji-p4-semantic-consolidation-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["semantic_consolidated_error"] < 0.05
    assert report["metrics"]["semantic_consolidated_error"] < report["metrics"]["episodic_nearest_error"]
    assert report["metrics"]["replay_lesion_error"] > 0.5
    assert report["metrics"]["episode_id_lesion_error"] < 0.05
    assert report["metrics"]["checkpoint_continuation_error"] < 0.05
