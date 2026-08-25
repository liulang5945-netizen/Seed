from __future__ import annotations

from scripts.training.eval_taiji_p4_episodic_recall import (
    build_corpus,
    evaluate_recall,
)


def test_p4_one_shot_recall_survives_episode_id_and_checkpoint_lesions() -> None:
    train, queries = build_corpus(cue_dim=8)
    report = evaluate_recall(train, queries)

    assert report["format"] == "taiji-p4-episodic-recall-v1"
    assert report["gate"]["passed"] is True
    assert report["conditions"]["full"]["action_recall"] == 1.0
    assert report["conditions"]["episode_id_lesion"]["action_recall"] == 1.0
    assert report["conditions"]["checkpoint_continuation"]["action_recall"] == 1.0
    assert report["conditions"]["retrieval_lesion"]["action_recall"] < 0.5
    assert report["conditions"]["write_lesion"]["action_recall"] < 0.5
