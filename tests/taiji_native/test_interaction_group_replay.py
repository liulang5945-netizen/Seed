from __future__ import annotations

from scripts.training.eval_taiji_interaction_groups import build_native_corpus
from taiji import InteractionGroupEvaluator, InteractionGroupEvaluatorConfig


def test_native_adapter_trace_projection_replays_before_group_admission() -> None:
    corpus, replay_records = build_native_corpus()

    assert len(corpus.train) == 8
    assert len(corpus.holdout) == 8
    assert all(record["replay_equal"] for record in replay_records)
    assert all(record["checkpoint_format"] == "taiji-native-v1" for record in replay_records)
    assert all(
        event.event_id.startswith("r2-s1-")
        for episode in (*corpus.train, *corpus.holdout)
        for event in episode.events
    )
    assert all(
        event.owner_id.startswith("native-owner:")
        for episode in (*corpus.train, *corpus.holdout)
        for event in episode.events
    )

    evaluation = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    ).evaluate(corpus)

    assert evaluation.metrics["holdout_direction_preserved"] is True
    assert evaluation.metrics["gate_passed"] is True
