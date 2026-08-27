from __future__ import annotations

import torch

from taiji import LearnedPerception, PerceptionConfig, TaijiConfig


def _config() -> TaijiConfig:
    return TaijiConfig(
        alphabet_size=257,
        boundary_symbol=256,
        perception=PerceptionConfig(
            feature_dim=12,
            local_window=3,
            minimum_assembly_duration=1,
            maximum_assembly_duration=3,
            boundary_threshold=1.0,
            change_gain=0.5,
            surprise_gain=0.5,
            learning_rate=0.1,
            seed_offset=101,
        ),
    )


def test_perception_emits_continuous_features_and_variable_duration_assemblies() -> None:
    perception = LearnedPerception(_config())
    events = [
        perception.observe(symbol, tick=index, stream_id="test", learn=False)
        for index, symbol in enumerate((97, 97, 97, 97, 97, 97))
    ]

    assert all(event.features.shape == (12,) for event in events)
    assert all(float(event.features.norm()) > 0.0 for event in events)
    assert all(event.duration >= 1 for event in events)
    assert [event.duration for event in events] == [1, 2, 3, 1, 2, 3]
    assert [event.boundary for event in events] == [False, False, True, False, False, True]
    assert all(event.assembly_id.startswith("test:assembly:") for event in events)


def test_perception_local_learning_changes_embedding_but_not_when_frozen() -> None:
    learned = LearnedPerception(_config())
    frozen = LearnedPerception(_config())
    before_learned = learned.embedding.weight.detach().clone()
    before_frozen = frozen.embedding.weight.detach().clone()

    for index, symbol in enumerate((97, 98, 99, 97)):
        learned.observe(symbol, tick=index, stream_id="learned", learn=True)
        frozen.observe(symbol, tick=index, stream_id="frozen", learn=False)

    assert not torch.equal(before_learned, learned.embedding.weight)
    assert torch.equal(before_frozen, frozen.embedding.weight)


def test_perception_predictive_fit_updates_local_representation() -> None:
    perception = LearnedPerception(_config())
    before_projection = perception.local_projection.weight.detach().clone()
    before_recency = perception.assembly_recency_logit.detach().clone()

    losses = perception.fit_predictive(
        ((97, 98, 99, 97, 98, 99), (99, 98, 97, 99, 98, 97)),
        epochs=2,
        learning_rate=0.01,
        multi_step_prediction_weight=0.05,
        multi_step_prediction_horizon=3,
        cross_assembly_prediction_weight=0.1,
        cross_assembly_negative_weight=0.025,
    )

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert not torch.equal(before_projection, perception.local_projection.weight)
    assert not torch.equal(before_recency, perception.assembly_recency_logit)


def test_boundary_after_segments_stop_at_the_next_runtime_boundary() -> None:
    perception = LearnedPerception(_config())

    segments = perception._boundary_after_segments(
        ((0, 2), (2, 5)),
        sequence_length=7,
        target_rows={2: 1, 5: 4},
    )

    assert segments == ((1, 2, 5), (4, 5, 7))


def test_training_assembly_rollout_matches_runtime_boundary_clock() -> None:
    sequence = (97, 97, 97, 97, 97, 97)
    perception = LearnedPerception(_config())
    features, _, _, _ = perception._sequence_features(sequence)

    training_spans = perception._rollout_assembly_spans(sequence, features)

    perception.reset_dynamics()
    runtime_spans: list[tuple[int, int]] = []
    start = 0
    for index, symbol in enumerate(sequence):
        event = perception.observe(symbol, tick=index, stream_id="alignment", learn=False)
        if event.boundary:
            runtime_spans.append((start, index + 1))
            start = index + 1

    assert training_spans == runtime_spans


def test_perception_checkpoint_restores_dynamic_assembly_and_weights() -> None:
    original = LearnedPerception(_config())
    for index, symbol in enumerate((97, 98)):
        original.observe(symbol, tick=index, stream_id="roundtrip", learn=True)
    restored = LearnedPerception(_config())
    restored.restore(original.checkpoint())

    left = original.observe(99, tick=2, stream_id="roundtrip", learn=True)
    right = restored.observe(99, tick=2, stream_id="roundtrip", learn=True)

    assert left.event_id == right.event_id
    assert left.duration == right.duration
    assert left.boundary == right.boundary
    assert torch.equal(left.features, right.features)
    for left_parameter, right_parameter in zip(
        original.parameter_tensors(), restored.parameter_tensors(), strict=False
    ):
        assert torch.equal(left_parameter, right_parameter)
