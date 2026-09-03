from __future__ import annotations

import torch

from taiji import (
    Outcome,
    Taiji,
    TaijiConfig,
    WorldAction,
    WorldInterventionCase,
    WorldObject,
    WorldState,
)
from taiji.foundation_tasks import (
    ContinualLearningCorpus,
    ContinualLearningTask,
    ContinualMemoryTask,
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    DelayedMemoryTask,
    GoalActionCorpus,
    GoalActionEpisode,
    GoalActionTask,
    MemoryEpisode,
    SequencePredictionCorpus,
    SequencePredictionTask,
    WorldTransitionCorpus,
    WorldTransitionTask,
    _persistent_digest,
)


def _config(seed: int) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        concept_capacity=8,
        seed=seed,
    )


def test_sequence_task_returns_a_real_measurement_and_no_holdout_mutation() -> None:
    corpus = SequencePredictionCorpus(
        train=(b"alpha-beta-" * 12),
        holdout=(b"alpha-gamma-" * 5),
        retention=(b"alpha-delta-" * 5),
    )

    measurement = SequencePredictionTask(
        _config(11), seeds=(11, 29, 47), epochs=1
    ).evaluate(corpus)

    assert measurement.ability_id == "b1_sequence_prediction"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_value is not None
    assert measurement.sample_counts == {
        "train": len(corpus.train),
        "holdout": len(corpus.holdout),
        "retention": len(corpus.retention),
    }
    assert measurement.holdout_updates == 0
    assert all(kind in measurement.baseline_metrics for kind in (
        "random",
        "frozen_parent",
        "simple_rule",
        "hash_only",
    ))
    assert any("seed_metrics" in item for item in measurement.evidence)


def test_delayed_memory_task_recalls_trained_cues_without_holdout_writes() -> None:
    train = tuple(
        MemoryEpisode(
            memory_id=f"train-{index}",
            cue=ord("A") + index,
            action=ord("0") + index % 2,
            outcome=ord("+") if index % 2 == 0 else ord("-"),
        )
        for index in range(8)
    )
    queries = tuple(
        DelayedMemoryQuery(
            query_id=f"query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"retention-{index}",
            cue=query.cue,
            expected_action=query.expected_action,
        )
        for index, query in enumerate(queries)
    )
    corpus = DelayedMemoryCorpus(train=train, holdout=queries, retention=retention)

    measurement = DelayedMemoryTask(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            memory_units=128,
            memory_fan_in=32,
            memory_meta_dim=32,
            memory_readout_fan_in=32,
            memory_iterations=3,
            seed=11,
        ),
        seeds=(11,),
    ).evaluate(corpus)

    assert measurement.ability_id == "b2_delayed_memory"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_direction == "higher_is_better"
    assert measurement.holdout_updates == 0
    assert measurement.sample_counts == {"train": 8, "holdout": 8, "retention": 8}
    assert "memory_lesion" in measurement.baseline_metrics


def test_b2_delayed_memory_identity_lesion_appears_in_baseline() -> None:
    """Identity lesion must be a distinct baseline from memory lesion.

    The current ``memory_lesion`` is a double ablation (``use_memory=False``
    disables both episodic memory AND the identity organ).  Adding a pure
    identity-organ lesion (``use_memory=True, use_identity=False``) allows
    the evaluation to attribute gains to either mechanism.
    """
    train = tuple(
        MemoryEpisode(
            memory_id=f"train-{index}",
            cue=ord("A") + index,
            action=ord("0") + index % 2,
            outcome=ord("+") if index % 2 == 0 else ord("-"),
        )
        for index in range(8)
    )
    queries = tuple(
        DelayedMemoryQuery(
            query_id=f"query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"retention-{index}",
            cue=query.cue,
            expected_action=query.expected_action,
        )
        for index, query in enumerate(queries)
    )
    corpus = DelayedMemoryCorpus(train=train, holdout=queries, retention=retention)

    measurement = DelayedMemoryTask(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            memory_units=128,
            memory_fan_in=32,
            memory_meta_dim=32,
            memory_readout_fan_in=32,
            memory_iterations=3,
            seed=11,
        ),
        seeds=(11,),
    ).evaluate(corpus)

    assert "identity_lesion" in measurement.baseline_metrics


def test_persistent_digest_responds_to_identity_organ_mutation() -> None:
    """The read-only audit digest must be sensitive to identity-organ writes.

    ``_persistent_digest`` currently only hashes fabric/motor/memory, making
    it blind to identity organ mutations during holdout or retention.
    """
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
        seed=11,
    )
    model = Taiji(config)
    model.observe(config.boundary_symbol, learn=False, learn_motor=False)
    model.observe(ord("X"), learn=False, learn_motor=False)
    context = model.fabric.cortical_context(
        model.snapshot().regions
    ).detach().clone()

    before = _persistent_digest(model)

    model.identity_organ.learn(
        context,
        action_symbol=ord("0"),
        outcome_symbol=ord("+"),
    )

    after = _persistent_digest(model)

    assert before != after, (
        "identity organ write should change the persistent digest"
    )


def test_b2_cue_can_exceed_the_alphabet_via_multi_symbol_context() -> None:
    """B2 must reach the manifest floor of 1,000 mutually distinct cues.

    ``MemoryEpisode.cue`` is a single symbol and ``alphabet_size`` is 257, so
    at most 256 distinct cues exist today. A multi-symbol ``context`` prefix
    must widen the key space without reusing any cue.
    """
    episodes = tuple(
        MemoryEpisode(
            memory_id=f"wide-{index}",
            cue=ord("A") + index % 2,
            action=ord("0") + index % 2,
            outcome=ord("+"),
            context=(ord("a") + index // 2,),
        )
        for index in range(512)
    )
    keys = {episode.recall_key for episode in episodes}

    assert len(keys) == 512, (
        "multi-symbol context must make 512 cues mutually distinct"
    )


def test_b2_write_and_query_are_separated_by_interference_filler() -> None:
    """The manifest declares ``cue_event_delay_interference_episode``.

    ``_write_episode`` and ``_recall_accuracy`` contain no delay or
    interference filler, so today's B2 measures immediate recall and the
    declared ``input_kind`` is an empty claim.
    """
    train = tuple(
        MemoryEpisode(
            memory_id=f"train-{index}",
            cue=ord("A") + index,
            action=ord("0") + index % 2,
            outcome=ord("+") if index % 2 == 0 else ord("-"),
        )
        for index in range(4)
    )
    queries = tuple(
        DelayedMemoryQuery(
            query_id=f"query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    corpus = DelayedMemoryCorpus(
        train=train,
        holdout=queries,
        retention=tuple(
            DelayedMemoryQuery(
                query_id=f"retention-{index}",
                cue=query.cue,
                expected_action=query.expected_action,
            )
            for index, query in enumerate(queries)
        ),
        interference_symbols=(ord("z"), ord("y"), ord("x")),
    )
    observed: list[int] = []
    task = DelayedMemoryTask(_config(11), seeds=(11,))
    model = Taiji(_config(11))
    original = model.observe

    def _record(symbol: int, **kwargs: object):
        observed.append(int(symbol))
        return original(symbol, **kwargs)

    model.observe = _record  # type: ignore[method-assign]
    task._recall_accuracy(
        model,
        corpus.holdout[:1],
        (ord("0"), ord("1")),
        use_memory=True,
        interference_symbols=corpus.interference_symbols,
    )

    assert observed[-3:] == [ord("z"), ord("y"), ord("x")], (
        "recall must observe interference filler between cue and readout"
    )


def test_b2_evaluate_threads_corpus_interference_into_every_read_channel() -> None:
    """A corpus-declared delay must reach taiji, both lesions and frozen parent.

    If ``evaluate`` drops ``interference_symbols`` the corpus claims a delay
    that no read channel ever experiences, and every reported margin is an
    immediate-recall number wearing a delayed-recall label.
    """
    train = tuple(
        MemoryEpisode(
            memory_id=f"train-{index}",
            cue=ord("A") + index,
            action=ord("0") + index % 2,
            outcome=ord("+") if index % 2 == 0 else ord("-"),
        )
        for index in range(4)
    )
    corpus = DelayedMemoryCorpus(
        train=train,
        holdout=tuple(
            DelayedMemoryQuery(
                query_id=f"holdout-{index}",
                cue=episode.cue,
                expected_action=episode.action,
            )
            for index, episode in enumerate(train)
        ),
        retention=tuple(
            DelayedMemoryQuery(
                query_id=f"retention-{index}",
                cue=episode.cue,
                expected_action=episode.action,
            )
            for index, episode in enumerate(train)
        ),
        interference_symbols=(ord("z"), ord("y")),
    )
    seen: list[tuple[int, ...]] = []
    original = DelayedMemoryTask._recall_accuracy

    def _spy(*args: object, **kwargs: object) -> float:
        seen.append(tuple(kwargs.get("interference_symbols", ())))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    DelayedMemoryTask._recall_accuracy = staticmethod(_spy)  # type: ignore[method-assign]
    try:
        DelayedMemoryTask(_config(11), seeds=(11,)).evaluate(corpus)
    finally:
        DelayedMemoryTask._recall_accuracy = staticmethod(original)  # type: ignore[method-assign]

    assert seen and all(channel == corpus.interference_symbols for channel in seen), (
        f"every read channel must observe the declared delay, got {seen}"
    )


def test_continual_memory_contract_measures_replay_against_no_replay() -> None:
    from scripts.training.eval_taiji_b5_memory import build_corpus

    corpus = build_corpus(train_count=8, holdout_count=4, retention_count=4)

    assert corpus.sample_counts == {
        "train": 16,
        "holdout": 8,
        "retention": 8,
        "phase_a_train": 8,
        "phase_a_holdout": 4,
        "phase_a_retention": 4,
        "phase_b_train": 8,
        "phase_b_holdout": 4,
        "phase_b_retention": 4,
        "replay_train": 8,
    }
    assert len(corpus.digest) == 64
    measurement = ContinualMemoryTask(_config(11), seeds=(11,)).evaluate(corpus)

    assert measurement.ability_id == "b5_continual_learning"
    assert measurement.status in {"passed", "failed"}
    assert measurement.primary_metric == "backward_transfer"
    assert measurement.holdout_updates == 0
    assert any("no_replay_counterfactual" in item for item in measurement.evidence)


def _world_case(case_id: str, *, position: float, action_kind: str) -> WorldInterventionCase:
    before = WorldState(
        tick=0,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position}),
        ),
    )
    action = WorldAction(
        action_id=case_id,
        kind=action_kind,
        tick=0,
        actor_id="agent",
        target_id="target",
        parameters={"amount": 1.0},
    )
    delta = 1.0 if action_kind == "push" else -1.0
    after = WorldState(
        tick=1,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position + delta}),
        ),
    )
    return WorldInterventionCase(
        case_id=case_id,
        initial=before,
        action=action,
        expected_state=after,
        expected_outcome=Outcome(
            intent_id=case_id,
            reward=delta,
            success=delta > 0,
            tick=1,
        ),
    )


def test_world_transition_task_uses_train_only_schema_and_reports_controls() -> None:
    train = tuple(
        _world_case(f"train-{index}", position=float(index), action_kind="push")
        for index in range(6)
    )
    holdout = tuple(
        _world_case(f"holdout-{index}", position=10.0 + index, action_kind="push")
        for index in range(3)
    )
    retention = tuple(
        _world_case(f"retention-{index}", position=20.0 + index, action_kind="push")
        for index in range(3)
    )

    measurement = WorldTransitionTask(epochs=10, seeds=(11,)).evaluate(
        WorldTransitionCorpus(train=train, holdout=holdout, retention=retention)
    )

    assert measurement.ability_id == "b3_world_transition"
    assert measurement.status in {"passed", "failed"}
    assert measurement.sample_counts == {"train": 6, "holdout": 3, "retention": 3}
    assert measurement.holdout_updates == 0
    assert set(("random", "frozen_parent", "simple_rule", "hash_only")).issubset(
        measurement.baseline_metrics
    )


def test_goal_action_task_uses_outcome_credit_without_holdout_writes() -> None:
    def episode(prefix: str, index: int) -> GoalActionEpisode:
        return GoalActionEpisode(
            episode_id=f"{prefix}-{index}",
            cue=65 + index % 2,
            preferred_action=48 + index % 2,
            alternate_action=49 - index % 2,
        )

    corpus = GoalActionCorpus(
        train=tuple(episode("train", index) for index in range(8)),
        holdout=tuple(episode("holdout", index) for index in range(4)),
        retention=tuple(episode("retention", index) for index in range(4)),
    )

    measurement = GoalActionTask(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            memory_units=128,
            memory_fan_in=32,
            memory_meta_dim=32,
            memory_readout_fan_in=32,
            memory_iterations=3,
            seed=11,
        ),
        seeds=(11,),
    ).evaluate(corpus)

    assert measurement.ability_id == "b4_goal_action"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_direction == "higher_is_better"
    assert measurement.holdout_updates == 0
    assert measurement.sample_counts == {"train": 8, "holdout": 4, "retention": 4}
    assert "credit_lesion" in measurement.baseline_metrics


def test_continual_task_records_checkpoint_continuation_and_replay_retention() -> None:
    corpus = ContinualLearningCorpus(
        phase_a_train=b"ABCD1234-" * 8,
        phase_a_holdout=b"ABCD1234+" * 4,
        phase_b_train=b"wxyz5678:" * 8,
        phase_b_holdout=b"wxyz5678;" * 4,
        retention=b"ABCD1234?" * 4,
    )

    measurement = ContinualLearningTask(
        _config(11),
        seeds=(11,),
        epochs=1,
        replay_epochs=1,
    ).evaluate(corpus)

    assert measurement.ability_id == "b5_continual_learning"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_direction == "higher_is_better"
    assert measurement.holdout_updates == 0
    assert measurement.sample_counts == {"train": 144, "holdout": 72, "retention": 36}
    assert "replay_lesion" in measurement.baseline_metrics
    assert any("continued_from_parent" in item for item in measurement.evidence)
