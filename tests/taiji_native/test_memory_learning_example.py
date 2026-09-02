from __future__ import annotations

import pytest

from taiji import (
    MEMORY_LEARNING_BYPASS_FIELDS,
    MEMORY_LEARNING_EXAMPLE_FIELDS,
    MEMORY_LEARNING_PARTITIONS,
    MemoryLearningCurriculum,
    MemoryLearningExample,
)

ACTIONS = (48, 49)
OUTCOMES = (43, 45)
COMBINATIONS = tuple((action, outcome) for action in ACTIONS for outcome in OUTCOMES)


def _example(
    prefix: str,
    index: int,
    *,
    partition: str,
    cue_key: int,
    combination: tuple[int, int],
    role: str,
) -> MemoryLearningExample:
    return MemoryLearningExample(
        example_id=f"{prefix}-{index}",
        partition=partition,
        cue_key=cue_key,
        action_value=combination[0],
        outcome_value=combination[1],
        role=role,
        episode_tag=f"{prefix}-{index}",
    )


def _stable_curriculum(*, repeats: int = 3) -> MemoryLearningCurriculum:
    train = tuple(
        _example(
            "train",
            slot * repeats + repeat,
            partition="train",
            cue_key=slot,
            combination=COMBINATIONS[slot % len(COMBINATIONS)],
            role="repeated_observation",
        )
        for slot in range(len(COMBINATIONS))
        for repeat in range(repeats)
    )
    holdout = tuple(
        _example(
            "holdout",
            slot,
            partition="holdout",
            cue_key=slot,
            combination=COMBINATIONS[slot % len(COMBINATIONS)],
            role="one_shot_experience",
        )
        for slot in range(len(COMBINATIONS))
    )
    retention = tuple(
        _example(
            "retention",
            slot,
            partition="retention",
            cue_key=slot,
            combination=COMBINATIONS[slot % len(COMBINATIONS)],
            role="one_shot_experience",
        )
        for slot in range(len(COMBINATIONS))
    )
    return MemoryLearningCurriculum(
        name="stable_key",
        train=train,
        holdout=holdout,
        retention=retention,
    )


def test_example_roundtrips_by_content_digest() -> None:
    example = _example(
        "train",
        0,
        partition="train",
        cue_key=7,
        combination=COMBINATIONS[0],
        role="repeated_observation",
    )
    restored = MemoryLearningExample.from_dict(example.to_dict())
    assert restored == example
    assert restored.digest == example.digest
    assert example.value_key == "48/43"
    assert not example.is_query
    assert tuple(example.to_dict()["fields"]) == MEMORY_LEARNING_EXAMPLE_FIELDS
    assert all(name in MEMORY_LEARNING_EXAMPLE_FIELDS for name in MEMORY_LEARNING_BYPASS_FIELDS)


def test_example_rejects_invalid_partition_role_and_symbols() -> None:
    with pytest.raises(ValueError, match="partition"):
        MemoryLearningExample(
            example_id="x", partition="replay", cue_key=0, action_value=48, outcome_value=43
        )
    with pytest.raises(ValueError, match="role"):
        MemoryLearningExample(
            example_id="x",
            partition="train",
            cue_key=0,
            action_value=48,
            outcome_value=43,
            role="rehearsed",
        )
    with pytest.raises(ValueError, match="negative"):
        MemoryLearningExample(
            example_id="x", partition="train", cue_key=-1, action_value=48, outcome_value=43
        )
    with pytest.raises(ValueError, match="non-empty"):
        MemoryLearningExample(
            example_id="  ", partition="train", cue_key=0, action_value=48, outcome_value=43
        )


def test_example_rejects_wrong_format_or_fields() -> None:
    payload = _example(
        "train", 0, partition="train", cue_key=0, combination=COMBINATIONS[0], role="repeated_observation"
    ).to_dict()
    broken = dict(payload)
    broken["format"] = "legacy"
    with pytest.raises(ValueError, match="format"):
        MemoryLearningExample.from_dict(broken)
    broken = dict(payload)
    broken["fields"] = ["cue_key"]
    with pytest.raises(ValueError, match="fields"):
        MemoryLearningExample.from_dict(broken)


def test_stable_curriculum_audit_reports_learnable_partitioned_data() -> None:
    curriculum = _stable_curriculum()
    audit = curriculum.audit()
    assert audit["well_formed"]
    assert audit["observed_key_value_deterministic"]
    assert audit["declaration_matches_observation"]
    assert audit["leakage"]["train_query_id_overlap"] == 0
    assert not audit["leakage"]["any_bypass_field_leak"]
    for partition in MEMORY_LEARNING_PARTITIONS:
        record = audit["partitions"][partition]
        assert record["factorial_complete"]
        assert record["key_value_deterministic"]
        assert record["role_labels_match_observation_counts"]
    assert audit["partitions"]["train"]["observations_per_key"]["min"] == 3.0
    for partition in ("holdout", "retention"):
        relation = audit["query_relations"][partition]
        assert relation["answer_matches_train_taught_value"]
        assert relation["ambiguous_key_query_count"] == 0


def test_curriculum_roundtrips_and_preserves_digest() -> None:
    curriculum = _stable_curriculum()
    restored = MemoryLearningCurriculum.from_dict(curriculum.to_dict())
    assert restored.digest == curriculum.digest
    assert restored.audit() == curriculum.audit()


def test_negative_control_must_declare_its_own_conflict() -> None:
    stable = _stable_curriculum()
    conflicting = tuple(
        MemoryLearningExample(
            example_id=example.example_id,
            partition=example.partition,
            cue_key=example.cue_key,
            action_value=COMBINATIONS[index % len(COMBINATIONS)][0],
            outcome_value=COMBINATIONS[index % len(COMBINATIONS)][1],
            role=example.role,
            episode_tag=example.episode_tag,
        )
        for index, example in enumerate(stable.train)
    )
    with pytest.raises(ValueError, match="declares determinism"):
        MemoryLearningCurriculum(
            name="conflict",
            train=conflicting,
            holdout=stable.holdout,
            retention=stable.retention,
        )
    negative = MemoryLearningCurriculum(
        name="conflict",
        train=conflicting,
        holdout=stable.holdout,
        retention=stable.retention,
        declared_key_value_deterministic=False,
    )
    audit = negative.audit()
    assert not audit["observed_key_value_deterministic"]
    assert audit["declaration_matches_observation"]
    assert audit["partitions"]["train"]["conflicting_key_count"] > 0
    assert audit["query_relations"]["holdout"]["ambiguous_key_query_count"] > 0


def test_curriculum_rejects_untrained_query_key_and_slot_mismatch() -> None:
    stable = _stable_curriculum()
    stray = MemoryLearningExample(
        example_id="holdout-stray",
        partition="holdout",
        cue_key=999,
        action_value=48,
        outcome_value=43,
        role="one_shot_experience",
    )
    with pytest.raises(ValueError, match="untrained cue key"):
        MemoryLearningCurriculum(
            name="stray",
            train=stable.train,
            holdout=(*stable.holdout, stray),
            retention=stable.retention,
        )
    with pytest.raises(ValueError, match="does not match its slot"):
        MemoryLearningCurriculum(
            name="slot",
            train=stable.train,
            holdout=stable.train[:1],
            retention=stable.retention,
        )


def test_audit_detects_a_bypass_field_that_reveals_the_answer() -> None:
    stable = _stable_curriculum()
    leaking = tuple(
        MemoryLearningExample(
            example_id=example.example_id,
            partition=example.partition,
            cue_key=example.cue_key,
            action_value=example.action_value,
            outcome_value=example.outcome_value,
            role=example.role,
            provenance=f"experienced-{example.value_key}",
            episode_tag=example.episode_tag,
        )
        for example in stable.train
    )
    curriculum = MemoryLearningCurriculum(
        name="leak",
        train=leaking,
        holdout=stable.holdout,
        retention=stable.retention,
    )
    audit = curriculum.audit()
    assert audit["leakage"]["bypass_field_value_leak"]["provenance"]
    assert audit["leakage"]["any_bypass_field_leak"]
    assert not audit["well_formed"]
