"""Contract tests for the N177 E1 state-dependent query course."""

from __future__ import annotations

import pytest

from taiji.joint_input_producer import TAG_NAMES
from taiji.joint_state_consumption_course import (
    E1_COLORS,
    E1_ENTITIES,
    E1_EPISODES_PER_FAMILY,
    E1_TURNS_PER_EPISODE,
    audit_course,
    build_fixture_episodes,
    build_sealed_split,
    build_train_episodes,
    e1_course_contract,
    e1_families,
    e1_split_digest,
    e1_turn_text,
    episode_from_json,
    episode_to_json,
)
from taiji.joint_t1_course import JOINT_T1_STOP_BYTES


def test_family_assignment_is_stratified_and_disjoint() -> None:
    contract = e1_course_contract()
    assignments = contract["family_assignments"]
    for entity in E1_ENTITIES:
        values = assignments[entity]
        assert len(values["train"]) == 4
        assert len(values["query"]) == 2
        assert len(values["holdout"]) == 2
        assert not set(values["train"]) & set(values["query"])
        assert not set(values["train"]) & set(values["holdout"])
        assert not set(values["query"]) & set(values["holdout"])
    assert contract["families_per_split"] == {"train": 32, "query": 16, "holdout": 16}
    assert contract["sealed_materialized"] is False


def test_structural_audit_passes() -> None:
    audit = audit_course()
    assert audit["ok"] is True
    assert all(item["passed"] for item in audit["checks"])


def test_train_and_fixture_episode_shape() -> None:
    train = build_train_episodes()
    fixture = build_fixture_episodes()
    assert len(train) == 32 * E1_EPISODES_PER_FAMILY
    assert len(fixture) == 2 * E1_EPISODES_PER_FAMILY
    assert len({episode.episode_id for episode in train}) == len(train)
    assert all(episode.partition == "train" for episode in train + fixture)
    assert all(len(episode.turns) == E1_TURNS_PER_EPISODE for episode in train)
    assert all(
        turn.answer_target.target_bytes.endswith(JOINT_T1_STOP_BYTES)
        for episode in train
        for turn in episode.turns
    )


def test_same_query_text_changes_answer_after_correction() -> None:
    train = build_train_episodes()
    for episode in train:
        pre_text, pre_answer = e1_turn_text(episode.turns[2])
        post_text, post_answer = e1_turn_text(episode.turns[4])
        assert pre_text == post_text
        assert pre_answer != post_answer
        pre_colours = {colour for colour in E1_COLORS if colour in pre_answer}
        post_colours = {colour for colour in E1_COLORS if colour in post_answer}
        assert pre_colours and post_colours
        assert pre_colours != post_colours
        correction_text, _ = e1_turn_text(episode.turns[3])
        assert post_colours <= {colour for colour in E1_COLORS if colour in correction_text}
        old_text, old_answer = e1_turn_text(episode.turns[6])
        assert old_answer != post_answer
        assert pre_colours <= {colour for colour in E1_COLORS if colour in old_answer}


def test_ambiguity_turn_has_no_value_tag() -> None:
    train = build_train_episodes()
    value_tags = {
        TAG_NAMES.index("B_VALUE"),
        TAG_NAMES.index("I_VALUE"),
    }
    for episode in train:
        tags = episode.turns[7].semantic_target.tag_ids
        assert not value_tags & set(tags)


def test_distractor_uses_a_different_entity() -> None:
    train = build_train_episodes()
    for episode in train:
        query_text, _ = e1_turn_text(episode.turns[2])
        distractor_text, _ = e1_turn_text(episode.turns[5])
        query_entities = {entity for entity in E1_ENTITIES if entity in query_text}
        distractor_entities = {entity for entity in E1_ENTITIES if entity in distractor_text}
        assert len(query_entities) == 1
        assert distractor_entities
        assert not query_entities & distractor_entities


def test_sealed_splits_are_not_materialized() -> None:
    with pytest.raises(ValueError):
        build_sealed_split("query")
    with pytest.raises(ValueError):
        build_sealed_split("holdout")


def test_episode_wire_round_trip_is_explicit_hex() -> None:
    episode = build_train_episodes()[0]
    wire = episode_to_json(episode)
    frame = wire["episode"]["turns"][0]["frame"]
    assert "payload" not in frame
    assert frame["payload_encoding"] == "hex"
    restored = episode_from_json(wire)
    assert restored.to_payload() == episode.to_payload()


def test_course_and_train_digests_are_deterministic() -> None:
    first_contract = e1_course_contract()
    second_contract = e1_course_contract()
    assert first_contract == second_contract
    first_train = build_train_episodes()
    second_train = build_train_episodes()
    assert e1_split_digest(first_train) == e1_split_digest(second_train)
    families = e1_families()
    assert sum(len(rows) for rows in families.values()) == len(E1_ENTITIES) * len(E1_COLORS)
