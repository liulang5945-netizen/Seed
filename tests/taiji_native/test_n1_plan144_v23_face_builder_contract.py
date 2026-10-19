from __future__ import annotations

import copy

import pytest

from scripts.training import build_taiji_n1_v23_confirmation_after_n143 as builder


def _excluded() -> tuple[set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.DIRECTIONS}
    return names, words


def test_v23_face_is_deterministic_balanced_and_covers_frozen_positions() -> None:
    excluded_names, excluded_words = _excluded()
    first = builder.build_face(excluded_names, excluded_words)
    second = builder.build_face(excluded_names, excluded_words)

    assert first == second
    assert first["seed"] == 123
    assert first["observations_per_item"] == 32
    checks = builder.validate_face(first, excluded_names, excluded_words)
    assert checks["accepted"] is True
    assert checks["items"] == 160
    assert all(count == 20 for count in checks["items_by_subset_and_kind"].values())
    assert checks["entity_count"] == len({name for item in first["episodes"] for name in item["entities"]})
    assert first["shape_counts"]["new_entity_count"] == checks["entity_count"]
    assert first["shape_counts"]["new_word_count"] > 0

    for item in first["episodes"]:
        local = int(item["id"].rsplit("-", 1)[-1])
        kind = item["kind"]
        if kind == "lookup":
            expected_start = builder.LOOKUP_POSITIONS[local % 5]
            expected_indices = [expected_start]
        elif kind in ("revision", "transition"):
            expected_start = builder.PAIR_POSITIONS[local % 5]
            expected_indices = [expected_start, expected_start + 1]
        else:
            expected_start = None
            expected_indices = []
        assert item["target_start_index"] == expected_start
        assert item["target_fact_indices"] == expected_indices


def test_v23_validator_rejects_context_shape_and_target_position_changes() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)

    wrong_rows = copy.deepcopy(face)
    wrong_rows["episodes"][0]["observations"].pop()
    with pytest.raises(builder.Refuse, match="observation counts differ"):
        builder.validate_face(wrong_rows, excluded_names, excluded_words)

    wrong_position = copy.deepcopy(face)
    wrong_position["episodes"][0]["target_start_index"] = 1
    with pytest.raises(builder.Refuse, match="target position/index mismatch"):
        builder.validate_face(wrong_position, excluded_names, excluded_words)

    wrong_answer = copy.deepcopy(face)
    wrong_answer["episodes"][0]["answer"] = "badval"
    with pytest.raises(builder.Refuse, match="gold target semantics mismatch"):
        builder.validate_face(wrong_answer, excluded_names, excluded_words)

    contaminated = copy.deepcopy(face)
    contaminated["episodes"][0]["entities"][0] = next(iter(excluded_names))
    with pytest.raises(builder.Refuse, match="entity overlaps"):
        builder.validate_face(contaminated, excluded_names, excluded_words)


def test_source_union_adds_the_consumed_v22_face_and_complete_v22_range() -> None:
    entities, words, source_hashes, summary = builder._collect_source_union()

    assert len(entities) == 31_722
    assert len(words) == 10_128
    assert summary["n1_143_entity_exclusion_count"] == 29_202
    assert summary["n1_143_word_exclusion_count"] == 8_208
    assert summary["consumed_v22_face_entity_count"] == 2_520
    assert summary["consumed_v22_face_material_value_count"] == 1_920
    assert summary["entity_overlap_with_old_union"] == 0
    assert summary["word_overlap_with_old_union"] == 0
    assert summary["partial_v22_sacrifice_queries_excluded"] == 960
    assert summary["partial_v22_confirmation_queries"] == 0
    assert builder._verify_hashes(source_hashes)


def test_frozen_plan_binding_matches_the_current_chinese_plan() -> None:
    assert builder.PLAN.is_file()
    assert builder.sha256(builder.PLAN) == builder.PLAN_SHA256
    assert builder.SEED == 123
    assert builder.OBSERVATIONS == 32
    assert builder.LOOKUP_POSITIONS == (0, 8, 16, 24, 31)
    assert builder.PAIR_POSITIONS == (0, 8, 15, 23, 30)
