from __future__ import annotations

import copy

import pytest

from scripts.training import build_taiji_n1_v21_confirmation_after_n141 as builder


def _excluded() -> tuple[set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.DIRECTIONS}
    return names, words


def test_v21_face_is_deterministic_balanced_and_source_disjoint() -> None:
    excluded_names, excluded_words = _excluded()
    first = builder.build_face(excluded_names, excluded_words)
    second = builder.build_face(excluded_names, excluded_words)

    assert first == second
    checks = builder.validate_face(first, excluded_names, excluded_words)
    assert checks["accepted"] is True
    assert checks["items"] == 160
    assert checks["entity_count"] == len({n for episode in first["episodes"] for n in episode["entities"]})
    assert all(count == 20 for count in checks["items_by_subset_and_kind"].values())
    assert first["shape_counts"]["new_entity_count"] == checks["entity_count"]
    assert first["shape_counts"]["new_word_count"] > 0
    assert all(count == 5 for count in first["shape_counts"]["direction_counts"].values())
    assert all(count == 10 for count in first["shape_counts"]["turn_counts"].values())


def test_face_checker_rejects_split_or_context_shape_changes() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)

    wrong_split = copy.deepcopy(face)
    wrong_split["episodes"][0]["split"] = "train"
    with pytest.raises(builder.Refuse, match="split must be holdout"):
        builder.validate_face(wrong_split, excluded_names, excluded_words)

    wrong_rows = copy.deepcopy(face)
    wrong_rows["episodes"][0]["observations"].pop()
    with pytest.raises(builder.Refuse, match="observation counts differ"):
        builder.validate_face(wrong_rows, excluded_names, excluded_words)


def test_face_checker_rejects_excluded_names_and_non_neighbor_entities() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)

    contaminated = copy.deepcopy(face)
    contaminated["episodes"][0]["entities"][0] = next(iter(excluded_names))
    with pytest.raises(builder.Refuse, match="entity overlaps"):
        builder.validate_face(contaminated, excluded_names, excluded_words)

    wrong_distance = copy.deepcopy(face)
    episode = wrong_distance["episodes"][0]
    replacement = list(episode["entities"][1])
    target = episode["entities"][0]
    match = next(index for index, (a, b) in enumerate(zip(target, replacement, strict=True)) if a == b)
    replacement[match] = "z" if target[match] != "z" else "y"
    episode["entities"][1] = "".join(replacement)
    with pytest.raises(builder.Refuse, match="Hamming-distance"):
        builder.validate_face(wrong_distance, excluded_names, excluded_words)


def test_source_union_reconstructs_every_prior_context_filler() -> None:
    entities, words, source_hashes, summary = builder._collect_source_union()

    assert len(entities) == 26_682
    assert len(words) == 6_288
    assert summary["synthetic_filler_counts"] == {
        "n139": 12_096,
        "n140": 7_776,
        "n141": 7_776,
    }
    assert summary["synthetic_filler_unique_union_count"] == 17_328
    assert summary["synthetic_filler_pairwise_overlap"] == {
        "n139_n140": 4_272,
        "n139_n141": 4_272,
        "n140_n141": 4_896,
    }
    assert len(source_hashes) == summary["source_hash_count"]
    assert builder._verify_hashes(source_hashes)
