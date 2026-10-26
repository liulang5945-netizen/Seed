"""Curriculum prototypes must isolate semantic combinations and hide oracle metadata."""

import copy

import pytest

from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    as_language_episode,
    build_rows,
    validate_rows,
)


def test_deterministic_curriculum_has_the_declared_scene_and_level_counts():
    rows = build_rows()
    assert rows == build_rows()
    report = validate_rows(rows)
    assert report["episodes"] == 1536 and report["families"] == 256
    assert report["unique_model_prompts"] == 1536
    assert report["split_counts"]["train"]["episodes"] == 960
    assert report["split_counts"]["calibration"]["episodes"] == 192
    assert report["split_counts"]["sealed"]["episodes"] == 384
    assert report["split_counts"]["train"]["by_level"] == {"F0": 320, "F1": 320, "F2": 320}


def test_held_primary_pair_never_leaks_into_training_distractor_facts():
    rows = build_rows()
    training_facts = set()
    held_primary = set()
    for row in rows:
        metadata = row["metadata"]
        if row["split"] == "train" and row["curriculum_level"] == "F1":
            training_facts.update(
                [tuple(metadata["primary_pair"]), tuple(metadata["distractor_pair"])]
            )
        elif row["split"] != "train":
            held_primary.add(tuple(metadata["primary_pair"]))
    assert not held_primary & training_facts


def test_oracle_metadata_and_response_are_never_serialized_as_the_model_prompt():
    row = next(row for row in build_rows() if row["metadata"]["query_kind"] == "location_update")
    first = as_language_episode(row)
    changed = copy.deepcopy(row)
    changed["metadata"]["answer_from_world"] = "秘密答案"
    changed["response"] = "另一个目标"
    assert as_language_episode(changed).prompt_bytes == first.prompt_bytes
    assert b"answer_from_world" not in first.prompt_bytes
    assert first.history and first.split == "train"


def test_changing_both_label_and_oracle_cannot_override_the_world_field():
    rows = build_rows()
    index = next(
        i for i, row in enumerate(rows) if row["metadata"]["query_kind"] == "primary_color"
    )
    rows[index]["metadata"]["answer_from_world"] = "错误颜色"
    rows[index]["response"] = "错误颜色。"
    with pytest.raises(ValueError, match="world field"):
        validate_rows(rows)


def test_cross_split_scenes_and_duplicate_episode_ids_are_rejected():
    rows = build_rows()
    rows[0]["split"] = "sealed"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_rows(rows)
    rows = build_rows()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows)


def test_native_episode_split_mapping_keeps_sealed_out_of_training():
    rows = build_rows()
    for split, expected in (("train", "train"), ("calibration", "dev"), ("sealed", "final")):
        row = next(row for row in rows if row["split"] == split)
        assert as_language_episode(row).split == expected
