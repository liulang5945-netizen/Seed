"""The next curriculum expands support without dropping hard target letters."""

from copy import deepcopy

import pytest

from scripts.training.build_taiji_n1_content_curriculum_v3 import build, validate


def test_v3_positive_alphabet_and_diverse_gold_values():
    manifest = build()
    assert manifest == build()
    result = validate(manifest)
    assert result["distinct_train_material_targets"] == 96
    assert min(result["positive_target_letter_counts"].values()) >= 8
    assert {key: sum(value.values()) for key, value in result["counts"].items()} == {
        "train": 192,
        "dev": 32,
        "holdout": 32,
    }


def test_v3_cross_split_value_leak_is_rejected():
    manifest = build()
    train = manifest["episodes"][0]
    contaminated = deepcopy(manifest)
    contaminated["episodes"][-1]["observations"][0] = train["observations"][0]
    with pytest.raises(ValueError, match="value leakage"):
        validate(contaminated)


def test_v3_rejects_closed_material_gold_vocabulary():
    manifest = build()
    for item in manifest["episodes"]:
        if item["split"] == "train" and item["kind"] in ("lookup", "revision"):
            item["answer"] = "abcdef"
    with pytest.raises(ValueError, match="alphabet coverage|small answer vocabulary"):
        validate(manifest)
