from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as builder


def _face():
    excluded_entities = {"zzzzzzzzzz", "yyyyyyyyyy"}
    excluded_words = {
        "zzzzzz",
        "yyyyyyy",
        "north",
        "east",
        "south",
        "west",
        "left",
        "right",
        "unknown",
    }
    manifest, summary = builder.build_face(excluded_entities, excluded_words)
    return manifest, summary, excluded_entities, excluded_words


def test_fresh_face_is_balanced_and_uses_nested_lengths() -> None:
    manifest, summary, excluded_entities, excluded_words = _face()

    assert manifest["base_item_count"] == 160
    assert manifest["items_by_kind"] == {kind: 40 for kind in builder.KINDS}
    assert manifest["queries_per_endpoint"] == 480
    assert manifest["expected_queries"] == 2_880
    assert summary["checks"]["accepted"] is True
    assert summary["checks"]["source_overlap"] is False

    for item in manifest["items"]:
        views = item["observations_by_length"]
        assert len(views["8"]) == 8
        assert len(views["16"]) == 16
        assert len(views["32"]) == 32
        assert views["8"] == views["16"][:8] == views["32"][:8]
        assert item["split"] == "diagnostic"
        assert builder.n132._query_subject(item["question"]) not in excluded_entities
        if item["kind"] in ("lookup", "revision"):
            assert item["answer"] not in excluded_words
        elif item["kind"] == "transition":
            assert item["answer"] in builder.DIRECTIONS
        else:
            assert item["answer"] == builder.UNKNOWN


def test_face_targets_are_first_and_all_neighbors_are_d1() -> None:
    manifest, _summary, _excluded_entities, _excluded_words = _face()
    for item in manifest["items"]:
        target = builder.n132._query_subject(item["question"])
        rows = item["observations_by_length"]["32"]
        target_indices = item["target_observation_indices"]
        if target_indices:
            assert target_indices == list(range(len(target_indices)))
        for row in rows:
            name = row.removesuffix(".").split()[0]
            if name != target:
                assert builder._hamming(target, name) == 1


def test_face_validator_rejects_a_non_nested_view() -> None:
    manifest, _summary, excluded_entities, excluded_words = _face()
    changed = {**manifest, "items": [dict(item) for item in manifest["items"]]}
    changed["items"][0] = {
        **changed["items"][0],
        "observations_by_length": dict(changed["items"][0]["observations_by_length"]),
    }
    changed["items"][0]["observations_by_length"]["16"] = list(
        changed["items"][0]["observations_by_length"]["16"]
    )
    changed["items"][0]["observations_by_length"]["16"][0] = "aaaaaaaaaa uses wrongval."

    with pytest.raises(builder.Refuse, match="not nested"):
        builder.validate_face(changed, excluded_entities, excluded_words)


def test_builder_cli_invocation_is_last() -> None:
    module = ast.parse(Path(builder.__file__).read_text(encoding="utf-8"))
    final = module.body[-1]
    assert isinstance(final, ast.If)
    assert "__name__" in ast.unparse(final.test)
