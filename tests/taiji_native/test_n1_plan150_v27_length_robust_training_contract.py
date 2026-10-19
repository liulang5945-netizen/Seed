from __future__ import annotations

import ast
import copy
from collections import Counter
from pathlib import Path

import pytest

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as builder
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as runner


def _word(index: int, length: int = 6) -> str:
    chars = ["a"] * length
    value = index
    for offset in range(length):
        value, remainder = divmod(value, 26)
        chars[-1 - offset] = chr(ord("a") + remainder)
    return "m" + "".join(chars[1:]) if length == 6 else "n" + "".join(chars[1:])


def _name(index: int) -> str:
    chars = ["a"] * builder.ENTITY_LENGTH
    value = index
    for offset in range(builder.ENTITY_LENGTH):
        value, remainder = divmod(value, 26)
        chars[-1 - offset] = chr(ord("a") + remainder)
    return "".join(chars)


def _items() -> list[dict]:
    rows = []
    for index in range(192):
        kind = builder.TRAIN_KINDS[index // 48]
        subject = _name(index + 1_000)
        first = _word(index + 10_000)
        second = _word(index + 20_000, 7)
        if kind == "lookup":
            distractor = _name(index + 30_000)
            observations = [f"{subject} uses {first}.", f"{distractor} uses {second}."]
            question, answer = f"material of {subject}?", first
            entities = [subject, distractor]
        elif kind == "revision":
            observations = [f"{subject} uses {first}.", f"{subject} now uses {second}."]
            question, answer = f"material of {subject}?", second
            entities = [subject]
        elif kind == "transition":
            observations = [f"{subject} faces north.", f"{subject} turns left."]
            question, answer = f"direction of {subject}?", "west"
            entities = [subject]
        else:
            alpha, beta = _name(index + 40_000), _name(index + 50_000)
            observations = [f"{alpha} uses {first}.", f"{beta} uses {second}."]
            question, answer = f"material of {subject}?", builder.ABSTENTION
            entities = [subject, alpha, beta]
        rows.append(
            {
                "id": f"train-{index:03d}",
                "split": "train",
                "kind": kind,
                "entities": entities,
                "observations": observations,
                "question": question,
                "answer": answer,
            }
        )
    return rows


def _scheduler(step: int, items: list[dict], _curriculum: list) -> tuple[dict, dict | None]:
    parent_index, within_parent = divmod(step, builder.UPDATES_PER_PARENT)
    route = "raw" if within_parent < 24 else "hard"
    item = copy.deepcopy(items[parent_index])
    item["parent_id"] = item["id"]
    item["id"] = f"{route}-{item['id']}"
    second = None
    if item["kind"] in ("lookup", "unknown"):
        beta = item["entities"][-1]
        second = copy.deepcopy(item)
        second["id"] = f"{route}-paired-{item['parent_id']}"
        second["kind"] = "lookup"
        second["question"] = f"material of {beta}?"
        second["answer"] = item["observations"][1].split()[2].removesuffix(".")
    return item, second


def _build() -> tuple[dict, dict]:
    inputs = {"items": _items(), "curriculum": []}
    g = {"length_ledger": {}, "scheduled": _scheduler}
    exclusions = {_name(90_000)}, {_word(90_000, 7)}
    schedule = builder.build_schedule(g, inputs, *exclusions)
    return inputs, schedule


def test_schedule_balances_each_parent_route_and_task_kind() -> None:
    _inputs, schedule = _build()
    manifest = schedule["manifest"]

    assert len(schedule["slots"]) == builder.UPDATES_PER_MODEL == 9_216
    assert manifest["updates_by_task_kind"] == {kind: 2_304 for kind in builder.TRAIN_KINDS}
    assert manifest["candidate_updates_by_length"] == {"8": 3_072, "16": 3_072, "32": 3_072}
    assert manifest["control_updates_by_length"] == {"8": 9_216, "16": 0, "32": 0}
    assert manifest["answerable_binding_queries_per_model"] == 11_520
    assert manifest["parent_count"] == 192

    counts = Counter(
        (row["parent_id"], row["route"], row["candidate_length"])
        for row in manifest["schedule_rows"]
    )
    assert set(counts.values()) == {8}
    assert all(row["control_length"] == 8 for row in manifest["schedule_rows"])
    assert builder.validate_schedule_manifest(manifest)["accepted"] is True


def test_context_bank_is_nested_target_first_and_exclusion_safe() -> None:
    inputs, schedule = _build()
    excluded_entities, excluded_words = {_name(90_000)}, {_word(90_000, 7)}
    source_names = set()
    source_words = set()
    for item in schedule["source_items"].values():
        names, words = builder._extract_observation_values(item)
        source_names.update(names)
        source_words.update(words)

    for key, item in schedule["source_items"].items():
        base = builder._base_context_rows(item)
        views = schedule["views"][key]
        assert views[8] == views[16][:8] == views[32][:8]
        assert views[8][: len(base)] == base
        subject = builder.n132._query_subject(item["question"])
        for row in views[32][len(base) :]:
            name, _, word = row.removesuffix(".").split()
            assert len(name) == builder.ENTITY_LENGTH
            assert builder.n149_builder.n148_builder.n147_builder._hamming(subject, name) == builder.DISTANCE
            assert name not in source_names | excluded_entities
            assert word not in source_words | excluded_words

    assert len(inputs["items"]) == 192
    assert schedule["manifest"]["context_bank"]["context_view_count"] == len(schedule["views"]) * 3


def test_unknown_pair_uses_the_same_augmented_train_tape() -> None:
    _inputs, schedule = _build()
    slot = next(row for row in schedule["slots"] if row["kind"] == "unknown")
    length = slot["candidate_length"]
    item = runner._apply_context(slot["item"], slot["item_key"], length, schedule["views"])
    second = runner._with_observations(slot["second"], item["observations"], length)

    assert item["split"] == second["split"] == "train"
    assert item["observations"] == second["observations"]
    assert item["question"] != second["question"]
    assert item["answer"] == builder.ABSTENTION
    assert second["kind"] == "lookup"


def test_train_only_score_denominators_and_cli_entrypoint_are_frozen() -> None:
    inputs = {"items": _items()}
    selected = runner._length_score_items(inputs)
    assert set(selected) == set(builder.LENGTHS)
    assert {length: len(rows) for length, rows in selected.items()} == {8: 96, 16: 96, 32: 96}
    assert runner.ORIGINAL_GATE_QUERIES_PER_ENDPOINT == 960
    assert runner.LENGTH_SCORE_QUERIES_PER_ENDPOINT == 288
    assert runner.EXPECTED_TOTAL_TRAIN_SCORE_QUERIES == 7_488
    assert runner.CPU_LIMIT_SECONDS == builder.CPU_LIMIT_SECONDS == 18_000

    module = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    final = module.body[-1]
    assert isinstance(final, ast.If)
    assert "__name__" in ast.unparse(final.test)


def test_schedule_manifest_rejects_changed_slots() -> None:
    _inputs, schedule = _build()
    changed = copy.deepcopy(schedule["manifest"])
    changed["schedule_rows"][0]["candidate_length"] = 32

    with pytest.raises(builder.Refuse, match="SHA mismatch"):
        builder.validate_schedule_manifest(changed)
