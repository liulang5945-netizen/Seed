from __future__ import annotations

import pytest

from scripts.training import (
    run_taiji_n1_plan141_interleaved_context_schedule_matched_continuation as runner,
)


def _name(index: int) -> str:
    chars = ["a"] * 10
    value = index
    for offset in range(10):
        value, remainder = divmod(value, 26)
        chars[-1 - offset] = chr(ord("a") + remainder)
    return "".join(chars)


def _records() -> list[dict]:
    records = []
    step = 0
    for parent_index in range(48):
        parent = f"train-{parent_index:03d}"
        for route in ("raw", "hard"):
            for _ in range(24):
                records.append({"step": step, "parent_id": parent, "route": route})
                step += 1
    return records


def _item(parent_index: int, subject: str, *, route: str) -> dict:
    return {
        "id": f"{route}-train-{parent_index:03d}",
        "kind": "transition",
        "split": "train",
        "entities": [subject],
        "observations": [
            f"{subject} faces south.",
            f"{subject} turns left.",
        ],
        "question": f"direction of {subject}?",
        "answer": "east",
    }


def _scheduled_targets(records: list[dict]) -> tuple[dict, list[dict]]:
    raw = {}
    hard = []
    hard_ordinal = 0
    for record in records:
        parent_index = int(record["parent_id"].removeprefix("train-"))
        if record["route"] == "raw":
            raw[(record["parent_id"], "raw")] = _item(
                parent_index, _name(100 + parent_index), route="raw"
            )
        else:
            hard.append(
                {
                    **record,
                    "item": _item(
                        parent_index,
                        _name(10_000 + hard_ordinal),
                        route="hard",
                    ),
                }
            )
            hard_ordinal += 1
    return raw, hard


def test_candidate_assignment_interleaves_two_row_baseline_and_balances_paths() -> None:
    assignment, summary = runner.build_context_assignment(_records())

    assert len(assignment) == 2_304
    assert summary["parents"] == 48
    assert summary["by_condition"] == {
        "baseline": 1_152,
        "8/d1": 192,
        "8/d2": 192,
        "8/d4": 192,
        "16/d1": 192,
        "16/d2": 192,
        "16/d4": 192,
    }
    assert summary["by_route_and_condition"]["raw/baseline"] == 576
    assert summary["by_route_and_condition"]["hard/baseline"] == 576
    for condition, _length, _distance in runner.TRANSITION_CONDITIONS:
        assert summary["by_route_and_condition"][f"raw/{condition}"] == 96
        assert summary["by_route_and_condition"][f"hard/{condition}"] == 96
    assert all(
        counts["baseline"] == 24
        and all(
            counts[name] == 4
            for name, _length, _distance in runner.TRANSITION_CONDITIONS
        )
        for counts in summary["per_parent"].values()
    )


def test_context_generation_tracks_each_scheduled_hard_target() -> None:
    records = _records()
    assignment, _assignment_summary = runner.build_context_assignment(records)
    raw, hard = _scheduled_targets(records)
    variants, summary = runner.build_context_variants(raw, hard, assignment, set())

    assert summary["raw_filler_entity_count"] == 2_016
    assert summary["hard_augmented_slots"] == 576
    assert summary["hard_filler_entity_count"] == 5_760
    assert summary["unique_new_entity_count"] == 7_776

    raw_item = raw[("train-000", "raw")]
    raw_variant = variants[("train-000", "raw")]
    for distance in (1, 2, 4):
        names = raw_variant["by_distance"][distance]["entities"]
        assert len(names) == 14
        assert all(runner._hamming(raw_variant["subject"], name) == distance for name in names)
    for condition, length, distance in runner.TRANSITION_CONDITIONS:
        changed = runner._apply_context(raw_item, raw_variant, condition)
        assert len(changed["observations"]) == length
        assert changed["observations"][:2] == raw_item["observations"]
        assert all(
            runner._hamming(raw_variant["subject"], name) == distance
            for name in changed["entities"][-(length - 2) :]
        )

    hard_record = next(
        row for row in hard if assignment[row["step"]]["condition"] == "8/d1"
    )
    subject = runner.n132._query_subject(hard_record["item"]["question"])
    hard_variant = variants[(hard_record["parent_id"], "hard", subject)]
    changed = runner._apply_context(hard_record["item"], hard_variant, "8/d1")
    assert changed["observations"][:2] == hard_record["item"]["observations"]
    assert len(changed["observations"]) == 8
    assert all(runner._hamming(subject, name) == 1 for name in changed["entities"][-6:])


def test_context_builder_rejects_unknown_condition() -> None:
    records = _records()
    assignment, _summary = runner.build_context_assignment(records)
    raw, hard = _scheduled_targets(records)
    variants, _variant_summary = runner.build_context_variants(raw, hard, assignment, set())

    with pytest.raises(ValueError, match="unknown augmented context"):
        runner._apply_context(raw[("train-000", "raw")], variants[("train-000", "raw")], "8/d3")


def test_frozen_gate_query_count_does_not_change_with_training_mix() -> None:
    assert runner.EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT == 960
    assert runner.EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT == 288
    assert runner.EXPECTED_TOTAL_TRAIN_QUERIES == 7_488


def test_each_parent_route_uses_the_frozen_interleaved_order() -> None:
    records = _records()
    assignment, _summary = runner.build_context_assignment(records)
    expected_round = []
    for condition, _length, _distance in runner.TRANSITION_CONDITIONS:
        expected_round.extend((runner.BASELINE_CONDITION, condition))
    expected = expected_round * 2

    for parent in sorted({row["parent_id"] for row in records}):
        for route in ("raw", "hard"):
            steps = sorted(
                int(row["step"])
                for row in records
                if row["parent_id"] == parent and row["route"] == route
            )
            assert [assignment[step]["condition"] for step in steps] == expected
