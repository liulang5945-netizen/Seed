"""PLAN-N1-99 variant, baseline-binding and one-shot contract tests; no model queries."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import eval_taiji_n1_unknown_abstention_factorial_diagnostic as diag


def base_items() -> list[dict]:
    items = []
    letters = diag.builder.LETTERS
    for index in range(diag.ITEMS_PER_CELL):
        subject = letters[index] * 10
        distractors = []
        for position in range(4):
            chars = list(subject)
            chars[position] = letters[(index + position + 1) % len(letters)]
            distractors.append("".join(chars))
        entities = [subject, *distractors]
        observations = [
            f"{name} uses {letters[(index + slot + 5) % 26]}bcdef."
            for slot, name in enumerate(distractors)
        ]
        items.append(
            {
                "id": f"v6-confirm-{index * 4 + 3:03d}",
                "split": "holdout",
                "kind": "unknown",
                "subset": "confirm",
                "entities": entities,
                "observations": observations,
                "question": f"material of {subject}?",
                "answer": "unknown",
            }
        )
    return items


def baseline_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        for seed in diag.SEEDS:
            for arm in diag.ARMS:
                exact = (seed + len(arm)) % 2 == 0
                output = b"unknown" if exact else b"abcdef"
                rows.append(
                    {
                        "id": item["id"],
                        "kind": "unknown",
                        "split": "holdout",
                        "seed": seed,
                        "arm": arm,
                        "target_symbols": list(b"unknown") + [256],
                        "output_hex": output.hex(),
                        "free_exact": exact,
                    }
                )
    return rows


def test_build_variants_is_deterministic_and_changes_only_registered_axes() -> None:
    items = base_items()
    first = diag.build_variants(items)
    second = diag.build_variants(items)
    assert first == second
    assert set(first) == {"B", "C", "D"}
    assert {cell: len(values) for cell, values in first.items()} == {"B": 5, "C": 5, "D": 5}

    for index, source in enumerate(items):
        b, c, d = (first[cell][index] for cell in ("B", "C", "D"))
        assert b["base_id"] == c["base_id"] == d["base_id"] == source["id"]
        assert b["question"] == c["question"] == d["question"] == source["question"]
        assert b["answer"] == c["answer"] == d["answer"] == "unknown"
        assert b["entities"][0] == c["entities"][0] == d["entities"][0] == source["entities"][0]
        assert len(b["observations"]) == len(d["observations"]) == 2
        assert len(c["observations"]) == 4
        assert b["observations"] == source["observations"][:2]
        assert d["observations"] == c["observations"][:2]
        assert [
            diag.candidate_value(o, n)
            for o, n in zip(b["observations"], b["entities"][1:], strict=True)
        ] == [
            diag.candidate_value(o, n)
            for o, n in zip(source["observations"][:2], source["entities"][1:3], strict=True)
        ]
        assert all(diag.hamming(source["entities"][0], name) == 2 for name in c["entities"][1:])
        assert all(diag.hamming(source["entities"][0], name) == 2 for name in d["entities"][1:])
        assert [
            diag.candidate_value(o, n)
            for o, n in zip(c["observations"], c["entities"][1:], strict=True)
        ] == [
            diag.candidate_value(o, n)
            for o, n in zip(source["observations"], source["entities"][1:], strict=True)
        ]
        assert all(source["entities"][0] not in observation for observation in c["observations"])


def test_variant_builder_refuses_if_v6_d1_base_shape_changed() -> None:
    broken = base_items()
    broken[0]["entities"][1] = "z" * 10
    with pytest.raises(diag.Refuse, match="d=1 shape"):
        diag.build_variants(broken)


def test_baseline_join_requires_one_exact_row_per_item_seed_arm() -> None:
    items = base_items()
    joined = diag._baseline_rows({"rows": baseline_rows(items)}, items)
    assert len(joined) == 45
    assert {row["condition"] for row in joined} == {"A"}
    assert {row["query_source"] for row in joined} == {"published_v6"}
    assert {row["base_id"] for row in joined} == {item["id"] for item in items}

    missing = baseline_rows(items)[:-1]
    with pytest.raises(diag.Refuse, match="exactly 45 rows"):
        diag._baseline_rows({"rows": missing}, items)

    duplicate = baseline_rows(items)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(diag.Refuse, match="45 rows|bijective"):
        diag._baseline_rows({"rows": duplicate}, items)


def test_output_paths_fail_closed_before_attempt_write(tmp_path, monkeypatch) -> None:
    report = tmp_path / "existing-report.json"
    report.write_text("{}", encoding="utf-8")
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(diag, "REPORT", report)
    monkeypatch.setattr(diag, "ATTEMPT", attempt)
    with pytest.raises(diag.Refuse, match="report path already exists"):
        diag.preconditions()
    assert not attempt.exists()


def test_readiness_mode_uses_no_checkpoint_load_or_query(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        diag,
        "preconditions",
        lambda: {
            "plan_sha256": "p",
            "face_sha256": "f",
            "baseline_report_sha256": "b",
            "items": [None] * 5,
            "endpoints": {(seed, arm): None for seed in diag.SEEDS for arm in diag.ARMS},
            "baseline_rows": [None] * 45,
        },
    )

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("readiness must not load checkpoints")

    monkeypatch.setattr(diag.torch, "load", forbidden_load)
    assert diag.run_readiness() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_loads"] == result["checkpoint_loads"] == 0
    assert result["new_diagnostic_queries"] == 0 and result["new_queries"] == 135
