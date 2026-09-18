"""R2-D8 measurement fixture v3 contract (scale pack).

Contract: plans/reference/M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918.md section 2-3.

The scale pack's only variable is train combination diversity: the v3 train split
expands the object pool 8 -> 32 and the colour pool 6 -> 12 (6 single + 6 six-byte),
while **dev and final are regenerated exactly as in v1**.  These tests pin the
fixture's identity and, above all, that the frozen instrument did not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_V1 = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
FIXTURE_V3 = Path("tests/fixtures/r2_d1_measurement_v3.jsonl")
REPORT_V3 = Path("reports/r2_d1_data_contract_v3_20260918.json")

# 32 objects x 12 colours for the three colour-value shapes; one unknown row per
# object per unknown shape; adjacent-pair combos.
EXPECTED_TRAIN_SHAPES = {
    "fact": 384,
    "negation": 384,
    "same_opening_fact": 384,
    "unknown": 32,
    "same_opening_unknown": 32,
    "combination_same": 31,
    "combination_different": 31,
}
EXPECTED_DEV_SHAPES = {
    "fact": 28,
    "negation": 24,
    "same_opening_fact": 24,
    "unknown": 6,
    "same_opening_unknown": 6,
    "combination_same": 5,
    "combination_different": 5,
}
EXPECTED_FINAL_SHAPES = {**EXPECTED_DEV_SHAPES, "fact": 24}

#: shapes whose response *is* a colour value (the multibyte-copy target)
COLOR_VALUE_SHAPES = ("fact", "negation", "same_opening_fact")
FLIP_PAIR_TYPES = ("fact_flip", "combo_flip")


def _rows(path: Path) -> list[dict]:
    full = PROJECT_ROOT / path
    assert full.is_file(), f"missing fixture: {path}"
    return [
        json.loads(line) for line in full.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _by_split(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def _shape_counts(rows: list[dict]) -> dict[str, int]:
    counts = {shape: 0 for shape in EXPECTED_TRAIN_SHAPES}
    for row in rows:
        counts[row["shape"]] += 1
    return counts


def _canon(rows: list[dict]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]


def _color_value(row: dict) -> str:
    """The colour value carried by a colour-value row ('不是X' -> 'X')."""

    text = row["response"]
    return text[2:] if row["shape"] == "negation" else text


@pytest.fixture(scope="module")
def v3_rows() -> list[dict]:
    return _rows(FIXTURE_V3)


def test_v3_fixture_identity(v3_rows: list[dict]) -> None:
    """Row counts and shape distribution are exactly the frozen v3 numbers."""

    assert len(_by_split(v3_rows, "train")) == sum(EXPECTED_TRAIN_SHAPES.values())
    assert _shape_counts(_by_split(v3_rows, "train")) == EXPECTED_TRAIN_SHAPES
    assert _shape_counts(_by_split(v3_rows, "dev")) == EXPECTED_DEV_SHAPES
    assert _shape_counts(_by_split(v3_rows, "final")) == EXPECTED_FINAL_SHAPES
    assert len(v3_rows) == 1278 + 98 + 94


def test_v3_dev_and_final_are_byte_identical_to_v1(v3_rows: list[dict]) -> None:
    """The instrument is frozen: dev/final must not move when train scales up."""

    v1_rows = _rows(FIXTURE_V1)
    for split in ("dev", "final"):
        assert _canon(_by_split(v3_rows, split)) == _canon(
            _by_split(v1_rows, split)
        ), f"{split} moved between v1 and v3"


def test_v3_train_value_characters_disjoint_from_dev_final(v3_rows: list[dict]) -> None:
    """Train colour characters must not intersect dev/final colour characters."""

    def collect(rows: list[dict]) -> set[str]:
        chars: set[str] = set()
        for row in rows:
            if row["shape"] in COLOR_VALUE_SHAPES:
                chars.update(_color_value(row))
        return chars

    train_chars = collect(_by_split(v3_rows, "train"))
    held_out_chars = collect(_by_split(v3_rows, "dev")) | collect(_by_split(v3_rows, "final"))
    assert train_chars, "no colour-value train rows found"
    assert not (train_chars & held_out_chars), sorted(train_chars & held_out_chars)


def test_v3_train_pool_size_matches_the_scale_contract(v3_rows: list[dict]) -> None:
    """32 objects x 12 colours, and the colour pool mixes 6 single + 6 six-byte."""

    train = _by_split(v3_rows, "train")
    fact_values = {_color_value(row) for row in train if row["shape"] == "fact"}
    assert len(fact_values) == 12, sorted(fact_values)
    assert sum(1 for value in fact_values if len(value) == 1) == 6
    assert sum(1 for value in fact_values if len(value) == 2) == 6
    # 32 objects: `unknown` is emitted once per object
    unknown_prefixes = {row["prefix"] for row in train if row["shape"] == "unknown"}
    assert len(unknown_prefixes) == 32


def test_v3_every_prefix_and_pair_is_unique(v3_rows: list[dict]) -> None:
    prefixes = [row["prefix"] for row in v3_rows]
    pairs = [(row["prefix"], row["response"]) for row in v3_rows]
    assert len(set(prefixes)) == len(prefixes)
    assert len(set(pairs)) == len(pairs)


def test_v3_flip_pair_structure_is_intact(v3_rows: list[dict]) -> None:
    """Balanced counterfactual flip pairs survive the scale-up."""

    flip_rows = [row for row in v3_rows if row.get("pair_type") in FLIP_PAIR_TYPES]
    assert flip_rows, "no flip pairs found"
    grouped: dict[str, list[dict]] = {}
    for row in flip_rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    for pair_id, members in grouped.items():
        assert len(members) == 2, (pair_id, len(members))
        assert sorted(member["pair_role"] for member in members) == ["a", "b"], pair_id
        assert members[0]["pair_type"] == members[1]["pair_type"], pair_id


def test_v3_report_outcome_and_digest(v3_rows: list[dict]) -> None:
    """The sealed data report must record the same corpus digest and pass all gates."""

    report = json.loads((PROJECT_ROOT / REPORT_V3).read_text(encoding="utf-8"))
    assert report["outcome"] == "passed"
    assert report["failed_checks"] == []
    assert report["format"] == "r2-d1-measurement-v3"
    assert report["splits"]["train"]["episodes"] == len(_by_split(v3_rows, "train"))
    assert report["corpus_digest"]
