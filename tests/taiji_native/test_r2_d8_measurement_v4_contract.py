"""R2-D8 scale pack, second step (v4) contract -- ((alpha) of the matched-dev routing).

Contract: plans/reference/M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918.md (section 2)
and the matched-dev routing sheet in
plans/reference/M5_R2_D8_MATCHED_DEV_RESULT_20260918.md (row 2 -> "(alpha) enlarge again").

v4 doubles the pools a second time: 64 objects x 24 colours (12 single-byte + 12
six-byte).  As with v3, **dev and final must remain byte-identical to v1** -- the
whole point is that only the train side scales.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_V1 = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
FIXTURE_V3 = Path("tests/fixtures/r2_d1_measurement_v3.jsonl")
FIXTURE_V4 = Path("tests/fixtures/r2_d1_measurement_v4.jsonl")
REPORT_V4 = Path("reports/r2_d1_data_contract_v4_20260918.json")

EXPECTED_TRAIN_SHAPES = {
    "fact": 1536,  # 64 objects x 24 colours
    "negation": 1536,
    "same_opening_fact": 1536,
    "unknown": 64,
    "same_opening_unknown": 64,
    "combination_same": 63,
    "combination_different": 63,
}
COLOR_VALUE_SHAPES = ("fact", "negation", "same_opening_fact")


def _rows(path: Path) -> list[dict]:
    full = PROJECT_ROOT / path
    assert full.is_file(), f"missing fixture: {path}"
    return [
        json.loads(line) for line in full.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _by_split(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def _canon(rows: list[dict]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]


def _color_value(row: dict) -> str:
    text = row["response"]
    return text[2:] if row["shape"] == "negation" else text


@pytest.fixture(scope="module")
def v4_rows() -> list[dict]:
    return _rows(FIXTURE_V4)


def test_v4_fixture_identity(v4_rows: list[dict]) -> None:
    train = _by_split(v4_rows, "train")
    counts = {shape: 0 for shape in EXPECTED_TRAIN_SHAPES}
    for row in train:
        counts[row["shape"]] += 1
    assert counts == EXPECTED_TRAIN_SHAPES
    assert len(train) == sum(EXPECTED_TRAIN_SHAPES.values()) == 4862
    assert len(_by_split(v4_rows, "dev")) == 98
    assert len(_by_split(v4_rows, "final")) == 94


def test_v4_dev_and_final_identical_to_v1(v4_rows: list[dict]) -> None:
    v1_rows = _rows(FIXTURE_V1)
    for split in ("dev", "final"):
        assert _canon(_by_split(v4_rows, split)) == _canon(
            _by_split(v1_rows, split)
        ), f"{split} moved between v1 and v4"


def test_v4_pool_size_and_colour_mix(v4_rows: list[dict]) -> None:
    train = _by_split(v4_rows, "train")
    fact_values = {_color_value(row) for row in train if row["shape"] == "fact"}
    assert len(fact_values) == 24, sorted(fact_values)
    assert sum(1 for value in fact_values if len(value) == 1) == 12
    assert sum(1 for value in fact_values if len(value) == 2) == 12
    unknown_prefixes = {row["prefix"] for row in train if row["shape"] == "unknown"}
    assert len(unknown_prefixes) == 64


def test_v4_train_value_characters_disjoint_from_dev_final(v4_rows: list[dict]) -> None:
    def collect(rows: list[dict]) -> set[str]:
        chars: set[str] = set()
        for row in rows:
            if row["shape"] in COLOR_VALUE_SHAPES:
                chars.update(_color_value(row))
        return chars

    train_chars = collect(_by_split(v4_rows, "train"))
    held_out = collect(_by_split(v4_rows, "dev")) | collect(_by_split(v4_rows, "final"))
    assert train_chars
    assert not (train_chars & held_out), sorted(train_chars & held_out)


def test_v4_is_a_strict_superset_of_v3_train_shape_rules(v4_rows: list[dict]) -> None:
    """v4 keeps v3's shape vocabulary and pair structure; only pools scale."""

    v3_rows = _rows(FIXTURE_V3)
    v3_shapes = {row["shape"] for row in _by_split(v3_rows, "train")}
    v4_shapes = {row["shape"] for row in _by_split(v4_rows, "train")}
    assert v3_shapes == v4_shapes
    v3_pair_types = {row.get("pair_type") for row in v3_rows if row.get("pair_type")}
    v4_pair_types = {row.get("pair_type") for row in v4_rows if row.get("pair_type")}
    assert v3_pair_types == v4_pair_types


def test_v4_every_prefix_and_pair_is_unique(v4_rows: list[dict]) -> None:
    prefixes = [row["prefix"] for row in v4_rows]
    pairs = [(row["prefix"], row["response"]) for row in v4_rows]
    assert len(set(prefixes)) == len(prefixes)
    assert len(set(pairs)) == len(pairs)


def test_v4_report_outcome_and_digest(v4_rows: list[dict]) -> None:
    report = json.loads((PROJECT_ROOT / REPORT_V4).read_text(encoding="utf-8"))
    assert report["outcome"] == "passed"
    assert report["failed_checks"] == []
    assert report["format"] == "r2-d1-measurement-v4"
    assert report["splits"]["train"]["episodes"] == len(_by_split(v4_rows, "train"))
    assert report["corpus_digest"]
