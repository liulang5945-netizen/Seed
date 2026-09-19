"""R2 content-binding static data contract (implementation gate).

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md §3

Pins the generated namespace ``r2_content_binding_v1``: quotas, pair
structure, in-group counterfactual relations, split-level holdout keys,
copy-mask semantics, value-length and position balances, sealed slice
denominators, the independent reference key and the fixed-policy floor that
must hold BEFORE any training starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.training.build_taiji_r2_content_binding_data import (
    EXPECTED_IN_GROUP,
    FLIP_GATE_CLASSES,
    GROUP_CLASSES,
    MARKERS,
    MAX_PREFIX_CHARS,
    MAX_RESPONSE_CHARS,
    POOLS,
    VALUE_CLASSES,
    reference_answer,
    split_material_body,
)
from scripts.training.eval_taiji_r2_content_binding import (
    build_policies,
    load_fixtures,
    score_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_REPORT = PROJECT_ROOT / "reports/r2_content_binding_v1/data_contract_v1_20260919.json"
BASELINE_REPORT = (
    PROJECT_ROOT / "reports/r2_content_binding_v1/static_policy_baselines_v1_20260919.json"
)
EXPECTED_GROUPS = {"train": 256, "calibration": 32, "sealed": 64}


@pytest.fixture(scope="module")
def fixtures() -> dict[str, list[dict]]:
    return load_fixtures()


def _groups(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record)
    return grouped


# --------------------------------------------------------------------------- #
# Identity and report gates
# --------------------------------------------------------------------------- #


def test_data_report_passed_and_digest_matches(fixtures) -> None:
    from taiji.internalization import content_digest

    report = json.loads(DATA_REPORT.read_text(encoding="utf-8"))
    assert report["outcome"] == "passed", report.get("failed_checks")
    assert report["namespace"] == "r2_content_binding_v1"
    observed = content_digest(
        [row for split in ("train", "calibration", "sealed") for row in fixtures[split]]
    )
    assert report["corpus_digest"] == observed


# --------------------------------------------------------------------------- #
# Quotas, pairs, uniqueness, holdout
# --------------------------------------------------------------------------- #


def test_quotas_and_pair_structure(fixtures) -> None:
    for split, quota in EXPECTED_GROUPS.items():
        records = fixtures[split]
        groups = _groups(records)
        assert len(records) == quota * 2 * len(GROUP_CLASSES)
        assert len(groups) == quota * len(GROUP_CLASSES)
        for members in groups.values():
            assert len(members) == 2
            assert {m["member"] for m in members} == {"a", "b"}
        counts = {cls: 0 for cls in GROUP_CLASSES}
        for group_id, members in groups.items():
            counts[members[0]["group_class"]] += 1
        assert all(count == quota for count in counts.values())


def test_prefix_and_pair_uniqueness_within_split(fixtures) -> None:
    for split, records in fixtures.items():
        prefixes = [r["prefix"] for r in records]
        pairs = [(r["prefix"], r["response"]) for r in records]
        assert len(set(prefixes)) == len(prefixes), split
        assert len(set(pairs)) == len(pairs), split
        assert len(set(g["group_id"] for g in records)) == len(prefixes) // 2


def test_group_and_word_level_holdout_across_splits(fixtures) -> None:
    train, calibration, sealed = (fixtures[key] for key in ("train", "calibration", "sealed"))
    train_prefixes = {r["prefix"] for r in train}
    calibration_prefixes = {r["prefix"] for r in calibration}
    sealed_prefixes = {r["prefix"] for r in sealed}
    assert not (train_prefixes & calibration_prefixes)
    assert not (train_prefixes & sealed_prefixes)
    assert not (calibration_prefixes & sealed_prefixes)
    # value words are split-disjoint by pool construction; assert on records
    train_values = set(POOLS["train"]["values_single"]) | set(POOLS["train"]["values_double"])
    calibration_values = set(POOLS["calibration"]["values_single"]) | set(
        POOLS["calibration"]["values_double"]
    )
    sealed_values = set(POOLS["sealed"]["values_single"]) | set(POOLS["sealed"]["values_double"])
    assert not (train_values & calibration_values)
    assert not (train_values & sealed_values)
    assert not (calibration_values & sealed_values)
    # objects likewise
    assert not (set(POOLS["train"]["objects"]) & set(POOLS["calibration"]["objects"]))
    assert not (set(POOLS["train"]["objects"]) & set(POOLS["sealed"]["objects"]))
    assert not (set(POOLS["calibration"]["objects"]) & set(POOLS["sealed"]["objects"]))


# --------------------------------------------------------------------------- #
# In-group counterfactual relations and copy masks
# --------------------------------------------------------------------------- #


def test_in_group_answer_relations(fixtures) -> None:
    for split, records in fixtures.items():
        for members in _groups(records).values():
            expected = EXPECTED_IN_GROUP[members[0]["group_class"]]
            same = members[0]["response"] == members[1]["response"]
            assert (expected == "equal") == same, (split, members[0]["group_id"])


def test_copy_mask_semantics(fixtures) -> None:
    for split, records in fixtures.items():
        for record in records:
            mask = record["copy_mask"]
            assert len(mask) == len(record["response"])
            if record["copyable"]:
                assert all(mask)
                material = record["material"]
                assert all(ch in material for ch in record["response"])
            else:
                assert not any(mask)
                assert record["response"] in {"相同", "不同", "是", "否", "未知"}


def test_value_length_and_position_balance(fixtures) -> None:
    for split, records in fixtures.items():
        singles = set(POOLS[split]["values_single"])
        for cls in VALUE_CLASSES:
            groups = [
                members
                for members in _groups(records).values()
                if members[0]["group_class"] == cls
            ]
            single = 0
            double = 0
            for members in groups:
                words = set()
                for member in members:
                    body = split_material_body(split, member["material"])
                    for token in body.replace("；", "，").split("，"):
                        for value in list(singles) + list(POOLS[split]["values_double"]):
                            if f"是{value}" in token or f"不是{value}" in token:
                                words.add(value)
                if words & singles:
                    single += 1
                else:
                    double += 1
            assert single == double, (split, cls, single, double)
        # negation queried-object balance and distractor position balance
        negation = [
            members for members in _groups(records).values()
            if members[0]["group_class"] == "negation_scope"
        ]
        queried_first = 0
        for members in negation:
            facts = {}
            body = split_material_body(split, members[0]["material"])
            for token in body.split("，"):
                for obj in POOLS[split]["objects"]:
                    if token.startswith(obj):
                        facts[obj] = True
                        break
            question_object = next(
                obj
                for obj in POOLS[split]["objects"]
                if MARKERS[split]["head"] + obj in members[0]["question"]
            )
            if next(iter(facts)) == question_object:
                queried_first += 1
        assert queried_first == len(negation) - queried_first
        distractor = [
            members for members in _groups(records).values()
            if members[0]["group_class"] == "distractor_invariant"
        ]
        before = 0
        for members in distractor:
            body = split_material_body(split, members[1]["material"])
            first_object = next(
                obj for obj in POOLS[split]["objects"] if body.startswith(obj)
            )
            question_object = next(
                obj
                for obj in POOLS[split]["objects"]
                if MARKERS[split]["head"] + obj in members[0]["question"]
            )
            if first_object != question_object:
                before += 1
        assert before == len(distractor) - before


def test_length_bounds(fixtures) -> None:
    for split, records in fixtures.items():
        assert max(len(r["prefix"]) for r in records) <= MAX_PREFIX_CHARS
        assert max(len(r["response"]) for r in records) <= MAX_RESPONSE_CHARS
        assert all(r["prefix"] == r["question"] + r["material"] for r in records)


# --------------------------------------------------------------------------- #
# Sealed slices (contract section 3.1: >= 64 copyable answers each)
# --------------------------------------------------------------------------- #


def test_sealed_slice_denominators(fixtures) -> None:
    train_text = "".join(
        r["question"] + r["material"] + r["response"] for r in fixtures["train"]
    )
    train_values = set(POOLS["train"]["values_single"]) | set(POOLS["train"]["values_double"])
    for split in ("calibration", "sealed"):
        copyable = [r for r in fixtures[split] if r["copyable"]]
        unseen_value = [r for r in copyable if r["response"] not in train_values]
        unseen_char = [
            r for r in copyable if any(ch not in train_text for ch in r["response"])
        ]
        assert len(unseen_value) >= 64, (split, len(unseen_value))
        assert len(unseen_char) >= 64, (split, len(unseen_char))
    sealed = json.loads(DATA_REPORT.read_text(encoding="utf-8"))
    detail = sealed["checks"]["copyable_slice_denominators_sealed"]["detail"]
    assert detail["copyable"] == 448
    assert detail["unseen_complete_value"] == 448
    assert detail["unseen_char"] >= 64


# --------------------------------------------------------------------------- #
# Reference key and fixed-policy floor (contract section 3.2)
# --------------------------------------------------------------------------- #


def test_reference_key_solves_every_item(fixtures) -> None:
    for split, records in fixtures.items():
        for record in records:
            assert reference_answer(record) == record["response"], record["id"]


def test_fixed_policies_below_pairwise_floor_on_flip_categories(fixtures) -> None:
    policies = build_policies()
    train = fixtures["train"]
    for name, predict in policies.items():
        scoring = score_records(train, predict)
        macro = scoring["flip_pairwise_macro"]
        assert macro is not None and macro <= 0.25, (name, macro)
        for cls in FLIP_GATE_CLASSES:
            assert scoring["per_class"][cls]["pair_rate"] <= 0.25, (name, cls)


def test_static_baseline_report_passed() -> None:
    report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    assert report["outcome"] == "passed"
    assert report["mode"] == "policies"
    for name, payload in report["results"].items():
        if name == "reference_key":
            for split in ("train", "calibration", "sealed"):
                assert payload[split]["flip_pairwise_macro"] == 1.0
        else:
            assert payload["train"]["flip_pairwise_macro"] <= 0.25
