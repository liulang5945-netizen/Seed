from __future__ import annotations

import copy

import pytest

from scripts.training import build_taiji_n1_v24_abstention_after_n146 as builder
from scripts.training import eval_taiji_n1_v24_abstention_after_n146 as evaluator


def _excluded() -> tuple[set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.n144_builder.DIRECTIONS}
    return names, words


def test_v24_face_is_deterministic_and_balances_matched_conditions() -> None:
    excluded_names, excluded_words = _excluded()
    first = builder.build_face(excluded_names, excluded_words)
    second = builder.build_face(excluded_names, excluded_words)

    assert first == second
    checks = builder.validate_face(first, excluded_names, excluded_words)
    assert checks["accepted"] is True
    assert checks["pairs"] == 40
    assert checks["items"] == 80
    assert checks["pairs_by_target_start"] == {0: 8, 8: 8, 16: 8, 24: 8, 31: 8}
    assert checks["known_answer_length_counts"] == {6: 20, 7: 20}
    assert checks["unique_entities"] == 1_320
    assert checks["unique_materials"] == 1_320


def test_known_unknown_pairs_differ_only_in_the_target_slot() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    by_pair: dict[str, dict[str, dict]] = {}
    for item in face["episodes"]:
        by_pair.setdefault(item["pair_id"], {})[item["pair_condition"]] = item

    for pair in by_pair.values():
        known = pair["known"]
        unknown = pair["unknown"]
        start = known["target_start_index"]
        target = known["entities"][0]
        assert unknown["entities"][0] == target
        assert known["question"] == unknown["question"]
        assert known["observations"][start] == f"{target} uses {known['answer']}."
        assert unknown["observations"][start].startswith(f"{unknown['entities'][-1]} uses ")
        assert all(
            known["observations"][index] == unknown["observations"][index]
            for index in range(builder.OBSERVATIONS)
            if index != start
        )
        assert all(not row.startswith(f"{target} ") for row in unknown["observations"])


def test_face_validator_rejects_a_changed_shared_context_row() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    changed = copy.deepcopy(face)
    changed["episodes"][1]["observations"][1] = "zzzzzzzzzz uses badval."

    with pytest.raises(builder.Refuse, match="paired contexts differ outside target slot"):
        builder.validate_face(changed, excluded_names, excluded_words)


def test_source_union_adds_the_consumed_v23_face_without_overlap() -> None:
    entities, words, source_hashes, summary = builder._collect_source_union()

    assert len(entities) == 36_802
    assert len(words) == 13_968
    assert summary["n1_144_entity_exclusion_count"] == 31_722
    assert summary["n1_144_word_exclusion_count"] == 10_128
    assert summary["consumed_v23_entity_count"] == 5_080
    assert summary["consumed_v23_material_count"] == 3_840
    assert summary["entity_overlap_with_previous_union"] == 0
    assert summary["word_overlap_with_previous_union"] == 0
    assert builder.n144_builder._verify_hashes(source_hashes)


def test_plan_seed_positions_and_long_context_contract_are_frozen() -> None:
    assert builder.sha256(builder.PLAN) == builder.PLAN_SHA256
    assert builder.SEED == 124
    assert builder.PAIRS == 40
    assert builder.OBSERVATIONS == 32
    assert builder.POSITIONS == (0, 8, 16, 24, 31)


def test_endpoint_registry_reuses_only_the_twelve_frozen_comparison_bundles() -> None:
    specs = evaluator._endpoint_specs()

    assert len(specs) == 12
    assert {spec["source"] for spec in specs} == {"N1-132", "N1-140", "N1-141"}
    assert all(
        spec["path"].is_file() and builder.sha256(spec["path"]) == spec["sha256"] for spec in specs
    )


def test_paired_summary_separates_known_correctness_from_unknown_abstention() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    endpoint_id = "N1-141/seed11/candidate"
    rows = []
    for item in face["episodes"]:
        pair = int(item["pair_id"].rsplit("-", 1)[-1])
        known = item["pair_condition"] == "known"
        if known and pair < 30:
            output = item["answer"]
            exact = True
        elif known:
            output = "unknown"
            exact = False
        elif pair < 38:
            output = "unknown"
            exact = True
        else:
            output = "wrongv"
            exact = False
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "id": item["id"],
                "output_hex": output.encode("utf-8").hex(),
                "free_exact": exact,
            }
        )

    summary = evaluator._paired_summary(rows, face)

    assert summary[endpoint_id]["pairs"] == 40
    assert summary[endpoint_id]["known_exact"] == 30
    assert summary[endpoint_id]["known_abstain_literal"] == 10
    assert summary[endpoint_id]["unknown_abstain_literal"] == 38
    assert summary[endpoint_id]["unknown_false_answer"] == 2
    assert summary[endpoint_id]["paired_outcomes"] == {
        "known_abstains_unknown_abstain": 8,
        "known_exact_unknown_abstain": 30,
        "other_pair_outcome": 2,
    }
