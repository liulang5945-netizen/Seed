from __future__ import annotations

import copy

import pytest

from scripts.training import build_taiji_n1_v26_abstention_length_after_n148 as builder
from scripts.training import eval_taiji_n1_v26_abstention_length_after_n148 as evaluator


def _excluded() -> tuple[set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.n148_builder.n147_builder.n144_builder.DIRECTIONS}
    return names, words


def test_v26_face_is_deterministic_nested_and_balanced_across_lengths() -> None:
    excluded_names, excluded_words = _excluded()
    first = builder.build_face(excluded_names, excluded_words)
    second = builder.build_face(excluded_names, excluded_words)
    checks = builder.validate_face(first, excluded_names, excluded_words)

    assert first == second
    assert checks["accepted"] is True
    assert checks["logical_pairs"] == 40
    assert checks["items"] == 240
    assert checks["items_by_length"] == {"8": 80, "16": 80, "32": 80}
    assert checks["known_answer_length_counts"] == {6: 20, 7: 20}
    assert checks["unique_entities"] == 1_320
    assert checks["unique_materials"] == 1_320


def test_face_validator_rejects_non_nested_contexts() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    changed = copy.deepcopy(face)
    item = next(
        row
        for row in changed["episodes"]
        if row["context_length"] == 16 and row["pair_condition"] == "known"
    )
    logical_id = item["logical_pair_id"]
    for row in changed["episodes"]:
        if row["logical_pair_id"] == logical_id and row["context_length"] == 16:
            row["observations"][5] = "zzzzzzzzzz uses wrongv."

    with pytest.raises(builder.Refuse, match="contexts are not nested"):
        builder.validate_face(changed, excluded_names, excluded_words)


def test_diagnostic_validator_requires_exact_nested_lengths_and_split() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    validated = evaluator.validate_diagnostic_items(face["episodes"])

    assert validated == {
        "items": 240,
        "logical_pairs": 40,
        "pair_groups": 120,
        "items_by_length": {8: 80, 16: 80, 32: 80},
        "conditions": {"known": 120, "unknown": 120},
    }

    invalid = [dict(item) for item in face["episodes"]]
    invalid[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="diagnostic split required"):
        evaluator.validate_diagnostic_items(invalid)


def test_measure_rejects_non_diagnostic_items_before_model_forward() -> None:
    class UntouchableModel:
        def __getattr__(self, _name: str):
            raise AssertionError("model methods must not run before the diagnostic split guard")

    item = {"split": "holdout", "answer": "abcdef"}
    with pytest.raises(ValueError, match="diagnostic split and frozen position arm"):
        evaluator.measure_diagnostic(UntouchableModel(), item, "position", "bad-split-probe")


def test_nested_length_summary_reports_known_abstention_and_unknown_specificity() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    endpoint_id = "N1-141/seed11/candidate"
    rows = []
    for item in face["episodes"]:
        length = item["context_length"]
        known = item["pair_condition"] == "known"
        if known and length == 8:
            output, exact = item["answer"], True
        elif known:
            output, exact = "unknown", False
        else:
            output, exact = "unknown", True
        rows.append({"endpoint_id": endpoint_id, "id": item["id"], "output_hex": output.encode().hex(), "free_exact": exact})

    summary = evaluator._paired_summary(rows, face)
    by_length = summary[endpoint_id]["by_context_length"]

    assert by_length["8"]["known_exact"] == 40
    assert by_length["8"]["unknown_abstain_literal"] == 40
    assert by_length["16"]["known_exact"] == 0
    assert by_length["16"]["known_abstain_literal"] == 40
    assert by_length["32"]["known_exact"] == 0
    assert all(by_length[str(length)]["unknown_false_answer"] == 0 for length in (8, 16, 32))
    assert len(summary[endpoint_id]["logical_pair_curves"]) == 40


def test_face_source_union_excludes_the_full_consumed_v25_face() -> None:
    entities, words, hashes, summary = builder._collect_source_union()

    assert len(entities) == 39_442
    assert len(words) == 16_608
    assert summary["n1_148_entity_exclusion_count"] == 38_122
    assert summary["n1_148_word_exclusion_count"] == 15_288
    assert summary["consumed_v25_entity_count"] == 1_320
    assert summary["consumed_v25_material_count"] == 1_320
    assert summary["entity_overlap_with_previous_union"] == 0
    assert summary["word_overlap_with_previous_union"] == 0
    assert builder.n148_builder.n147_builder.n144_builder._verify_hashes(hashes)


def test_plan_and_frozen_endpoint_registry_are_bound() -> None:
    assert builder.sha256(builder.PLAN) == builder.PLAN_SHA256
    specs = evaluator._endpoint_specs()
    assert len(specs) == 12
    assert {spec["source"] for spec in specs} == {"N1-132", "N1-140", "N1-141"}
