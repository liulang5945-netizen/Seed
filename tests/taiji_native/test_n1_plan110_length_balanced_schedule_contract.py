from __future__ import annotations

import copy

import pytest

from scripts.training import build_taiji_n1_length_balanced_training_schedule as schedule


def _item(kind: str, answer: str, observations: list[str]) -> dict:
    target = "targetname"
    question = f"material of {target}?" if kind != "transition" else f"direction of {target}?"
    entities = [target, "othername"]
    if kind in ("lookup", "revision", "unknown"):
        entities = [target, "othername"]
    else:
        entities = [target, "othername"]
    return {
        "answer": answer,
        "entities": entities,
        "id": f"parent-{kind}",
        "kind": kind,
        "observations": observations,
        "question": question,
        "split": "train",
    }


@pytest.fixture(scope="module")
def frozen_candidate_schedule():
    return schedule.build_from_frozen_sources()


def test_lookup_and_revision_transform_every_material_consistently():
    mapping = {"abcdef": "qwertyu", "ghijkl": "asdfghj", "mnopqr": "zxcvbnm"}
    lookup = _item("lookup", "abcdef", ["targetname uses abcdef.", "othername uses ghijkl."])
    transformed = schedule.transform_material_item(lookup, 7, mapping)
    assert transformed["answer"] == "qwertyu"
    assert transformed["observations"] == [
        "targetname uses qwertyu.",
        "othername uses asdfghj.",
    ]

    revision = _item(
        "revision",
        "ghijkl",
        ["targetname uses abcdef.", "targetname now uses ghijkl."],
    )
    transformed_revision = schedule.transform_material_item(revision, 7, mapping)
    assert transformed_revision["answer"] == "asdfghj"
    assert transformed_revision["observations"] == [
        "targetname uses qwertyu.",
        "targetname now uses asdfghj.",
    ]


def test_unknown_label_and_transition_inputs_remain_semantically_unchanged():
    mapping = {"abcdef": "qwertyu", "ghijkl": "asdfghj"}
    unknown = _item(
        "unknown",
        "unknown",
        ["othername uses abcdef.", "nearname uses ghijkl."],
    )
    transformed_unknown = schedule.transform_material_item(unknown, 7, mapping)
    assert transformed_unknown["answer"] == "unknown"
    assert transformed_unknown["observations"] == [
        "othername uses qwertyu.",
        "nearname uses asdfghj.",
    ]

    transition = _item("transition", "south", ["targetname faces east.", "targetname turns right."])
    transformed_transition = schedule.transform_material_item(transition, None, mapping)
    assert transformed_transition == transition
    with pytest.raises(ValueError, match="length-neutral"):
        schedule.transform_material_item(transition, 6, mapping)


def test_candidate_value_map_is_deterministic_injective_and_disjoint():
    items = [
        _item("lookup", "abcdef", ["targetname uses abcdef.", "othername uses ghijkl."]),
        _item("lookup", "ghijkl", ["targetname uses ghijkl.", "othername uses abcdef."]),
    ]
    first = schedule._collect_disjoint_values(
        items, excluded_names={"blocked"}, excluded_values={"forbidden"}, seed=9001
    )
    second = schedule._collect_disjoint_values(
        items, excluded_names={"blocked"}, excluded_values={"forbidden"}, seed=9001
    )
    assert first == second
    assert set(first) == {"abcdef", "ghijkl"}
    assert len(set(first.values())) == len(first)
    assert all(len(value) == 7 for value in first.values())
    assert not (set(first.values()) & {"blocked", "forbidden", "abcdef", "ghijkl"})


def test_real_train_only_candidate_schedule_has_complete_50_50_raw_and_hard_balance(
    frozen_candidate_schedule,
):
    candidate, checks = frozen_candidate_schedule
    assert checks["accepted"] is True
    assert checks["parameter_updates"] == checks["model_queries"] == checks["checkpoint_loads"] == 0
    assert len(candidate["items"]) == len(candidate["raw_views"]) == 3072
    assert candidate["summary"]["raw_material_length_counts"] == {
        "lookup/len6": 384,
        "lookup/len7": 384,
        "revision/len6": 384,
        "revision/len7": 384,
        "transition/lenNone": 768,
        "unknown/len6": 384,
        "unknown/len7": 384,
    }
    assert (
        candidate["summary"]["hard_material_length_counts"]
        == candidate["summary"]["raw_material_length_counts"]
    )
    assert len(candidate["value_map"]) > 0
    assert len(set(candidate["value_map"].values())) == len(candidate["value_map"])
    assert checks["exclusion_provenance"]["v7_v8_confirmation_payload_fields_decoded"] is False


def test_contract_audit_rejects_raw_hard_length_drift(frozen_candidate_schedule):
    candidate, _ = frozen_candidate_schedule
    damaged = copy.deepcopy(candidate)
    current = damaged["raw_views"][0]["material_length_chars"]
    damaged["raw_views"][0]["material_length_chars"] = 6 if current == 7 else 7
    audit = schedule.audit_schedule(damaged, schedule._source_bundle()[0], set())
    assert audit["accepted"] is False
    assert any(
        "raw view and frozen length assignment disagree" in error for error in audit["errors"]
    )
