from __future__ import annotations

import pytest

from scripts.training import audit_taiji_n1_v10_output_geometry as audit


def test_levenshtein_distance_is_character_level_and_symmetric():
    assert audit.levenshtein_distance("kitten", "sitting") == 3
    assert audit.levenshtein_distance("abc", "abc") == 0
    assert audit.levenshtein_distance("", "seed") == 4
    assert audit.levenshtein_distance("ab", "ba") == 2


def test_nearest_material_values_preserve_target_and_distractor_roles():
    item = {
        "answer": "current",
        "kind": "revision",
        "question": "material of target?",
        "observations": [
            "target uses stale.",
            "distractor uses observed.",
            "target now uses current.",
        ],
    }
    stale = audit.nearest_observed_values("stale", item)
    assert stale["distance"] == 0
    assert stale["values"] == [
        {"entity": "target", "role": "stale_target_material", "value": "stale"}
    ]

    distractor = audit.nearest_observed_values("observed", item)
    assert distractor["distance"] == 0
    assert distractor["values"] == [
        {"entity": "distractor", "role": "other_entity_observed_value", "value": "observed"}
    ]


def test_transition_candidates_use_faces_and_turns_values():
    item = {
        "answer": "south",
        "kind": "transition",
        "question": "direction of target?",
        "observations": [
            "target faces east.",
            "target turns right.",
            "distractor faces west.",
        ],
    }
    target_value = audit.nearest_observed_values("east", item)
    assert target_value["distance"] == 0
    assert target_value["values"] == [
        {"entity": "target", "role": "target_entity_relation_value", "value": "east"}
    ]
    other_value = audit.nearest_observed_values("west", item)
    assert other_value["distance"] == 0
    assert other_value["values"] == [
        {"entity": "distractor", "role": "other_entity_observed_value", "value": "west"}
    ]


def test_matched_family_requires_every_position_and_constant_template_content():
    items = [
        {
            "answer": "answer",
            "question": "material of target?",
            "entities": ["target", "distractor"],
            "observations": ["target uses answer.", f"filler{i} uses value."],
            "target_row_position_1based": position,
        }
        for position, i in enumerate(range(12), start=1)
    ]
    # The observations above intentionally differ, so this is not a valid matched family.
    with pytest.raises(ValueError, match="observation multiset"):
        audit.validate_matched_family(items, list(range(1, 13)))

    shared = ["target uses answer.", "distractor uses value."]
    valid = [
        {
            "answer": "answer",
            "question": "material of target?",
            "entities": ["target", "distractor"],
            "observations": shared if position % 2 else list(reversed(shared)),
            "target_row_position_1based": position,
        }
        for position in range(1, 13)
    ]
    audit.validate_matched_family(valid, list(range(1, 13)))


def test_frozen_face_and_result_are_fully_bound_and_analyze_all_rows():
    face = audit.read_json(audit.FACE)
    result = audit.read_json(audit.RESULT)
    items = audit._validate_face_and_result(face, result)
    rows, templates = audit._analyze_rows(items, result["rows"])
    assert len(items) == 248
    assert len(rows) == 2232
    assert len(templates) == 180
    assert all(row["answer_edit_distance"] == 0 for row in rows if row["free_exact"])
    assert all(row["answer_edit_distance"] > 0 for row in rows if not row["free_exact"])
    assert all(row["position_variants"] in {11, 12} for row in templates)


def test_frozen_face_validation_rejects_missing_result_rows():
    face = audit.read_json(audit.FACE)
    result = audit.read_json(audit.RESULT)
    result["rows"].pop()
    with pytest.raises(ValueError, match="complete 2,232-row"):
        audit._validate_face_and_result(face, result)
