from __future__ import annotations

import copy

import pytest

from scripts.training import audit_taiji_n1_v21_face_semantics as audit
from scripts.training import build_taiji_n1_v21_confirmation_after_n141 as builder


def _face() -> tuple[dict, set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.DIRECTIONS}
    return builder.build_face(names, words), names, words


def test_semantic_audit_recomputes_targets_unknown_and_material_inventory() -> None:
    face, names, words = _face()

    result = audit.audit_face(face, {"excluded_entities": names, "excluded_word_values": words})

    assert result["items"] == 160
    assert result["target_row_count_by_kind"] == {
        "lookup": 40,
        "revision": 80,
        "transition": 80,
        "unknown": 0,
    }
    assert result["material_target_answer_count"] == 80
    assert result["material_revision_stale_value_count"] == 40
    assert result["material_filler_value_count"] == 1800
    assert result["material_value_count"] == 1920
    assert result["unknown_literal_answer_counts"] == {
        "confirm/unknown": 20,
        "sacrifice/unknown": 20,
    }
    assert all(count == 5 for count in result["transition_direction_counts"].values())
    assert all(count == 10 for count in result["transition_turn_counts"].values())


def test_semantic_audit_rejects_wrong_transition_answer() -> None:
    face, names, words = _face()
    item = next(row for row in face["episodes"] if row["kind"] == "transition")
    item["answer"] = "east" if item["answer"] != "east" else "west"

    with pytest.raises(ValueError, match="transition answer does not follow facts"):
        audit.audit_face(face, {"excluded_entities": names, "excluded_word_values": words})


def test_semantic_audit_rejects_target_row_index_mismatch() -> None:
    face, names, words = _face()
    changed = copy.deepcopy(face)
    item = next(row for row in changed["episodes"] if row["kind"] == "lookup")
    item["target_fact_indices"] = [item["target_fact_indices"][0] + 1]

    with pytest.raises(ValueError, match="target fact index ledger mismatch"):
        audit.audit_face(changed, {"excluded_entities": names, "excluded_word_values": words})
