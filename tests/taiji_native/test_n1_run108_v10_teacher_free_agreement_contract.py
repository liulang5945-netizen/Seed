from __future__ import annotations

import pytest

from scripts.training import audit_taiji_n1_v10_teacher_free_agreement as audit


def test_paired_cells_are_mutually_exclusive():
    assert audit._paired_cell(True, True) == "both_exact"
    assert audit._paired_cell(True, False) == "free_only_exact"
    assert audit._paired_cell(False, True) == "teacher_only_exact"
    assert audit._paired_cell(False, False) == "both_wrong"


def test_summary_conserves_joint_denominator():
    rows = [
        {"paired_cell": "both_exact", "free_exact": True, "teacher_exact": True},
        {"paired_cell": "free_only_exact", "free_exact": True, "teacher_exact": False},
        {"paired_cell": "teacher_only_exact", "free_exact": False, "teacher_exact": True},
        {"paired_cell": "both_wrong", "free_exact": False, "teacher_exact": False},
    ]
    summary = audit._summary(rows)
    assert summary["denominator"] == 4
    assert sum(summary[cell] for cell in audit.PAIRED_CELLS) == 4
    assert summary["disagreements"] == 2
    assert summary["free_exact"] == summary["teacher_exact"] == 2


def test_stored_v10_rows_have_complete_teacher_free_agreement():
    face = audit.read_json(audit.FACE)
    result = audit.read_json(audit.RESULT)
    rows = audit._validate_and_collect(face, result)
    summary = audit._summary(rows)
    assert len(rows) == 2232
    assert summary == {
        "denominator": 2232,
        "both_exact": 981,
        "free_only_exact": 0,
        "teacher_only_exact": 0,
        "both_wrong": 1251,
        "free_exact": 981,
        "teacher_exact": 981,
        "disagreements": 0,
    }
    assert sum(row["paired_cell"] == "both_wrong" for row in rows) == 1251


def test_frozen_face_result_validation_rejects_missing_or_mismatched_rows():
    face = audit.read_json(audit.FACE)
    result = audit.read_json(audit.RESULT)
    result["rows"].pop()
    with pytest.raises(ValueError, match="complete 2,232-row"):
        audit._validate_and_collect(face, result)

    result = audit.read_json(audit.RESULT)
    result["rows"][0]["teacher_exact"] = "true"
    with pytest.raises(ValueError, match="not a boolean"):
        audit._validate_and_collect(face, result)


def test_marginal_strata_retain_all_denominators():
    rows = [
        {
            "seed": 11,
            "arm": "position",
            "kind": "lookup",
            "material_answer_length_chars": 6,
            "target_row_position_1based": 1,
            "free_exact": True,
            "teacher_exact": True,
            "paired_cell": "both_exact",
        },
        {
            "seed": 11,
            "arm": "position",
            "kind": "lookup",
            "material_answer_length_chars": 6,
            "target_row_position_1based": 2,
            "free_exact": False,
            "teacher_exact": False,
            "paired_cell": "both_wrong",
        },
    ]
    strata = audit._strata(rows, ("seed", "arm", "kind", "material_answer_length_chars"))
    assert len(strata) == 1
    assert strata[0]["denominator"] == 2
    assert strata[0]["both_exact"] == 1
    assert strata[0]["both_wrong"] == 1
