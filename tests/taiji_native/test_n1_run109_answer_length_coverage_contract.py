from __future__ import annotations

import pytest

from scripts.training import audit_taiji_n1_answer_length_coverage as audit


def test_frozen_schedule_is_train_only_and_kind_condition_counts_match():
    schedule = audit.read_json(audit.SCHEDULE)
    entries = audit._validate_schedule(schedule)
    assert len(entries) == 3072
    assert all(entry["item"]["split"] == "train" for entry in entries)
    assert all(
        entry["second_question"] is None or entry["second_question"]["split"] == "train"
        for entry in entries
    )


def test_length_histograms_separate_known_material_from_unknown_label():
    schedule = audit.read_json(audit.SCHEDULE)
    face = audit.read_json(audit.FACE)
    entries = audit._validate_schedule(schedule)
    episodes = audit._validate_face(face)
    hist = audit._histograms(entries, episodes)
    support = hist["known_answer_length_support"]
    assert support["lookup"] == {
        "train_schedule_item_answer_lengths": [6],
        "diagnostic_face_answer_lengths": [6, 7],
        "diagnostic_lengths_absent_from_train_schedule": [7],
    }
    assert support["revision"] == {
        "train_schedule_item_answer_lengths": [6],
        "diagnostic_face_answer_lengths": [6, 7],
        "diagnostic_lengths_absent_from_train_schedule": [7],
    }
    unknown = next(
        row
        for row in hist["schedule_by_kind_source_field"]
        if row["kind"] == "unknown" and row["source_field"] == "item"
    )
    assert unknown["answer_length_counts"] == {"7": 768}
    assert "abstention label" in hist["unknown_label_note"]


def test_nontrain_payload_is_rejected_before_length_counting():
    schedule = audit.read_json(audit.SCHEDULE)
    schedule["items"][0]["item"]["split"] = "dev"
    with pytest.raises(ValueError, match="non-train payload"):
        audit._validate_schedule(schedule)


def test_diagnostic_face_requires_frozen_kind_counts_and_split():
    face = audit.read_json(audit.FACE)
    face["episodes"][0]["split"] = "holdout"
    with pytest.raises(ValueError, match="non-diagnostic"):
        audit._validate_face(face)


def test_report_histograms_preserve_each_source_denominator():
    schedule = audit.read_json(audit.SCHEDULE)
    face = audit.read_json(audit.FACE)
    hist = audit._histograms(audit._validate_schedule(schedule), audit._validate_face(face))
    for row in hist["schedule_by_kind_source_field"]:
        assert row["denominator"] == sum(map(int, row["answer_length_counts"].values()))
    for row in hist["diagnostic_face_by_kind"]:
        assert row["denominator"] == sum(map(int, row["answer_length_counts"].values()))
