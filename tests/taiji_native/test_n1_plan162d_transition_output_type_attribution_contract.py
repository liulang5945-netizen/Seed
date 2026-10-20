from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from scripts.training import (
    analyze_taiji_n1_plan162d_transition_output_type_attribution as analyzer,
)


def test_plan_digest_matches_frozen_source() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256
    assert hashlib.sha256(analyzer.N162C_CLOSEOUT.read_bytes()).hexdigest() == analyzer.N162C_CLOSEOUT_SHA256
    assert hashlib.sha256(analyzer.N162C_REPORT.read_bytes()).hexdigest() == analyzer.N162C_REPORT_SHA256


def test_transition_target_reconstruction_uses_face_turn_rule() -> None:
    item = {
        "id": "train-002",
        "kind": "transition",
        "split": "train",
        "answer": "east",
        "observations": ["entity faces north.", "entity turns right."],
    }
    assert analyzer._input_transition(item) == ("north", "right", "east")
    changed = {**item, "answer": "west"}
    with pytest.raises(analyzer.Refuse, match="target formula mismatch"):
        analyzer._input_transition(changed)


def test_error_categories_separate_directional_failure_modes() -> None:
    assert analyzer._classify_output("65617374", "boundary", "east", "north", "right") == "correct_exact"
    assert analyzer._classify_output("6e6f727468", "boundary", "east", "north", "right") == "initial_direction_copy"
    assert analyzer._classify_output("77657374", "boundary", "east", "north", "right") == "inverse_turn_answer"
    assert analyzer._classify_output("736f757468", "boundary", "east", "north", "right") == "opposite_initial_direction"
    assert analyzer._classify_output("756e6b6e6f776e", "boundary", "east", "north", "right") == "literal_unknown"
    assert analyzer._classify_output("", "empty", "east", "north", "right") == "empty"
    assert analyzer._classify_output("ff", "boundary", "east", "north", "right") == "non_utf8_bytes"


def test_endpoint_binding_checks_the_frozen_view_denominator() -> None:
    rows = []
    for seed, arm in analyzer.ENDPOINTS:
        for view, count in (("original", 48), ("rename", 48), ("factorial_original", 192), ("factorial_rename", 192)):
            rows.extend({"seed": seed, "arm": "position", "evaluation_set": view} for _ in range(count))
    bound = analyzer._endpoint_bind(rows)
    assert len(bound) == 2_880
    assert Counter((row["seed"], row["experiment_arm"]) for row in bound)[(11, "surface_control")] == 480
    assert Counter((row["seed"], row["experiment_arm"]) for row in bound)[(13, "surface_augmented")] == 480


def test_outputs_remain_inside_repository() -> None:
    for path in (analyzer.PLAN, analyzer.N162B_RESULT, analyzer.PROGRESS, analyzer.N162C_REPORT, analyzer.REPORT):
        assert Path(path).resolve().is_relative_to(analyzer.ROOT)
