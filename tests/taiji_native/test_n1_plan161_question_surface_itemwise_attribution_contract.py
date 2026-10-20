from __future__ import annotations

import hashlib

from scripts.training import (
    analyze_taiji_n1_plan161_question_surface_itemwise_attribution as analyzer,
)


def test_offline_attribution_inputs_and_denominators_are_frozen() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256
    report = analyzer.calculate()
    assert report["row_count"] == 2_880
    assert report["pair_count"] == 1_440
    assert len(report["summary_by_branch_seed_kind_template_length"]) == 144
    assert len(report["paired_correctness_by_kind_template_length"]) == 24
    assert report["teacher_free_exact_mismatches"] == 0
    assert report["model_loads"] == 0
    assert report["checkpoint_loads"] == 0
    assert report["model_queries"] == 0
    assert report["parameter_updates"] == 0
    assert report["can_promote"] is False


def test_output_classification_prioritizes_old_revision_value() -> None:
    base = {
        "kind": "revision",
        "answer": "newval",
        "observations_by_length": {
            "8": ["target uses oldval.", "target now uses newval."]
        },
    }
    row = {"raw_hex": b"oldval".hex(), "stop_reason": "boundary", "context_length": 8}
    assert analyzer.output_class(row, base) == "stale_revision_value"


def test_output_classification_separates_abstention_and_direction_tokens() -> None:
    base = {"kind": "lookup", "answer": "matval", "observations_by_length": {"8": ["target uses matval."]}}
    unknown_row = {"raw_hex": b"unknown".hex(), "stop_reason": "boundary", "context_length": 8}
    direction_row = {"raw_hex": b"north".hex(), "stop_reason": "boundary", "context_length": 8}
    assert analyzer.output_class(unknown_row, base) == "literal_unknown"
    assert analyzer.output_class(direction_row, base) == "direction_or_turn_token"
