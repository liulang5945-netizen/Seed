from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.training import audit_taiji_n1_plan162f_transcript_integrity as auditor
from scripts.training import (
    run_taiji_n1_plan162f_transition_template_crossover_diagnostic as engine,
)


def test_plan_and_prior_runner_drift_are_frozen() -> None:
    assert hashlib.sha256(auditor.PLAN.read_bytes()).hexdigest() == auditor.PLAN_SHA256
    assert auditor.F_RUNNER_SHA_AT_COMPLETION != auditor.F_CURRENT_RUNNER_SHA
    assert hashlib.sha256(auditor.F_ENGINE.read_bytes()).hexdigest() == auditor.F_CURRENT_RUNNER_SHA


def test_expected_query_keyspace_preserves_nine_endpoints_and_two_surfaces() -> None:
    endpoints = [{"lineage": lineage, "seed": seed} for lineage, seed in (("n155a_candidate", 11), ("surface_control", 11), ("surface_augmented", 11))]
    cases = {
        ("original", "train-002", "original", "train-002:original"): {
            "view": "original",
            "parent_id": "train-002",
            "condition": "original",
            "case_id": "train-002:original",
            "item": {"question": "direction of erifmmmlgg?"},
            "expected_answer": "east",
            "initial_direction": "north",
            "turn": "right",
        }
    }
    expected = auditor._expected_queries(cases, endpoints)
    assert len(expected) == 6
    assert {row["question"] for row in expected.values()} == {
        "which direction does erifmmmlgg face after turning?",
        "what direction is erifmmmlgg facing after its turn?",
    }


def test_transcript_verifier_recomputes_raw_answer_and_progress_pair() -> None:
    expected = {
        ("n155a_candidate", 11, "original", "train-002", "original", "train-002:original", template_id): {
            "question": template.format(entity="erifmmmlgg"),
            "expected_answer": "east",
            "initial_direction": "north",
            "turn": "right",
        }
        for template_id, template in engine.TEMPLATES.items()
    }
    rows = []
    for template_id, values in (("direction_after_turning", "which direction does erifmmmlgg face after turning?"), ("direction_after_turn", "what direction is erifmmmlgg facing after its turn?")):
        rows.append({
            "lineage": "n155a_candidate",
            "seed": 11,
            "view": "original",
            "parent_id": "train-002",
            "condition": "original",
            "case_id": "train-002:original",
            "question_template_id": template_id,
            "question": values,
            "expected_answer": "east",
            "initial_direction": "north",
            "turn": "right",
            "raw_hex": "65617374",
            "output_hex": "65617374",
            "stop_reason": "boundary",
            "free_exact": True,
            "teacher_exact": True,
            "output_type": "correct_exact",
            "original_question_raw_hex": "65617374",
            "original_question_free_exact": True,
            "original_question_output_type": "correct_exact",
            "id": "train-002",
            "kind": "transition",
            "split": "train",
            "context_length": None,
            "answer_length": 4,
            "teacher_tokens_correct": 5,
            "teacher_token_count": 5,
            "generated_steps": 5,
        })
    paired_summary, paired = engine._summarize(rows)
    progress = [{"event": "transition_template_query", **row} for row in rows]
    checked = auditor._validate_transcript({"output_rows": rows, "summary_by_endpoint_view_condition_template": paired_summary, "paired_template_type_transitions": paired}, progress, expected)
    assert checked["result_progress_rows_match"] is True
    assert checked["output_types_recomputed"] == 2


def test_outputs_are_repository_local() -> None:
    for path in (auditor.PLAN, auditor.F_PLAN, auditor.F_RESULT, auditor.F_PROGRESS, auditor.REPORT):
        assert Path(path).resolve().is_relative_to(auditor.ROOT)
