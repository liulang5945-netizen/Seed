from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.training import build_taiji_n1_plan162_question_surface_schedule_and_face as builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as runner


def test_repair_addendum_and_original_failure_receipt_are_bound() -> None:
    assert hashlib.sha256(runner.REPAIR_ADDENDUM.read_bytes()).hexdigest() == runner.REPAIR_ADDENDUM_SHA256
    assert hashlib.sha256(runner.ORIGINAL_RESULT.read_bytes()).hexdigest() == runner.ORIGINAL_RESULT_SHA256
    assert hashlib.sha256(runner.ORIGINAL_ATTEMPT.read_bytes()).hexdigest() == runner.ORIGINAL_ATTEMPT_SHA256
    assert hashlib.sha256(runner.ORIGINAL_PROGRESS.read_bytes()).hexdigest() == runner.ORIGINAL_PROGRESS_SHA256
    assert hashlib.sha256(runner.ORIGINAL_PARTIAL_CHECKPOINT.read_bytes()).hexdigest() == runner.ORIGINAL_PARTIAL_SHA256


def test_original_question_fallback_uses_captured_parser_after_patch() -> None:
    original = n132._query_subject
    n132._query_subject = runner._question_subject
    try:
        assert n132._query_subject("material of erifmmmlgg?") == "erifmmmlgg"
        assert runner._question_subject("direction of qwertyuiop?") == "qwertyuiop"
    finally:
        n132._query_subject = original


def test_all_four_augmented_training_surfaces_parse_without_recursion() -> None:
    for template in builder.TRAINING_TEMPLATES.values():
        question = template.format(entity="erifmmmlgg")
        assert runner._question_subject(question) == "erifmmmlgg"


def test_corrected_attempt_paths_preserve_first_attempt_evidence() -> None:
    assert runner.ATTEMPT != runner.ORIGINAL_ATTEMPT
    assert runner.PROGRESS != runner.ORIGINAL_PROGRESS
    assert runner.RESULT != runner.ORIGINAL_RESULT
    assert runner.CHECKPOINT_ROOT != builder.CHECKPOINT_ROOT
    assert runner.ZERO_PREFLIGHT != runner.ORIGINAL_ZERO_PREFLIGHT
    assert runner.ORIGINAL_PARTIAL_CHECKPOINT.is_file()


def test_budget_accounts_for_the_zero_update_first_attempt() -> None:
    assert runner.PRIOR_FAILED_STAGE_CPU_SECONDS == 270.4375
    assert runner.PRIOR_LINKED_CPU_SECONDS + runner.PRIOR_FAILED_STAGE_CPU_SECONDS == 11_328.78125
    assert runner.CPU_LIMIT_SECONDS == 12_000.0
    assert runner.LINKED_CPU_LIMIT_SECONDS == 24_000.0
    assert runner.CPU_LIMIT_SECONDS - runner.PRIOR_FAILED_STAGE_CPU_SECONDS == 11_729.5625


def test_new_execution_artifacts_remain_inside_repository() -> None:
    for path in (runner.REPAIR_ADDENDUM, runner.REPAIR_TEST, runner.CHECKPOINT_ROOT, runner.ATTEMPT, runner.PROGRESS, runner.RESULT):
        assert Path(path).resolve().is_relative_to(builder.ROOT)
