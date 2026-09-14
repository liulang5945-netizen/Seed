from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r6_formal_aggregate import (
    _learning_formal_ready,
)
from scripts.training.eval_taiji_m4v2_r6_formal_single_cell import (
    _summarize_k_measurement,
)


def test_feedback_admission_does_not_change_task_success() -> None:
    admitted = _summarize_k_measurement(
        runner_passed=True,
        task_executed=True,
        outcome_success=True,
        projection_accepted=True,
        feedback_admitted=True,
        lesion_k3=False,
        training_update_steps=0,
        candidate_training_performed=False,
    )
    not_admitted = _summarize_k_measurement(
        runner_passed=True,
        task_executed=True,
        outcome_success=True,
        projection_accepted=True,
        feedback_admitted=False,
        lesion_k3=False,
        training_update_steps=0,
        candidate_training_performed=False,
    )

    assert admitted["task_success_rate"] == 1.0
    assert not_admitted["task_success_rate"] == 1.0
    assert admitted["feedback_admitted"] is True
    assert not_admitted["feedback_admitted"] is False


def test_unattempted_task_is_not_imputed_as_zero_success() -> None:
    measurement = _summarize_k_measurement(
        runner_passed=True,
        task_executed=False,
        outcome_success=False,
        projection_accepted=False,
        feedback_admitted=False,
        lesion_k3=False,
        training_update_steps=0,
        candidate_training_performed=False,
    )

    assert measurement["task_success_rate"] is None
    assert measurement["task_measurement_status"] == "not_attempted"
    assert measurement["learning_eligible"] is False


def test_zero_update_canary_is_not_learning_evidence() -> None:
    measurement = _summarize_k_measurement(
        runner_passed=True,
        task_executed=True,
        outcome_success=True,
        projection_accepted=True,
        feedback_admitted=True,
        lesion_k3=False,
        training_update_steps=0,
        candidate_training_performed=False,
    )

    assert measurement["task_success_rate"] == 1.0
    assert measurement["learning_update_applied"] is False
    assert measurement["learning_eligible"] is False


def test_wiring_canary_cannot_enter_learning_formal_gate() -> None:
    assert (
        _learning_formal_ready(
            {
                "run_kind": "wiring-canary",
                "training_performed": False,
                "candidate_training_performed": False,
            }
        )
        is False
    )
    assert (
        _learning_formal_ready(
            {
                "run_kind": "learning-formal",
                "training_performed": True,
                "candidate_training_performed": True,
            }
        )
        is True
    )
