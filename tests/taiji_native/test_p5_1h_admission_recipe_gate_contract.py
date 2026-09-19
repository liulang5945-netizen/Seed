"""P5.1h admission recipe gate contract (zero-training gates).

Contract: plans/reference/M5_P5_1H_ADMISSION_PACKAGE_DRAFT_20260919.md §6

Pins the calibration runner's decision surface WITHOUT running any
consolidation: the four pre-registered lines as a pure function, the
training-budget gate that refuses the sweep before touching any data, and
(discoverable data provided) the independent a-gate slice chaining after the
P5.1g a-gate with no overlap and in-vocabulary pruning.
"""

from __future__ import annotations

import pytest

from scripts.training.eval_taiji_p5_1h_admission_recipe_gate import (
    FROZEN_TICK_MAJORITY,
    INDEPENDENT_MARGIN,
    RETENTION_LINE,
    evaluate_lines,
    run_sweep,
)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _metrics(**overrides):
    base = dict(
        retention_accuracy=0.52,
        independent_accuracy=0.60,
        semantic_retention_before=1.0,
        semantic_retention_after=0.0004,
        lesion_accuracy=0.0,
        roundtrip_preserved=True,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Gate 1: the four pre-registered lines as a pure decision function
# --------------------------------------------------------------------------- #


def test_line_evaluation_all_pass_and_thresholds() -> None:
    verdict = evaluate_lines(**_metrics())
    assert verdict["all_pass"] is True
    assert verdict["thresholds"]["retention_line"] == RETENTION_LINE == 0.5
    assert verdict["thresholds"]["independent_accuracy_line"] == pytest.approx(
        FROZEN_TICK_MAJORITY + INDEPENDENT_MARGIN
    )  # 0.517: 0.367 baseline + 0.15 margin


def test_line_evaluation_each_failure_is_detected() -> None:
    assert evaluate_lines(**_metrics(retention_accuracy=0.49))["lines"]["L1_retention"] is False
    assert (
        evaluate_lines(**_metrics(independent_accuracy=0.51))["lines"]["L2_independent_transfer"]
        is False
    )  # 0.51 - 0.367 = 0.143 < 0.15
    assert (
        evaluate_lines(**_metrics(semantic_retention_after=1.1))["lines"]["L3_semantic_retention"]
        is False
    )
    assert (
        evaluate_lines(**_metrics(lesion_accuracy=0.2))["lines"]["L4_lesion_roundtrip"] is False
    )
    assert (
        evaluate_lines(**_metrics(roundtrip_preserved=False))["lines"]["L4_lesion_roundtrip"]
        is False
    )


# --------------------------------------------------------------------------- #
# Gate 2: training-budget gate refuses before touching any data
# --------------------------------------------------------------------------- #


def test_budget_gate_refuses_without_approval() -> None:
    with pytest.raises(SystemExit, match="budget"):
        run_sweep(budget_approved=False)


# --------------------------------------------------------------------------- #
# Gate 3: independent slice chaining (contract section 6; needs local corpus)
# --------------------------------------------------------------------------- #


def test_independent_slice_chains_after_p51g_agate() -> None:
    data_file = PROJECT_ROOT / "data/ultradata/SFT-Agent-2609/data/Tool_Use/Tool_Use_part-1-of-9.jsonl"
    if not data_file.is_file():
        pytest.skip("P5.1g corpus file not present on this machine")
    from eval_taiji_p5_1g_real_corpus_quota_budget_gate import (
        TRAIN_COUNT,
        _sample_agate,
        _sample_arm,
    )

    sourced_path = data_file
    sourced_sample = _sample_arm(sourced_path)
    train_vocabulary = frozenset(
        f"tool.{name}"
        for trajectory in sourced_sample.trajectories[:TRAIN_COUNT]
        for name in trajectory.tool_calls
    )
    p51g_agate = _sample_agate(sourced_path, train_vocabulary, after_line=sourced_sample.last_line)
    independent = _sample_agate(sourced_path, train_vocabulary, after_line=p51g_agate.last_line)
    assert len(independent.trajectories) == len(p51g_agate.trajectories) == 16
    p51g_lines = {t.line_index for t in p51g_agate.trajectories}
    independent_lines = {t.line_index for t in independent.trajectories}
    assert not (p51g_lines & independent_lines)
    assert min(independent_lines) > max(p51g_lines)
    for trajectory in independent.trajectories:
        assert frozenset(f"tool.{name}" for name in trajectory.tool_calls) <= train_vocabulary
