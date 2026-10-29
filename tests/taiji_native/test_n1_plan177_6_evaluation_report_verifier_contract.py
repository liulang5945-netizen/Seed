"""N177.6 评价报告验收器（`verify_taiji_n1_plan177_6_t1_evaluation_report.py`）的契约册。

验收器不重算任何数，只判"这份报告可不可以被读作结论"。所以本册钉的是它**会拒绝**：
每个缺口都要被点名，而不是笼统地 rc=1。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTRUMENT = REPO / "scripts/training/verify_taiji_n1_plan177_6_t1_evaluation_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1776_report_verifier", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _module()


def _gate_names(verifier) -> list[str]:
    return list(verifier.ABSOLUTE_GATES) + ["f1_false_write_rate"]


def _report(verifier, arms: int = 15) -> dict[str, Any]:
    gates = {name: True for name in _gate_names(verifier)}
    means = {name: 0.5 for name in _gate_names(verifier)}
    contrasts = {
        control: {
            "paired_seed_deltas": [{"seed": seed, "delta": 0.2} for seed in range(3)],
            "median_delta": 0.2,
            "positive_seed_count": 3,
            "minimum_delta": 0.2,  # 与逐 seed 差值一致（三个都是 0.2）
            "passed": True,
        }
        for control in ("no_update", "parser_permuted")
    }
    return {
        "format": "taiji-n177-6-t1-evaluation-v1",
        "status": "complete",
        "arms_completed": arms,
        #: 与真实报告同形：这是**计数**，不是逐臂列表（第一版 fixture 用列表，掩盖了这个形状差）。
        "fully_scored_holdout_arms": arms,
        "sealed_reads": None,
        "absolute_gates": gates,
        "absolute_means": means,
        "learning_contrasts": contrasts,
        "state_write_diagnostics": [
            {
                "seed": seed,
                "paired_family_count": 1,
                "paired_family_deltas": [
                    {"family_id": "f", "write_enabled": 1.0, "write_disabled": 0.5, "delta": 0.5}
                ],
                "write_enabled_holdout": {"f1_state_query_score": 1.0},
                "write_disabled_holdout": {"f1_state_query_score": 0.5},
            }
            for seed in range(3)
        ],
        "f0_retention": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
        "architecture_attribution": {
            "information_path": "x",
            "semantic_feature_bridge": "x",
            "credit_path": "x",
            "state_lifecycle": "x",
            "conclusion": "x",
        },
        "parameters_updated_during_scoring": False,
        "query_and_holdout": [
            {"split": "query", "arms": []},
            {
                "split": "holdout",
                "arms": [
                    {
                        "mode": "correct",
                        "seed": index % 3,
                        "by_level": {
                            "F0": {
                                "family_macro": {
                                    key: 0.5
                                    for key in (
                                        "answer_exact_rate",
                                        "valid_utf8_rate",
                                        "stop_marker_rate",
                                    )
                                }
                            },
                            "F1": {
                                "family_macro": {
                                    key: 0.5
                                    for key in (
                                        "answer_exact_rate",
                                        "f1_state_query_score",
                                        "entity_span_exact_rate",
                                        "attribute_span_exact_rate",
                                        "value_span_exact_rate",
                                        "f1_post_correction_state_exact",
                                        "f1_post_correction_answer_exact",
                                        "f1_unrelated_keep_exact",
                                        "f1_false_write_rate",
                                    )
                                }
                            },
                        },
                    }
                    for index in range(arms)
                ],
            },
        ],
    }


def _state(attempts: int = 1, superseded: str | None = None) -> dict[str, Any]:
    return {"evaluation_attempts": attempts, "evaluation_report_superseded": superseded}


def test_a_complete_report_passes(verifier) -> None:
    assert verifier.verify(_report(verifier), _state()) == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda report: report.update(arms_completed=14), "arms_completed"),
        (lambda report: report.update(fully_scored_holdout_arms=14), "fully_scored_holdout_arms"),
        (
            lambda report: report["query_and_holdout"][1]["arms"].pop(),
            "holdout arms=14",
        ),
        (
            lambda report: report["query_and_holdout"][1]["arms"][0]["by_level"]["F1"][
                "family_macro"
            ].pop("f1_state_query_score"),
            "missing f1_state_query_score",
        ),
        (lambda report: report["absolute_gates"].pop("f1_value_span_exact"), "f1_value_span_exact"),
        (
            lambda report: report["absolute_means"].pop("f0_valid_utf8"),
            "absolute_means missing f0_valid_utf8",
        ),
        (lambda report: report.update(state_write_diagnostics=[]), "state_write_diagnostics"),
        (lambda report: report["learning_contrasts"].pop("parser_permuted"), "parser_permuted"),
        (lambda report: report.update(f0_retention=[{"seed": 0}]), "f0_retention"),
        (lambda report: report["architecture_attribution"].pop("credit_path"), "credit_path"),
        (
            lambda report: report.update(parameters_updated_during_scoring=True),
            "parameters_updated",
        ),
    ],
)
def test_each_gap_is_named(verifier, mutate, needle: str) -> None:
    report = _report(verifier)
    mutate(report)
    problems = verifier.verify(report, _state())
    assert any(needle in problem for problem in problems), problems


def test_aggregates_must_agree_with_the_paired_deltas(verifier) -> None:
    report = _report(verifier)
    report["learning_contrasts"]["no_update"]["median_delta"] = 0.5  # 与逐 seed 差值不符
    assert any("median_delta disagrees" in p for p in verifier.verify(report, _state()))

    report = _report(verifier)
    report["learning_contrasts"]["no_update"]["paired_seed_deltas"][0]["delta"] = -0.2
    #: 逐 seed 差里出现 −0.2 之后，报告里那个 passed=True 就不再成立。
    assert any("passed flag disagrees" in p for p in verifier.verify(report, _state()))


def test_state_write_delta_must_equal_the_two_sides(verifier) -> None:
    report = _report(verifier)
    report["state_write_diagnostics"][0]["paired_family_deltas"][0]["delta"] = 0.9
    problems = verifier.verify(report, _state())
    assert any("write_enabled - write_disabled" in p for p in problems), problems


def test_reading_holdout_twice_is_refused(verifier) -> None:
    assert any(
        "evaluation_attempts" in problem
        for problem in verifier.verify(_report(verifier), _state(attempts=2))
    )
    assert any(
        "superseded" in problem
        for problem in verifier.verify(_report(verifier), _state(superseded="old.json"))
    )
