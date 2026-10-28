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
            "minimum_delta": 0.1,
            "passed": True,
        }
        for control in ("no_update", "parser_permuted")
    }
    return {
        "format": "taiji-n177-6-t1-evaluation-v1",
        "status": "complete",
        "arms_completed": arms,
        "fully_scored_holdout_arms": [{"mode": "correct", "seed": index} for index in range(arms)],
        "sealed_reads": None,
        "absolute_gates": gates,
        "absolute_means": means,
        "learning_contrasts": contrasts,
        "state_write_diagnostics": [{"mode": "correct", "seed": 0}],
        "f0_retention": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
        "architecture_attribution": {
            "information_path": "x",
            "semantic_feature_bridge": "x",
            "credit_path": "x",
            "state_lifecycle": "x",
            "conclusion": "x",
        },
        "parameters_updated_during_scoring": False,
    }


def _state(attempts: int = 1, superseded: str | None = None) -> dict[str, Any]:
    return {"evaluation_attempts": attempts, "evaluation_report_superseded": superseded}


def test_a_complete_report_passes(verifier) -> None:
    assert verifier.verify(_report(verifier), _state()) == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda report: report.update(arms_completed=14), "arms_completed"),
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


def test_reading_holdout_twice_is_refused(verifier) -> None:
    assert any(
        "evaluation_attempts" in problem
        for problem in verifier.verify(_report(verifier), _state(attempts=2))
    )
    assert any(
        "superseded" in problem
        for problem in verifier.verify(_report(verifier), _state(superseded="old.json"))
    )
