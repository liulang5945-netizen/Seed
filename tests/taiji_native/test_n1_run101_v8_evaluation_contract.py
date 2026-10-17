from __future__ import annotations

import json

import pytest

from scripts.training import eval_taiji_n1_hard_neighbor_v8 as evaluator


def _range_cells(usable: int) -> list[dict]:
    result = []
    for index, (seed, arm) in enumerate(
        (seed, arm) for seed in evaluator.SEEDS for arm in evaluator.ARMS
    ):
        rows = []
        for kind in evaluator.KINDS:
            for item in range(5):
                error = index < usable and kind != "unknown" and item < 2
                if index < usable and kind == "transition" and item == 2:
                    error = True
                rows.append(
                    {
                        "id": f"{index}-{kind}-{item}",
                        "kind": kind,
                        "free_exact": not error,
                    }
                )
        result.append({"seed": seed, "arm": arm, "rows": rows})
    return result


def test_range_gate_uses_six_addressable_errors_on_six_of_nine_endpoints():
    passing = evaluator.range_lines(_range_cells(6))
    failing = evaluator.range_lines(_range_cells(5))
    assert passing["range_check_passed"] is True
    assert passing["usable_endpoints"] == 6
    assert passing["decodable_items_per_endpoint"] == 15
    assert failing["range_check_passed"] is False
    assert failing["usable_endpoints"] == 5
    assert all(value["reported_only"] for value in passing["per_class"].values())


def test_confirmation_refuses_when_range_gate_has_not_passed(tmp_path, monkeypatch):
    range_path = tmp_path / "range.json"
    report = {
        "complete": True,
        "status": "range_not_met",
        "range_check_passed": False,
        "confirm_subset_queried": False,
        "queried_subsets": ["sacrifice"],
        "items_queried": 180,
        "cpu_seconds": 1.0,
        "face_sha256": "unneeded-on-failed-gate",
    }
    range_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(evaluator, "RANGE_REPORT", range_path)
    with pytest.raises(ValueError, match="range gate has not passed"):
        evaluator._verify_range_pass()


def test_confirmation_attempt_is_not_written_before_range_eligibility(tmp_path, monkeypatch):
    range_path = tmp_path / "range.json"
    range_path.write_text(
        json.dumps(
            {
                "complete": True,
                "status": "range_not_met",
                "range_check_passed": False,
                "confirm_subset_queried": False,
                "queried_subsets": ["sacrifice"],
                "items_queried": 180,
                "cpu_seconds": 1.0,
                "face_sha256": "unused",
            }
        ),
        encoding="utf-8",
    )
    attempt = tmp_path / "confirm_attempt.json"
    monkeypatch.setattr(evaluator, "RANGE_REPORT", range_path)
    monkeypatch.setattr(evaluator, "CONFIRM_REPORT", tmp_path / "confirm.json")
    monkeypatch.setattr(evaluator, "CONFIRM_ATTEMPT", attempt)
    with pytest.raises(ValueError, match="range gate has not passed"):
        evaluator.run_confirmation()
    assert not attempt.exists()
