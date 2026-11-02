"""Recompute the step-10 headline numbers from the raw rows in the same reports.

The card (10.11) only checks that the four exits are recorded. This guard goes one
level deeper: every published aggregate must be recoverable from the per-row data
stored next to it, so a reader never has to trust a hand-carried number.

One unit trap is pinned here on purpose: 10.4's ``totals`` are **sums** over the six
fixtures (each fixture scores 1.0 minus 0.01 per step or a negative penalty for an
illegal move), not means. The mean is 0.468 for both the subject and the random
arm, and reading the sum as a mean would make the report look like it contradicts
itself.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORT_10_4 = REPO / "reports" / "taiji_core_subject_selection_10_4_20261101.json"
REPORT_10_5 = REPO / "reports" / "taiji_core_goal_conditioned_10_5_20261101.json"
REPORT_10_6 = REPO / "reports" / "taiji_core_action_verdict_10_6_20261101.json"
REPORT_10_7 = REPO / "reports" / "taiji_core_goal_training_10_7_20261101.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _total_mismatches(report: dict) -> dict:
    mismatches = {}
    for arm in ("subject", "greedy", "random"):
        recomputed = sum(fixture[arm]["score"] for fixture in report["fixtures"])
        if abs(report["totals"][arm] - recomputed) > 1e-9:
            mismatches[arm] = recomputed
    return mismatches


def test_10_4_totals_are_fixture_sums():
    report = _load(REPORT_10_4)
    fixtures = report["fixtures"]
    assert len(fixtures) == 6
    assert _total_mismatches(report) == {}
    assert report["totals"]["subject"] == pytest.approx(report["totals"]["random"], abs=1e-9)
    assert report["totals"]["subject"] < report["totals"]["greedy"]


def test_10_4_totals_guard_can_fail():
    report = _load(REPORT_10_4)
    report["totals"]["subject"] += 1.0
    assert _total_mismatches(report) == {"subject": pytest.approx(2.81, abs=1e-9)}


def test_10_4_selection_is_one_constant_object():
    report = _load(REPORT_10_4)
    tables = [table for fixture in report["fixtures"] for table in fixture["subject"]["tables"]]
    assert len(tables) == 19
    picks = [min(table, key=lambda row: row["nll"])["move"][0] for table in tables]
    counts = Counter(picks)
    assert len(counts) == 1, counts
    assert max(counts.values()) == 19


def test_10_5_contrasts_recompute():
    report = _load(REPORT_10_5)
    for row in report["contrast"]:
        assert row["contrast"] == pytest.approx(
            row["nll_mismatched"] - row["nll_matched"], abs=1e-12
        )
    first, second = report["contrast"]
    assert report["gap"]["gap"] == pytest.approx(
        abs(first["contrast"] - second["contrast"]), abs=1e-12
    )
    assert report["gap"]["correct"] == pytest.approx(first["contrast"], abs=1e-12)
    assert report["gap"]["other"] == pytest.approx(second["contrast"], abs=1e-12)
    assert all(row["contrast"] == 0.0 for row in report["null_control"])


def test_10_6_verdicts_recompute():
    report = _load(REPORT_10_6)
    rows = report["rows"]
    assert len(rows) == 2
    for row in rows:
        assert row["verdict"] == pytest.approx(row["nll_no"] - row["nll_yes"], abs=1e-12)
        assert row["answer_gap"] == pytest.approx(row["nll_yes"] - row["nll_no"], abs=1e-12)
    assert report["gap"] == pytest.approx(abs(rows[0]["verdict"] - rows[1]["verdict"]), abs=1e-12)
    assert report["gap"] < report["declared"]["gap_floor"]


def test_10_7_hit_counts_recompute():
    report = _load(REPORT_10_7)
    untrained = report["untrained_held_out"]
    assert untrained["hits"] == sum(1 for row in untrained["rows"] if row["hit"])
    assert untrained["hits"] == 3
    for run in report["runs"]:
        assert run["held_out_hits"] == sum(1 for row in run["held_out"] if row["hit"])
        assert run["held_out_hits"] == 3
        assert len(run["held_out"]) == 6
    fast = [run for run in report["runs"] if run["lr"] == pytest.approx(1e-4)][0]
    ratio = fast["last_loss"] / fast["first_loss"]
    assert 0.60 < ratio <= 0.70
