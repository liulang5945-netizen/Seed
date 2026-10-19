from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.training import analyze_taiji_n1_plan154_fresh_context_itemwise_attribution as analysis


def _rows() -> list[dict]:
    rows = []
    for seed in analysis.SEEDS:
        for branch in analysis.BRANCHES:
            for kind in analysis.KINDS:
                for index in range(40):
                    for length in analysis.LENGTHS:
                        if kind == "unknown":
                            exact = branch == "eight_row_control" or length >= 16
                        elif kind == "lookup":
                            exact = length < 32
                        else:
                            exact = True
                        rows.append(
                            {
                                "seed": seed,
                                "branch": branch,
                                "event": "diagnostic_row",
                                "split": "diagnostic",
                                "id": f"v28-{kind}-{index:03d}-l{length}",
                                "parent_id": f"v28-{kind}-{index:03d}",
                                "kind": kind,
                                "context_length": length,
                                "free_exact": exact,
                                "output_hex": analysis.UNKNOWN_HEX
                                if kind == "unknown" and exact
                                else "6f74686572",
                            }
                        )
    return rows


def test_six_endpoint_length_face_is_fully_paired() -> None:
    rows = _rows()
    summary = analysis.summarize_rows(rows)

    assert summary["coverage"] == {
        "rows": 2_880,
        "endpoint_length_kind_strata": 72,
        "items_per_stratum": 40,
        "complete": True,
    }
    assert len(summary["by_branch_length_kind"]) == 24
    assert len(summary["matched_candidate_control_by_seed_length_kind"]) == 36
    assert len(summary["within_branch_transitions_by_seed_kind"]) == 72


def test_candidate_control_and_within_branch_transitions_keep_pair_counts() -> None:
    rows = _rows()
    summary = analysis.summarize_rows(rows)
    paired = next(
        row
        for row in summary["matched_candidate_control_by_seed_length_kind"]
        if row["seed"] == 11 and row["context_length"] == 8 and row["kind"] == "unknown"
    )
    assert paired["pairs"] == 40
    assert paired["control_only_correct"] == 40
    assert paired["candidate_only_correct"] == 0

    transition = next(
        row
        for row in summary["within_branch_transitions_by_seed_kind"]
        if row["seed"] == 11
        and row["branch"] == "length_mixed"
        and row["kind"] == "lookup"
        and row["from_length"] == 8
        and row["to_length"] == 32
    )
    assert transition["pairs"] == 40
    assert transition["correct_to_incorrect"] == 40


def test_analyzer_rejects_missing_duplicate_and_wrong_split_rows() -> None:
    rows = _rows()
    with pytest.raises(analysis.Refuse, match="expected 2880"):
        analysis.summarize_rows(rows[:-1])
    with pytest.raises(analysis.Refuse, match="duplicate"):
        analysis.summarize_rows([*rows[:-1], rows[0]])
    wrong_split = [*rows]
    wrong_split[0] = {**wrong_split[0], "split": "holdout"}
    with pytest.raises(analysis.Refuse, match="diagnostic_row"):
        analysis.summarize_rows(wrong_split)


def test_cli_guard_follows_analysis_functions() -> None:
    module = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    assert "__name__" in ast.unparse(guard.test)
