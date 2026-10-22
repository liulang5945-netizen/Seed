from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.training import analyze_taiji_n1_plan152_length_itemwise_attribution as analysis


def _rows() -> list[dict]:
    rows = []
    for seed in analysis.SEEDS:
        for branch in analysis.BRANCHES:
            for length in analysis.LENGTHS:
                for kind in analysis.KINDS:
                    for index in range(24):
                        if kind == "unknown":
                            exact = branch == "eight_row_control" or length >= 16
                        elif kind == "lookup":
                            exact = length < 32
                        else:
                            exact = True
                        rows.append(
                            {
                                "event": "length_score_row",
                                "split": "train",
                                "seed": seed,
                                "branch": branch,
                                "context_length": length,
                                "kind": kind,
                                "id": f"train-{kind}-{index:02d}",
                                "free_exact": exact,
                                "output_hex": (
                                    analysis.UNKNOWN_HEX
                                    if kind == "unknown" and exact
                                    else "6f74686572"
                                ),
                            }
                        )
    return rows


def test_row_contract_and_denominators_are_complete() -> None:
    rows = _rows()
    assert len(rows) == 1_728
    assert analysis.verify_length_rows(rows) == {
        "rows": 1_728,
        "endpoint_length_kind_strata": 72,
        "items_per_endpoint_length_kind": 24,
        "candidate_control_paired_items_per_seed_length": {
            f"{seed}/{length}": 96 for seed in analysis.SEEDS for length in analysis.LENGTHS
        },
        "complete": True,
    }


def test_candidate_control_and_length_transitions_are_paired() -> None:
    rows = _rows()
    paired = analysis.matched_candidate_control(rows)
    lookup_32 = next(
        row for row in paired if row["context_length"] == 32 and row["kind"] == "lookup"
    )
    assert lookup_32["pairs"] == 72
    assert lookup_32["both_incorrect"] == 72

    unknown_8 = next(
        row for row in paired if row["context_length"] == 8 and row["kind"] == "unknown"
    )
    assert unknown_8["pairs"] == 72
    assert unknown_8["control_only_correct"] == 72

    transitions = analysis.within_branch_transitions(rows)
    lookup = next(
        row
        for row in transitions
        if row["branch"] == "length_mixed"
        and row["kind"] == "lookup"
        and row["from_length"] == 8
        and row["to_length"] == 32
    )
    assert lookup["pairs"] == 72
    assert lookup["correct_to_incorrect"] == 72


def test_row_validator_rejects_missing_duplicate_or_non_train_rows() -> None:
    rows = _rows()
    with pytest.raises(analysis.Refuse, match="expected 1728"):
        analysis.verify_length_rows(rows[:-1])
    with pytest.raises(analysis.Refuse, match="duplicate"):
        analysis.verify_length_rows([*rows[:-1], rows[0]])
    changed = [*rows]
    changed[0] = {**changed[0], "split": "holdout"}
    with pytest.raises(analysis.Refuse, match="train length_score_row"):
        analysis.verify_length_rows(changed)


def test_cli_entrypoint_follows_all_analysis_helpers() -> None:
    module = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
    final = module.body[-1]
    assert isinstance(final, ast.If)
    assert "__name__" in ast.unparse(final.test)
