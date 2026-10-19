from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.training import run_taiji_n1_plan153a_fresh_context_result_binding_recovery as recovery


def _endpoint_rows(seed: int, branch: str) -> list[dict]:
    rows = []
    for kind in recovery.KINDS:
        for index in range(40):
            for length in recovery.LENGTHS:
                exact = (index + length) % 3 != 0
                rows.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "event": "diagnostic_row",
                        "id": f"v28-{kind}-{index:03d}-l{length}",
                        "kind": kind,
                        "context_length": length,
                        "free_exact": exact,
                        "output_hex": recovery.builder.UNKNOWN.encode("utf-8").hex()
                        if kind == "unknown" and exact
                        else "6f74686572",
                    }
                )
    return rows


def _all_rows() -> list[dict]:
    return [row for seed, branch in recovery.ENDPOINTS for row in _endpoint_rows(seed, branch)]


def test_n153a_summary_is_scoped_to_one_completed_endpoint() -> None:
    endpoint_rows = _endpoint_rows(11, "length_mixed")
    summary = recovery._summarize_endpoint(endpoint_rows, 11, "length_mixed")
    assert len(endpoint_rows) == 480
    assert len(summary) == 12
    assert all(row["items"] == 40 for row in summary)

    retained = []
    recovery._append_endpoint_rows(retained, endpoint_rows)
    assert retained == endpoint_rows


def test_cumulative_retry_budget_is_frozen_to_one_lost_endpoint() -> None:
    assert recovery.PRIOR_QUERY_COUNT == 480
    assert recovery.EXPECTED_NEW_QUERIES == 2_880
    assert recovery.EXPECTED_CUMULATIVE_QUERIES == 3_360
    assert recovery.PRIOR_CPU_SECONDS == 113.78125
    assert recovery.CPU_LIMIT_SECONDS == 3_600


def test_all_six_endpoint_rows_are_required_for_combined_completion() -> None:
    rows = _all_rows()
    assert len(rows) == 2_880
    assert recovery.validate_remaining_endpoints(rows) == {
        "endpoint_count": 6,
        "queries_per_endpoint": 480,
        "new_queries": 2_880,
        "complete": True,
    }

    with pytest.raises(recovery.Refuse, match="requires 2880"):
        recovery.validate_remaining_endpoints(rows[:-1])


def test_cli_guard_follows_all_recovery_helpers() -> None:
    module = ast.parse(Path(recovery.__file__).read_text(encoding="utf-8"))
    final = module.body[-1]
    assert isinstance(final, ast.If)
    assert "__name__" in ast.unparse(final.test)
