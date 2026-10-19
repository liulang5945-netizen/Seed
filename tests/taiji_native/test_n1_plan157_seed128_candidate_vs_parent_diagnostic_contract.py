from __future__ import annotations

import hashlib

import pytest

from scripts.training import build_taiji_n1_plan157_seed128_candidate_vs_parent_face as builder
from scripts.training import run_taiji_n1_plan157_seed128_candidate_vs_parent_diagnostic as runner


def test_plan_digest_and_query_denominator_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert len(builder.ENDPOINTS) == 6
    assert builder.QUERIES_PER_ENDPOINT == 480
    assert builder.TOTAL_QUERIES == 2_880
    assert builder.PRIOR_LINKED_CPU_SECONDS == 9_328.078125
    assert builder.LINKED_CPU_LIMIT_SECONDS == 18_000.0


def test_candidate_and_parent_endpoints_use_distinct_one_shot_paths() -> None:
    assert builder.FACE.name == "n1_plan157_v30_candidate_vs_parent_seed128_20261020.json"
    assert builder.ATTEMPT != builder.N155A_RESULT
    assert builder.RESULT != builder.N155A_RESULT
    assert builder.PROGRESS != builder.N155A_PROGRESS


def test_fresh_summary_requires_all_endpoint_length_kind_strata() -> None:
    rows = []
    for seed, branch in builder.ENDPOINTS:
        for length in builder.LENGTHS:
            for kind in builder.KINDS:
                for index in range(builder.ITEMS_PER_KIND):
                    rows.append(
                        {
                            "seed": seed,
                            "branch": branch,
                            "context_length": length,
                            "kind": kind,
                            "free_exact": index % 2 == 0,
                            "output_hex": "756e6b6e6f776e" if index % 3 == 0 else "",
                        }
                    )
    summary = runner._fresh_summary(rows)
    assert len(summary) == 72
    assert all(row["items"] == 40 for row in summary)
    assert all(row["free_exact"] == 20 for row in summary)
    with pytest.raises(runner.Refuse, match="stratum mismatch"):
        runner._fresh_summary(rows[:-1])

