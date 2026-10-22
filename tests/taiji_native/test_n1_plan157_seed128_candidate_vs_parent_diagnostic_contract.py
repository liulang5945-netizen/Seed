from __future__ import annotations

import hashlib

import pytest

from scripts.training import build_taiji_n1_plan157_seed128_candidate_vs_parent_face as builder
from scripts.training import run_taiji_n1_plan157_seed128_candidate_vs_parent_diagnostic as runner


def test_plan_digest_and_query_denominator_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert (
        hashlib.sha256(builder.CORRECTION_ADDENDUM.read_bytes()).hexdigest()
        == builder.CORRECTION_SHA256
    )
    assert len(builder.ENDPOINTS) == 6
    assert builder.QUERIES_PER_ENDPOINT == 480
    assert builder.TOTAL_QUERIES == 2_880
    assert builder.PRIOR_RESULT_LINKED_CPU_SECONDS == 9_502.03125
    assert builder.BINDING_FAILURE_CPU_RESERVE_SECONDS == 5.0
    assert builder.PRIOR_LINKED_CPU_SECONDS == 9_507.03125
    assert builder.LINKED_CPU_LIMIT_SECONDS == 18_000.0
    assert builder.PRIOR_QUERIES == 480
    assert builder.TOTAL_QUERIES - builder.PRIOR_QUERIES == 2_400


def test_partial_face_continuation_reuses_exactly_the_sealed_endpoint() -> None:
    endpoints, sources, exclusions = builder.source_context()
    assert len(endpoints) == 6
    assert builder.PRIOR_LINKED_CPU_SECONDS == 9_507.03125
    assert builder.BINDING_FAILURE_CPU_RESERVE_SECONDS == 5.0
    assert builder.PRIOR_QUERIES == 480
    assert len(sources) >= 590
    assert exclusions["exclusion_summary"]["prior_n153_face_sha256"] == builder.sha256(
        builder.n153_builder.FACE
    )
    assert not builder.ATTEMPT.exists()
    assert not builder.PROGRESS.exists()
    assert not builder.RESULT.exists()


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


def test_fresh_summary_accepts_one_complete_endpoint_continuation_cell() -> None:
    rows = []
    for length in builder.LENGTHS:
        for kind in builder.KINDS:
            for index in range(builder.ITEMS_PER_KIND):
                rows.append(
                    {
                        "seed": 11,
                        "branch": "unknown8_rebalanced",
                        "context_length": length,
                        "kind": kind,
                        "free_exact": index % 2 == 0,
                        "output_hex": "",
                    }
                )
    summary = runner._fresh_summary(rows)
    assert len(summary) == 12
    assert all(row["seed"] == 11 and row["branch"] == "unknown8_rebalanced" for row in summary)
    assert all(row["items"] == 40 for row in summary)
