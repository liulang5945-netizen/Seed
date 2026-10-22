from __future__ import annotations

import hashlib

from scripts.training import build_taiji_n1_plan159_seed129_candidate_parent_face as builder
from scripts.training import run_taiji_n1_plan159_seed129_candidate_parent_diagnostic as runner


def test_plan_and_seed129_replication_denominators_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert (
        hashlib.sha256(builder.PRIOR_CORRECTION_ADDENDUM.read_bytes()).hexdigest()
        == builder.PRIOR_CORRECTION_SHA256
    )
    assert (
        hashlib.sha256(builder.CORRECTION_ADDENDUM.read_bytes()).hexdigest()
        == builder.CORRECTION_SHA256
    )
    assert builder.QUERIES_PER_ENDPOINT == 480
    assert builder.TOTAL_QUERIES == 2_880
    assert builder.PRIOR_LINKED_CPU_SECONDS == 9_956.421875
    assert builder.LINKED_CPU_LIMIT_SECONDS == 18_000.0
    assert builder.PRIOR_QUERIES == 480
    assert builder.TOTAL_QUERIES - builder.PRIOR_QUERIES == 2_400
    assert builder.FACE.name == "n1_plan159_v31_candidate_parent_seed129_20261020.json"
    assert builder.PRIOR_RESULT != runner.RESULT
    assert builder.PRIOR_PROGRESS != runner.PROGRESS


def test_seed128_face_values_are_added_to_replication_exclusions() -> None:
    face = {
        "items": [
            {
                "entities": ["abcdexyz"],
                "answer": "copper",
                "observations_by_length": {"32": ["abcdexyz uses copper.", "uvwxyzab uses iron."]},
            }
        ]
    }
    entities, words = builder._face_values(face)
    assert entities == {"abcdexyz", "uvwxyzab"}
    assert words == {"copper", "iron"}


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
                            "output_hex": (
                                "756e6b6e6f776e" if kind == "unknown" and index % 2 == 0 else ""
                            ),
                        }
                    )
    summary = runner._fresh_summary(rows)
    assert len(summary) == 72
    assert all(row["items"] == 40 for row in summary)
    assert all(row["free_exact"] == 20 for row in summary)
    assert all(row["literal_unknown"] == 20 for row in summary if row["kind"] == "unknown")


def test_fresh_summary_accepts_single_endpoint_continuation_data() -> None:
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
                        "output_hex": (
                            "756e6b6e6f776e" if kind == "unknown" and index % 2 == 0 else ""
                        ),
                    }
                )
    summary = runner._fresh_summary(rows)
    assert len(summary) == 12
    assert all(row["seed"] == 11 and row["branch"] == "unknown8_rebalanced" for row in summary)
    assert all(row["items"] == 40 for row in summary)
