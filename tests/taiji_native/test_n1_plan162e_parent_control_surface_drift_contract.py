from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.training import analyze_taiji_n1_plan162e_parent_control_surface_drift as analyzer


def test_plan_digest_matches_frozen_source() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256


def test_n155a_candidate_rows_bind_by_seed_and_branch() -> None:
    rows = [
        {
            "seed": seed,
            "branch": branch,
            "event": "original_train_gate_row",
            "id": f"train-{seed}-{branch}-{index}",
        }
        for seed in analyzer.SEEDS
        for branch in ("unknown8_rebalanced", "uniform_length_replay")
        for index in range(analyzer.GATE_ROWS_PER_ENDPOINT)
    ]
    bound = analyzer._n155a_gate_rows(rows)
    assert len(bound) == len(analyzer.SEEDS) * analyzer.GATE_ROWS_PER_ENDPOINT
    assert {(row["seed"], row["lineage_arm"]) for row in bound} == {
        (11, "n155a_candidate"),
        (12, "n155a_candidate"),
        (13, "n155a_candidate"),
    }


def test_three_way_gate_trajectory_separates_continuation_and_surface() -> None:
    rows = [
        {
            "seed": 11,
            "lineage_arm": "n155a_candidate",
            "evaluation_set": "original",
            "kind": "transition",
            "id": "train-002",
            "condition": None,
            "case_id": None,
            "free_exact": True,
            "raw_hex": "65617374",
        },
        {
            "seed": 11,
            "lineage_arm": "surface_control",
            "evaluation_set": "original",
            "kind": "transition",
            "id": "train-002",
            "condition": None,
            "case_id": None,
            "free_exact": True,
            "raw_hex": "65617374",
        },
        {
            "seed": 11,
            "lineage_arm": "surface_augmented",
            "evaluation_set": "original",
            "kind": "transition",
            "id": "train-002",
            "condition": None,
            "case_id": None,
            "free_exact": False,
            "raw_hex": "77657374",
        },
    ]
    summary, trajectories, examples = analyzer._three_way_gate(rows)
    assert summary[0]["parent_to_control_regression"] == 0
    assert summary[0]["control_to_augmented_regression"] == 1
    assert trajectories == [
        {
            "n155a_correct": True,
            "surface_control_correct": True,
            "surface_augmented_correct": False,
            "items": 1,
        }
    ]
    assert examples[0]["parent_id"] == "train-002"


def test_three_way_length_trajectory_matches_all_arms() -> None:
    rows = [
        {
            "seed": 12,
            "lineage_arm": arm,
            "context_length": 8,
            "kind": "unknown",
            "id": "train-003",
            "split": "train",
            "free_exact": exact,
        }
        for arm, exact in (
            ("n155a_candidate", True),
            ("surface_control", False),
            ("surface_augmented", True),
        )
    ]
    summary = analyzer._three_way_length(rows)
    assert summary == [
        {
            "seed": 12,
            "kind": "unknown",
            "context_length": 8,
            "items": 1,
            "n155a_exact": 1,
            "control_exact": 0,
            "augmented_exact": 1,
            "parent_to_control_regression": 1,
            "parent_to_control_improvement": 0,
            "control_to_augmented_regression": 0,
            "control_to_augmented_improvement": 1,
        }
    ]


def test_outputs_are_repository_local() -> None:
    for path in (
        analyzer.PLAN,
        analyzer.N155A_RESULT,
        analyzer.N155A_PROGRESS,
        analyzer.N162B_RESULT,
        analyzer.N162B_PROGRESS,
        analyzer.REPORT,
    ):
        assert Path(path).resolve().is_relative_to(analyzer.ROOT)
