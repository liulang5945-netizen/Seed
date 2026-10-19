from __future__ import annotations

import pytest

from scripts.training import build_taiji_n1_v21_confirmation_after_n141 as face_builder
from scripts.training import eval_taiji_n1_v21_confirmation_after_n141 as evaluator

ENDPOINTS = [
    ("N1-132", "parent", seed) for seed in (11, 12, 13)
] + [
    ("N1-140", "candidate", seed) for seed in (11, 12, 13)
] + [
    ("N1-141", branch, seed)
    for seed in (11, 12, 13)
    for branch in ("candidate", "control")
]


def _range_rows(qualified: int) -> list[dict]:
    rows = []
    for endpoint_index, (source, branch, seed) in enumerate(ENDPOINTS):
        endpoint_id = f"{source}/seed{seed}/{branch}"
        has_floor = endpoint_index < qualified
        for kind in evaluator.KINDS:
            for item in range(20):
                error = has_floor and kind != "unknown" and item < 8
                rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "subset": "sacrifice",
                        "kind": kind,
                        "id": f"sacrifice-{kind}-{item:02d}",
                        "free_exact": not error,
                    }
                )
    return rows


def _confirmation_rows() -> list[dict]:
    rows = []
    for source, branch, seed in ENDPOINTS:
        endpoint_id = f"{source}/seed{seed}/{branch}"
        if source == "N1-141" and branch == "candidate":
            per_kind = (13, 13, 12, 12)
        elif source == "N1-140":
            per_kind = (10, 10, 10, 10)
        elif source == "N1-141" and branch == "control":
            per_kind = (7, 6, 6, 6)
        else:
            per_kind = (7, 6, 6, 6)
        for kind, correct_count in zip(evaluator.KINDS, per_kind, strict=True):
            for item in range(20):
                rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "source": source,
                        "branch": branch,
                        "seed": seed,
                        "subset": "confirm",
                        "kind": kind,
                        "id": f"confirm-{kind}-{item:02d}",
                        "free_exact": item < correct_count,
                    }
                )
    return rows


def test_range_gate_uses_answerable_error_floor_and_two_thirds_endpoint_rule() -> None:
    passed = evaluator._range_summary(_range_rows(8))
    stopped = evaluator._range_summary(_range_rows(7))

    assert passed["qualified_endpoints"] == 8
    assert passed["range_check_passed"] is True
    assert passed["by_endpoint"]["N1-132/seed11/parent"]["answerable_items"] == 60
    assert passed["by_endpoint"]["N1-132/seed11/parent"]["unknown_items"] == 20
    assert stopped["qualified_endpoints"] == 7
    assert stopped["range_check_passed"] is False


def test_confirmation_gate_compares_all_four_families_on_same_items() -> None:
    summary = evaluator.confirmation_summary(_confirmation_rows())

    assert summary["confirmation_pass"] is True
    assert summary["candidate_exact"] == 150
    assert summary["n141_control_exact"] == 75
    assert summary["n132_parent_exact"] == 75
    assert summary["n140_block_candidate_exact"] == 120
    assert all(summary["gates"].values())


def test_confirmation_gate_fails_candidate_per_seed_threshold() -> None:
    rows = _confirmation_rows()
    candidate_rows = [row for row in rows if row["endpoint_id"] == "N1-141/seed11/candidate"]
    for row in [row for row in candidate_rows if row["free_exact"]][:11]:
        row["free_exact"] = False
    summary = evaluator.confirmation_summary(rows)

    assert summary["confirmation_pass"] is False
    assert summary["gates"]["candidate_each_seed_at_least_40_of_80"] is False


def test_range_and_confirmation_require_identical_item_sets() -> None:
    range_rows = _range_rows(8)
    range_rows[0]["id"] = "different-sacrifice-item"
    with pytest.raises(ValueError, match="same 80 unique sacrifice IDs"):
        evaluator._range_summary(range_rows)

    confirmation_rows = _confirmation_rows()
    confirmation_rows[0]["id"] = "different-confirm-item"
    with pytest.raises(ValueError, match="same 80 unique confirmation IDs"):
        evaluator.confirmation_summary(confirmation_rows)


def test_endpoint_registry_binds_all_three_sources_and_seed_pairs() -> None:
    specs = evaluator._endpoint_specs()

    assert len(specs) == 12
    assert len({spec["endpoint_id"] for spec in specs}) == 12
    assert {spec["source"] for spec in specs} == {"N1-132", "N1-140", "N1-141"}
    assert all(spec["seed"] in (11, 12, 13) and spec["arm"] == "position" for spec in specs)
    assert all(spec["path"].is_file() and face_builder.sha256(spec["path"]) == spec["sha256"] for spec in specs)
