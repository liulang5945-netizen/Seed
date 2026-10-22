from __future__ import annotations

import copy

import pytest

from scripts.training import analyze_taiji_n1_v23_itemwise_errors_after_n144 as analyzer
from scripts.training import seal_ast


def _synthetic_inputs() -> tuple[list[dict], dict]:
    face_items = []
    positions = {
        "lookup": analyzer.EXPECTED_FACE_POSITIONS["lookup"],
        "revision": analyzer.EXPECTED_FACE_POSITIONS["revision"],
        "transition": analyzer.EXPECTED_FACE_POSITIONS["transition"],
        "unknown": (),
    }
    for kind in analyzer.KINDS:
        for local in range(20):
            face_items.append(
                {
                    "id": f"v23-confirm-{kind}-{local:03d}",
                    "subset": "confirm",
                    "split": "holdout",
                    "kind": kind,
                    "target_start_index": (
                        positions[kind][local % len(positions[kind])] if positions[kind] else None
                    ),
                    "answer_length": 6 + local % 2 if kind in ("lookup", "revision") else None,
                }
            )

    rows = []
    for endpoint_id, (source, branch, seed) in analyzer.expected_endpoints().items():
        if source == "N1-141" and branch == "candidate":
            correct_per_kind = 10
        elif source == "N1-132":
            correct_per_kind = 8
        elif source == "N1-140":
            correct_per_kind = 12
        else:
            correct_per_kind = 9
        for item in face_items:
            local = int(item["id"].rsplit("-", 1)[-1])
            rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "source": source,
                    "branch": branch,
                    "seed": seed,
                    "subset": "confirm",
                    "kind": item["kind"],
                    "id": item["id"],
                    "free_exact": local < correct_per_kind,
                }
            )
    return rows, {
        "schema": "n1-v23-confirmation-face-v1",
        "seed": 123,
        "plan_sha256": analyzer.N144_PLAN_SHA256,
        "episodes": face_items,
    }


def test_summary_recomputation_matches_expected_family_and_gate_counts() -> None:
    rows, _face = _synthetic_inputs()
    summary = analyzer.recompute_confirmation_summary(rows)

    assert summary["candidate_exact"] == 120
    assert summary["n132_parent_exact"] == 96
    assert summary["n140_block_candidate_exact"] == 144
    assert summary["n141_control_exact"] == 108
    assert summary["candidate_advantage_over_n132_parent"] == 24
    assert summary["candidate_advantage_over_n140_block"] == -24
    assert summary["candidate_advantage_over_n141_control"] == 12
    assert summary["gates"]["candidate_each_seed_at_least_40_of_80"] is True
    assert summary["gates"]["candidate_each_seed_each_kind_at_least_8_of_20"] is True
    assert summary["confirmation_pass"] is False


def test_joined_attribution_covers_position_length_and_paired_outcome_cells() -> None:
    rows, face = _synthetic_inputs()
    attribution = analyzer.build_attribution(rows, face)

    assert len(attribution["per_endpoint_kind"]) == 12
    assert attribution["per_endpoint_kind"]["N1-141/seed11/candidate"]["lookup"] == {
        "items": 20,
        "exact": 10,
        "exact_rate": 0.5,
    }
    assert len(attribution["n141_candidate_by_target_position"]) == 45
    assert all(row["items"] == 4 for row in attribution["n141_candidate_by_target_position"])
    assert len(attribution["n141_candidate_by_answer_length"]) == 12
    assert all(row["items"] == 10 for row in attribution["n141_candidate_by_answer_length"])
    assert len(attribution["n141_candidate_paired_transitions"]) == 36

    parent = next(
        row
        for row in attribution["n141_candidate_paired_transitions"]
        if row["comparison"] == "n132_parent" and row["seed"] == 11 and row["kind"] == "lookup"
    )
    assert parent["matrix"] == {
        "candidate_correct_comparator_correct": 8,
        "candidate_correct_comparator_wrong": 2,
        "candidate_wrong_comparator_correct": 0,
        "candidate_wrong_comparator_wrong": 10,
    }


def test_join_refuses_incomplete_or_misaligned_consumed_rows() -> None:
    rows, face = _synthetic_inputs()

    with pytest.raises(analyzer.Refuse, match="960 rows"):
        analyzer.validate_and_join(rows[:-1], face)

    altered = copy.deepcopy(rows)
    altered[0]["id"] = "not-in-v23-face"
    with pytest.raises(analyzer.Refuse, match="no face item"):
        analyzer.validate_and_join(altered, face)


def test_join_refuses_face_position_that_is_outside_the_frozen_schedule() -> None:
    rows, face = _synthetic_inputs()
    changed = copy.deepcopy(face)
    changed["episodes"][0]["target_start_index"] = 3

    with pytest.raises(analyzer.Refuse, match="unexpected target position"):
        analyzer.validate_and_join(rows, changed)


def test_plan_and_consumed_report_hash_constants_are_current() -> None:
    assert analyzer.sha256(analyzer.PLAN) == analyzer.PLAN_SHA256
    assert analyzer.sha256(analyzer.N144_PLAN) == analyzer.N144_PLAN_SHA256
    # 元封印碰撞修复: FACE 清单自身也是重冻结对象，字节钉切 git 历史核验
    assert seal_ast.verify_path_pin(analyzer.FACE, analyzer.FACE_SHA256)
    assert analyzer.sha256(analyzer.RESULT) == analyzer.RESULT_SHA256
