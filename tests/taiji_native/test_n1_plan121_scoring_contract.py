from __future__ import annotations

import json
from hashlib import sha256

import pytest

from scripts.training import eval_taiji_n1_v11_after_n120 as evaluator


def _cohort_rows(*, position_per_kind: dict[str, int], control_per_kind: int) -> list[dict]:
    rows = []
    for seed in evaluator.SEEDS:
        for arm in evaluator.ARMS:
            for kind in evaluator.KINDS:
                per_kind = position_per_kind[kind] if arm == "position" else control_per_kind
                for item_index in range(5):
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "kind": kind,
                            "id": f"{kind}-{item_index}",
                            "split": "holdout",
                            "free_exact": item_index < per_kind,
                        }
                    )
    return rows


def test_range_gate_requires_six_of_nine_usable_candidate_endpoints() -> None:
    rows = []
    for index in range(9):
        selected = []
        for item_index in range(20):
            kind = evaluator.KINDS[item_index % len(evaluator.KINDS)]
            selected.append(
                {
                    "id": f"i{item_index}",
                    "kind": kind,
                    "split": "holdout",
                    "free_exact": True,
                }
            )
        addressable = [row for row in selected if row["kind"] != "unknown"]
        for row in addressable[:6]:
            row["free_exact"] = False
        rows.append(
            {
                "seed": evaluator.SEEDS[index % 3],
                "arm": evaluator.ARMS[index // 3],
                "rows": selected,
            }
        )

    result = evaluator._range_summary(rows)

    assert result["usable_endpoints"] == 9
    assert result["range_check_passed"] is True
    for row in rows[0]["rows"]:
        row["free_exact"] = True
    for row in [item for item in rows[0]["rows"] if item["kind"] != "unknown"][:5]:
        row["free_exact"] = False
    result = evaluator._range_summary(rows)
    assert result["usable_endpoints"] == 8


def test_confirmation_gates_keep_seed_kind_and_each_control_thresholds() -> None:
    passing = evaluator._cohort_gates(
        _cohort_rows(position_per_kind=dict.fromkeys(evaluator.KINDS, 4), control_per_kind=2),
        "synthetic",
    )
    assert all(passing["gates"].values())
    assert passing["all_four_gates"] is True

    one_kind_short = evaluator._cohort_gates(
        _cohort_rows(
            position_per_kind={"lookup": 1, "revision": 5, "transition": 5, "unknown": 5},
            control_per_kind=2,
        ),
        "synthetic",
    )
    assert one_kind_short["gates"]["each_position_seed"] is True
    assert one_kind_short["gates"]["each_position_seed_kind"] is False
    assert one_kind_short["gates"]["advantage_over_terminal"] is True
    assert one_kind_short["gates"]["advantage_over_cut"] is True


def test_one_shot_query_modes_require_explicit_owner_approval() -> None:
    with pytest.raises(PermissionError, match="owner approval"):
        evaluator.run_range_check(owner_approved=False)
    with pytest.raises(PermissionError, match="owner approval"):
        evaluator.run_confirmation(owner_approved=False)


def test_scorer_readiness_rejects_a_source_exclusion_regression(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source-readiness.json"
    source.write_text("{}", encoding="utf-8")
    checkpoint_hashes = {}
    for index in range(36):
        checkpoint = tmp_path / f"checkpoint-{index}.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        checkpoint_hashes[str(checkpoint)] = sha256(checkpoint.read_bytes()).hexdigest()

    report = {
        "complete": True,
        "status": "ready_for_owner_approval",
        "schema": "n1-plan121-scorer-zero-query-readiness-v6",
        "plan_sha256": evaluator.N121_PLAN_SHA256,
        "source_readiness_sha256": sha256(source.read_bytes()).hexdigest(),
        "excluded_entity_count": 1918,
        "excluded_material_count": 1633,
        "n110_material_value_map_sha256": (
            "dba1d9eb8fec6d60e6b1ceca496bce3b3386b05d1539ad88696177db0b9e9ba8"
        ),
        "n110_schedule_sha256": "39edaf58020f6cf18680838a6d7576eafef5c7c9286cfd1ea40b93a581851380",
        "n110_material_value_map_entries": 288,
        "n110_mapped_training_values_excluded": 288,
        "n121_source_correction_sha256": "896dad354e6c81b7a1157746fe4e4bc5dcc8fcd266a7707dc09f476d7bb7aba7",
        "v7_confirm_payload_fields_decoded": False,
        "v8_confirm_payload_fields_decoded": False,
        "checkpoint_hash_count": 36,
        "checkpoint_hashes": checkpoint_hashes,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "face_generated": False,
        "source_hashes": {},
    }
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(evaluator, "SCORER_READINESS", readiness)
    monkeypatch.setattr(evaluator, "FACE_READINESS", source)

    report["n110_mapped_training_values_excluded"] = 287
    readiness.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or incomplete"):
        evaluator._verify_scorer_readiness()
