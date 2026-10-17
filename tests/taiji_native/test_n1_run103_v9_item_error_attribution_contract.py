from __future__ import annotations

import json

import pytest

from scripts.training import audit_taiji_n1_hard_neighbor_v9_item_errors as audit


def _row(seed: int, arm: str, item: int, *, correct: bool, output: str) -> dict:
    return {
        "seed": seed,
        "arm": arm,
        "id": f"item-{item:02d}",
        "kind": audit.KINDS[item // 5],
        "family": audit.NEW_FAMILY,
        "free_exact": correct,
        "output_hex": output,
    }


def _complete_rows() -> list[dict]:
    rows = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            for item in range(20):
                rows.append(
                    _row(
                        seed,
                        arm,
                        item,
                        correct=(seed == 11 and arm == "position") or (item % 2 == 0),
                        output=f"{seed:02x}{item:02x}{int(arm == 'position'):02x}",
                    )
                )
    return rows


def test_pair_summary_partitions_correctness_and_counts_exact_output_agreement():
    position = [
        _row(11, "position", index, correct=value, output=output)
        for index, (value, output) in enumerate(
            ((True, "aa"), (True, "bb"), (False, "cc"), (False, "dd"), (False, "ee"))
        )
    ]
    cut = [
        _row(11, "cut", index, correct=value, output=output)
        for index, (value, output) in enumerate(
            ((True, "aa"), (False, "ff"), (True, "cc"), (False, "dd"), (False, "99"))
        )
    ]
    result = audit.compare_pair(position, cut)
    assert result == {
        "both_correct": 1,
        "position_only_correct": 1,
        "control_only_correct": 1,
        "both_wrong": 2,
        "output_bytes_identical": 3,
        "denominator": 5,
    }


def test_complete_attribution_has_fixed_seed_kind_denominators_without_rejudging_gates():
    result = audit.analyze_rows(_complete_rows())
    assert result["rows_analyzed"] == 180
    assert result["per_kind_denominator_per_seed_pair"] == 5
    assert result["per_seed_denominator_per_arm_pair"] == 20
    assert result["overall_denominator_per_arm_pair"] == 60
    cut = result["comparisons"]["position_vs_cut"]
    assert set(cut["per_seed_kind"]) == {
        f"{seed}_{kind}" for seed in audit.SEEDS for kind in audit.KINDS
    }
    assert all(cell["denominator"] == 5 for cell in cut["per_seed_kind"].values())
    assert all(cell["denominator"] == 20 for cell in cut["per_seed"].values())
    assert cut["overall"]["denominator"] == 60
    assert "gate" not in cut and "passed" not in cut


def test_attribution_rejects_missing_or_duplicate_seed_arm_items():
    rows = _complete_rows()
    with pytest.raises(ValueError, match="exactly twenty rows"):
        audit.analyze_rows(rows[:-1])
    with pytest.raises(ValueError, match="duplicate seed/arm/item"):
        audit.analyze_rows(rows + [rows[0]])


def test_attribution_rejects_prior_family_rows():
    rows = _complete_rows()
    rows[0] = dict(rows[0], family=audit.PRIOR_FAMILY)
    with pytest.raises(ValueError, match="only the v9 new family"):
        audit.analyze_rows(rows)


def test_stream_reader_skips_prior_family_payload_without_decoding(tmp_path, monkeypatch):
    source = tmp_path / "confirmation.json"
    metadata = {
        "schema": "n1-hard-neighbor-v9-confirmation-result-v1",
        "status": "complete",
        "complete": True,
        "passed": False,
        "confirm_items": 20,
        "model_queries": 360,
        "v9_confirm_model_queries": 360,
        "v9_new_family_model_queries": 180,
        "prior_family_descriptive_model_queries": 180,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "sources_unchanged": True,
        "can_promote": False,
        "TRAIN_SHA": None,
    }
    rows = [
        _row(11, "position", 0, correct=True, output="0102"),
        {
            "seed": 11,
            "arm": "position",
            "id": "old-item",
            "kind": "lookup",
            "family": audit.PRIOR_FAMILY,
            "free_exact": False,
            "output_hex": "PRIVATE_PRIOR_OUTPUT_MUST_NOT_DECODE",
        },
    ]
    source.write_text(json.dumps({**metadata, "rows": rows}), encoding="utf-8")
    original_loads = audit.json.loads
    decoded_payloads = []

    def guarded_loads(value, *args, **kwargs):
        decoded_payloads.append(value)
        assert "PRIVATE_PRIOR_OUTPUT_MUST_NOT_DECODE" not in value
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(audit.json, "loads", guarded_loads)
    selected_metadata, selected_rows, skipped = audit.stream_confirmation(source)
    assert selected_metadata["status"] == "complete"
    assert len(selected_rows) == 1
    assert selected_rows[0]["family"] == audit.NEW_FAMILY
    assert skipped == {audit.PRIOR_FAMILY: 1}
    assert decoded_payloads
