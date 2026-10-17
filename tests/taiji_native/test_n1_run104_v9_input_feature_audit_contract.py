from __future__ import annotations

import json

import pytest

from scripts.training import audit_taiji_n1_hard_neighbor_v9_input_features as audit


def _episode(item_id: str, subset: str, kind: str = "lookup") -> dict:
    target = f"target-{item_id}"
    rows = ["other uses value.", f"{target} uses answer."]
    if kind == "revision":
        rows = [f"{target} uses old.", "other uses value.", f"{target} now uses answer."]
    elif kind == "transition":
        rows = [f"{target} faces north.", "other faces east.", f"{target} turns left."]
    elif kind == "unknown":
        rows = ["other uses value."]
    return {
        "answer": "answer" if kind in {"lookup", "revision"} else "unknown",
        "entities": [target, "other"],
        "id": item_id,
        "kind": kind,
        "observations": rows,
        "question": f"material of {target}?",
        "split": "holdout",
        "subset": subset,
    }


def _confirmation_metadata() -> dict:
    return {
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


def test_feature_extraction_uses_first_target_row_and_only_material_answer_length():
    lookup = audit.extract_input_features(_episode("lookup", "confirm"))
    assert lookup["target_row_position_1based"] == 2
    assert lookup["target_observation_positions_1based"] == [2]
    assert lookup["material_answer_length_chars"] == 6

    revision = audit.extract_input_features(_episode("revision", "confirm", "revision"))
    assert revision["target_row_position_1based"] == 1
    assert revision["target_observation_positions_1based"] == [1, 3]
    assert revision["material_answer_length_chars"] == 6

    transition = audit.extract_input_features(_episode("transition", "confirm", "transition"))
    assert transition["target_row_position_1based"] == 1
    assert transition["target_observation_positions_1based"] == [1, 3]
    assert transition["material_answer_length_chars"] is None

    unknown = audit.extract_input_features(_episode("unknown", "confirm", "unknown"))
    assert unknown["target_row_position_1based"] is None
    assert unknown["target_observation_positions_1based"] == []
    assert unknown["material_answer_length_chars"] is None


def test_manifest_reader_never_decodes_sacrifice_payloads(tmp_path, monkeypatch):
    source = tmp_path / "manifest.json"
    items = [_episode(f"confirm-{i}", "confirm") for i in range(20)]
    items += [_episode(f"sacrifice-{i}", "sacrifice") for i in range(20)]
    items[20]["answer"] = "SACRIFICE_ANSWER_MUST_NOT_DECODE"
    items[20]["observations"] = ["SACRIFICE_OBSERVATIONS_MUST_NOT_DECODE"]
    source.write_text(
        json.dumps({"schema": "test", "items": len(items), "episodes": items}),
        encoding="utf-8",
    )

    original_loads = audit.json.loads
    decoded = []

    def guarded_loads(value, *args, **kwargs):
        decoded.append(value)
        assert "SACRIFICE_ANSWER_MUST_NOT_DECODE" not in value
        assert "SACRIFICE_OBSERVATIONS_MUST_NOT_DECODE" not in value
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(audit.json, "loads", guarded_loads)
    confirm, skipped = audit.stream_manifest(source)
    assert len(confirm) == 20
    assert skipped == 20
    assert all("SACRIFICE_" not in value for value in decoded)


def test_confirmation_reader_skips_prior_payload_and_unneeded_v9_fields(tmp_path, monkeypatch):
    source = tmp_path / "confirmation.json"
    rows = [
        {
            "family": audit.PRIOR_FAMILY,
            "output_hex": "PRIOR_PAYLOAD_MUST_NOT_DECODE",
            "id": "old-item",
        },
        {
            "family": audit.NEW_FAMILY,
            "id": "v9-item",
            "kind": "lookup",
            "seed": 11,
            "arm": "position",
            "free_exact": True,
            "output_hex": "UNUSED_V9_OUTPUT_MUST_NOT_DECODE",
        },
    ]
    source.write_text(json.dumps({**_confirmation_metadata(), "rows": rows}), encoding="utf-8")

    original_loads = audit.json.loads
    decoded = []

    def guarded_loads(value, *args, **kwargs):
        decoded.append(value)
        assert "PRIOR_PAYLOAD_MUST_NOT_DECODE" not in value
        assert "UNUSED_V9_OUTPUT_MUST_NOT_DECODE" not in value
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(audit.json, "loads", guarded_loads)
    metadata, selected, skipped = audit.stream_confirmation(source)
    assert metadata["status"] == "complete"
    assert selected == [
        {
            "family": audit.NEW_FAMILY,
            "id": "v9-item",
            "kind": "lookup",
            "seed": 11,
            "arm": "position",
            "free_exact": True,
        }
    ]
    assert skipped == {audit.PRIOR_FAMILY: 1}
    assert all("MUST_NOT_DECODE" not in value for value in decoded)


def test_feature_grouping_preserves_nine_endpoint_denominators_without_gates():
    items = {}
    for index in range(20):
        kind = audit.KINDS[index // 5]
        item_id = f"item-{index:02d}"
        items[item_id] = {
            "id": item_id,
            "kind": kind,
            "target_row_position_1based": index % 3 + 1 if kind != "unknown" else None,
            "material_answer_length_chars": 6 if kind in {"lookup", "revision"} else None,
        }
    rows = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            for item_id, features in items.items():
                rows.append(
                    {
                        "family": audit.NEW_FAMILY,
                        "id": item_id,
                        "kind": features["kind"],
                        "seed": seed,
                        "arm": arm,
                        "free_exact": seed == 13 and arm == "position",
                    }
                )

    result = audit.group_outcomes(items, rows)
    assert result["confirmation_items"] == 20
    assert result["confirmation_outcome_rows"] == 180
    assert all(
        endpoint["denominator"] == endpoint["free_exact_correct"] + endpoint["free_exact_incorrect"]
        for cell in result["feature_cells"]
        for seed in cell["per_seed_arm"].values()
        for endpoint in seed.values()
    )
    assert all("passed" not in cell and "gate" not in cell for cell in result["feature_cells"])


def test_grouping_rejects_missing_endpoint_outcomes():
    items = {
        f"item-{index:02d}": {
            "id": f"item-{index:02d}",
            "kind": audit.KINDS[index // 5],
            "target_row_position_1based": None,
            "material_answer_length_chars": None,
        }
        for index in range(20)
    }
    rows = [
        {
            "family": audit.NEW_FAMILY,
            "id": item_id,
            "kind": features["kind"],
            "seed": seed,
            "arm": arm,
            "free_exact": False,
        }
        for seed in audit.SEEDS
        for arm in audit.ARMS
        for item_id, features in items.items()
    ]
    with pytest.raises(ValueError, match="same 20 v9 confirmation items"):
        audit.group_outcomes(items, rows[:-1])


def test_protocol_deviation_is_explicitly_carried_into_receipts():
    assert audit.PRE_EXECUTION_SCOPE_DEVIATION == {
        "occurred": True,
        "affected_manifest_items": 20,
        "boundary": "v9 sacrifice item payload fields were decoded during setup inspection",
        "used_in_analysis": False,
        "reader_decodes_sacrifice_payloads": False,
        "v7_v8_or_v3_v5_v6_sources_read": False,
    }
