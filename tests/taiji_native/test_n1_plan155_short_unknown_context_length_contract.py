from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.training import build_taiji_n1_plan155_short_unknown_context_length as builder
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as runner


def _base_schedule() -> dict:
    slots = []
    step = 0
    for index in range(192):
        parent = f"train-{index:03d}"
        kind = builder.KINDS[index // 48]
        for route in ("raw", "hard"):
            for ordinal in range(24):
                item = {
                    "id": f"{route}-{parent}",
                    "parent_id": parent,
                    "split": "train",
                    "kind": kind,
                }
                slots.append(
                    {
                        "step": step,
                        "parent_id": parent,
                        "route": route,
                        "item": item,
                        "item_key": f"item-{parent}-{route}-{ordinal}",
                        "second_key": None,
                        "second_id": None,
                    }
                )
                step += 1
    return {
        "slots": slots,
        "views": {
            8: ("short-context-view",),
            16: ("medium-context-view",),
            32: ("long-context-view",),
        },
        "manifest": {
            "answerable_binding_queries_per_model": 11_520,
            "schedule_sha256": "base-schedule-hash",
        },
    }


def test_plan_digest_matches_frozen_builder_binding() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert (
        hashlib.sha256(builder.CORRECTION_ADDENDUM.read_bytes()).hexdigest()
        == builder.CORRECTION_SHA256
    )


def test_corrected_schedule_retains_bound_context_views() -> None:
    base = _base_schedule()
    schedule = builder.build_schedule(base)
    assert schedule["views"] is base["views"]


def test_corrective_run_keeps_prior_one_shot_outputs_separate() -> None:
    assert runner.ATTEMPT != builder.PRIOR_FAILED_ATTEMPT
    assert runner.PROGRESS != builder.PRIOR_FAILED_PROGRESS
    assert runner.RESULT != builder.PRIOR_FAILED_RESULT
    assert builder.CHECKPOINT_ROOT != builder.PRIOR_CHECKPOINT_ROOT


def test_unknown_only_short_context_rebalance_matches_frozen_counts() -> None:
    schedule = builder.build_schedule(_base_schedule())
    manifest = schedule["manifest"]

    assert len(schedule["rows"]) == builder.UPDATES_PER_MODEL == 9_216
    assert manifest["updates_by_kind"] == {kind: 2_304 for kind in builder.KINDS}
    assert manifest["candidate_updates_by_kind_length"] == {
        "lookup": {"8": 768, "16": 768, "32": 768},
        "revision": {"8": 768, "16": 768, "32": 768},
        "transition": {"8": 768, "16": 768, "32": 768},
        "unknown": {"8": 1_152, "16": 576, "32": 576},
    }
    assert manifest["control_updates_by_length"] == {"8": 3_072, "16": 3_072, "32": 3_072}
    assert manifest["answerable_binding_queries_per_model"] == 11_520
    assert builder.validate_schedule(manifest)["accepted"] is True


def test_unknown_slot_order_is_interleaved_per_parent_route() -> None:
    schedule = builder.build_schedule(_base_schedule())
    rows = schedule["manifest"]["schedule_rows"]
    unknown = [row for row in rows if row["kind"] == "unknown"]
    expected = list(builder.UNKNOWN_LENGTH_PATTERN)
    for parent in sorted({row["parent_id"] for row in unknown}):
        for route in ("raw", "hard"):
            slots = sorted(
                (row for row in unknown if row["parent_id"] == parent and row["route"] == route),
                key=lambda row: row["route_ordinal"],
            )
            assert [row["candidate_length"] for row in slots] == expected
            assert len(slots) == 24


def test_parent_kind_mismatch_and_schedule_mutation_are_rejected() -> None:
    invalid = _base_schedule()
    invalid["slots"][0]["item"]["kind"] = "revision"
    with pytest.raises(builder.Refuse, match="multiple task kinds"):
        builder.build_schedule(invalid)

    schedule = builder.build_schedule(_base_schedule())
    changed = dict(schedule["manifest"])
    changed["schedule_rows"] = list(changed["schedule_rows"])
    changed["schedule_rows"][0] = {**changed["schedule_rows"][0], "candidate_length": 32}
    with pytest.raises(builder.Refuse, match="digest mismatch"):
        builder.validate_schedule(changed)


def test_progress_events_use_one_append_position_for_training_and_scores() -> None:
    stream = io.StringIO()
    runner._write_jsonl(stream, {"event": "training_update", "step": 0})
    runner._write_jsonl(stream, {"event": "original_train_gate_row", "id": "train-0"})
    runner._write_jsonl(stream, {"event": "length_score_row", "id": "train-0"})
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["event"] for row in rows] == [
        "training_update",
        "original_train_gate_row",
        "length_score_row",
    ]

    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "append_jsonl" not in source
    assert 'PROGRESS.open("x"' in source
