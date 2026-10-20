from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from scripts.training import build_taiji_n1_plan162_question_surface_schedule_and_face as builder
from scripts.training import run_taiji_n1_plan162_question_surface_matched_continuation as runner


def _synthetic_base_schedule() -> dict:
    slots = []
    rows = []
    step = 0
    known_pattern = (8, 16, 32) * 8
    unknown_pattern = (8, 16, 8, 32) * 6
    for index in range(192):
        parent = f"train-{index:03d}"
        kind = builder.KINDS[index // 48]
        for route in ("raw", "hard"):
            for ordinal in range(24):
                length = (unknown_pattern if kind == "unknown" else known_pattern)[ordinal]
                item = {
                    "id": f"{route}-{parent}-{ordinal:02d}",
                    "parent_id": parent,
                    "split": "train",
                    "kind": kind,
                    "question": f"material of entity{index:03d}?",
                }
                slot = {
                    "step": step,
                    "parent_id": parent,
                    "route": route,
                    "item": item,
                    "item_key": f"key-{parent}-{route}-{ordinal}",
                    "second": None,
                    "second_id": None,
                    "second_key": None,
                }
                slots.append(slot)
                rows.append({
                    "step": step,
                    "parent_id": parent,
                    "route": route,
                    "route_ordinal": ordinal,
                    "kind": kind,
                    "candidate_length": length,
                    "item_id": item["id"],
                    "item_key": slot["item_key"],
                    "second_id": None,
                    "second_key": None,
                })
                step += 1
    return {
        "slots": slots,
        "rows": rows,
        "views": {"fixture": {8: [], 16: [], 32: []}},
        "manifest": {"schedule_sha256": "fixture"},
    }


def test_plan_and_budget_addendum_hashes_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert hashlib.sha256(builder.BUDGET_ADDENDUM.read_bytes()).hexdigest() == builder.BUDGET_ADDENDUM_SHA256


def test_matched_schedule_preserves_parent_routes_and_per_parent_ratios() -> None:
    schedule = builder.build_schedule(_synthetic_base_schedule())
    manifest = schedule["manifest"]
    assert builder.validate_schedule(manifest)["accepted"] is True
    assert len(manifest["schedule_rows"]) == builder.UPDATES_PER_MODEL == 9_216
    assert manifest["total_updates"] == 55_296
    assert manifest["train_score_queries"] == 7_488

    groups = defaultdict(list)
    for row in manifest["schedule_rows"]:
        groups[(row["parent_id"], row["kind"], row["context_length"])].append(row)
        assert row["control_template_id"] == "original"
    for (parent, kind, length), rows in groups.items():
        values = Counter(row["augmented_template_id"] for row in rows)
        first, second = builder._template_pair(kind)
        total = builder._expected_length_counts(kind)[length]
        assert values == Counter({"original": total // 2, first: total // 4, second: total // 4})
        assert len({row["route"] for row in rows}) == 2


def test_surface_question_parser_covers_training_templates_and_original_form() -> None:
    assert runner._question_subject("material of erifmmmlgg?") == "erifmmmlgg"
    expected = {
        "material_active": "what material does erifmmmlgg use?",
        "material_passive": "which material is used by erifmmmlgg?",
        "direction_after_turning": "which direction does erifmmmlgg face after turning?",
        "direction_after_turn": "what direction is erifmmmlgg facing after its turn?",
    }
    for template_id, question in expected.items():
        assert runner._question_subject(question) == "erifmmmlgg"
        assert builder.TRAINING_TEMPLATES[template_id].format(entity="erifmmmlgg") == question


def test_seed132_holdout_templates_are_distinct_from_training_templates() -> None:
    assert set(builder.HOLDOUT_TEMPLATES.values()).isdisjoint(set(builder.TRAINING_TEMPLATES.values()))
    for kind in builder.KINDS:
        template_ids = [builder._face_template_id(kind, index) for index in range(40)]
        assert set(Counter(template_ids).values()) == {20}


def test_all_outputs_stay_in_plan_bound_locations() -> None:
    for path in (
        builder.SOURCE_AUDIT,
        builder.SCHEDULE,
        builder.READINESS,
        builder.FACE,
        builder.RESULT,
        builder.CHECKPOINT_ROOT,
    ):
        assert path.is_relative_to(builder.ROOT)
    assert Path(runner.__file__).resolve() == builder.RUNNER.resolve()
