from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_plan163_transition_template_schedule as schedule_builder


def _base_schedule_rows() -> list[dict]:
    rows = []
    step = 0
    for kind in ("lookup", "revision", "transition", "unknown"):
        for parent_index in range(48):
            parent = f"{kind}-parent-{parent_index:02d}"
            route_length_counts = (
                ("raw", 8, 8),
                ("hard", 8, 8),
                ("raw", 16, 8),
                ("hard", 16, 8),
                ("raw", 32, 8),
                ("hard", 32, 8),
            )
            if kind == "unknown":
                route_length_counts = (
                    ("raw", 8, 12),
                    ("hard", 8, 12),
                    ("raw", 16, 6),
                    ("hard", 16, 6),
                    ("raw", 32, 6),
                    ("hard", 32, 6),
                )
            for route, length, count in route_length_counts:
                templates = ["original"] * count
                if kind == "transition":
                    half = count // 2
                    templates = (
                        ["original"] * half
                        + ["direction_after_turn"] * (count // 4)
                        + ["direction_after_turning"] * (count // 4)
                    )
                for template in templates:
                    rows.append(
                        {
                            "step": step,
                            "parent_id": parent,
                            "route": route,
                            "kind": kind,
                            "context_length": length,
                            "item_id": f"item-{step}",
                            "item_key": f"key-{step}",
                            "second_id": f"second-{step}" if kind == "unknown" else None,
                            "second_key": f"second-key-{step}" if kind == "unknown" else None,
                            "augmented_template_id": template,
                        }
                    )
                    step += 1
    return rows


def test_only_transition_turning_template_is_replaced_and_denominators_hold() -> None:
    base_rows = _base_schedule_rows()

    derived = schedule_builder.derive_arm_rows(base_rows)
    summary = schedule_builder.validate_derived_rows(base_rows, derived)

    assert len(base_rows) == 9_216
    assert summary["replaced_transition_slots"] == 576
    assert Counter(int(row["context_length"]) for row in derived) == Counter(
        {8: 3_456, 16: 2_880, 32: 2_880}
    )
    for base, row in zip(base_rows, derived, strict=True):
        assert row["n163_control_template_id"] == base["augmented_template_id"]
        expected = base["augmented_template_id"]
        if base["kind"] == "transition" and expected == schedule_builder.BAD_TEMPLATE:
            expected = schedule_builder.PREFERRED_TEMPLATE
        assert row["n163_candidate_template_id"] == expected


def test_plan_hash_matches_the_frozen_contract() -> None:
    assert schedule_builder.sha256(schedule_builder.PLAN) == schedule_builder.PLAN_SHA256
    assert (
        schedule_builder.sha256(schedule_builder.REBINDING_PLAN)
        == schedule_builder.REBINDING_PLAN_SHA256
    )
    assert (
        schedule_builder.sha256(schedule_builder.SUPERSEDED_MANIFEST)
        == schedule_builder.SUPERSEDED_MANIFEST_SHA256
    )
    assert (
        schedule_builder.sha256(schedule_builder.SUPERSEDED_READINESS)
        == schedule_builder.SUPERSEDED_READINESS_SHA256
    )
