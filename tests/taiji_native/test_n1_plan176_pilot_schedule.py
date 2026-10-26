"""Ordering comparisons preserve every example's exact exposure."""

from collections import Counter

from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    COLORS,
    OBJECTS,
    build_rows,
    pair_split,
)
from scripts.training.evaluate_taiji_n1_plan176_foundation_pilot import color_variants
from scripts.training.run_taiji_n1_plan176_foundation_pilot import ARMS, schedule


def test_three_arms_share_multiset_and_only_curriculum_arms_share_order():
    rows = [
        r for r in build_rows() if r["split"] == "train" and r["curriculum_level"] in {"F0", "F1"}
    ]
    orders = {arm: schedule(rows, arm) for arm in ARMS}
    assert all(len(value) == 2560 for value in orders.values())
    for value in orders.values():
        assert Counter(i for batch in value for i in batch) == Counter({i: 16 for i in range(640)})
    assert orders["curriculum_v3"] == orders["curriculum_v4"]
    assert orders["flat_v3"] != orders["curriculum_v3"]
    assert all(
        rows[i]["curriculum_level"] == "F0" for b in orders["curriculum_v3"][:640] for i in b
    )
    assert any(rows[i]["curriculum_level"] == "F1" for b in orders["flat_v3"][:640] for i in b)


def test_sealed_counterfactual_remains_an_unseen_pair_and_irrelevant_target_is_fixed():
    row = next(
        r
        for r in build_rows()
        if r["split"] == "sealed" and r["metadata"]["query_kind"] == "primary_color"
    )
    related, irrelevant = color_variants(row)
    assert related["response"] != row["response"]
    oi = OBJECTS.index(row["metadata"]["primary_pair"][0])
    assert pair_split(oi, COLORS.index(related["response"].rstrip("。"))) == "sealed"
    assert irrelevant["response"] == row["response"]
    assert irrelevant["user_input"] != row["user_input"]
