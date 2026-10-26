"""The final face is selected once, by family, before any body is read."""

from scripts.training.evaluate_taiji_n1_plan175_v3_holdout import (
    AUTHORIZED_ANNOTATION_TRANSITION,
    migration_candidate_source_matches,
    select_one_per_family,
)


def test_source_transition_accepts_only_the_explicitly_authorized_pair():
    old, new = AUTHORIZED_ANNOTATION_TRANSITION
    assert migration_candidate_source_matches(old, old)
    assert migration_candidate_source_matches(old, new)
    assert not migration_candidate_source_matches(new, old)
    assert not migration_candidate_source_matches(old, "0" * 64)
    assert not migration_candidate_source_matches("0" * 64, new)


def test_holdout_selection_is_deterministic_and_keeps_one_prompt_per_family():
    rows = [{"family": f"family-{index}", "prompt_id": f"prompt-{index}"} for index in range(64)]
    rows.extend(
        [
            {"family": "family-0", "prompt_id": "prompt-0-sibling"},
            {"family": "family-1", "prompt_id": "prompt-1-sibling"},
        ]
    )
    selected = select_one_per_family(rows)
    assert selected == select_one_per_family(list(reversed(rows)))
    assert len(selected) == 64
    assert len({row["family"] for row in selected}) == 64
    assert all(sum(item["family"] == row["family"] for item in selected) == 1 for row in rows[:64])
