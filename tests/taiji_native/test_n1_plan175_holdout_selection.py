"""The final face is selected once, by family, before any body is read."""

from scripts.training.evaluate_taiji_n1_plan175_v3_holdout import select_one_per_family


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
