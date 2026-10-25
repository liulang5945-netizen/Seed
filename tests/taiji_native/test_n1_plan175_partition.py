"""Exercise isolation and deterministic selection before any body access."""

import pytest

from scripts.training.audit_taiji_n1_plan175_train_quality import check_chain
from scripts.training.partition_taiji_n1_plan175_metadata import partition


def source():
    rows = {}
    for i in range(5):
        rows[f"q{i}"] = {"parent_id": None, "role": "user", "lang": "zh"}
        rows[f"a{i}"] = {"parent_id": f"q{i}", "role": "assistant", "lang": "zh", "rank": 0}
    return rows


def test_deterministic_order_and_old_family_isolation():
    rows = source()
    old = [{"episode_id": "oasst:a0", "split": "train"}, {"episode_id": "oasst:a1", "split": "dev"}]
    result = partition(rows, old, 2)
    assert result == partition(dict(reversed(list(rows.items()))), old, 2)
    assert "q0" not in result["holdout_families"]
    assert "q1" in result["protected_families"]
    assert all(
        x["family"] != "q1" for x in result["train_candidates"] + result["holdout_candidates"]
    )
    assert not ({x["family"] for x in result["train_candidates"]} & set(result["holdout_families"]))


def test_no_fallback_to_lower_rank_and_no_duplicate_prompt_targets():
    rows = source()
    rows["b0"] = {**rows["a0"], "rank": 1}
    result = partition(rows, [], 1)
    assert len(result["train_candidates"]) + len(result["holdout_candidates"]) == 5
    rows["b0"]["rank"] = 0
    with pytest.raises(ValueError, match="multiple rank-zero"):
        partition(rows, [], 1)


def test_unknown_old_id_and_inadequate_fresh_pool_fail_closed():
    with pytest.raises(ValueError, match="missing ancestor"):
        partition(source(), [{"episode_id": "oasst:missing", "split": "dev"}], 1)
    with pytest.raises(ValueError, match="insufficient"):
        partition(source(), [], 64)


def test_full_target_limit_does_not_silently_truncate_or_drop_context():
    chain = [
        {"role": "user", "lang": "zh", "text": "请解释。"},
        {"role": "assistant", "lang": "zh", "text": "一个完整回答。"},
    ]
    assert check_chain(chain) == "structurally_eligible_not_semantically_verified"
    chain[-1]["text"] = "汉" * 171
    assert check_chain(chain) == "target_over_512_bytes"
    chain[0]["deleted"] = True
    assert check_chain(chain) == "invalid_context"
