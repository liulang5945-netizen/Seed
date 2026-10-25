"""Distinct prompts and root isolation must survive multiple response branches."""

import pytest

from scripts.training.audit_taiji_n1_plan175_source_metadata import analyze


def fixture():
    return {
        "q": {"message_id": "q", "role": "prompter", "lang": "zh", "parent_id": None},
        "a": {"message_id": "a", "role": "assistant", "lang": "zh", "parent_id": "q", "rank": 0},
        "b": {"message_id": "b", "role": "assistant", "lang": "zh", "parent_id": "q", "rank": 1},
        "next": {"message_id": "next", "role": "user", "lang": "zh", "parent_id": "a"},
        "c": {"message_id": "c", "role": "assistant", "lang": "zh", "parent_id": "next", "rank": 0},
    }


def test_reply_count_does_not_inflate_independent_prompt_count():
    result = analyze(fixture(), [])
    assert result["metadata_zh_reply_candidates"] == 3
    assert result["distinct_prompt_parent_nodes"] == 2
    assert result["root_families"] == 1


def test_protected_family_excludes_siblings_and_descendants():
    result = analyze(fixture(), [{"episode_id": "oasst2:b", "split": "dev"}])
    assert result["pool_reply_candidates"] == 0
    assert result["old_nontrain_root_families"] == 1


def test_unknown_old_source_is_reported_and_deleted_nodes_do_not_count():
    rows = fixture()
    rows["a"]["deleted"] = True
    result = analyze(rows, [{"episode_id": "oasst2:missing", "split": "dev"}])
    assert result["old_episode_ids_unmapped"] == 1
    assert result["metadata_zh_reply_candidates"] == 2


def test_parent_cycles_are_rejected():
    rows = fixture()
    rows["q"]["parent_id"] = "a"
    with pytest.raises(ValueError, match="cycle"):
        analyze(rows, [])
