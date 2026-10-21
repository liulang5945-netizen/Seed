from __future__ import annotations

from scripts.data_prep.build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus import (
    _episode_for_message,
)


def _conversation() -> dict[str, dict[str, object]]:
    rows = (
        {
            "message_id": "user-1",
            "message_tree_id": "tree-1",
            "parent_id": None,
            "role": "prompter",
            "text": "你好",
        },
        {
            "message_id": "assistant-1",
            "message_tree_id": "tree-1",
            "parent_id": "user-1",
            "role": "assistant",
            "text": "你好！",
        },
        {
            "message_id": "user-2",
            "message_tree_id": "tree-1",
            "parent_id": "assistant-1",
            "role": "prompter",
            "text": "你好吗？",
        },
        {
            "message_id": "assistant-2",
            "message_tree_id": "tree-1",
            "parent_id": "user-2",
            "role": "assistant",
            "text": "我很好，谢谢。",
        },
    )
    return {str(row["message_id"]): dict(row) for row in rows}


def test_user_turn_is_a_role_conditioned_target_after_prior_pairs() -> None:
    rows = _conversation()
    episode = _episode_for_message(rows["user-2"], rows)

    assert episode.target_role == "user"
    assert episode.user_input == ""
    assert episode.history == (("你好", "你好！"),)
    assert episode.prefix_text.endswith("<|history_assistant|>\n你好！\n<|user|>\n")
    assert episode.target_text == "你好吗？\n<|end|>\n"


def test_assistant_turn_keeps_current_user_and_prior_history() -> None:
    rows = _conversation()
    episode = _episode_for_message(rows["assistant-2"], rows)

    assert episode.target_role == "assistant"
    assert episode.user_input == "你好吗？"
    assert episode.history == (("你好", "你好！"),)
    assert episode.prefix_text.endswith(
        "<|history_assistant|>\n你好！\n<|user|>\n你好吗？\n<|assistant|>\n"
    )
    assert episode.target_text == "我很好，谢谢。\n<|end|>\n"
