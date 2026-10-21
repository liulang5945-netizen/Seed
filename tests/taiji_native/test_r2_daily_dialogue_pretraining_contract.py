from __future__ import annotations

import pytest

from taiji.language_alignment import LanguageEpisode, LanguageEpisodeCorpus


def _episode(*, target_role: str = "assistant") -> LanguageEpisode:
    return LanguageEpisode(
        episode_id="pretrain-1",
        family_id="tree-1",
        task_family="full_turn_pretraining",
        split="train",
        user_input="你好" if target_role == "assistant" else "",
        response="你好！",
        system="你在进行自然的中文对话。",
        history=(("你好", "你好！"),),
        target_role=target_role,
    )


def test_assistant_episode_payload_and_prefix_remain_legacy_compatible() -> None:
    episode = _episode()

    assert "target_role" not in episode.to_payload()
    assert episode.prefix_text.endswith("<|user|>\n你好\n<|assistant|>\n")
    assert LanguageEpisode.from_payload(episode.to_payload()).digest == episode.digest


def test_user_target_episode_ends_at_user_role_marker_and_roundtrips() -> None:
    episode = _episode(target_role="user")

    assert episode.user_input == ""
    assert episode.prefix_text.endswith("<|user|>\n")
    assert episode.target_text == "你好！\n<|end|>\n"
    assert episode.to_payload()["target_role"] == "user"
    assert LanguageEpisode.from_payload(episode.to_payload()).digest == episode.digest


def test_assistant_target_without_safe_parent_uses_none_prompt_marker() -> None:
    episode = LanguageEpisode(
        episode_id="pretrain-orphan",
        family_id="tree-2",
        task_family="full_turn_pretraining",
        split="train",
        user_input="",
        response="这是一段回复。",
        target_role="assistant",
    )

    assert episode.prefix_text.endswith("<|user|>\n<|none|>\n<|assistant|>\n")


def test_unknown_dialogue_target_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_role must be user or assistant"):
        _episode(target_role="system")


def test_train_only_pretraining_corpus_requires_explicit_opt_in() -> None:
    episode = _episode(target_role="assistant")
    episode = LanguageEpisode(
        episode_id=episode.episode_id,
        family_id=episode.family_id,
        task_family="full_turn_pretraining",
        split="train",
        user_input=episode.user_input,
        response=episode.response,
        system=episode.system,
        history=episode.history,
    )

    with pytest.raises(ValueError, match="needs a dev split"):
        LanguageEpisodeCorpus((episode,))

    corpus = LanguageEpisodeCorpus((episode,), require_eval_splits=False)
    assert corpus.sample_counts == {"train": 1, "dev": 0, "final": 0, "retention": 0}
