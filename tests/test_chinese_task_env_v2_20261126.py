"""Unit tests for closed-vocabulary Chinese continuous-task v2."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")

import torch

from taiji.chinese_task_env_v2 import (
    ACTION_TO_ID,
    ACTIONS,
    HELDOUT_SET,
    MAX_TOKENS,
    PAD_ID,
    PLACE_TOKEN_IDS,
    VOCAB,
    VOCAB_TO_ID,
    ChineseTaskEnvV2,
    eval_policy_v2,
    oracle_policy_v2,
    random_policy_v2,
)


def test_vocab_pad_clean_and_no_action_tokens() -> None:
    assert VOCAB[0] == "<PAD>" and PAD_ID == 0
    assert VOCAB_TO_ID["请"] != PAD_ID
    assert len(VOCAB) == len(set(VOCAB))
    for act in ACTIONS:
        assert act not in VOCAB_TO_ID


def test_coreference_turns_have_zero_place_tokens() -> None:
    coref_seen = 0
    non_coref_seen = 0
    for seed in range(1, 30):
        env = ChineseTaskEnvV2(seed)
        obs = env.reset()
        assert len(obs["tokens"]) == MAX_TOKENS
        assert set(obs) == {"tokens", "step", "turns", "pending"}
        while True:
            tokens = env.current_tokens()
            if env.current_is_coref:
                coref_seen += 1
                assert not (set(tokens) & PLACE_TOKEN_IDS), f"place leak seed={seed}"
            else:
                non_coref_seen += 1
            _, _, done = env.step(env.intended_action)
            if done:
                break
    assert coref_seen >= 2 * 29 and non_coref_seen >= 29


def test_heldout_split_disjoint_and_baselines() -> None:
    for s in range(100, 120):
        tr = ChineseTaskEnvV2(s, heldout=False)
        tr.reset()
        assert tr.turns_data[0]["anchor"] not in HELDOUT_SET
        he = ChineseTaskEnvV2(s, heldout=True)
        he.reset()
        assert he.turns_data[0]["anchor"] in HELDOUT_SET

    seeds = list(range(5000, 5030))
    ora = eval_policy_v2(oracle_policy_v2, seeds)
    rnd = eval_policy_v2(random_policy_v2, seeds, gen=torch.Generator().manual_seed(0))
    assert ora["mean_step_acc"] == 1.0 and ora["coref_acc"] == 1.0
    assert rnd["mean_step_acc"] <= 0.22
    assert len(ACTIONS) == len(ACTION_TO_ID) == 9
