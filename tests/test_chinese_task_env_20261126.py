"""Tests for the closed-vocabulary Chinese continuous-task harness (`taiji-chinese-continuous-v1`)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")

from taiji.chinese_task_env import (
    ACTION_TO_ID,
    ACTIONS,
    MAX_TOKENS,
    PAD_ID,
    VOCAB,
    VOCAB_TO_ID,
    ChineseTaskEnv,
    eval_policy,
    random_policy,
    scripted_oracle_policy,
)


def test_deterministic_episode() -> None:
    a = ChineseTaskEnv(11)
    b = ChineseTaskEnv(11)
    oa, ob = a.reset(), b.reset()
    assert oa["tokens"] == ob["tokens"]
    for _ in range(oa["turns"]):
        assert a.intended_action == b.intended_action
        a.step(a.intended_action)
        b.step(b.intended_action)


def test_observation_shape_and_vocab() -> None:
    env = ChineseTaskEnv(5)
    obs = env.reset()
    assert len(obs["tokens"]) == MAX_TOKENS
    assert all(0 <= t < len(VOCAB) for t in obs["tokens"])
    assert obs["turns"] in (3, 4, 5)
    assert obs["pending"] == obs["turns"] - 1


def test_pending_decreases_and_state_tracks_place() -> None:
    env = ChineseTaskEnv(7)
    obs = env.reset()
    pending = [obs["pending"]]
    while True:
        o, _, done = env.step(env.intended_action)
        if done:
            break
        pending.append(o["pending"])
    assert pending == sorted(pending, reverse=True)


def test_scripted_oracle_high_random_low() -> None:
    seeds = list(range(4000, 4030))
    oracle = eval_policy(scripted_oracle_policy, seeds)
    rnd = eval_policy(random_policy, seeds)
    assert oracle["mean_step_acc"] >= 0.99
    assert oracle["episode_success"] >= 0.99
    assert rnd["mean_step_acc"] <= 0.30


def test_no_task_id_and_action_ids() -> None:
    env = ChineseTaskEnv(3)
    obs = env.reset()
    assert set(obs) == {"tokens", "step", "turns", "pending"}
    assert len(ACTIONS) == len(ACTION_TO_ID) == 8
    assert VOCAB_TO_ID["请"] == 0
    assert env.step(env.intended_action)[1] == 1.0
    assert env.step(7)[1] in (0.0, 1.0)
    assert PAD_ID == 0


def test_action_phrases_are_renderable_tokens() -> None:
    from taiji.chinese_task_env import ACTIONS

    for action in ACTIONS:
        assert action in VOCAB_TO_ID, f"action phrase {action!r} missing from VOCAB"
    env = ChineseTaskEnv(9)
    env.reset()
    tokens = env._render(0)
    assert VOCAB_TO_ID[env.plan[0]["action"]] in tokens
