"""Unit tests for the N33.1 session simulator (`taiji/choice_simulator.py`).

Zero-training fixture checks: six actions, determinism, scripted baseline beating random,
reward sanity (rest restores energy when tired), and state feature bounds.
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.choice_simulator import (
    CHOICES,
    HORIZON,
    STATE_DIM,
    ChoiceSessionEnv,
    State,
    evaluate_policy,
    random_policy,
    run_episode,
    scripted_policy,
    seed_from,
)


def test_six_choices_and_state_dim() -> None:
    assert len(CHOICES) == 6
    assert len(set(CHOICES)) == 6
    assert State().vector().shape == (STATE_DIM,)


def test_deterministic_same_seed() -> None:
    a = run_episode(scripted_policy, 123)
    b = run_episode(scripted_policy, 123)
    assert a == b
    env1 = ChoiceSessionEnv(seed_from("x", 1))
    env2 = ChoiceSessionEnv(seed_from("x", 1))
    s1 = env1.reset()
    s2 = env2.reset()
    assert s1.vector().tolist() == s2.vector().tolist()


def test_scripted_beats_random() -> None:
    seeds = list(range(40))
    scripted = evaluate_policy(scripted_policy, seeds)
    rand_gen = torch.Generator().manual_seed(7)
    random = evaluate_policy(lambda s: random_policy(s, rand_gen), seeds)
    assert scripted["mean_return"] > random["mean_return"] + 1.0


def test_rest_restores_energy() -> None:
    env = ChoiceSessionEnv(1)
    state = env.reset()
    state.energy = 0.2
    env.state = state
    before = state.energy
    state2, reward, _ = env.step("rest")
    assert state2.energy > before
    assert reward > 0


def test_state_features_bounded() -> None:
    env = ChoiceSessionEnv(5)
    state = env.reset()
    for action in CHOICES * 4:
        state, _, done = env.step(action)
        vec = state.vector()
        assert torch.all(vec >= 0.0) and torch.all(vec <= 1.0)
        if done:
            state = env.reset()


def test_wait_requires_no_pending() -> None:
    env = ChoiceSessionEnv(9)
    state = env.reset()
    state.pending = 0.0
    env.state = state
    _, reward_wait, _ = env.step("wait_wake")
    env2 = ChoiceSessionEnv(9)
    env2.reset()
    _, reward_bad, _ = env2.step("wait_wake")
    assert reward_wait > reward_bad


def test_horizon_terminates() -> None:
    env = ChoiceSessionEnv(3)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step("continue_task")
        steps += 1
        assert steps <= HORIZON
    assert steps == HORIZON
