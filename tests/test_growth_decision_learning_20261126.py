"""Tests for learned growth decisions (zero heavy training)."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, r"E:\Seed")

from taiji.growth_decision_learning import (
    ACTIONS,
    GrowthDecisionPolicy,
    features,
    greedy_action,
    run_episode,
)


def test_episode_deterministic() -> None:
    deadline = time.process_time() + 60
    a = run_episode(21, "grow8", deadline)
    b = run_episode(21, "grow8", deadline)
    assert a["combined"] == b["combined"]
    assert a["units"] == b["units"] == 8


def test_actions_change_capacity_and_cost() -> None:
    deadline = time.process_time() + 60
    small = run_episode(3, "no_grow", deadline)
    big = run_episode(3, "grow24", deadline)
    assert small["units"] == 0 and big["units"] == 24
    assert big["reward"] == round(big["combined"] - 0.001 * 24, 6)


def test_policy_shapes_and_greedy() -> None:
    policy = GrowthDecisionPolicy(5)
    deadline = time.process_time() + 60
    probe = run_episode(9, "no_grow", deadline)
    feats = features(probe)
    logits = policy(feats)
    assert logits.shape == (len(ACTIONS),)
    assert 0 <= greedy_action(policy, probe) < len(ACTIONS)
