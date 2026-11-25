"""Unit tests for the N33.2 cross-session environment (`taiji/cross_session_env.py`)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.cross_session_env import (
    ACTIONS,
    STATE_DIM,
    CrossSessionEnv,
    evaluate_policy_with_store,
    random_policy,
    run_session1,
    scripted_persistent_policy,
)


def test_session1_and_env_determinism() -> None:
    r1, h1 = run_session1(6005)
    r2, h2 = run_session1(6005)
    assert r1.digest() == r2.digest()
    assert h1 == h2
    e1 = CrossSessionEnv(6005, loaded_record=r1)
    assert e1.obs().shape == (STATE_DIM,)


def test_store_roundtrip_and_scripted_upper_bound() -> None:
    seeds = list(range(6000, 6060))
    scripted = evaluate_policy_with_store(scripted_persistent_policy, seeds, mode="normal")
    assert scripted["store_roundtrip_exact"] is True
    assert scripted["post_s2_persist_exact"] is True
    assert scripted["resolve_rate"] >= 0.95
    assert scripted["mean_return"] >= 4.0

    gen = torch.Generator().manual_seed(17)
    rand = evaluate_policy_with_store(lambda o: random_policy(o, gen), seeds, mode="normal")
    assert rand["mean_return"] < -2.0
    for act in ACTIONS:
        const = evaluate_policy_with_store(lambda _o, a=act: a, seeds, mode="normal")
        assert const["mean_return"] < 0.0


def test_scripted_drops_on_zero_and_swap_counterfactuals() -> None:
    seeds = list(range(6000, 6060))
    normal = evaluate_policy_with_store(scripted_persistent_policy, seeds, mode="normal")
    zeroed = evaluate_policy_with_store(scripted_persistent_policy, seeds, mode="zero_cross")
    swapped = evaluate_policy_with_store(scripted_persistent_policy, seeds, mode="swap_cross")
    assert normal["mean_return"] - zeroed["mean_return"] >= 5.0
    assert normal["mean_return"] - swapped["mean_return"] >= 5.0
    assert zeroed["resolve_rate"] == 0.0
