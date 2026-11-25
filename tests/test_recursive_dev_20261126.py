"""Unit tests for N33.4 recursive development environment & model (`taiji/recursive_dev*.py`)."""

from __future__ import annotations

import inspect
import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.recursive_dev import (
    POLICY_IN_DIM,
    ConsequenceModel,
    InterventionPolicy,
    build_policy_input,
)
from taiji.recursive_dev_env import (
    ACTIONS,
    HORIZON,
    STATE_DIM,
    RecursiveDevEnv,
    collect_logged_experience,
    evaluate_recursive_with_store,
    random_policy,
    scripted_oracle_policy,
)


def test_recursive_env_determinism_and_reset() -> None:
    e1 = RecursiveDevEnv(9001)
    e2 = RecursiveDevEnv(9001)
    s1 = e1.reset()
    s2 = e2.reset()
    assert s1.shape == (STATE_DIM,)
    assert torch.equal(s1, s2)
    for act in ("stabilize_fragile", "boost_s0", "boost_s1", "consolidate_all"):
        o1, i1, d1 = e1.step(act)
        o2, i2, d2 = e2.step(act)
        assert torch.equal(o1, o2)
        assert i1 == i2
        assert d1 == d2


def test_c0_calibration_and_store_roundtrip() -> None:
    seeds = list(range(9000, 9060))
    oracle = evaluate_recursive_with_store(scripted_oracle_policy, seeds)
    assert oracle["store_roundtrip_exact"] is True
    assert oracle["mean_net_gain"] >= 0.16
    assert oracle["mean_improving_steps"] >= 3.0

    fixed_parent = evaluate_recursive_with_store(lambda _o: "noop_hold", seeds)
    assert fixed_parent["mean_net_gain"] == 0.0

    gen = torch.Generator().manual_seed(13)
    rand = evaluate_recursive_with_store(lambda o: random_policy(o, gen), seeds)
    assert oracle["mean_net_gain"] - rand["mean_net_gain"] >= 0.05

    for act in ACTIONS:
        const = evaluate_recursive_with_store(lambda _o, a=act: a, seeds)
        assert oracle["mean_net_gain"] - const["mean_net_gain"] >= 0.05


def test_model_signatures_and_experience_collector() -> None:
    assert list(inspect.signature(ConsequenceModel.forward).parameters) == [
        "self",
        "state_and_action",
    ]
    assert list(inspect.signature(InterventionPolicy.forward).parameters) == ["self", "x"]
    cm = ConsequenceModel(seed=1)
    pol = InterventionPolicy(seed=2)
    s = RecursiveDevEnv(9005).reset()
    pin = build_policy_input(s, cm, mask_preds=False)
    pin_masked = build_policy_input(s, cm, mask_preds=True)
    assert pin.shape == (POLICY_IN_DIM,)
    assert torch.equal(pin_masked[STATE_DIM:], torch.zeros(POLICY_IN_DIM - STATE_DIM))
    logits, val = pol(pin)
    assert logits.shape == (len(ACTIONS),)
    assert val.shape == ()

    tr = collect_logged_experience(list(range(9100, 9105)), gen_seed=5)
    assert len(tr) == 5 * HORIZON
