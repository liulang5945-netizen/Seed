"""Unit tests for state-dependent growth decision v2 (`taiji-growth-decision-v2`)."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, r"E:\Seed")

import torch

from taiji.growth_decision_learning_v2 import (
    ACTIONS,
    COST_PER_UNIT,
    GrowthActorCritic,
    finish_from_warmup,
    greedy_action_v2,
    prepare_warmup,
    snapshot_features,
)


def test_warmup_snapshot_fork_deterministic() -> None:
    dl = time.process_time() + 60.0
    s1 = prepare_warmup(11, hard=True, deadline=dl)
    s2 = prepare_warmup(11, hard=True, deadline=dl)
    assert s1.val_err == s2.val_err and s1.p1_acc == s2.p1_acc
    r1 = finish_from_warmup(s1, "grow16", dl)
    r2 = finish_from_warmup(s2, "grow16", dl)
    assert r1 == r2
    assert r1["reward"] == round(r1["combined"] - COST_PER_UNIT * 16, 6)


def test_regime_contrast_easy_vs_hard() -> None:
    dl = time.process_time() + 60.0
    easy_snap = prepare_warmup(7, hard=False, deadline=dl)
    hard_snap = prepare_warmup(7, hard=True, deadline=dl)
    assert easy_snap.val_err < hard_snap.val_err
    e_no = finish_from_warmup(easy_snap, "no_grow", dl)
    e_g24 = finish_from_warmup(easy_snap, "grow24", dl)
    assert e_no["reward"] > e_g24["reward"]
    h_no = finish_from_warmup(hard_snap, "no_grow", dl)
    h_g16 = finish_from_warmup(hard_snap, "grow16", dl)
    assert h_g16["reward"] > h_no["reward"]


def test_actor_critic_shapes_and_determinism() -> None:
    dl = time.process_time() + 60.0
    snap = prepare_warmup(3, hard=False, deadline=dl)
    p1 = GrowthActorCritic(5)
    p2 = GrowthActorCritic(5)
    l1, v1 = p1(snapshot_features(snap))
    l2, v2 = p2(snapshot_features(snap))
    assert l1.shape == (len(ACTIONS),) and v1.dim() == 0
    assert torch.equal(l1, l2) and torch.equal(v1, v2)
    assert 0 <= greedy_action_v2(p1, snap) < len(ACTIONS)
