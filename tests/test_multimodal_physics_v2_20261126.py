"""Unit tests for multimodal physics v2 (`taiji-multimodal-physics-v2`)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")

import torch

from taiji.multimodal_physics_env_v2 import (
    ACTIONS,
    FAMILIES,
    HORIZON,
    OUTCOMES,
    ConsequenceNetV2,
    MultimodalPhysicsEnvV2,
    ablate_barrier_collision_recall,
    apply_physics_v2,
    collect_v2,
    eval_metrics_v2,
    train_net_v2,
)


def test_deterministic_and_no_family_id_in_obs() -> None:
    a, b = MultimodalPhysicsEnvV2(17), MultimodalPhysicsEnvV2(17)
    oa, ob = a.reset(family="ramp_bumper"), b.reset(family="ramp_bumper")
    assert oa == ob
    assert set(oa) == {"vision", "proprio", "audio", "t"}
    assert len(oa["vision"]) == 6
    assert len(oa["proprio"]) == 3
    assert len(oa["audio"]) == 2
    for _ in range(HORIZON):
        ra = a.step("push_right")
        rb = b.step("push_right")
        assert ra == rb
        if ra[3]:
            break


def test_shared_law_independent_of_family_label() -> None:
    env = MultimodalPhysicsEnvV2(5)
    env.reset(family="ramp_bumper")
    assert env.state is not None
    s1 = env.state
    s2 = type(s1)(
        family="rough_collision",
        pos=s1.pos,
        vel=s1.vel,
        barrier_pos=s1.barrier_pos,
        slope_angle=s1.slope_angle,
        roughness=s1.roughness,
        prev_impulse=s1.prev_impulse,
        t=s1.t,
        last_outcome=s1.last_outcome,
    )
    assert s1.observe() == s2.observe()
    for act in ACTIONS:
        n1, o1 = apply_physics_v2(s1, act)
        n2, o2 = apply_physics_v2(s2, act)
        assert o1 == o2 and abs(n1.pos - n2.pos) < 1e-9 and abs(n1.vel - n2.vel) < 1e-9


def test_both_families_produce_collisions_and_all_outcomes() -> None:
    coll_idx = OUTCOMES.index("碰撞")
    for fam in FAMILIES:
        x, y = collect_v2(101, fam, 120)
        frac = float((y == coll_idx).float().mean())
        assert frac >= 0.10, f"family={fam} collision frac too low: {frac}"
        assert x.shape[1] == 15


def test_net_and_barrier_ablation_behavior() -> None:
    x_tr, y_tr = collect_v2(7, "ramp_bumper", 60)
    net = train_net_v2(x_tr, y_tr, seed=7, steps=300)
    m = eval_metrics_v2(net, x_tr, y_tr)
    assert m["accuracy"] >= 0.85
    assert m["collision_recall"] >= 0.60
    abl = ablate_barrier_collision_recall(net, x_tr, y_tr)
    assert abl["collision_recall_drop"] >= 0.40
    net2 = ConsequenceNetV2(7)
    assert torch.equal(next(ConsequenceNetV2(7).parameters()), next(net2.parameters()))
