"""Tests for the synthetic multimodal + physics environment (zero training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")

from taiji.multimodal_physics_env import (
    ACTIONS,
    FAMILIES,
    HORIZON,
    OUTCOMES,
    TASK_VERSION,
    CorrectablePredictor,
    MultimodalPhysicsEnv,
    OraclePredictor,
    RandomPredictor,
    eval_predictor,
)


def test_deterministic_episodes() -> None:
    a, b = MultimodalPhysicsEnv(31), MultimodalPhysicsEnv(31)
    oa, ob = a.reset(), b.reset()
    assert oa == ob
    for _ in range(HORIZON):
        ra = a.step("push_right")
        rb = b.step("push_right")
        assert ra == rb
        if ra[3]:
            break


def test_modalities_and_no_task_id() -> None:
    env = MultimodalPhysicsEnv(7)
    obs = env.reset()
    assert set(obs) == {"vision", "proprio", "audio", "t"}
    assert len(obs["vision"]) == 6
    assert len(obs["proprio"]) == 3
    assert len(obs["audio"]) == 2
    assert env.state is not None and env.state.family in FAMILIES
    assert TASK_VERSION == "taiji-multimodal-physics-v1"


def test_physics_invariants() -> None:
    env = MultimodalPhysicsEnv(11)
    env.reset()
    env.state.pos = 0.2
    env.state.vel = 0.0
    env.state.family = "incline"
    nxt, outcome = __import__("taiji.multimodal_physics_env", fromlist=["_apply"])._apply(
        env.state, "push_right"
    )
    assert nxt.pos > 0.2
    assert outcome == "向右滑动"
    env.reset()
    env.state.family = "collision"
    env.state.pos = env.state.block_b - 0.05
    env.state.vel = 0.0
    nxt2, outcome2 = __import__("taiji.multimodal_physics_env", fromlist=["_apply"])._apply(
        env.state, "push_right"
    )
    assert outcome2 == "碰撞"


def test_outcome_labels_and_rewards() -> None:
    env = MultimodalPhysicsEnv(3)
    env.reset()
    for _ in range(HORIZON):
        for action in ACTIONS:
            assert env.true_outcome(action) in OUTCOMES
        obs, outcome, reward, done = env.step("push_soft")
        assert outcome in OUTCOMES and reward in (0.0, 1.0)
        if done:
            break


def test_oracle_high_random_low() -> None:
    seeds = list(range(100, 120))
    oracle = eval_predictor(OraclePredictor(), seeds)
    rnd = eval_predictor(RandomPredictor(0), seeds)
    assert oracle["accuracy"] >= 0.99
    assert rnd["accuracy"] <= 0.35


def test_correctable_predictor_improves() -> None:
    pred = CorrectablePredictor()
    early = None
    for seed in range(60):
        env = MultimodalPhysicsEnv(seed)
        env.reset()
        while True:
            for action in ACTIONS:
                truth = env.true_outcome(action)
                predicted = pred.predict(env, action)
                pred.observe(env, action, truth, predicted)
            obs, outcome, reward, done = env.step(ACTIONS[env.state.t % len(ACTIONS)])
            if done:
                break
        if seed == 1:
            early = pred.accuracy()
    late = pred.accuracy()
    assert early is not None and late >= early + 0.05
    assert late >= 0.70
    assert pred.total > 0
