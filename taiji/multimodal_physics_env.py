"""Synthetic multimodal + physics environment (`taiji-multimodal-physics-v1`, zero training).

Facility for the "多模态感知、行动和真实结果反馈 → 可修正的物理因果理解" clause (coverage gap E).
Two dynamics families share one interface:

* ``incline``: a ball on a ramp; actions push it left/right/soft or hold; the ball slides with
  friction and can stop.
* ``collision``: a movable block A and a fixed block B; pushing A right can make it collide.

Each step exposes three modalities (vision / proprio / audio), a discrete causal outcome
(``向右滑动``/``向左滑动``/``停下``/``碰撞``/``无变化``) and a scalar reward. A **scripted oracle
predictor** (true-physics lookahead, control only), a uniform random predictor, and a
**correctable count-based predictor** (updates from observed feedback) are provided for
calibration; nothing is learned by a neural model here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

TASK_VERSION = "taiji-multimodal-physics-v1"
FAMILIES = ("incline", "collision")
ACTIONS = ("push_left", "push_right", "push_soft", "hold")
OUTCOMES = ("向右滑动", "向左滑动", "停下", "碰撞", "无变化")
HORIZON = 8


@dataclass
class PhysicsState:
    family: str
    pos: float
    vel: float
    block_b: float
    target: float
    t: int = 0
    last_outcome: str = "无变化"
    history: list[str] = field(default_factory=list)

    def vision(self) -> list[float]:
        return [
            self.pos,
            self.block_b,
            self.target,
            1.0 if self.family == "incline" else 0.0,
            float(self.t),
            float(abs(self.vel)),
        ]

    def proprio(self) -> list[float]:
        return [self.vel, 1.0 if abs(self.vel) > 0.05 else 0.0, float(self.pos)]

    def audio(self) -> list[float]:
        return [1.0 if self.last_outcome == "碰撞" else 0.0, 1.0 if abs(self.vel) > 0.1 else 0.0]

    def observe(self) -> dict[str, list[float]]:
        return {
            "vision": self.vision(),
            "proprio": self.proprio(),
            "audio": self.audio(),
            "t": self.t,
        }


def _apply(state: PhysicsState, action: str) -> tuple[PhysicsState, str]:
    """Deterministic transition; returns next state and the observed causal outcome."""
    pos, vel = state.pos, state.vel
    collide = False
    if action == "push_right":
        vel += 0.30
    elif action == "push_left":
        vel -= 0.30
    elif action == "push_soft":
        vel += 0.12
    else:
        vel *= 0.5
    vel *= 0.85
    pos += vel
    if state.family == "collision":
        if pos >= state.block_b:
            collide = True
            pos = state.block_b - 0.01
            vel = 0.0
        pos = min(1.0, max(0.0, pos))
    else:
        pos = min(1.0, max(0.0, pos))
        if pos in (0.0, 1.0):
            vel = -vel * 0.4
    delta = pos - state.pos
    if collide:
        outcome = "碰撞"
    elif delta > 0.02:
        outcome = "向右滑动"
    elif delta < -0.02:
        outcome = "向左滑动"
    elif abs(vel) > 0.1:
        outcome = "停下"
    else:
        outcome = "无变化"
    nxt = PhysicsState(
        family=state.family,
        pos=pos,
        vel=vel,
        block_b=state.block_b,
        target=state.target,
        t=state.t + 1,
        last_outcome=outcome,
        history=[*state.history, outcome],
    )
    return nxt, outcome


class MultimodalPhysicsEnv:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(int(seed))
        self.state: PhysicsState | None = None

    def reset(self) -> dict[str, Any]:
        family = FAMILIES[self.rng.randrange(len(FAMILIES))]
        if family == "incline":
            pos = self.rng.uniform(0.05, 0.35)
            block_b = 1.0
            target = 0.85
        else:
            pos = self.rng.uniform(0.05, 0.35)
            block_b = self.rng.uniform(0.55, 0.75)
            target = block_b
        self.state = PhysicsState(family=family, pos=pos, vel=0.0, block_b=block_b, target=target)
        return self.state.observe()

    def observe(self) -> dict[str, Any]:
        assert self.state is not None
        return self.state.observe()

    def true_outcome(self, action: str) -> str:
        assert self.state is not None
        _, outcome = _apply(self.state, action)
        return outcome

    def step(self, action: str) -> tuple[dict[str, Any], str, float, bool]:
        assert self.state is not None
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        self.state, outcome = _apply(self.state, action)
        reward = 1.0 if outcome in ("向右滑动", "碰撞") else 0.0
        done = self.state.t >= HORIZON
        return self.state.observe(), outcome, reward, done


# ---------------------------------------------------------------------- predictors
class OraclePredictor:
    """Control only: peeks at true physics (never used as a learned policy)."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, env: MultimodalPhysicsEnv, action: str) -> str:
        self.calls += 1
        return env.true_outcome(action)


class RandomPredictor:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def predict(self, env: MultimodalPhysicsEnv, action: str) -> str:
        return OUTCOMES[self.rng.randrange(len(OUTCOMES))]


class CorrectablePredictor:
    """Count-based predictor updated from real feedback (correctable, not neural)."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.correct = 0
        self.total = 0

    @staticmethod
    def _key(env: MultimodalPhysicsEnv, action: str) -> tuple[str, str, str]:
        assert env.state is not None
        pos_bucket = "pos_high" if env.state.pos >= 0.5 else "pos_low"
        return (env.state.family, pos_bucket, action)

    def predict(self, env: MultimodalPhysicsEnv, action: str) -> str:
        key = self._key(env, action)
        best, best_n = "无变化", -1
        for outcome in OUTCOMES:
            n = self.counts.get((key, outcome), 0)
            if n > best_n:
                best, best_n = outcome, n
        return best

    def observe(self, env: MultimodalPhysicsEnv, action: str, truth: str, predicted: str) -> None:
        key = self._key(env, action)
        self.counts[(key, truth)] = self.counts.get((key, truth), 0) + 1
        self.total += 1
        if predicted == truth:
            self.correct += 1

    def accuracy(self) -> float:
        return round(self.correct / self.total, 4) if self.total else 0.0


def eval_predictor(predictor, seeds: list[int]) -> dict[str, float]:
    correct = total = 0
    for seed in seeds:
        env = MultimodalPhysicsEnv(seed)
        env.reset()
        while True:
            for action in ACTIONS:
                truth = env.true_outcome(action)
                pred = predictor.predict(env, action)
                total += 1
                if pred == truth:
                    correct += 1
            obs, outcome, reward, done = env.step(ACTIONS[env.state.t % len(ACTIONS)])
            if done:
                break
    return {"n": total, "accuracy": round(correct / max(1, total), 4)}
