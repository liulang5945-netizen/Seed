"""Synthetic multimodal + physics v2 environment & learner (`taiji-multimodal-physics-v2`).

Fixes the v1 defects found during audit:
1. No `family_id` or `task_id` in observations (`vision[3]` is now physical `slope_angle`,
   never an environment family flag).
2. Both dynamics families (`ramp_bumper` and `rough_collision`) share one continuous physical
   contact/kinematics law with strictly disjoint `(slope_angle, roughness)` regimes and each
   produces frequent collisions (`>= 10%`), sliding, stopping and no-change outcomes.
3. Explicit per-outcome recall (especially `"碰撞"` recall), macro-F1, fair cross-family baseline
   comparison, and a barrier-geometry ablation gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

TASK_VERSION = "taiji-multimodal-physics-v2"
FAMILIES = ("ramp_bumper", "rough_collision")
ACTIONS = ("push_left", "push_right", "push_soft", "hold")
OUTCOMES = ("向右滑动", "向左滑动", "停下", "碰撞", "无变化")
HORIZON = 8
IN_DIM = 6 + 3 + 2 + len(ACTIONS)  # 15
N_OUTCOMES = len(OUTCOMES)


@dataclass
class PhysicsStateV2:
    family: str
    pos: float
    vel: float
    barrier_pos: float
    slope_angle: float
    roughness: float
    prev_impulse: float = 0.0
    t: int = 0
    last_outcome: str = "无变化"

    def vision(self) -> list[float]:
        return [
            self.pos,
            self.barrier_pos,
            max(0.0, self.barrier_pos - self.pos),
            self.slope_angle,
            self.roughness,
            float(abs(self.vel)),
        ]

    def proprio(self) -> list[float]:
        return [self.vel, 1.0 if abs(self.vel) > 0.05 else 0.0, self.prev_impulse]

    def audio(self) -> list[float]:
        return [1.0 if self.last_outcome == "碰撞" else 0.0, 1.0 if abs(self.vel) > 0.10 else 0.0]

    def observe(self) -> dict[str, Any]:
        return {
            "vision": self.vision(),
            "proprio": self.proprio(),
            "audio": self.audio(),
            "t": self.t,
        }


_IMPULSE = {
    "push_left": -0.30,
    "push_right": 0.30,
    "push_soft": 0.12,
    "hold": 0.0,
}


def apply_physics_v2(state: PhysicsStateV2, action: str) -> tuple[PhysicsStateV2, str]:
    """Shared continuous physical transition law (independent of family label)."""
    impulse = _IMPULSE[action]
    vel_next = (state.vel + impulse - 0.12 * state.slope_angle) * (1.0 - 0.50 * state.roughness)
    if action == "hold":
        vel_next *= 0.65
    pos_next = state.pos + vel_next
    collide = False
    if pos_next >= state.barrier_pos:
        collide = True
        pos_next = max(0.0, state.barrier_pos - 0.01)
        vel_next = 0.0
    elif pos_next < 0.0:
        pos_next = 0.0
        vel_next = -0.35 * vel_next
    delta = pos_next - state.pos
    if collide:
        outcome = "碰撞"
    elif delta > 0.02:
        outcome = "向右滑动"
    elif delta < -0.02:
        outcome = "向左滑动"
    elif abs(state.vel) > 0.08 or abs(vel_next) > 0.08:
        outcome = "停下"
    else:
        outcome = "无变化"
    nxt = PhysicsStateV2(
        family=state.family,
        pos=pos_next,
        vel=vel_next,
        barrier_pos=state.barrier_pos,
        slope_angle=state.slope_angle,
        roughness=state.roughness,
        prev_impulse=impulse,
        t=state.t + 1,
        last_outcome=outcome,
    )
    return nxt, outcome


class MultimodalPhysicsEnvV2:
    """Deterministic multimodal physics env without any task_id or family_id leakage."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(int(seed))
        self.state: PhysicsStateV2 | None = None

    def reset(self, *, family: str | None = None) -> dict[str, Any]:
        if family is None:
            family = FAMILIES[self.rng.randrange(len(FAMILIES))]
        elif family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        if family == "ramp_bumper":
            slope_angle = self.rng.uniform(0.15, 0.40)
            roughness = self.rng.uniform(0.10, 0.25)
            barrier_pos = self.rng.uniform(0.50, 0.85)
            pos = self.rng.uniform(0.10, barrier_pos - 0.12)
        else:
            slope_angle = 0.0
            roughness = self.rng.uniform(0.35, 0.60)
            barrier_pos = self.rng.uniform(0.38, 0.68)
            pos = self.rng.uniform(0.05, barrier_pos - 0.10)
        self.state = PhysicsStateV2(
            family=family,
            pos=pos,
            vel=0.0,
            barrier_pos=barrier_pos,
            slope_angle=slope_angle,
            roughness=roughness,
        )
        return self.state.observe()

    def observe(self) -> dict[str, Any]:
        assert self.state is not None
        return self.state.observe()

    def true_outcome(self, action: str) -> str:
        assert self.state is not None
        _, outcome = apply_physics_v2(self.state, action)
        return outcome

    def step(self, action: str) -> tuple[dict[str, Any], str, float, bool]:
        assert self.state is not None
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        self.state, outcome = apply_physics_v2(self.state, action)
        reward = 1.0 if outcome in ("向右滑动", "碰撞") else 0.0
        done = self.state.t >= HORIZON
        return self.state.observe(), outcome, reward, done


def _action_onehot(action: str) -> list[float]:
    return [1.0 if action == a else 0.0 for a in ACTIONS]


def collect_v2(seed: int, family: str, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect deterministic `(modalities + action_onehot, outcome)` exploratory rollouts."""
    xs: list[list[float]] = []
    ys: list[int] = []
    for ep in range(n_episodes):
        ep_seed = seed * 1_000_000 + ep
        env = MultimodalPhysicsEnvV2(ep_seed)
        roll_rng = random.Random(ep_seed ^ 0xA5A5)
        env.reset(family=family)
        assert env.state is not None
        for _ in range(HORIZON):
            obs = env.observe()
            base = [*obs["vision"], *obs["proprio"], *obs["audio"]]
            for action in ACTIONS:
                truth = env.true_outcome(action)
                xs.append([*base, *_action_onehot(action)])
                ys.append(OUTCOMES.index(truth))
            chosen = ACTIONS[roll_rng.randrange(len(ACTIONS))]
            if env.step(chosen)[3]:
                break
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


class ConsequenceNetV2(nn.Module):
    def __init__(self, seed: int, hidden: int = 128) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) ^ 0x33)
            self.net = nn.Sequential(
                nn.Linear(IN_DIM, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, N_OUTCOMES),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


def train_net_v2(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    *,
    steps: int = 1600,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> ConsequenceNetV2:
    net = ConsequenceNetV2(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed) ^ 0xBEEF)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
        loss = nn.functional.cross_entropy(net(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return net


def eval_metrics_v2(net: ConsequenceNetV2, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        pred = net(x).argmax(dim=-1)
    acc = round(float((pred == y).float().mean()), 4)
    per_class_recall: dict[str, float] = {}
    f1s: list[float] = []
    for idx, label in enumerate(OUTCOMES):
        true_m = y == idx
        pred_m = pred == idx
        tp = int((true_m & pred_m).sum())
        n_true = int(true_m.sum())
        n_pred = int(pred_m.sum())
        recall = tp / n_true if n_true > 0 else 0.0
        prec = tp / n_pred if n_pred > 0 else 0.0
        f1 = (2.0 * prec * recall / (prec + recall)) if (prec + recall) > 0.0 else 0.0
        per_class_recall[label] = round(recall, 4)
        if n_true > 0:
            f1s.append(f1)
    coll_idx = OUTCOMES.index("碰撞")
    return {
        "n": int(y.shape[0]),
        "accuracy": acc,
        "collision_recall": per_class_recall["碰撞"],
        "collision_count": int((y == coll_idx).sum()),
        "macro_f1": round(sum(f1s) / max(1, len(f1s)), 4),
        "per_class_recall": per_class_recall,
    }


def ablate_barrier_collision_recall(
    net: ConsequenceNetV2, x: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
    """Counterfactual barrier removal (`vision[1]=1.5`, `vision[2]=1.5`: barrier out of reach)."""
    x_abl = x.clone()
    x_abl[:, 1] = 1.5
    x_abl[:, 2] = 1.5
    m_full = eval_metrics_v2(net, x, y)
    m_abl = eval_metrics_v2(net, x_abl, y)
    return {
        "full_collision_recall": m_full["collision_recall"],
        "ablated_collision_recall": m_abl["collision_recall"],
        "collision_recall_drop": round(m_full["collision_recall"] - m_abl["collision_recall"], 4),
    }


class CorrectablePredictorV2:
    """Observable-only count predictor (no family_id)."""

    def __init__(self) -> None:
        self.counts: dict[tuple[tuple[int, int, str], str], int] = {}

    @staticmethod
    def _key(env: MultimodalPhysicsEnvV2, action: str) -> tuple[int, int, str]:
        assert env.state is not None
        gap = env.state.barrier_pos - env.state.pos
        gap_bucket = 0 if gap < 0.15 else (1 if gap < 0.35 else 2)
        vel_sign = 1 if env.state.vel > 0.05 else (-1 if env.state.vel < -0.05 else 0)
        return (vel_sign, gap_bucket, action)

    def predict(self, env: MultimodalPhysicsEnvV2, action: str) -> str:
        key = self._key(env, action)
        best, best_n = "无变化", -1
        for outcome in OUTCOMES:
            n = self.counts.get((key, outcome), 0)
            if n > best_n:
                best, best_n = outcome, n
        return best

    def observe(self, env: MultimodalPhysicsEnvV2, action: str, truth: str) -> None:
        key = self._key(env, action)
        self.counts[(key, truth)] = self.counts.get((key, truth), 0) + 1


def eval_count_baseline_v2(
    train_seed: int,
    train_family: str,
    eval_seed: int,
    eval_family: str,
    n_train_ep: int,
    n_eval_ep: int,
) -> dict[str, float]:
    """Fit count predictor on train_family and evaluate frozen counts on eval_family."""
    predictor = CorrectablePredictorV2()
    for ep in range(n_train_ep):
        ep_seed = train_seed * 1_000_000 + ep
        env = MultimodalPhysicsEnvV2(ep_seed)
        roll_rng = random.Random(ep_seed ^ 0xA5A5)
        env.reset(family=train_family)
        assert env.state is not None
        for _ in range(HORIZON):
            for action in ACTIONS:
                predictor.observe(env, action, env.true_outcome(action))
            if env.step(ACTIONS[roll_rng.randrange(len(ACTIONS))])[3]:
                break

    correct = total = coll_true = coll_hit = 0
    for ep in range(n_eval_ep):
        ep_seed = eval_seed * 1_000_000 + ep
        env = MultimodalPhysicsEnvV2(ep_seed)
        roll_rng = random.Random(ep_seed ^ 0xA5A5)
        env.reset(family=eval_family)
        assert env.state is not None
        for _ in range(HORIZON):
            for action in ACTIONS:
                truth = env.true_outcome(action)
                pred = predictor.predict(env, action)
                total += 1
                correct += int(pred == truth)
                if truth == "碰撞":
                    coll_true += 1
                    coll_hit += int(pred == "碰撞")
            if env.step(ACTIONS[roll_rng.randrange(len(ACTIONS))])[3]:
                break
    return {
        "n": total,
        "accuracy": round(correct / max(1, total), 4),
        "collision_recall": round(coll_hit / max(1, coll_true), 4),
    }
