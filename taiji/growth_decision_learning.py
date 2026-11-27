"""Learned growth decisions (`taiji-growth-decision-v1`, zero heavy training).

Addresses the v4 clause "选择与调整须由主体内部学得的策略产生": here the decision **whether/how
much to grow** is made by a learned bandit policy that observes the training state, instead of
the scripted `GrowthPressureController`. One decision per episode, after phase-1 training and a
warm-up of phase-2 training:

* actions: ``no_grow`` / ``grow8`` / ``grow16`` / ``grow24`` (units added neutrally);
* reward: final combined accuracy − 0.001 × units grown (real benefit minus cost);
* tasks come in two difficulty regimes (easy prototype phase 2 vs hard parity phase 2) so the
  optimal decision is state-dependent;
* controls: fixed growth sizes, no growth, and the scripted pressure rule.

Exploration seeds only; no confirmation seed, check or final is read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from taiji.continual_growth_task import (  # type: ignore[attr-defined]
    IN_DIM,
    NUM_CLASSES,
    make_split,
)
from taiji.plastic_growth import PlasticGrowthNet

SCHEMA = "taiji-growth-decision-v1"
ACTIONS = ("no_grow", "grow8", "grow16", "grow24")
GROW_K = {"no_grow": 0, "grow8": 8, "grow16": 16, "grow24": 24}
COST_PER_UNIT = 0.001
P1_STEPS = 250
P2_WARMUP = 150
P2_STEPS = 250
BATCH = 128

_EASY_PROTOS = torch.tensor(
    [
        [1.4, -1.2, 1.2, 0.0, -1.4, 0.6, 0.2, -0.8],
        [-1.2, 1.4, -1.0, 1.0, 0.4, -1.4, 0.8, 0.2],
        [0.2, 0.4, 1.4, -1.4, 0.8, 0.2, -1.4, 0.6],
    ],
    dtype=torch.float32,
)


def _make_phase2_easy(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 3, (n,), generator=gen, dtype=torch.long) + 3
    x_lo = _EASY_PROTOS[y - 3] + 0.35 * torch.randn((n, 8), generator=gen)
    x_hi = 0.20 * torch.randn((n, 8), generator=gen)
    return torch.cat([x_lo, x_hi], dim=1), y


def make_task(seed: int, hard: bool):
    """Task data; ``hard=True`` uses the parity phase 2, else the easy prototype phase 2."""
    data = make_split(seed)
    if hard:
        return data
    gen = torch.Generator().manual_seed(seed * 31 + 3)
    p2_tr_x, p2_tr_y = _make_phase2_easy(3072, gen)
    p2_va_x, p2_va_y = _make_phase2_easy(1024, gen)
    return {
        "phase1": data["phase1"],
        "phase2": {
            "train": {"x": p2_tr_x, "y": p2_tr_y},
            "val": {"x": p2_va_x, "y": p2_va_y},
        },
    }


def _batches(
    gen: torch.Generator, x: torch.Tensor, y: torch.Tensor, n: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for _ in range(n):
        idx = torch.randint(0, x.shape[0], (BATCH,), generator=gen)
        out.append((x[idx], y[idx]))
    return out


def run_episode(seed: int, action: str, deadline: float, *, hard: bool = True) -> dict:
    """Train one episode and return the decision context, final accuracy and reward."""
    data = make_task(seed, hard=hard)
    p1_tr, p1_va = data["phase1"]["train"], data["phase1"]["val"]
    p2_tr, p2_va = data["phase2"]["train"], data["phase2"]["val"]

    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    gen1 = torch.Generator().manual_seed(seed * 7919 + 1)
    gen2 = torch.Generator().manual_seed(seed * 7919 + 2)

    net.train()
    for xb, yb in _batches(gen1, p1_tr["x"], p1_tr["y"], P1_STEPS):
        if time.process_time() > deadline:
            raise TimeoutError("growth-decision p1")
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def acc(x: torch.Tensor, y: torch.Tensor) -> float:
        net.eval()
        with torch.no_grad():
            pred = net(x).argmax(dim=-1)
        net.train()
        return float((pred == y).float().mean())

    p1_acc = acc(p1_va["x"], p1_va["y"])

    for xb2, yb2 in _batches(gen2, p2_tr["x"], p2_tr["y"], P2_WARMUP):
        idx = torch.randint(0, p1_tr["x"].shape[0], (BATCH // 2,), generator=gen2)
        xb = torch.cat([xb2[: BATCH // 2], p1_tr["x"][idx]], dim=0)
        yb = torch.cat([yb2[: BATCH // 2], p1_tr["y"][idx]], dim=0)
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    val_err = 1.0 - acc(p2_va["x"], p2_va["y"])
    k = GROW_K[action]
    if k > 0:
        net.grow_neutral_(k, seed=seed * 13 + k)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for xb2, yb2 in _batches(gen2, p2_tr["x"], p2_tr["y"], P2_STEPS):
        if time.process_time() > deadline:
            raise TimeoutError("growth-decision p2")
        idx = torch.randint(0, p1_tr["x"].shape[0], (BATCH // 2,), generator=gen2)
        xb = torch.cat([xb2[: BATCH // 2], p1_tr["x"][idx]], dim=0)
        yb = torch.cat([yb2[: BATCH // 2], p1_tr["y"][idx]], dim=0)
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    p1_after = acc(p1_va["x"], p1_va["y"])
    p2_acc = acc(p2_va["x"], p2_va["y"])
    combined = 0.5 * (p1_after + p2_acc)
    reward = combined - COST_PER_UNIT * k
    return {
        "seed": seed,
        "hard": hard,
        "action": action,
        "p1_acc": round(p1_acc, 6),
        "val_err": round(val_err, 6),
        "p2_acc": round(p2_acc, 6),
        "p1_after": round(p1_after, 6),
        "combined": round(combined, 6),
        "reward": round(reward, 6),
        "units": k,
    }


class GrowthDecisionPolicy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.net = nn.Sequential(
                nn.Linear(4, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, len(ACTIONS)),
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(feats))


def features(result: dict) -> torch.Tensor:
    return torch.tensor(
        [result["val_err"], result["p1_acc"], result["units"] / 24.0, 1.0 - result["val_err"]],
        dtype=torch.float32,
    )


@dataclass
class TrainResult:
    policy: GrowthDecisionPolicy
    rewards: list[float]


def train_policy(
    task_specs: list[tuple[int, bool]], seed: int, episodes: int, deadline: float
) -> TrainResult:
    policy = GrowthDecisionPolicy(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed * 999 + 7)
    rewards: list[float] = []
    baseline = 0.0
    for ep in range(episodes):
        task_seed, hard = task_specs[ep % len(task_specs)]
        probe = run_episode(task_seed, "no_grow", deadline, hard=hard)
        feats = features(probe)
        logits = policy(feats)
        dist = F.softmax(logits, dim=-1)
        a = int(torch.multinomial(dist, 1, generator=gen).item())
        res = run_episode(task_seed, ACTIONS[a], deadline, hard=hard)
        r = res["reward"]
        rewards.append(r)
        baseline = 0.9 * baseline + 0.1 * r
        adv = r - baseline
        logp = torch.log(dist[a].clamp(min=1e-8))
        loss = -logp * adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    policy.eval()
    return TrainResult(policy=policy, rewards=rewards)


def greedy_action(policy: GrowthDecisionPolicy, result: dict) -> int:
    with torch.no_grad():
        logits = policy(features(result))
    return int(torch.argmax(logits, dim=-1).item())
