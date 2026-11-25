"""Learned multimodal consequence prediction over the physics facility (`taiji-multimodal-learning-v1`).

Zero-heavy-training supervised facility: collects `(vision, proprio, audio, action) -> outcome`
samples from :mod:`taiji.multimodal_physics_env` and trains a small MLP consequence predictor on
one dynamics family, evaluating both in-family and cross-family generalisation. Controls: scripted
oracle (true lookahead) and the correctable count baseline from the environment module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from taiji.multimodal_physics_env import (  # noqa: E402
    ACTIONS,
    HORIZON,
    OUTCOMES,
    CorrectablePredictor,
    MultimodalPhysicsEnv,
)

SCHEMA = "taiji-multimodal-learning-v1"
IN_DIM = 6 + 3 + 2 + len(ACTIONS)  # 15
N_OUTCOMES = len(OUTCOMES)


def _action_onehot(action: str) -> list[float]:
    return [1.0 if action == a else 0.0 for a in ACTIONS]


def collect(seed: int, family: str, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic dataset of (modality + action one-hot, outcome label) samples."""
    xs: list[list[float]] = []
    ys: list[int] = []
    for ep in range(n_episodes):
        env = MultimodalPhysicsEnv(seed * 1_000_000 + ep)
        env.reset()
        assert env.state is not None
        env.state.family = family
        for _ in range(HORIZON):
            if env.state is None:
                break
            obs = env.observe()
            base = [*obs["vision"], *obs["proprio"], *obs["audio"]]
            for action in ACTIONS:
                truth = env.true_outcome(action)
                xs.append([*base, *_action_onehot(action)])
                ys.append(OUTCOMES.index(truth))
            done = env.step(ACTIONS[env.state.t % len(ACTIONS)])[3]
            if done:
                break
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


class ConsequenceNet(nn.Module):
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
        return self.net(x)


def train_net(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    *,
    steps: int = 1200,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> ConsequenceNet:
    net = ConsequenceNet(seed)
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


def accuracy(net: ConsequenceNet, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = net(x).argmax(dim=-1)
    return round(float((pred == y).float().mean()), 4)


def count_baseline_accuracy(seed: int, family: str, n_episodes: int) -> float:
    """Correctable count predictor evaluated on the same family (control)."""
    predictor = CorrectablePredictor()
    correct = total = 0
    for ep in range(n_episodes):
        env = MultimodalPhysicsEnv(seed * 1_000_000 + ep)
        env.reset()
        assert env.state is not None
        env.state.family = family
        for _ in range(HORIZON):
            if env.state is None:
                break
            for action in ACTIONS:
                truth = env.true_outcome(action)
                pred = predictor.predict(env, action)
                total += 1
                correct += int(pred == truth)
                predictor.observe(env, action, truth, pred)
            if env.step(ACTIONS[env.state.t % len(ACTIONS)])[3]:
                break
    return round(correct / max(1, total), 4)


@dataclass
class Report:
    in_family: float
    cross_family: float
    count_baseline: float
    n_train: int
    n_eval: int
