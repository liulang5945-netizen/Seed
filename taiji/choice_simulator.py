"""Session simulator v2 for N33.1: six autonomous-choice actions over pending goals/state.

v2 rebalance (D-20261125-558): wait_wake no longer pays a per-step premium (+0.15 instead of
+0.9), task progress is gated by uncertainty, and verify/ask_help/rest each have a clear
state where they dominate. N33.1.01 was replayed on v2 before N33.1.02 used it.

Original docstring below.


This is the zero-training fixture for the N33.1 line (learned autonomous selection).
Scripted heuristics here are baselines only: the N33 requirement is that the policy is
learned inside the subject, not hand-coded.

Choices (six):
  continue_task  work on the pending task (progress)
  recall_learn   recall/learn from experience (knowledge gap)
  verify         check/verify an uncertain belief (uncertainty)
  ask_help       ask for help (when uncertainty is high and verification is not enough)
  rest           rest (restores energy)
  wait_wake      wait with a wake condition (only meaningful when nothing pending)

State vector (7 floats, all in [0,1]):
  pending, unresolved, energy, uncertainty, wait_remaining, progress, elapsed

Reward is shaped so the six actions are all useful in some state, and so a state-blind
policy is clearly worse than a state-aware one. Deterministic given a 64-bit seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

CHOICES: tuple[str, ...] = (
    "continue_task",
    "recall_learn",
    "verify",
    "ask_help",
    "rest",
    "wait_wake",
)
SIM_VERSION = "taiji-choice-session-v3"
CHOICE_INDEX = {name: i for i, name in enumerate(CHOICES)}
STATE_DIM = 7
HORIZON = 24


def seed_from(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**62)


@dataclass
class State:
    pending: float = 1.0
    unresolved: float = 0.0
    energy: float = 1.0
    uncertainty: float = 0.2
    wait_remaining: float = 0.0
    progress: float = 0.0
    elapsed: float = 0.0

    def vector(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.pending,
                self.unresolved,
                self.energy,
                self.uncertainty,
                self.wait_remaining,
                self.progress,
                self.elapsed,
            ],
            dtype=torch.float32,
        )


class ChoiceSessionEnv:
    """Tiny deterministic session simulator with a 6-action choice space."""

    def __init__(self, seed: int = 0) -> None:
        self.gen = torch.Generator().manual_seed(int(seed) % (2**63))
        self.state = State()
        self.t = 0

    def reset(self) -> State:
        u = float(torch.rand((), generator=self.gen))
        self.t = 0
        self.state = State(
            pending=1.0 if u < 0.7 else 0.0,
            unresolved=float(torch.randint(0, 4, (), generator=self.gen)) / 3.0,
            energy=0.35 + 0.65 * float(torch.rand((), generator=self.gen)),
            uncertainty=0.1 + 0.8 * float(torch.rand((), generator=self.gen)),
            wait_remaining=0.0,
            progress=0.0,
            elapsed=0.0,
        )
        return self.state

    def _clip(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def step(self, action: str) -> tuple[State, float, bool]:
        if action not in CHOICE_INDEX:
            raise ValueError(f"unknown action {action!r}")
        s = self.state
        noise = 0.05 * (float(torch.rand((), generator=self.gen)) - 0.5)
        reward = -0.1  # per-step cost so idling is never free
        if action == "continue_task":
            if s.pending > 0:
                gain = 0.25 * (0.5 + s.energy) * (1.0 - 0.8 * s.uncertainty)
                s.progress = self._clip(s.progress + gain + noise)
                s.energy = self._clip(s.energy - 0.12)
                reward += 1.3 * gain
            else:
                reward -= 0.8
        elif action == "recall_learn":
            if s.unresolved > 0:
                gain = 0.25 * (1.0 - s.progress)
                s.unresolved = self._clip(s.unresolved - 0.35)
                s.energy = self._clip(s.energy - 0.03)
                reward += 0.8 * gain + 0.5
            elif s.progress < 0.2:
                reward += 0.2
            else:
                reward -= 0.5
        elif action == "verify":
            if s.uncertainty > 0.25:
                drop = 0.3 * (0.5 + s.energy)
                s.uncertainty = self._clip(s.uncertainty - drop)
                s.energy = self._clip(s.energy - 0.05)
                reward += 1.1 * drop
            else:
                reward -= 0.4
        elif action == "ask_help":
            if s.uncertainty > 0.55:
                s.uncertainty = self._clip(s.uncertainty - 0.5)
                s.progress = self._clip(s.progress + 0.05)
                reward += 0.9
            else:
                reward -= 0.7
        elif action == "rest":
            if s.energy < 0.6:
                s.energy = self._clip(s.energy + 0.3)
                reward += 0.7
            else:
                reward -= 0.4
        elif action == "wait_wake":
            if s.pending == 0:
                s.wait_remaining = 1.0
                reward += 0.15
            else:
                reward -= 0.6

        # Environment drift: energy decays, uncertainty drifts, pending task may finish.
        s.energy = self._clip(s.energy - 0.02 + noise)
        s.uncertainty = self._clip(s.uncertainty + 0.02 + noise)
        if s.pending > 0 and s.progress >= 1.0:
            s.pending = 0.0
            s.progress = 1.0
            reward += 2.0
        if s.wait_remaining > 0:
            s.wait_remaining = self._clip(s.wait_remaining - 0.1)
        self.t += 1
        s.elapsed = self.t / HORIZON
        return s, reward, self.t >= HORIZON


# ------------------------------------------------------------------ baselines


def random_policy(state: State, gen: torch.Generator | None = None) -> str:
    g = gen if gen is not None else torch.Generator().manual_seed(0)
    return CHOICES[int(torch.randint(0, len(CHOICES), (), generator=g))]


def scripted_policy(state: State) -> str:
    """Hand-coded heuristic baseline (comparison only; not the N33 deliverable)."""
    if state.energy < 0.35:
        return "rest"
    if state.pending == 0:
        return "wait_wake"
    if state.uncertainty > 0.55:
        return "ask_help"
    if state.energy > 0.55 and state.uncertainty < 0.45 and state.unresolved <= 0.3:
        return "continue_task"
    if state.uncertainty > 0.3:
        return "verify"
    if state.unresolved > 0.3 or state.progress < 0.2:
        return "recall_learn"
    return "continue_task"


def run_episode(policy, seed: int) -> dict:
    """Run one episode; ``policy`` maps State -> choice string."""
    env = ChoiceSessionEnv(seed)
    state = env.reset()
    total = 0.0
    for _ in range(HORIZON):
        action = policy(state)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    return {
        "return": round(total, 6),
        "progress": round(state.progress, 6),
        "pending": state.pending,
        "energy": round(state.energy, 6),
        "uncertainty": round(state.uncertainty, 6),
        "steps": env.t,
    }


def evaluate_policy(policy, seeds: list[int]) -> dict:
    runs = [run_episode(policy, s) for s in seeds]
    returns = [r["return"] for r in runs]
    return {
        "n": len(runs),
        "mean_return": round(sum(returns) / len(returns), 6),
        "min_return": round(min(returns), 6),
        "max_return": round(max(returns), 6),
    }
