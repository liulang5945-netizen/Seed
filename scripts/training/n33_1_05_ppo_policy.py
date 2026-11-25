"""N33.1.05 (exploration): PPO actor-critic policy on the calibrated autonomous-choice fixture.

Pre-registered: plans/reference/DESIGN-N33-1-05_20261125.md (D-20261125-563).
Gates were calibrated in N33.1.04 against the scripted baseline (D-20261125-564 result).
Policy is learned only from environment rewards; the scripted policy is never a training signal.

Usage:
  python scripts/training/n33_1_05_ppo_policy.py --smoke
  python scripts/training/n33_1_05_ppo_policy.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.training.n33_1_04_gate_calibration import (  # noqa: E402
    EVAL_SEEDS,
    GATE_G1_RANDOM,
    GATE_G1_SCRIPTED_RATIO,
    gate_eval,
)
from taiji.choice_simulator import (  # noqa: E402
    CHOICES,
    HORIZON,
    ChoiceSessionEnv,
    evaluate_policy,
    random_policy,
    scripted_policy,
)

CARD = "N33.1.05"
TRAIN_SEEDS = (1001, 1002, 1003)
EPISODES = 1500
SMOKE_EPISODES = 30
ROLLOUT_EPISODES = 8
PPO_EPOCHS = 3
CLIP = 0.2
GAMMA = 0.95
LAM = 0.95
LR = 3e-3
VALUE_COEF = 0.5
ENTROPY_START = 0.05
ENTROPY_END = 0.005
CAP_CPU = 80.0
GATE_G3_DROP = 1.5
OUT = ROOT / "reports" / "n33_1_05_ppo.json"


class ActorCritic(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.body = nn.Sequential(nn.Linear(7, 64), nn.Tanh())
            self.pi = nn.Linear(64, len(CHOICES))
            self.v = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def collect(policy: ActorCritic, train_seed: int, ep_start: int) -> dict[str, Any]:
    states: list[torch.Tensor] = []
    actions: list[int] = []
    logps: list[torch.Tensor] = []
    rewards: list[float] = []
    values: list[torch.Tensor] = []
    dones: list[float] = []
    for ep in range(ep_start, ep_start + ROLLOUT_EPISODES):
        env = ChoiceSessionEnv(train_seed * 100_000 + ep)
        state = env.reset()
        for _ in range(HORIZON):
            x = state.vector()
            logits, value = policy(x)
            dist = F.softmax(logits, dim=-1)
            a = int(torch.multinomial(dist, 1).item())
            states.append(x)
            actions.append(a)
            logps.append(torch.log(dist[a].clamp(min=1e-8)))
            values.append(value)
            rewards.append(0.0)
            state2, reward, done = env.step(CHOICES[a])
            rewards[-1] = reward
            dones.append(float(done))
            state = state2
            if done:
                break
    # GAE per episode over the flattened stream (episode boundaries marked by dones).
    adv = torch.zeros(len(rewards))
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * nonterminal - float(values[t])
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv[t] = gae
        next_value = float(values[t])
    return {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "logps": torch.stack(logps).detach(),
        "returns": (adv + torch.stack(values).detach()),
        "adv": adv,
    }


def train_policy(train_seed: int, episodes: int, deadline: float) -> ActorCritic:
    policy = ActorCritic(train_seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    total_updates = max(1, episodes // ROLLOUT_EPISODES)
    ep = 0
    for update in range(total_updates):
        if time.process_time() > deadline:
            raise TimeoutError(f"seed {train_seed} update {update}")
        batch = collect(policy, train_seed, ep)
        ep += ROLLOUT_EPISODES
        frac = update / total_updates
        entropy_coef = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * frac
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        old_logps = batch["logps"]
        for _ in range(PPO_EPOCHS):
            logits, values = policy(batch["states"])
            dist = F.softmax(logits, dim=-1)
            logps = torch.log(
                dist.gather(1, batch["actions"].unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            )
            ratio = torch.exp(logps - old_logps)
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            pi_loss = -(torch.min(ratio * adv, clipped * adv)).mean()
            v_loss = VALUE_COEF * F.mse_loss(values, batch["returns"].detach())
            entropy = -(dist * torch.log(dist.clamp(min=1e-8))).sum(-1).mean()
            loss = pi_loss + v_loss - entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return policy


def greedy(policy: ActorCritic, zero_state: bool = False):
    def act(state) -> str:
        vec = torch.zeros(7) if zero_state else state.vector()
        with torch.no_grad():
            logits, _ = policy(vec)
        return CHOICES[int(logits.argmax().item())]

    return act


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()

    gen = torch.Generator().manual_seed(7)
    random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
    scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
    baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    per_seed: dict[str, Any] = {}
    deadline = started + CAP_CPU
    for seed in TRAIN_SEEDS:
        policy = train_policy(seed, episodes, deadline)
        ev = gate_eval(greedy(policy), baselines)
        zero_norm = evaluate_policy(greedy(policy, zero_state=True), EVAL_SEEDS)
        drop = ev["eval"]["mean_return"] - zero_norm["mean_return"]
        g3 = drop >= GATE_G3_DROP
        per_seed[str(seed)] = {
            "return_normal": ev["eval"]["mean_return"],
            "return_zeroed_state": zero_norm["mean_return"],
            "state_drop": round(drop, 4),
            "probes": ev["probes"],
            "g1": ev["g1"],
            "g2": ev["g2"],
            "g3": bool(g3),
            "positive": bool(ev["g1"] and ev["g2"] and g3),
        }

    cpu = round(time.process_time() - started, 2)
    if args.smoke:
        exit_ = "smoke"
    elif all(v["positive"] for v in per_seed.values()):
        exit_ = "ppo_positive"
    elif all(v["g1"] for v in per_seed.values()):
        exit_ = "ppo_partial"
    else:
        exit_ = "ppo_negative"
    payload = {
        "schema": "taiji-n33-1-05-v1",
        "card": CARD,
        "design": "DESIGN-N33-1-05_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "algorithm": "ppo_actor_critic",
        "episodes": episodes,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": EVAL_SEEDS,
        "baselines": {"random": random_base, "scripted": scripted_base},
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_ratio": GATE_G1_SCRIPTED_RATIO,
            "g3_drop": GATE_G3_DROP,
        },
        "per_seed": per_seed,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "scripted_used_in_training": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "cpu": cpu, "baselines": payload["baselines"], "per_seed": per_seed},
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("ppo_positive", "ppo_partial", "ppo_negative", "smoke") else 1


if __name__ == "__main__":
    raise SystemExit(main())
