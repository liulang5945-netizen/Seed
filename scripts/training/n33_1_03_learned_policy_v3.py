"""N33.1.03 (exploration): learn the six-way autonomous-choice policy on simulator v3.

Pre-registered: plans/reference/DESIGN-N33-1-03_20261125.md (D-20261125-559).
Only one variable vs N33.1.02: the fixture recall rule (v3: recall pays only when unresolved > 0).
The policy (MLP 7->64->6) is trained only on environment returns via REINFORCE with a mean
baseline and entropy bonus. The hand-coded scripted policy is never used as a training signal;
it only appears as an evaluation baseline. No confirmation seeds are read.

Usage:
  python scripts/training/n33_1_03_learned_policy_v3.py --smoke
  python scripts/training/n33_1_03_learned_policy_v3.py
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

from taiji.choice_simulator import (  # noqa: E402
    CHOICE_INDEX,
    CHOICES,
    HORIZON,
    ChoiceSessionEnv,
    State,
    evaluate_policy,
    random_policy,
    scripted_policy,
)

CARD = "N33.1.03"
TRAIN_SEEDS = (1001, 1002, 1003)
EPISODES = 4000
SMOKE_EPISODES = 200
EVAL_SEEDS = list(range(5000, 5040))
GATE_G1_RANDOM = 10.0
GATE_G1_SCRIPTED = -2.0
GATE_G2_SHARE = 0.50
GATE_G3_DROP = 1.5
CAP_CPU = 80.0
LR = 1e-2
HIDDEN = 64
ENTROPY = 0.01

OUT = ROOT / "reports" / "n33_1_03_policy.json"


class Policy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(7, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, len(CHOICES))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits


def train_policy(train_seed: int, episodes: int) -> Policy:
    policy = Policy(train_seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    base = 0.0
    for ep in range(episodes):
        env = ChoiceSessionEnv(train_seed * 10_000 + ep)
        state = env.reset()
        logps = []
        entropies = []
        rewards = []
        for _ in range(HORIZON):
            logits = policy(state.vector())
            dist = F.softmax(logits, dim=-1)
            action_idx = int(torch.multinomial(dist, 1).item())
            logp = torch.log(dist[action_idx].clamp(min=1e-8))
            state, reward, done = env.step(CHOICES[action_idx])
            logps.append(logp)
            entropies.append(-(dist * torch.log(dist.clamp(min=1e-8))).sum())
            rewards.append(reward)
            if done:
                break
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + 0.95 * g
            returns.append(g)
        returns.reverse()
        ret = torch.tensor(returns, dtype=torch.float32)
        base = 0.9 * base + 0.1 * float(ret.mean())
        adv = ret - base
        if len(logps) < 2:
            continue
        loss = -(torch.stack(logps) * adv.detach()).mean() - ENTROPY * torch.stack(entropies).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return policy


def learned_policy(policy: Policy, zero_state: bool = False):
    def act(state: State) -> str:
        vec = torch.zeros(7) if zero_state else state.vector()
        with torch.no_grad():
            logits = policy(vec)
        idx = int(logits.argmax().item())
        return CHOICES[idx]

    return act


def probes(policy: Policy) -> dict[str, Any]:
    """Behaviour probes over evaluation episodes (greedy actions)."""
    low_energy_rest = []
    high_unc_verify = []
    pending_work = []
    hist = {c: 0 for c in CHOICES}
    with torch.no_grad():
        for seed in EVAL_SEEDS:
            env = ChoiceSessionEnv(seed)
            state = env.reset()
            for _ in range(HORIZON):
                if state.energy < 0.35:
                    low_energy_rest.append(
                        int(policy(state.vector()).argmax().item()) == CHOICE_INDEX["rest"]
                    )
                if state.uncertainty > 0.55:
                    idx = int(policy(state.vector()).argmax().item())
                    high_unc_verify.append(
                        idx in (CHOICE_INDEX["verify"], CHOICE_INDEX["ask_help"])
                    )
                if state.pending > 0 and state.progress < 0.3:
                    idx = int(policy(state.vector()).argmax().item())
                    pending_work.append(
                        idx in (CHOICE_INDEX["continue_task"], CHOICE_INDEX["recall_learn"])
                    )
                action = CHOICES[int(policy(state.vector()).argmax().item())]
                hist[action] += 1
                state, _, done = env.step(action)
                if done:
                    break
    return {
        "n_low_energy": len(low_energy_rest),
        "rest_share_low_energy": round(sum(low_energy_rest) / max(1, len(low_energy_rest)), 4),
        "n_high_uncertainty": len(high_unc_verify),
        "verify_or_help_share_high_unc": round(
            sum(high_unc_verify) / max(1, len(high_unc_verify)), 4
        ),
        "n_pending": len(pending_work),
        "continue_or_recall_share_pending": round(sum(pending_work) / max(1, len(pending_work)), 4),
        "action_histogram": hist,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()

    random_gen = torch.Generator().manual_seed(7)
    random_base = evaluate_policy(lambda s: random_policy(s, random_gen), EVAL_SEEDS)
    scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    per_seed: dict[str, Any] = {}
    for seed in TRAIN_SEEDS:
        policy = train_policy(seed, episodes)
        normal = evaluate_policy(learned_policy(policy), EVAL_SEEDS)
        zeroed = evaluate_policy(learned_policy(policy, zero_state=True), EVAL_SEEDS)
        pr = probes(policy)
        drop = normal["mean_return"] - zeroed["mean_return"]
        g1 = (
            normal["mean_return"] >= random_base["mean_return"] + GATE_G1_RANDOM
            and normal["mean_return"] >= scripted_base["mean_return"] + GATE_G1_SCRIPTED
        )
        g2 = (
            pr["rest_share_low_energy"] >= GATE_G2_SHARE
            and pr["verify_or_help_share_high_unc"] >= GATE_G2_SHARE
            and pr["continue_or_recall_share_pending"] >= GATE_G2_SHARE
        )
        g3 = drop >= GATE_G3_DROP
        per_seed[str(seed)] = {
            "return_normal": normal["mean_return"],
            "return_zeroed_state": zeroed["mean_return"],
            "state_drop": round(drop, 4),
            "probes": pr,
            "g1": bool(g1),
            "g2": bool(g2),
            "g3": bool(g3),
            "positive": bool(g1 and g2 and g3),
        }

    cpu = round(time.process_time() - started, 2)
    all_g1 = all(v["g1"] for v in per_seed.values())
    positives = all(v["positive"] for v in per_seed.values())
    if args.smoke:
        exit_ = "smoke"
    elif positives:
        exit_ = "learned_policy_positive"
    elif all_g1:
        exit_ = "learned_policy_partial"
    else:
        exit_ = "learned_policy_negative"
    payload = {
        "schema": "taiji-n33-1-02-v1",
        "card": CARD,
        "design": "DESIGN-N33-1-02_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "episodes": episodes,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": EVAL_SEEDS,
        "baselines": {"random": random_base, "scripted": scripted_base},
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_gap": GATE_G1_SCRIPTED,
            "g2_share": GATE_G2_SHARE,
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
    return (
        0
        if exit_
        in ("learned_policy_positive", "learned_policy_partial", "learned_policy_negative", "smoke")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
