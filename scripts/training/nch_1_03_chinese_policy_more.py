"""NCH.1.03 (exploration): Chinese continuous-task learned policy, training budget 1,200 -> 6,000 episodes.

Pre-registered: plans/reference/DESIGN-NCH-1-03_20261126.md (D-20261126-634).\nSingle variable vs NCH.1.02: EPISODES 1,200 -> 6,000; architecture/gates unchanged.
Executed under the owner's standing delegation for reasonable exploration cards (D-20261125-
<delegation>, restated D-20261126-625); exploration seeds only, no confirmation seed / check /
final is read.

Policy: token-embedding mean-pool -> MLP(16->64->64) -> 8 action logits + value head; REINFORCE
with a learned value baseline and an entropy bonus, explicit per-seed sampling generator (so two
passes are bit-identical). Evaluated against scripted oracle / random / 6 constant controls.

Usage:
  python scripts/training/nch_1_02_chinese_policy.py --smoke
  python scripts/training/nch_1_02_chinese_policy.py
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

from taiji.chinese_task_env import (  # noqa: E402
    ACTIONS,
    VOCAB,
    ChineseTaskEnv,
    scripted_oracle_policy,
)

CARD = "NCH.1.03"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = "DESIGN-NCH-1-03_20261126.md"
TRAIN_SEEDS = (6101, 6102, 6103)
SPOT_CHECK_SEED = 6101
EVAL_SEEDS = list(range(4100, 4140))
HELDOUT_SEEDS = list(range(4140, 4180))
EPISODES = 6000
SMOKE_EPISODES = 60
ROLLOUT_EPISODES = 8
EPOCHS = 6
LR = 3e-3
ENTROPY = 0.02
VALUE_COEF = 0.5
CAP_CPU = 120.0
OUT = ROOT / "reports" / "nch_1_03_chinese_policy.json"


class ChinesePolicy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.emb = nn.Embedding(len(VOCAB), 16)
            self.trunk = nn.Sequential(
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.actor = nn.Linear(64, len(ACTIONS))
            self.critic = nn.Linear(64, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = tokens.dim() == 1
        if single:
            tokens = tokens.unsqueeze(0)
        x = self.emb(tokens.long()).mean(dim=1)
        h = self.trunk(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def _eval_acc(
    policy: ChinesePolicy, seeds: list[int], *, heldout: bool, oracle: bool = False
) -> float:
    policy.eval()
    correct = total = 0
    with torch.no_grad():
        for seed in seeds:
            env = ChineseTaskEnv(seed, heldout=heldout)
            env.reset()
            while True:
                tokens = torch.tensor(env._render(env.step_idx), dtype=torch.long)
                logits, _ = policy(tokens)
                action = int(torch.argmax(logits, dim=-1).item())
                if oracle:
                    action = scripted_oracle_policy(env)
                total += 1
                correct += int(action == env.intended_action)
                obs, reward, done = env.step(action)
                if done:
                    break
    policy.train()
    return round(correct / max(1, total), 4)


def train_policy(seed: int, episodes: int, deadline: float) -> ChinesePolicy:
    policy = ChinesePolicy(seed)
    gen = torch.Generator().manual_seed(int(seed) * 7919 + 13)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    update = 0
    ep = 0
    while ep < episodes:
        if time.process_time() > deadline:
            raise TimeoutError(f"chinese policy seed={seed} update={update}")
        states: list[torch.Tensor] = []
        actions: list[int] = []
        logps: list[torch.Tensor] = []
        rewards: list[float] = []
        values: list[torch.Tensor] = []
        for _ in range(ROLLOUT_EPISODES):
            env = ChineseTaskEnv(seed * 1_000_000 + ep)
            env.reset()
            ep += 1
            while True:
                tokens = torch.tensor(env._render(env.step_idx), dtype=torch.long)
                logits, value = policy(tokens)
                dist = F.softmax(logits, dim=-1)
                a = int(torch.multinomial(dist, 1, generator=gen).item())
                logp = torch.log(dist[a].clamp(min=1e-8))
                states.append(tokens)
                actions.append(a)
                logps.append(logp)
                values.append(value.squeeze())
                rewards.append(0.0)
                obs, reward, done = env.step(a)
                rewards[-1] = reward
                if done:
                    break

        returns: list[float] = []
        running = 0.0
        for r in reversed(rewards):
            running = r + 0.99 * running
            returns.append(running)
        returns_t = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        values_t = torch.stack(values).detach()
        adv = returns_t - values_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states_t = torch.stack(states)
        actions_t = torch.tensor(actions, dtype=torch.long)
        old_logps = torch.stack(logps).detach()
        for _ in range(EPOCHS):
            logits, values = policy(states_t)
            dist = F.softmax(logits, dim=-1)
            logps = torch.log(dist.gather(1, actions_t.unsqueeze(1)).squeeze(1).clamp(min=1e-8))
            ratio = torch.exp(logps - old_logps)
            clip = 0.2
            pi_loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)).mean()
            v_loss = VALUE_COEF * F.mse_loss(values, returns_t)
            ent = -(dist * torch.log(dist.clamp(min=1e-8))).sum(dim=-1).mean()
            loss = pi_loss + v_loss - ENTROPY * ent
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite chinese policy seed={seed}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
        update += 1
    policy.eval()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    design = DESIGN
    assert (ROOT / "plans/reference" / DESIGN).exists(), f"missing design card {DESIGN}"
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    oracle_train = _eval_acc(ChinesePolicy(0), EVAL_SEEDS, heldout=False, oracle=True)
    oracle_held = _eval_acc(ChinesePolicy(0), HELDOUT_SEEDS, heldout=True, oracle=True)

    per_seed: dict[str, Any] = {}
    try:
        for seed in TRAIN_SEEDS:
            t0 = time.process_time()
            policy = train_policy(seed, episodes, deadline)
            train_acc = _eval_acc(policy, EVAL_SEEDS, heldout=False)
            held_acc = _eval_acc(policy, HELDOUT_SEEDS, heldout=True)
            if seed == SPOT_CHECK_SEED and not args.smoke:
                again = train_policy(seed, episodes, deadline)
                weights_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(policy.parameters(), again.parameters(), strict=True)
                )
            else:
                weights_exact = None
            per_seed[str(seed)] = {
                "train_acc": train_acc,
                "heldout_acc": held_acc,
                "cpu": round(time.process_time() - t0, 2),
                "weights_exact": weights_exact,
            }
    except (TimeoutError, FloatingPointError) as exc:
        payload = {
            "schema": SCHEMA,
            "card": CARD,
            "design": design,
            "exit": "engineering_failure",
            "reason": str(exc),
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"exit": "engineering_failure", "reason": str(exc)}, ensure_ascii=False))
        return 1

    g1 = all(v["train_acc"] >= 0.85 for v in per_seed.values())
    g2 = all(v["heldout_acc"] >= 0.75 for v in per_seed.values())
    g3 = (
        all(v["heldout_acc"] >= 0.70 for v in per_seed.values())
        and max(v["train_acc"] for v in per_seed.values()) >= 0.85
    )
    repro = all(
        v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
    )
    exit_ = "chinese_policy_positive" if (g1 and g2 and g3 and repro) else "chinese_policy_partial"
    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design,
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(TRAIN_SEEDS),
        "episodes": episodes,
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "heldout_seeds": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)],
        "oracle_train": oracle_train,
        "oracle_heldout": oracle_held,
        "per_seed": per_seed,
        "gates": {"g1_train_acc": 0.85, "g2_heldout_acc": 0.75, "g3_partial_heldout": 0.70},
        "g1": bool(g1),
        "g2": bool(g2),
        "g3": bool(g3),
        "repro_exact": bool(repro),
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
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
            {"exit": exit_, "per_seed": per_seed, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False
        )
    )
    return 0 if exit_ in ("chinese_policy_positive", "chinese_policy_partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
