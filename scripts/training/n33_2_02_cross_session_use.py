"""N33.2.02 (exploration): genuine cross-session state use via SessionStore vs stateless PPO.

Pre-registered: plans/reference/DESIGN-N33-2-02_20261125.md (D-20261125-580).
Evaluates whether a PPO policy trained on Session 2 can read restored Session 1 state
(capsule + pending + experience via taiji/session_store.py) to beat an isomorphic stateless
PPO baseline and suffer clear counterfactual drops under zeroed or swapped Session 1 records.

Usage:
  python scripts/training/n33_2_02_cross_session_use.py --smoke
  python scripts/training/n33_2_02_cross_session_use.py
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

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji.cross_session_env import (  # noqa: E402
    ACTIONS,
    CROSS_DIM,
    ENV_VERSION,
    HORIZON,
    OBS_S2_DIM,
    STATE_DIM,
    CrossSessionEnv,
    evaluate_policy_with_store,
    random_policy,
    scripted_persistent_policy,
    seed_from,
)

CARD = "N33.2.02"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
TRAIN_SEEDS = (2001, 2002, 2003)
EVAL_SEEDS = list(range(6000, 6100))
EPISODES = 4000
SMOKE_EPISODES = 48
ROLLOUT_EPISODES = 16
PPO_EPOCHS = 3
CLIP = 0.2
GAMMA = 0.95
LAM = 0.95
LR = 3e-3
VALUE_COEF = 0.5
ENTROPY_START = 0.05
ENTROPY_END = 0.005
CAP_CPU = 80.0
SPOT_CHECK_SEED = 2001

GATE_STATELESS_GAP = 3.0
GATE_SCRIPTED_RATIO = 0.85
GATE_RESOLVE_RATE = 0.80
GATE_COUNTERFACTUAL_DROP = 3.0

OUT = ROOT / "reports" / "n33_2_02_cross_session.json"


class ActorCritic(nn.Module):
    """Compact MLP actor-critic for the 15-dim cross-session task."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.body = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.Tanh())
            self.pi = nn.Linear(64, len(ACTIONS))
            self.v = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def collect_det(
    policy: ActorCritic,
    train_seed: int,
    ep_start: int,
    sample_gen: torch.Generator,
    *,
    stateless: bool,
) -> dict[str, torch.Tensor]:
    """Collect one rollout batch with explicit deterministic sampling generator."""
    states: list[torch.Tensor] = []
    actions: list[int] = []
    logps: list[torch.Tensor] = []
    rewards: list[float] = []
    values: list[torch.Tensor] = []
    dones: list[float] = []

    for ep in range(ep_start, ep_start + ROLLOUT_EPISODES):
        env = CrossSessionEnv(train_seed * 100_000 + ep, zero_cross=stateless)
        obs = env.reset()
        for _ in range(HORIZON):
            logits, value = policy(obs)
            dist = F.softmax(logits, dim=-1)
            a = int(torch.multinomial(dist, 1, generator=sample_gen).item())
            states.append(obs)
            actions.append(a)
            logps.append(torch.log(dist[a].clamp(min=1e-8)))
            values.append(value)
            obs, reward, done = env.step(ACTIONS[a])
            rewards.append(reward)
            dones.append(float(done))
            if done:
                break

    adv = torch.zeros(len(rewards))
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        v_t = float(values[t].detach())
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * nonterminal - v_t
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv[t] = gae
        next_value = v_t
    return {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "logps": torch.stack(logps).detach(),
        "returns": adv + torch.stack(values).detach(),
        "adv": adv,
    }


def train_policy_det(
    train_seed: int,
    *,
    stateless: bool,
    episodes: int,
    deadline: float,
) -> ActorCritic:
    """Train one deterministic PPO policy (`persistent` or isomorphic `stateless`)."""
    arm = "stateless" if stateless else "persistent"
    policy = ActorCritic(train_seed)
    sample_gen = torch.Generator().manual_seed(seed_from(CARD, arm, train_seed, "sample"))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    total_updates = max(1, episodes // ROLLOUT_EPISODES)
    ep = 0
    for update in range(total_updates):
        if time.process_time() > deadline:
            raise TimeoutError(f"{arm} seed={train_seed} update={update}")
        batch = collect_det(policy, train_seed, ep, sample_gen, stateless=stateless)
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {arm} seed={train_seed} update={update}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return policy


def greedy(policy: ActorCritic) -> Any:
    def act(obs: torch.Tensor) -> str:
        with torch.no_grad():
            logits, _ = policy(obs)
        return ACTIONS[int(logits.argmax().item())]

    return act


def calibrate_environment() -> dict[str, Any]:
    """Run C0 zero-training calibration through real SessionStore."""
    scripted_norm = evaluate_policy_with_store(
        scripted_persistent_policy, EVAL_SEEDS, mode="normal"
    )
    scripted_zero = evaluate_policy_with_store(
        scripted_persistent_policy, EVAL_SEEDS, mode="zero_cross"
    )
    scripted_swap = evaluate_policy_with_store(
        scripted_persistent_policy, EVAL_SEEDS, mode="swap_cross"
    )
    gen = torch.Generator().manual_seed(seed_from(CARD, "calib-random"))
    rand_ev = evaluate_policy_with_store(lambda o: random_policy(o, gen), EVAL_SEEDS, mode="normal")
    constants = {
        a: evaluate_policy_with_store(lambda _o, a=a: a, EVAL_SEEDS, mode="normal") for a in ACTIONS
    }
    drop_zero = scripted_norm["mean_return"] - scripted_zero["mean_return"]
    drop_swap = scripted_norm["mean_return"] - scripted_swap["mean_return"]
    c0_pass = (
        scripted_norm["store_roundtrip_exact"]
        and scripted_norm["post_s2_persist_exact"]
        and scripted_norm["mean_return"] >= 4.0
        and scripted_norm["resolve_rate"] >= 0.90
        and drop_zero >= 4.0
        and drop_swap >= 4.0
        and rand_ev["mean_return"] < 0.0
        and all(v["mean_return"] < 0.0 for v in constants.values())
    )
    return {
        "c0_pass": bool(c0_pass),
        "scripted_normal": scripted_norm,
        "scripted_zero_cross_mean": scripted_zero["mean_return"],
        "scripted_swap_cross_mean": scripted_swap["mean_return"],
        "scripted_drop_zero": round(drop_zero, 4),
        "scripted_drop_swap": round(drop_swap, 4),
        "random_true": rand_ev,
        "constants_mean_returns": {a: v["mean_return"] for a, v in constants.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    calib = calibrate_environment()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "env_version": ENV_VERSION,
        "dims": {"obs_s2": OBS_S2_DIM, "cross": CROSS_DIM, "total": STATE_DIM},
        "episodes": episodes,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "gates": {
            "stateless_gap": GATE_STATELESS_GAP,
            "scripted_ratio": GATE_SCRIPTED_RATIO,
            "resolve_rate": GATE_RESOLVE_RATE,
            "counterfactual_drop": GATE_COUNTERFACTUAL_DROP,
        },
        "calibration": calib,
    }

    if not calib["c0_pass"]:
        payload["exit"] = "engineering_failure"
        payload["reason"] = "C0 environment calibration failed"
    else:
        scripted_mean = calib["scripted_normal"]["mean_return"]
        per_seed: dict[str, Any] = {}
        repro_exact = False
        store_ok = True
        try:
            for seed in TRAIN_SEEDS:
                s_t0 = time.process_time()
                pol_pers = train_policy_det(
                    seed, stateless=False, episodes=episodes, deadline=deadline
                )
                pol_stat = train_policy_det(
                    seed, stateless=True, episodes=episodes, deadline=deadline
                )

                if seed == SPOT_CHECK_SEED:
                    pol_pers2 = train_policy_det(
                        seed, stateless=False, episodes=episodes, deadline=deadline
                    )
                    repro_exact = all(
                        torch.equal(a, b)
                        for a, b in zip(pol_pers.parameters(), pol_pers2.parameters(), strict=True)
                    )

                ev_norm = evaluate_policy_with_store(greedy(pol_pers), EVAL_SEEDS, mode="normal")
                ev_zero = evaluate_policy_with_store(
                    greedy(pol_pers), EVAL_SEEDS, mode="zero_cross"
                )
                ev_swap = evaluate_policy_with_store(
                    greedy(pol_pers), EVAL_SEEDS, mode="swap_cross"
                )
                ev_stat = evaluate_policy_with_store(
                    greedy(pol_stat), EVAL_SEEDS, mode="zero_cross"
                )

                store_ok = (
                    store_ok
                    and ev_norm["store_roundtrip_exact"]
                    and ev_norm["post_s2_persist_exact"]
                    and ev_zero["store_roundtrip_exact"]
                    and ev_swap["store_roundtrip_exact"]
                    and ev_stat["store_roundtrip_exact"]
                )

                gap_stateless = ev_norm["mean_return"] - ev_stat["mean_return"]
                drop_zero = ev_norm["mean_return"] - ev_zero["mean_return"]
                drop_swap = ev_norm["mean_return"] - ev_swap["mean_return"]
                g1 = (
                    gap_stateless >= GATE_STATELESS_GAP
                    and ev_norm["mean_return"] >= GATE_SCRIPTED_RATIO * scripted_mean
                    and ev_norm["resolve_rate"] >= GATE_RESOLVE_RATE
                )
                g2 = drop_zero >= GATE_COUNTERFACTUAL_DROP and drop_swap >= GATE_COUNTERFACTUAL_DROP
                per_seed[str(seed)] = {
                    "persistent_normal": ev_norm,
                    "stateless_mean": ev_stat["mean_return"],
                    "stateless_resolve_rate": ev_stat["resolve_rate"],
                    "zero_cross_mean": ev_zero["mean_return"],
                    "swap_cross_mean": ev_swap["mean_return"],
                    "gap_vs_stateless": round(gap_stateless, 4),
                    "drop_zero_cross": round(drop_zero, 4),
                    "drop_swap_cross": round(drop_swap, 4),
                    "g1": bool(g1),
                    "g2": bool(g2),
                    "positive": bool(g1 and g2),
                    "cpu": round(time.process_time() - s_t0, 2),
                }

            payload["per_seed"] = per_seed
            payload["repro_exact"] = bool(repro_exact)
            payload["store_checks_exact"] = bool(store_ok)

            if not repro_exact or not store_ok:
                payload["exit"] = "engineering_failure"
                payload["reason"] = f"repro_exact={repro_exact}, store_ok={store_ok}"
            elif args.smoke:
                payload["exit"] = "smoke"
            elif all(v["positive"] for v in per_seed.values()):
                payload["exit"] = "cross_session_positive"
            elif any(v["positive"] or v["g1"] or v["g2"] for v in per_seed.values()):
                payload["exit"] = "cross_session_partial"
            else:
                payload["exit"] = "cross_session_negative"
        except (TimeoutError, FloatingPointError) as exc:
            payload["per_seed"] = per_seed
            payload["exit"] = "engineering_failure"
            payload["reason"] = str(exc)

    payload["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    payload["discipline"] = {
        "check_reads": 0,
        "final_reads": 0,
        "confirm_reads": 0,
        "scripted_used_in_training": False,
        "continual_plasticity_or_growth": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        k: payload.get(k)
        for k in ("exit", "repro_exact", "store_checks_exact", "cost")
        if payload.get(k) is not None
    }
    summary["c0"] = {
        "c0_pass": calib["c0_pass"],
        "scripted_mean": calib["scripted_normal"]["mean_return"],
        "scripted_resolve_rate": calib["scripted_normal"]["resolve_rate"],
        "random_mean": calib["random_true"]["mean_return"],
    }
    if "per_seed" in payload:
        summary["per_seed"] = {
            s: {
                k: v[k]
                for k in (
                    "stateless_mean",
                    "gap_vs_stateless",
                    "drop_zero_cross",
                    "drop_swap_cross",
                    "g1",
                    "g2",
                    "positive",
                    "cpu",
                )
            }
            for s, v in payload["per_seed"].items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if payload["exit"] != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
