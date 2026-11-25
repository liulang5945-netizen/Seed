"""N33.1.08 (exploration): deterministic-sampling PPO replication on the frozen N33.1.07 probe set.

Pre-registered: plans/reference/DESIGN-N33-1-08_20261125.md (D-20261125-569).
Single variable vs N33.1.06: explicit per-seed rollout generator for torch.multinomial so training
is bit-exact reproducible across independent passes; evaluated on the frozen N33.1.07 fixed probe.

Usage:
  python scripts/training/n33_1_08_ppo_deterministic.py --smoke
  python scripts/training/n33_1_08_ppo_deterministic.py
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
from scripts.training.n33_1_04_gate_calibration import (  # noqa: E402
    EVAL_SEEDS,
    GATE_G1_RANDOM,
    GATE_G1_SCRIPTED_RATIO,
    GATE_G2_SHARE,
    gate_eval,
)
from scripts.training.n33_1_06_ppo_more_episodes import (  # noqa: E402
    CLIP,
    ENTROPY_END,
    ENTROPY_START,
    EPISODES,
    GAMMA,
    GATE_G3_DROP,
    LAM,
    LR,
    PPO_EPOCHS,
    ROLLOUT_EPISODES,
    TRAIN_SEEDS,
    VALUE_COEF,
    ActorCritic,
    greedy,
)
from scripts.training.n33_1_07_fixed_probe import (  # noqa: E402
    K_PER_DOMAIN,
    PROBE_ENV_SEEDS,
    build_probe_set,
    fixed_g2,
)
from taiji.choice_simulator import (  # noqa: E402
    CHOICES,
    HORIZON,
    SIM_VERSION,
    ChoiceSessionEnv,
    evaluate_policy,
    random_policy,
    scripted_policy,
    seed_from,
)

CARD = "N33.1.08"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = f"DESIGN-{CARD_SLUG}_20261125.md"
SMOKE_EPISODES = 24
CAP_CPU = 90.0
OUT = ROOT / "reports" / "n33_1_08_ppo_det.json"
EXPECTED_POOL_SIZES = {"rest": 969, "verify": 358, "work": 2247}


def collect_det(
    policy: ActorCritic,
    train_seed: int,
    ep_start: int,
    sample_gen: torch.Generator,
) -> dict[str, Any]:
    """Collect one rollout batch using an explicit deterministic sampling generator."""
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
            a = int(torch.multinomial(dist, 1, generator=sample_gen).item())
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


def train_policy_det(train_seed: int, episodes: int, deadline: float) -> ActorCritic:
    """Train one ActorCritic with explicit sampling RNG (independent of global RNG)."""
    policy = ActorCritic(train_seed)
    sample_gen = torch.Generator().manual_seed(seed_from(CARD, train_seed, "sample"))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    total_updates = max(1, episodes // ROLLOUT_EPISODES)
    ep = 0
    for update in range(total_updates):
        if time.process_time() > deadline:
            raise TimeoutError(f"seed {train_seed} update {update}")
        batch = collect_det(policy, train_seed, ep, sample_gen)
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
                raise FloatingPointError(f"non-finite loss seed={train_seed} update={update}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    probe = build_probe_set()
    sets = probe["sets"]
    scripted_probe = fixed_g2(scripted_policy, sets)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "algorithm": "ppo_actor_critic_deterministic",
        "fixture": SIM_VERSION,
        "episodes": episodes,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": EVAL_SEEDS,
        "probe": {
            "source_card": "N33.1.07",
            "env_seeds": [PROBE_ENV_SEEDS[0], PROBE_ENV_SEEDS[-1]],
            "k_per_domain": K_PER_DOMAIN,
            "pool_sizes": probe["pool_sizes"],
            "scripted_shares": scripted_probe["shares"],
        },
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_ratio": GATE_G1_SCRIPTED_RATIO,
            "g2_share": GATE_G2_SHARE,
            "g3_drop": GATE_G3_DROP,
        },
    }

    if probe["pool_sizes"] != EXPECTED_POOL_SIZES or not scripted_probe["g2"]:
        payload["exit"] = "engineering_failure"
        payload["reason"] = f"frozen probe mismatch: {probe['pool_sizes']}"
    else:
        gen = torch.Generator().manual_seed(7)
        random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
        scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
        baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}
        payload["baselines"] = {"random": random_base, "scripted": scripted_base}

        per_seed: dict[str, Any] = {}
        try:
            for seed in TRAIN_SEEDS:
                s_t0 = time.process_time()
                pol1 = train_policy_det(seed, episodes, deadline)
                ev1 = gate_eval(greedy(pol1), baselines)
                zero1 = evaluate_policy(greedy(pol1, zero_state=True), EVAL_SEEDS)
                fg1 = fixed_g2(greedy(pol1), sets)

                pol2 = train_policy_det(seed, episodes, deadline)
                ev2 = gate_eval(greedy(pol2), baselines)
                zero2 = evaluate_policy(greedy(pol2, zero_state=True), EVAL_SEEDS)
                fg2 = fixed_g2(greedy(pol2), sets)

                weights_exact = all(
                    torch.equal(p1, p2)
                    for p1, p2 in zip(pol1.parameters(), pol2.parameters(), strict=True)
                )
                metrics_exact = (
                    ev1["eval"]["mean_return"] == ev2["eval"]["mean_return"]
                    and zero1["mean_return"] == zero2["mean_return"]
                    and fg1["shares"] == fg2["shares"]
                )
                repro_exact = bool(weights_exact and metrics_exact)
                drop = ev1["eval"]["mean_return"] - zero1["mean_return"]
                g3 = drop >= GATE_G3_DROP
                per_seed[str(seed)] = {
                    "weights_exact": bool(weights_exact),
                    "metrics_exact": bool(metrics_exact),
                    "repro_exact": repro_exact,
                    "return_normal": ev1["eval"]["mean_return"],
                    "return_zeroed_state": zero1["mean_return"],
                    "state_drop": round(drop, 4),
                    "g1": ev1["g1"],
                    "g2_fixed": fg1["g2"],
                    "g3": bool(g3),
                    "fixed_shares": fg1["shares"],
                    "trajectory_probes_for_reference": ev1["probes"],
                    "trajectory_g2_for_reference": ev1["g2"],
                    "positive": bool(repro_exact and ev1["g1"] and fg1["g2"] and g3),
                    "cpu_2pass": round(time.process_time() - s_t0, 2),
                }

            payload["per_seed"] = per_seed
            all_repro = all(v["repro_exact"] for v in per_seed.values())
            payload["repro_exact"] = bool(all_repro)
            if args.smoke:
                payload["exit"] = "smoke" if all_repro else "repro_failure"
            elif not all_repro:
                payload["exit"] = "repro_failure"
            elif all(v["positive"] for v in per_seed.values()):
                payload["exit"] = "det_ppo_positive"
            elif all(v["g1"] and v["g3"] for v in per_seed.values()):
                payload["exit"] = "det_ppo_partial"
            else:
                payload["exit"] = "det_ppo_negative"
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        k: payload.get(k) for k in ("exit", "repro_exact", "cost") if payload.get(k) is not None
    }
    if "per_seed" in payload:
        summary["per_seed"] = {
            s: {
                k: v[k]
                for k in (
                    "repro_exact",
                    "return_normal",
                    "state_drop",
                    "g1",
                    "g2_fixed",
                    "g3",
                    "fixed_shares",
                    "positive",
                    "cpu_2pass",
                )
            }
            for s, v in payload["per_seed"].items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if payload["exit"] not in ("repro_failure", "engineering_failure") else 3


if __name__ == "__main__":
    raise SystemExit(main())
