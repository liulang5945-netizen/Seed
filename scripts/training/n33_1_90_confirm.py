"""N33.1.90 (confirmation): PPO autonomous-choice policy confirmed on 3 unread seeds.

Pre-registered: plans/reference/DESIGN-N33-1-90_20261125.md (D-20261125-568).
Runs the frozen N33.1.06 configuration (fixture v4) on confirmation seeds
20261501-20261503 and evaluates G1/G2/G3 with the frozen probe set (K=150, N33.1.07).
Scope is strictly the simulated session environment; no check/final, no cross-session,
no growth claims.

Usage:
  python scripts/training/n33_1_90_confirm.py --smoke
  python scripts/training/n33_1_90_confirm.py
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

from scripts.training.n31_guard import preflight  # noqa: E402
from scripts.training.n33_1_05_ppo_policy import (  # noqa: E402
    EPISODES,
    GATE_G1_RANDOM,
    GATE_G1_SCRIPTED_RATIO,
    GATE_G3_DROP,
    greedy,
    train_policy,
)
from taiji.choice_probes import probe_shares  # noqa: E402
from taiji.choice_simulator import evaluate_policy, random_policy, scripted_policy  # noqa: E402

CARD = "N33.1.90"
CONFIRM_SEEDS = (20261501, 20261502, 20261503)
EVAL_SEEDS = list(range(5000, 5040))
SMOKE_EPISODES = 30
CAP_CPU = 60.0
OUT = ROOT / "reports" / "n33_1_90_confirm.json"
PROBE_SHARES_MIN = 0.50


def evaluator(seed: int, episodes: int, baselines: dict[str, float], *, probe_fn):
    policy = train_policy(seed, episodes, time.process_time() + CAP_CPU)
    normal = evaluate_policy(greedy(policy), EVAL_SEEDS)
    zeroed = evaluate_policy(greedy(policy, zero_state=True), EVAL_SEEDS)
    shares = probe_fn(greedy(policy))
    drop = normal["mean_return"] - zeroed["mean_return"]
    g1 = (
        normal["mean_return"] >= baselines["random"] + GATE_G1_RANDOM
        and normal["mean_return"] >= GATE_G1_SCRIPTED_RATIO * baselines["scripted"]
    )
    g2 = (
        min(
            shares["rest_share"], shares["verify_or_help_share"], shares["continue_or_recall_share"]
        )
        >= PROBE_SHARES_MIN
    )
    g3 = drop >= GATE_G3_DROP
    return {
        "seed": seed,
        "return_normal": normal["mean_return"],
        "return_zeroed_state": zeroed["mean_return"],
        "state_drop": round(drop, 4),
        "probes": shares,
        "g1": bool(g1),
        "g2": bool(g2),
        "g3": bool(g3),
        "positive": bool(g1 and g2 and g3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()
    design = preflight(CARD, list(CONFIRM_SEEDS), record=not args.smoke).name

    gen = torch.Generator().manual_seed(7)
    random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
    scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
    baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    if args.smoke:
        t0 = time.process_time()
        train_policy(CONFIRM_SEEDS[0], SMOKE_EPISODES, started + CAP_CPU)
        train_cpu = time.process_time() - t0
        updates = max(1, SMOKE_EPISODES // 8)
        per_update = train_cpu / updates
        projected = round(per_update * (EPISODES // 8) * 3 * 1.15 + 3.0, 2)
        cpu = round(time.process_time() - started, 2)
        payload = {
            "smoke": True,
            "projected_full_cpu": projected,
            "train_cpu": round(train_cpu, 3),
            "cpu": cpu,
            "cap_cpu": CAP_CPU,
            "exit": "smoke",
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"exit": "smoke", "projected_full_cpu": projected, "cpu": cpu, "cap_cpu": CAP_CPU}
            )
        )
        return 0 if projected <= CAP_CPU else 3

    per_seed: dict[str, Any] = {}
    for seed in CONFIRM_SEEDS:
        per_seed[str(seed)] = evaluator(seed, episodes, baselines, probe_fn=probe_shares)
    cpu = round(time.process_time() - started, 2)
    if all(v["positive"] for v in per_seed.values()):
        exit_ = "confirm_positive"
    elif all(v["g1"] for v in per_seed.values()):
        exit_ = "confirm_partial"
    else:
        exit_ = "confirm_unstable"
    payload = {
        "schema": "taiji-n33-1-90-v1",
        "card": CARD,
        "design": design,
        "created_utc": datetime.now(UTC).isoformat(),
        "fixture": "taiji-choice-session-v4",
        "confirm_seeds": list(CONFIRM_SEEDS),
        "episodes": episodes,
        "eval_seeds": EVAL_SEEDS,
        "baselines": {"random": random_base, "scripted": scripted_base},
        "probe_set": "n33_1_probe_states_20261125 (K=150)",
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_ratio": GATE_G1_SCRIPTED_RATIO,
            "g2_share": PROBE_SHARES_MIN,
            "g3_drop": GATE_G3_DROP,
        },
        "per_seed": per_seed,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": len(CONFIRM_SEEDS),
            "deterministic": True,
        },
        "scope": "simulated session env v4 only; no real cross-session/growth/check/final",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exit": exit_, "cpu": cpu, "per_seed": per_seed}, ensure_ascii=False))
    return 0 if exit_ in ("confirm_positive", "confirm_partial", "confirm_unstable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
