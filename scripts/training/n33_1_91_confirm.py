"""N33.1.91 (confirmation, redone): deterministic PPO autonomous-choice policy on 3 unread seeds.

Pre-registered: plans/reference/DESIGN-N33-1-91_20261125.md (D-20261125-575).
Replaces the invalid N33.1.90 (D-574). Uses the deterministic sampler from N33.1.08 and the
frozen N33.1.07 fixed probe set (K=150). Confirmation seeds 20261511-20261513 are read once.
Scope: taiji-choice-session-v4 simulated sessions only; no check/final, no real cross-session,
no growth claims.

Usage:
  python scripts/training/n33_1_91_confirm.py --smoke
  python scripts/training/n33_1_91_confirm.py
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
from scripts.training.n33_1_04_gate_calibration import (  # noqa: E402
    EVAL_SEEDS,
    GATE_G1_RANDOM,
    GATE_G1_SCRIPTED_RATIO,
    GATE_G2_SHARE,
    gate_eval,
)
from scripts.training.n33_1_06_ppo_more_episodes import (  # noqa: E402
    EPISODES,
    GATE_G3_DROP,
    greedy,
)
from scripts.training.n33_1_07_fixed_probe import (  # noqa: E402
    K_PER_DOMAIN,
    PROBE_ENV_SEEDS,
    build_probe_set,
    fixed_g2,
)
from scripts.training.n33_1_08_ppo_deterministic import train_policy_det  # noqa: E402
from taiji.choice_simulator import (  # noqa: E402
    SIM_VERSION,
    evaluate_policy,
    random_policy,
    scripted_policy,
)

CARD = "N33.1.91"
CONFIRM_SEEDS = (20261511, 20261512, 20261513)
SPOT_CHECK_SEED = 20261511
SMOKE_EPISODES = 24
CAP_CPU = 60.0
OUT = ROOT / "reports" / "n33_1_91_confirm.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    design = preflight(CARD, list(CONFIRM_SEEDS), record=not args.smoke).name
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    probe = build_probe_set()
    sets = probe["sets"]
    scripted_probe = fixed_g2(scripted_policy, sets)
    gen = torch.Generator().manual_seed(7)
    random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
    scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
    baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}

    if not scripted_probe["g2"]:
        payload = {
            "schema": "taiji-n33-1-91-v1",
            "card": CARD,
            "exit": "engineering_failure",
            "reason": "frozen probe mismatch",
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1

    per_seed: dict[str, Any] = {}
    weights_exact = False
    try:
        for seed in CONFIRM_SEEDS:
            t0 = time.process_time()
            pol = train_policy_det(seed, episodes, deadline)
            ev = gate_eval(greedy(pol), baselines)
            zero = evaluate_policy(greedy(pol, zero_state=True), EVAL_SEEDS)
            fg = fixed_g2(greedy(pol), sets)
            drop = ev["eval"]["mean_return"] - zero["mean_return"]
            g3 = drop >= GATE_G3_DROP
            per_seed[str(seed)] = {
                "return_normal": ev["eval"]["mean_return"],
                "return_zeroed_state": zero["mean_return"],
                "state_drop": round(drop, 4),
                "g1": ev["g1"],
                "g2_fixed": fg["g2"],
                "fixed_shares": fg["shares"],
                "g3": bool(g3),
                "positive": bool(ev["g1"] and fg["g2"] and g3),
                "cpu": round(time.process_time() - t0, 2),
            }
            if seed == SPOT_CHECK_SEED:
                pol2 = train_policy_det(seed, episodes, deadline)
                weights_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(pol.parameters(), pol2.parameters(), strict=True)
                )
    except (TimeoutError, FloatingPointError) as exc:
        payload = {
            "schema": "taiji-n33-1-91-v1",
            "card": CARD,
            "design": design,
            "exit": "engineering_failure",
            "reason": str(exc),
            "per_seed": per_seed,
            "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"exit": "engineering_failure", "reason": str(exc)}))
        return 1

    all_positive = all(v["positive"] for v in per_seed.values())
    all_g1_g3 = all(v["g1"] and v["g3"] for v in per_seed.values())
    if not weights_exact:
        exit_ = "engineering_failure"
    elif all_positive:
        exit_ = "confirm_positive"
    elif all_g1_g3:
        exit_ = "confirm_partial"
    else:
        exit_ = "confirm_unstable"

    payload = {
        "schema": "taiji-n33-1-91-v1",
        "card": CARD,
        "design": design,
        "created_utc": datetime.now(UTC).isoformat(),
        "fixture": SIM_VERSION,
        "confirm_seeds": list(CONFIRM_SEEDS),
        "episodes": episodes,
        "eval_seeds": EVAL_SEEDS,
        "probe": {
            "source_card": "N33.1.07",
            "env_seeds": [PROBE_ENV_SEEDS[0], PROBE_ENV_SEEDS[-1]],
            "k_per_domain": K_PER_DOMAIN,
            "pool_sizes": probe["pool_sizes"],
            "scripted_shares": scripted_probe["shares"],
        },
        "baselines": {"random": random_base, "scripted": scripted_base},
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_ratio": GATE_G1_SCRIPTED_RATIO,
            "g2_share": GATE_G2_SHARE,
            "g3_drop": GATE_G3_DROP,
        },
        "per_seed": per_seed,
        "repro_spot_check": {"seed": SPOT_CHECK_SEED, "exact": weights_exact},
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": len(CONFIRM_SEEDS),
            "deterministic_sampler": True,
        },
        "scope": "taiji-choice-session-v4 simulated sessions only; no real cross-session/growth/check/final",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "cpu": payload["cost"]["cpu"],
                "repro_spot": weights_exact,
                "per_seed": per_seed,
            },
            ensure_ascii=False,
        )
    )
    return (
        0
        if exit_
        in ("confirm_positive", "confirm_partial", "confirm_unstable", "engineering_failure")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
