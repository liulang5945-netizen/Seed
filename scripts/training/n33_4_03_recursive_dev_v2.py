"""N33.4.02 (exploration): consequence-model recursive development vs masked & permuted controls.

Pre-registered: plans/reference/DESIGN-N33-4-03_20261126.md (D-20261126-603).
Single variable vs N33.4.02: outcome-end [damage, cost, delta_u] loss weights standardised by relative
target variance, CM steps 600 -> 1,200, logged train episodes 300 -> 600; environment and all gates unchanged.
Trains ConsequenceModel and model-based InterventionPolicy (T=4 imagined rollouts) from
random exploratory logs on seeds (4001, 4002, 4003), comparing `full_recursive` against
`masked_outcome`, `permuted_outcome`, and `fixed_parent` on EVAL_SEEDS (9000..9099).
Zero check/final/confirm reads (20261711-13 untouched).

Usage:
  python scripts/training/n33_4_02_recursive_dev.py --smoke
  python scripts/training/n33_4_02_recursive_dev.py
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
from scripts.training.n33_4_01_recursive_env_check import (
    EVAL_SEEDS,
    calibrate_recursive_env,
)  # noqa: E402
from taiji.recursive_dev import (  # noqa: E402
    ARCH_VERSION,
    greedy_recursive_policy,
    train_consequence_model,
    train_policy_in_model,
)
from taiji.recursive_dev_env import (  # noqa: E402
    ENV_VERSION,
    RecursiveDevEnv,
    collect_logged_experience,
    evaluate_recursive_with_store,
    seed_from,
)

CARD = "N33.4.03"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
TRAIN_SEEDS = (4001, 4002, 4003)
SPOT_CHECK_SEED = 4001
CM_STEPS = 1200
POL_EPISODES = 1600
SMOKE_CM_STEPS = 30
SMOKE_POL_EPISODES = 32
CAP_CPU = 30.0
OUT = ROOT / "reports" / "n33_4_03_recursive_dev.json"


def _run_seed_bundle(
    seed: int,
    *,
    cm_steps: int,
    pol_episodes: int,
    eval_seeds: list[int],
    deadline: float,
) -> tuple[dict[str, Any], Any, Any]:
    log_train_seeds = [seed * 1000 + i for i in range(600)]
    log_val_seeds = [seed * 1000 + 500 + i for i in range(100)]
    train_tr = collect_logged_experience(log_train_seeds, seed_from(CARD, seed, "log-tr"))
    val_tr = collect_logged_experience(log_val_seeds, seed_from(CARD, seed, "log-va"))
    init_states = torch.stack([RecursiveDevEnv(s).reset() for s in log_train_seeds])

    arms: dict[str, Any] = {}
    full_cm = None
    full_pol = None
    for mode in ("full", "masked", "permuted"):
        cm, cm_stats = train_consequence_model(
            train_tr,
            val_tr,
            seed,
            mode=mode,
            steps=cm_steps,
            outcome_weight_scale=True,
            deadline=deadline,
        )
        pol = train_policy_in_model(
            cm, init_states, seed, mode=mode, episodes=pol_episodes, deadline=deadline
        )
        ev = evaluate_recursive_with_store(
            greedy_recursive_policy(pol, cm, mask_preds=(mode == "masked")),
            eval_seeds,
        )
        arms[mode] = {"cmodel": cm_stats, "eval": ev}
        if mode == "full":
            full_cm, full_pol = cm, pol
    return arms, full_cm, full_pol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    cm_steps = SMOKE_CM_STEPS if args.smoke else CM_STEPS
    pol_eps = SMOKE_POL_EPISODES if args.smoke else POL_EPISODES

    calib = calibrate_recursive_env(EVAL_SEEDS)
    fixed_parent = evaluate_recursive_with_store(lambda _o: "noop_hold", EVAL_SEEDS)

    per_seed: dict[str, Any] = {}
    repro_exact = False
    store_exact = bool(calib["c0_pass"] and fixed_parent["store_roundtrip_exact"])
    oracle_gain = calib["oracle"]["mean_net_gain"]

    if not calib["c0_pass"]:
        exit_ = "engineering_failure"
    else:
        try:
            for seed in TRAIN_SEEDS:
                s_t0 = time.process_time()
                arms, full_cm, full_pol = _run_seed_bundle(
                    seed,
                    cm_steps=cm_steps,
                    pol_episodes=pol_eps,
                    eval_seeds=EVAL_SEEDS,
                    deadline=deadline,
                )
                if seed == SPOT_CHECK_SEED:
                    _, cm2, pol2 = _run_seed_bundle(
                        seed,
                        cm_steps=cm_steps,
                        pol_episodes=pol_eps,
                        eval_seeds=EVAL_SEEDS,
                        deadline=deadline,
                    )
                    repro_exact = all(
                        torch.equal(a, b)
                        for a, b in zip(full_cm.parameters(), cm2.parameters(), strict=True)
                    ) and all(
                        torch.equal(a, b)
                        for a, b in zip(full_pol.parameters(), pol2.parameters(), strict=True)
                    )

                for v in arms.values():
                    store_exact = store_exact and v["eval"]["store_roundtrip_exact"]

                fg = arms["full"]["eval"]["mean_net_gain"]
                mg = arms["masked"]["eval"]["mean_net_gain"]
                pg = arms["permuted"]["eval"]["mean_net_gain"]
                g_vs_mask = round(fg - mg, 6)
                g_vs_perm = round(fg - pg, 6)
                g1 = (
                    arms["full"]["cmodel"]["delta_u_r2"] >= 0.85
                    and arms["permuted"]["cmodel"]["delta_u_r2"] <= 0.10
                )
                g2 = (
                    fg >= 0.15
                    and fg >= 0.80 * oracle_gain
                    and arms["full"]["eval"]["mean_improving_steps"] >= 3.0
                )
                g3 = g_vs_mask >= 0.02 and g_vs_perm >= 0.02
                per_seed[str(seed)] = {
                    "full": arms["full"],
                    "masked": arms["masked"],
                    "permuted": arms["permuted"],
                    "gain_over_masked": g_vs_mask,
                    "gain_over_permuted": g_vs_perm,
                    "g1": bool(g1),
                    "g2": bool(g2),
                    "g3": bool(g3),
                    "positive": bool(g1 and g2 and g3),
                    "cpu": round(time.process_time() - s_t0, 2),
                }

            all_g4 = repro_exact and store_exact
            if not all_g4:
                exit_ = "engineering_failure"
            elif args.smoke:
                exit_ = "smoke"
            elif all(v["positive"] for v in per_seed.values()):
                exit_ = "recursive_dev_positive"
            elif any(v["positive"] or (v["g1"] and v["g3"]) for v in per_seed.values()):
                exit_ = "recursive_dev_partial"
            else:
                exit_ = "recursive_dev_negative"
        except (TimeoutError, FloatingPointError) as exc:
            exit_ = "engineering_failure"
            per_seed["error"] = str(exc)

    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "arch_version": ARCH_VERSION,
        "env_version": ENV_VERSION,
        "cm_steps": cm_steps,
        "pol_episodes": pol_eps,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "calibration": calib,
        "fixed_parent": fixed_parent,
        "repro_spot_check": {"seed": SPOT_CHECK_SEED, "exact": bool(repro_exact)},
        "store_checks_exact": bool(store_exact),
        "per_seed": per_seed,
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

    summary: dict[str, Any] = {
        "exit": exit_,
        "repro_spot_check": payload["repro_spot_check"],
        "store_checks_exact": store_exact,
        "oracle_gain": oracle_gain,
        "cost": payload["cost"],
    }
    if all(isinstance(v, dict) and "full" in v for v in per_seed.values()):
        summary["per_seed"] = {
            s: {
                "r2_full": v["full"]["cmodel"]["delta_u_r2"],
                "r2_perm": v["permuted"]["cmodel"]["delta_u_r2"],
                "full_gain": v["full"]["eval"]["mean_net_gain"],
                "full_imp_steps": v["full"]["eval"]["mean_improving_steps"],
                "masked_gain": v["masked"]["eval"]["mean_net_gain"],
                "permuted_gain": v["permuted"]["eval"]["mean_net_gain"],
                "over_mask": v["gain_over_masked"],
                "over_perm": v["gain_over_permuted"],
                "positive": v["positive"],
                "cpu": v["cpu"],
            }
            for s, v in per_seed.items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
