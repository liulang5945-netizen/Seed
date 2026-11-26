"""PHY.2.01 (exploration): multimodal physics causal transfer v2 without family leakage.

Design: plans/reference/DESIGN-PHY-2-01_20261126.md (D-20261126-652).
Trains ConsequenceNetV2 on `ramp_bumper` only and evaluates on held-out `ramp_bumper` plus
unseen `rough_collision` (zero overlap in slope/roughness, zero family_id in obs), checking
per-outcome collision recall, fair cross-family count baseline, and counterfactual barrier removal.
Exploration seeds 6411-6413; zero confirmation seed / check / final reads.

Usage: python scripts/training/phy_2_01_causal_transfer.py [--smoke]
Output: reports/phy_2_01_causal_transfer.json
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

from taiji.multimodal_physics_env_v2 import (  # noqa: E402
    FAMILIES,
    OUTCOMES,
    ablate_barrier_collision_recall,
    collect_v2,
    eval_count_baseline_v2,
    eval_metrics_v2,
    train_net_v2,
)

CARD = "PHY.2.01"
TRAIN_SEEDS = (6411, 6412, 6413)
SPOT_CHECK_SEED = 6411
TRAIN_EPISODES = 1000
EVAL_EPISODES = 150
STEPS = 1600
SMOKE_TRAIN_EPISODES = 100
SMOKE_EVAL_EPISODES = 40
SMOKE_STEPS = 200
CAP_CPU = 30.0
OUT = ROOT / "reports" / "phy_2_01_causal_transfer.json"


def _calibrate(n_ep: int) -> dict[str, Any]:
    coll_idx = OUTCOMES.index("碰撞")
    fracs = {}
    for fam in FAMILIES:
        _, y = collect_v2(900, fam, min(n_ep, 200))
        fracs[fam] = round(float((y == coll_idx).float().mean()), 4)
    c0_pass = all(v >= 0.10 for v in fracs.values())
    return {"c0_pass": c0_pass, "collision_fraction": fracs, "oracle_acc": 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    n_train = SMOKE_TRAIN_EPISODES if args.smoke else TRAIN_EPISODES
    n_eval = SMOKE_EVAL_EPISODES if args.smoke else EVAL_EPISODES
    steps = SMOKE_STEPS if args.smoke else STEPS

    calib = _calibrate(n_train)
    per_seed: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    if not calib["c0_pass"]:
        exit_ = "engineering_failure"
    else:
        for seed in TRAIN_SEEDS:
            t0 = time.process_time()
            x_tr, y_tr = collect_v2(seed, "ramp_bumper", n_train)
            x_in, y_in = collect_v2(seed + 100, "ramp_bumper", n_eval)
            x_cross, y_cross = collect_v2(seed + 200, "rough_collision", n_eval)
            net = train_net_v2(x_tr, y_tr, seed=seed, steps=steps)

            m_in = eval_metrics_v2(net, x_in, y_in)
            m_cross = eval_metrics_v2(net, x_cross, y_cross)
            base_in = eval_count_baseline_v2(
                seed, "ramp_bumper", seed + 100, "ramp_bumper", n_train, n_eval
            )
            base_cross = eval_count_baseline_v2(
                seed, "ramp_bumper", seed + 200, "rough_collision", n_train, n_eval
            )
            abl = ablate_barrier_collision_recall(net, x_cross, y_cross)

            exact = None
            if seed == SPOT_CHECK_SEED and not args.smoke:
                again = train_net_v2(x_tr, y_tr, seed=seed, steps=steps)
                exact = all(
                    torch.equal(a, b)
                    for a, b in zip(net.parameters(), again.parameters(), strict=True)
                )

            per_seed[str(seed)] = {
                "in_family": m_in,
                "cross_family": m_cross,
                "count_in_family": base_in,
                "count_cross_family": base_cross,
                "barrier_ablation": abl,
                "weights_exact": exact,
                "cpu": round(time.process_time() - t0, 2),
                "n_train": int(x_tr.shape[0]),
                "n_eval": int(x_in.shape[0]),
            }

        checks = {
            "c0_calibration": bool(calib["c0_pass"]),
            "g1_in_family": all(
                v["in_family"]["accuracy"] >= 0.90 and v["in_family"]["collision_recall"] >= 0.85
                for v in per_seed.values()
            ),
            "g2_cross_family": all(
                v["cross_family"]["accuracy"] >= 0.80
                and v["cross_family"]["collision_recall"] >= 0.70
                and v["cross_family"]["macro_f1"] >= 0.65
                for v in per_seed.values()
            ),
            "g3_beats_count_both": all(
                v["in_family"]["accuracy"] >= v["count_in_family"]["accuracy"] + 0.05
                and v["cross_family"]["accuracy"] >= v["count_cross_family"]["accuracy"] + 0.10
                for v in per_seed.values()
            ),
            "g4_barrier_ablation": all(
                v["barrier_ablation"]["collision_recall_drop"] >= 0.35 for v in per_seed.values()
            ),
            "g5_deterministic": all(
                v["weights_exact"] is True
                for v in per_seed.values()
                if v["weights_exact"] is not None
            ),
        }
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "multimodal_v2_positive"
        else:
            exit_ = "multimodal_v2_partial"

    payload = {
        "schema": "taiji-phy-2-01-v1",
        "card": CARD,
        "design": "DESIGN-PHY-2-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "train_seeds": list(TRAIN_SEEDS),
        "train_family": "ramp_bumper",
        "eval_families": ["ramp_bumper (held-out)", "rough_collision (cross-family)"],
        "calibration": calib,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "oracle_used_by_subject": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "calibration": calib,
                "per_seed": {
                    k: {
                        "in_acc": v["in_family"]["accuracy"],
                        "in_coll": v["in_family"]["collision_recall"],
                        "cross_acc": v["cross_family"]["accuracy"],
                        "cross_coll": v["cross_family"]["collision_recall"],
                        "cross_f1": v["cross_family"]["macro_f1"],
                        "count_cross": v["count_cross_family"]["accuracy"],
                        "abl_drop": v["barrier_ablation"]["collision_recall_drop"],
                    }
                    for k, v in per_seed.items()
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("multimodal_v2_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
