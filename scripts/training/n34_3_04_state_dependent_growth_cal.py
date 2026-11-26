"""N34.3.04 (exploration): state-dependent growth decision v2 with calibrated oracle gate.

Design: plans/reference/DESIGN-N34-3-04_20261126.md (D-20261126-657).
Single change vs N34.3.03: calibrate G2 to the reachable oracle envelope
(`learned - no_grow >= 0.025` and `>= 0.85 * (oracle_per_type - no_grow)`).
Exploration seeds 6611-6613; zero confirmation seed / check / final reads.

Usage: python scripts/training/n34_3_04_state_dependent_growth_cal.py [--smoke]
Output: reports/n34_3_04_state_dependent_growth_cal.json
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
from scripts.training.n34_3_03_state_dependent_growth import (  # noqa: E402
    CAP_CPU,
    N_EVAL_PER_TYPE,
    N_TRAIN_PER_TYPE,
    N_UPDATES,
    SMOKE_TASKS_PER_TYPE,
    SMOKE_UPDATES,
    SPOT_CHECK_SEED,
    TRAIN_SEEDS,
    _build_snapshots,
    _calibration_from_table,
    _eval_table,
    _evaluate_all,
)
from taiji.growth_decision_learning_v2 import train_actor_critic  # noqa: E402

CARD = "N34.3.04"
DESIGN = "DESIGN-N34-3-04_20261126.md"
OUT = ROOT / "reports" / "n34_3_04_state_dependent_growth_cal.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    preflight(CARD, list(TRAIN_SEEDS), record=False)
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    n_train = SMOKE_TASKS_PER_TYPE if args.smoke else N_TRAIN_PER_TYPE
    n_eval = SMOKE_TASKS_PER_TYPE if args.smoke else N_EVAL_PER_TYPE
    n_updates = SMOKE_UPDATES if args.smoke else N_UPDATES

    per_seed: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    try:
        for seed in TRAIN_SEEDS:
            t0 = time.process_time()
            train_snaps = _build_snapshots(seed * 1000, n_train, deadline)
            eval_snaps = _build_snapshots(seed * 1000 + 500, n_eval, deadline)
            eval_tbl = _eval_table(eval_snaps, deadline)
            calib = _calibration_from_table(eval_snaps, eval_tbl)
            if not calib["c0_pass"] and not args.smoke:
                per_seed[str(seed)] = {"calibration": calib, "c0_pass": False}
                raise RuntimeError(f"C0 calibration failed on seed={seed}: {calib}")

            pol = train_actor_critic(train_snaps, seed=seed, deadline=deadline, n_updates=n_updates)
            ev = _evaluate_all(pol, eval_snaps, eval_tbl)

            exact = None
            if seed == SPOT_CHECK_SEED and not args.smoke:
                again = train_actor_critic(
                    train_snaps, seed=seed, deadline=deadline, n_updates=n_updates
                )
                exact = all(
                    torch.equal(a, b)
                    for a, b in zip(pol.parameters(), again.parameters(), strict=True)
                )

            per_seed[str(seed)] = {
                "calibration": calib,
                **ev,
                "weights_exact": exact,
                "cpu": round(time.process_time() - t0, 2),
            }

        c0_all = all(v["calibration"]["c0_pass"] for v in per_seed.values())
        g1 = all(
            v["learned"]["mean_reward"]
            >= max(
                v["baselines"]["no_grow"],
                v["baselines"]["grow8"],
                v["baselines"]["grow16"],
                v["baselines"]["grow24"],
            )
            + 0.008
            and v["learned"]["mean_reward"] >= v["baselines"]["scripted_pressure"] - 0.01
            for v in per_seed.values()
        )
        g2 = all(
            (v["learned"]["mean_reward"] - v["baselines"]["no_grow"]) >= 0.025
            and (v["learned"]["mean_reward"] - v["baselines"]["no_grow"])
            >= 0.85 * (v["calibration"]["oracle_per_type"] - v["baselines"]["no_grow"]) - 1e-9
            for v in per_seed.values()
        )
        g3 = all(
            v["learned"]["easy_no_grow_rate"] >= 0.70 and v["learned"]["hard_grow_rate"] >= 0.70
            for v in per_seed.values()
        )
        g4 = all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        )
        checks = {
            "c0_calibration": c0_all,
            "g1_beats_fixed_and_scripted": g1,
            "g2_oracle_calibrated_gain": g2,
            "g3_state_dependent": g3,
            "g4_deterministic": g4,
        }
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "growth_decision_v2_positive"
        else:
            exit_ = "growth_decision_v2_partial"
    except (TimeoutError, FloatingPointError, RuntimeError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = str(exc)

    payload = {
        "schema": "taiji-n34-3-04-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "train_seeds": list(TRAIN_SEEDS),
        "n_train_per_type": n_train,
        "n_eval_per_type": n_eval,
        "n_updates": n_updates,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "scripted_rule_is_control_only": True,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "per_seed": per_seed, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("growth_decision_v2_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
