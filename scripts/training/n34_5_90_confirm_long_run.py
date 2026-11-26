"""N34.5.90 (confirmation): single-subject 100-session trained durability on 20261811-13.

Pre-registered: plans/reference/DESIGN-N34-5-90_20261126.md (D-20261126-662).
Freezes the exact N34.5.01 configuration and executes one formal confirmation pass on
`20261811, 20261812, 20261813` (`record=True`). Zero check/final reads.

Usage:
  python scripts/training/n34_5_90_confirm_long_run.py --smoke
  python scripts/training/n34_5_90_confirm_long_run.py
Output:
  reports/n34_5_90_confirm_long_run.json (formal mode only)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.n31_guard import find_design, preflight  # noqa: E402
from scripts.training.n34_5_01_long_run_joint import _eval_gates  # noqa: E402
from taiji.long_run_trained_subject import RunConfig, run_long_run_subject  # noqa: E402

CARD = "N34.5.90"
CONFIRM_SEEDS = (20261811, 20261812, 20261813)
SMOKE_SEEDS = (6911, 6912, 6913)
CAP_CPU = 160.0
OUT = ROOT / "reports" / "n34_5_90_confirm_long_run.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        design_file = find_design(CARD).name
        seeds = SMOKE_SEEDS
        cfg = RunConfig(
            n_sessions=12,
            events_per_session=20,
            crash_sessions=(3, 8),
            p1_end_session=4,
            p2_end_session=10,
            dormancy_session=11,
            ch_train_end_session=6,
            ch_updates_per_session=2,
            phy_steps_early=10,
            phy_steps_mid=10,
            grow_p1_steps=10,
            grow_p2_steps=10,
            grow_maint_steps=2,
        )
    else:
        design_file = preflight(CARD, list(CONFIRM_SEEDS), record=True).name
        seeds = CONFIRM_SEEDS
        cfg = RunConfig()

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    spot = seeds[0]
    per_seed: dict[str, Any] = {}
    spot_match = False
    checks: dict[str, bool] = {}

    try:
        for seed in seeds:
            t0 = time.process_time()
            with tempfile.TemporaryDirectory() as tmp1:
                res = run_long_run_subject(tmp1, seed, deadline, cfg=cfg)
            if seed == spot and not args.smoke:
                with tempfile.TemporaryDirectory() as tmp2:
                    again = run_long_run_subject(tmp2, seed, deadline, cfg=cfg)
                spot_match = (
                    res["durability"]["chain_head"] == again["durability"]["chain_head"]
                    and res["durability"]["record_digest"] == again["durability"]["record_digest"]
                )
            elif seed == spot and args.smoke:
                spot_match = True
            res["cpu"] = round(time.process_time() - t0, 2)
            per_seed[str(seed)] = res

        checks = _eval_gates(
            per_seed,
            spot_match,
            n_sessions=cfg.n_sessions,
            n_events=cfg.n_sessions * cfg.events_per_session,
        )
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "confirm_positive"
        else:
            exit_ = "confirm_partial"
    except (TimeoutError, FloatingPointError, RuntimeError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = str(exc)

    payload = {
        "schema": "taiji-n34-5-90-v1",
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0 if args.smoke else len(CONFIRM_SEEDS),
            "confirmation_seeds": [] if args.smoke else list(CONFIRM_SEEDS),
            "oracle_used_by_subject": False,
        },
    }
    if not args.smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "exit": exit_,
        "checks": checks,
        "cpu": payload["cost"]["cpu"],
        "seeds": {
            k: {
                "sess": v["durability"]["sessions"],
                "events": v["durability"]["events"],
                "crashes": v["durability"]["crash_recoveries"],
                "ch_ho": v["chinese"]["heldout"]["coref_acc"],
                "phy_cr": v["physics"]["cross_family"]["accuracy"],
                "phy_coll": v["physics"]["cross_family"]["collision_recall"],
                "p1_drop": v["growth"]["p1_retention_drop"],
                "p2": v["growth"]["p2_acc"],
                "ecl_db": v["credibility"]["delta_brier"],
                "ecl_dp": v["credibility"]["delta_post_abs"],
                "unc_ece": v["uncertainty"]["coupled_ece"],
                "unc_bg": v["uncertainty"]["brier_vs_static"],
                "cpu": v["cpu"],
            }
            for k, v in per_seed.items()
            if isinstance(v, dict) and "durability" in v
        },
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ in ("confirm_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
