"""N34.5.01 (exploration): single-subject 100-session trained durability & joint lifecycle.

Design: plans/reference/DESIGN-N34-5-01_20261126.md (D-20261126-661).
Runs one persistent `LongRunHarness` + `SessionStore` subject per seed across 100 sessions,
>=10,000 events, 10 crash recoveries, >=20 pending handoffs, 2/2 sealed split blocks, and all 5
learned mechanisms persisted via `record.capsule` (`ch`, `phy`, `grow`, `ecl`, `unc`).
Exploration seeds 6801-6803; zero confirmation seed / check / final reads.

Usage: python scripts/training/n34_5_01_long_run_joint.py [--smoke]
Output: reports/n34_5_01_long_run_joint.json
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

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji.long_run_trained_subject import RunConfig, run_long_run_subject  # noqa: E402

CARD = "N34.5.01"
DESIGN = "DESIGN-N34-5-01_20261126.md"
TRAIN_SEEDS = (6801, 6802, 6803)
SPOT_CHECK_SEED = 6801
CAP_CPU = 160.0
OUT = ROOT / "reports" / "n34_5_01_long_run_joint.json"


def _eval_gates(
    per_seed: dict[str, Any], spot_match: bool, *, n_sessions: int, n_events: int
) -> dict[str, bool]:
    g1 = all(
        v["durability"]["sessions"] == n_sessions
        and v["durability"]["events"] >= n_events
        and v["durability"]["crash_recoveries"] == len(RunConfig.crash_sessions)
        and v["durability"]["unplanned_rescues"] == 0
        and v["durability"]["chain_valid"] is True
        and v["durability"]["resolved_pending"] >= 20
        and v["durability"]["sealed_blocks"] == 2
        for v in per_seed.values()
    )
    g2 = all(
        v["chinese"]["train"]["mean_step_acc"] >= 0.85
        and v["chinese"]["train"]["coref_acc"] >= 0.80
        and v["chinese"]["heldout"]["mean_step_acc"] >= 0.80
        and v["chinese"]["heldout"]["coref_acc"] >= 0.75
        for v in per_seed.values()
    )
    g3 = all(
        v["physics"]["in_family"]["accuracy"] >= 0.90
        and v["physics"]["in_family"]["collision_recall"] >= 0.85
        and v["physics"]["cross_family"]["accuracy"] >= 0.80
        and v["physics"]["cross_family"]["collision_recall"] >= 0.70
        and v["physics"]["barrier_ablation"]["collision_recall_drop"] >= 0.35
        for v in per_seed.values()
    )
    g4 = all(
        v["growth"]["growth_events"] == 1
        and v["growth"]["grew_session"] >= 20
        and v["growth"]["birth_perturbation"] <= 1e-6
        and v["growth"]["dormancy_drop"] >= 0.50
        and v["growth"]["wake_diff"] <= 1e-6
        and v["growth"]["p1_after_acc"] >= 0.95
        and v["growth"]["p1_retention_drop"] <= 0.02
        and v["growth"]["p2_acc"] >= 0.90
        for v in per_seed.values()
    )
    g5 = all(
        v["credibility"]["delta_brier"] <= -0.02 and v["credibility"]["delta_post_abs"] <= -0.05
        for v in per_seed.values()
    )
    g6 = all(
        v["uncertainty"]["coupled_ece"] <= 0.10 and v["uncertainty"]["brier_vs_static"] <= -0.05
        for v in per_seed.values()
    )
    return {
        "g1_durability_and_recovery": g1,
        "g2_chinese_endurance": g2,
        "g3_physics_causal_endurance": g3,
        "g4_plastic_growth_retention": g4,
        "g5_credibility_drift_tracking": g5,
        "g6_coupled_uncertainty_calibration": g6,
        "g7_reproducibility": bool(spot_match),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    preflight(CARD, list(TRAIN_SEEDS), record=False)
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU

    cfg = (
        RunConfig(
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
        if args.smoke
        else RunConfig()
    )

    per_seed: dict[str, Any] = {}
    spot_match = False
    checks: dict[str, bool] = {}

    try:
        for seed in TRAIN_SEEDS:
            t0 = time.process_time()
            with tempfile.TemporaryDirectory() as tmp1:
                res = run_long_run_subject(tmp1, seed, deadline, cfg=cfg)
            if seed == SPOT_CHECK_SEED and not args.smoke:
                with tempfile.TemporaryDirectory() as tmp2:
                    again = run_long_run_subject(tmp2, seed, deadline, cfg=cfg)
                spot_match = (
                    res["durability"]["chain_head"] == again["durability"]["chain_head"]
                    and res["durability"]["record_digest"] == again["durability"]["record_digest"]
                )
            elif seed == SPOT_CHECK_SEED and args.smoke:
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
            exit_ = "long_run_joint_positive"
        else:
            exit_ = "long_run_joint_partial"
    except (TimeoutError, FloatingPointError, RuntimeError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = str(exc)

    payload = {
        "schema": "taiji-n34-5-01-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "train_seeds": list(TRAIN_SEEDS),
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
    summary = {
        "exit": exit_,
        "checks": checks,
        "cpu": payload["cost"]["cpu"],
        "seeds": {
            k: {
                "sess": v["durability"]["sessions"],
                "events": v["durability"]["events"],
                "crashes": v["durability"]["crash_recoveries"],
                "ch_tr": v["chinese"]["train"]["coref_acc"],
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
    return 0 if exit_ in ("long_run_joint_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
