"""N33.3.91 (confirmation, Revision 1): formal confirmation of N33.3.04 on 20261701-03.

Pre-registered: plans/reference/DESIGN-N33-3-91_20261125.md (D-20261125-597).
Exercises the single allowed Section 5.1 Design Revision confirmation on unread seeds
(20261701, 20261702, 20261703), leaving backup 20261711-20261713 untouched. Zero check/final.

Usage:
  python scripts/training/n33_3_91_confirm.py --smoke
  python scripts/training/n33_3_91_confirm.py
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
from scripts.training.n33_3_02_plastic_growth_train import (  # noqa: E402
    P1_STEPS,
    P2_STEPS,
    SMOKE_STEPS,
)
from scripts.training.n33_3_04_plastic_growth_rev1 import (  # noqa: E402
    CHECK_EVERY,
    LR_BASE,
    LR_IMMATURE,
    PREALLOC_LRS,
    run_arm_rev1,
)
from taiji.continual_growth_task import TASK_VERSION, make_split  # noqa: E402
from taiji.plastic_growth import ARCH_VERSION  # noqa: E402

CARD = "N33.3.91"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
CONFIRM_SEEDS = (20261701, 20261702, 20261703)
SPOT_CHECK_SEED = 20261701
CAP_CPU = 30.0
OUT = ROOT / "reports" / "n33_3_91_confirm.json"
SCOPE = (
    "taiji-plastic-growth-v1 + taiji-continual-growth-v1 + taiji-session-store-v1 two-phase "
    "synthetic task loop only; Section 5.1 Revision 1; no large-scale/open-task/real-multimodal/check/final claims"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(CONFIRM_SEEDS), record=not args.smoke).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    p1_steps = SMOKE_STEPS if args.smoke else P1_STEPS
    p2_steps = SMOKE_STEPS if args.smoke else P2_STEPS

    per_seed: dict[str, Any] = {}
    repro_exact = False
    try:
        for seed in CONFIRM_SEEDS:
            s_t0 = time.process_time()
            data = make_split(seed)
            small_res, _ = run_arm_rev1(
                seed,
                "fixed_small",
                data,
                lr=LR_BASE,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            grow_res, grow_net = run_arm_rev1(
                seed,
                "growth_plastic",
                data,
                lr=LR_BASE,
                lr_post_growth=LR_IMMATURE,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            pre_runs = {
                str(pre_lr): run_arm_rev1(
                    seed,
                    "prealloc_equal",
                    data,
                    lr=pre_lr,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    deadline=deadline,
                )[0]
                for pre_lr in PREALLOC_LRS
            }
            pre_best = max(pre_runs.values(), key=lambda r: r["combined_acc"])

            if seed == SPOT_CHECK_SEED:
                _, grow_net2 = run_arm_rev1(
                    seed,
                    "growth_plastic",
                    data,
                    lr=LR_BASE,
                    lr_post_growth=LR_IMMATURE,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    deadline=deadline,
                )
                repro_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(grow_net.parameters(), grow_net2.parameters(), strict=True)
                ) and torch.equal(grow_net.stage, grow_net2.stage)

            vs_small = round(grow_res["p2_acc"] - small_res["p2_acc"], 6)
            vs_prealloc = round(grow_res["combined_acc"] - pre_best["combined_acc"], 6)
            g_eng = bool(
                grow_res["store_roundtrip_exact"]
                and small_res["store_roundtrip_exact"]
                and all(r["store_roundtrip_exact"] for r in pre_runs.values())
                and grow_res["birth_perturbation"] <= 1e-6
                and grow_res["wake_diff"] == 0.0
                and grow_res["p1_growth_events"] == 0
                and (args.smoke or grow_res["p2_growth_events"] == 1)
            )
            g_press = bool(
                grow_res["p2_acc"] >= 0.85
                and vs_small >= 0.12
                and grow_res["p2_dormancy_drop"] >= 0.15
                and grow_res["p1_retention_drop"] <= 0.02
            )
            per_seed[str(seed)] = {
                "fixed_small": small_res,
                "growth_plastic": grow_res,
                "prealloc_runs": pre_runs,
                "prealloc_best": pre_best,
                "p2_gain_over_small": vs_small,
                "combined_diff_vs_prealloc_best": vs_prealloc,
                "g_eng": g_eng,
                "g_pressure": g_press,
                "cpu": round(time.process_time() - s_t0, 2),
            }

        all_eng = repro_exact and all(v["g_eng"] for v in per_seed.values())
        all_press = all(v["g_pressure"] for v in per_seed.values())
        diffs = [v["combined_diff_vs_prealloc_best"] for v in per_seed.values()]
        if not all_eng:
            exit_ = "engineering_failure"
        elif args.smoke:
            exit_ = "smoke"
        elif not all_press:
            exit_ = "confirm_pressure_not_met"
        elif all(d >= 0.05 for d in diffs):
            exit_ = "confirm_adopt"
        elif all(d >= -0.02 for d in diffs):
            exit_ = "confirm_required_inplace"
        else:
            exit_ = "confirm_required_revise"
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
        "task_version": TASK_VERSION,
        "check_every": CHECK_EVERY,
        "lrs": {"base": LR_BASE, "immature": LR_IMMATURE, "prealloc_grid": list(PREALLOC_LRS)},
        "steps": {"p1": p1_steps, "p2": p2_steps},
        "confirm_seeds": list(CONFIRM_SEEDS),
        "repro_spot_check": {"seed": SPOT_CHECK_SEED, "exact": bool(repro_exact)},
        "per_seed": per_seed,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0 if args.smoke else len(CONFIRM_SEEDS),
        },
        "scope": SCOPE,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "exit": exit_,
        "repro_spot_check": payload["repro_spot_check"],
        "cost": payload["cost"],
    }
    if all(isinstance(v, dict) and "growth_plastic" in v for v in per_seed.values()):
        summary["per_seed"] = {
            s: {
                "small_p2": v["fixed_small"]["p2_acc"],
                "grew_at": v["growth_plastic"]["grew_at_step"],
                "grow_p1_drop": v["growth_plastic"]["p1_retention_drop"],
                "grow_p2": v["growth_plastic"]["p2_acc"],
                "grow_comb": v["growth_plastic"]["combined_acc"],
                "prealloc_best_comb": v["prealloc_best"]["combined_acc"],
                "prealloc_best_lr": v["prealloc_best"]["lr"],
                "vs_prealloc_best": v["combined_diff_vs_prealloc_best"],
                "dormancy_drop": v["growth_plastic"]["p2_dormancy_drop"],
                "g_eng": v["g_eng"],
                "g_pressure": v["g_pressure"],
            }
            for s, v in per_seed.items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
