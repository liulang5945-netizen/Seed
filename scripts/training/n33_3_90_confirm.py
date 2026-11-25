"""N33.3.90 (confirmation): one-shot confirmation of continual plasticity & growth on 20261611-13.

Pre-registered: plans/reference/DESIGN-N33-3-90_20261125.md (D-20261125-592).
Runs the frozen N33.3.03 recipe (`run_arm_v2`) across unread confirmation seeds
(20261611, 20261612, 20261613) with a 2-pass bit-exact spot-check on 20261611.
Zero check/final reads.

Usage:
  python scripts/training/n33_3_90_confirm.py --smoke
  python scripts/training/n33_3_90_confirm.py
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
from scripts.training.n33_3_03_plastic_growth_v2 import (  # noqa: E402
    CHECK_EVERY,
    P1_STEPS,
    P2_STEPS,
    SMOKE_STEPS,
    run_arm_v2,
)
from taiji.continual_growth_task import TASK_VERSION, make_split  # noqa: E402
from taiji.plastic_growth import ARCH_VERSION  # noqa: E402

CARD = "N33.3.90"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
CONFIRM_SEEDS = (20261611, 20261612, 20261613)
SPOT_CHECK_SEED = 20261611
CAP_CPU = 30.0
OUT = ROOT / "reports" / "n33_3_90_confirm.json"
SCOPE = (
    "taiji-plastic-growth-v1 + taiji-continual-growth-v1 + taiji-session-store-v1 two-phase "
    "synthetic task loop only; no large-scale/open-task/real-multimodal/check/final claims"
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
            small_res, _ = run_arm_v2(
                seed, "fixed_small", data, p1_steps=p1_steps, p2_steps=p2_steps, deadline=deadline
            )
            grow_res, grow_net = run_arm_v2(
                seed,
                "growth_plastic",
                data,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            pre_res, _ = run_arm_v2(
                seed,
                "prealloc_equal",
                data,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            if seed == SPOT_CHECK_SEED:
                _, grow_net2 = run_arm_v2(
                    seed,
                    "growth_plastic",
                    data,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    deadline=deadline,
                )
                repro_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(grow_net.parameters(), grow_net2.parameters(), strict=True)
                ) and torch.equal(grow_net.stage, grow_net2.stage)

            vs_small = round(grow_res["p2_acc"] - small_res["p2_acc"], 6)
            vs_prealloc = round(grow_res["combined_acc"] - pre_res["combined_acc"], 6)
            g_eng = bool(
                grow_res["store_roundtrip_exact"]
                and small_res["store_roundtrip_exact"]
                and pre_res["store_roundtrip_exact"]
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
                "prealloc_equal": pre_res,
                "p2_gain_over_small": vs_small,
                "combined_diff_vs_prealloc": vs_prealloc,
                "g_eng": g_eng,
                "g_pressure": g_press,
                "cpu": round(time.process_time() - s_t0, 2),
            }

        all_eng = repro_exact and all(v["g_eng"] for v in per_seed.values())
        all_press = all(v["g_pressure"] for v in per_seed.values())
        diffs = [v["combined_diff_vs_prealloc"] for v in per_seed.values()]
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
                "prealloc_comb": v["prealloc_equal"]["combined_acc"],
                "vs_prealloc": v["combined_diff_vs_prealloc"],
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
