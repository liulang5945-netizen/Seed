"""N33.4.01 (zero training): C0 calibration & store-lineage check for taiji-recursive-dev-v1.

Pre-registered: plans/reference/DESIGN-N33-4-01_20261126.md (D-20261126-599).
Runs zero-training calibration on EVAL_SEEDS (9000..9099) verifying scripted oracle >= 3.0
improving steps, net gain >= 0.16, >= 0.05 ahead of random & all 6 constants, fixed_parent=0.0,
and bit-exact SessionStore lineage persistence. Zero confirm/check/final reads.

Usage: python scripts/training/n33_4_01_recursive_env_check.py
Output: reports/n33_4_01_env_check.json
"""

from __future__ import annotations

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

from taiji.recursive_dev_env import (  # noqa: E402
    ACTIONS,
    ENV_VERSION,
    HORIZON,
    STATE_DIM,
    evaluate_recursive_with_store,
    random_policy,
    scripted_oracle_policy,
    seed_from,
)

CARD = "N33.4.01"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = f"DESIGN-{CARD_SLUG}_20261126.md"
EVAL_SEEDS = list(range(9000, 9100))
CAP_CPU = 10.0
OUT = ROOT / "reports" / "n33_4_01_env_check.json"


def calibrate_recursive_env(eval_seeds: list[int]) -> dict[str, Any]:
    """Run C0 zero-training calibration on `taiji-recursive-dev-v1`."""
    oracle = evaluate_recursive_with_store(scripted_oracle_policy, eval_seeds)
    gen = torch.Generator().manual_seed(seed_from(CARD, "calib-random"))
    rand_ev = evaluate_recursive_with_store(lambda o: random_policy(o, gen), eval_seeds)
    constants = {a: evaluate_recursive_with_store(lambda _o, a=a: a, eval_seeds) for a in ACTIONS}

    oracle_gain = oracle["mean_net_gain"]
    margins = [oracle_gain - rand_ev["mean_net_gain"]] + [
        oracle_gain - v["mean_net_gain"] for v in constants.values()
    ]
    c0_pass = (
        oracle["store_roundtrip_exact"]
        and rand_ev["store_roundtrip_exact"]
        and oracle_gain >= 0.16
        and oracle["mean_improving_steps"] >= 3.0
        and constants["noop_hold"]["mean_net_gain"] == 0.0
        and min(margins) >= 0.05
    )
    return {
        "c0_pass": bool(c0_pass),
        "oracle": oracle,
        "random_true": rand_ev,
        "constants_net_gains": {a: v["mean_net_gain"] for a, v in constants.items()},
        "constants_improving_steps": {a: v["mean_improving_steps"] for a, v in constants.items()},
        "min_margin_over_baselines": round(min(margins), 6),
    }


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    calib = calibrate_recursive_env(EVAL_SEEDS)
    exit_ = "env_ready" if calib["c0_pass"] else "env_failure"
    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": DESIGN,
        "env_version": ENV_VERSION,
        "state_dim": STATE_DIM,
        "horizon": HORIZON,
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "created_utc": datetime.now(UTC).isoformat(),
        "calibration": calib,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "oracle_gain": calib["oracle"]["mean_net_gain"],
                "oracle_imp_steps": calib["oracle"]["mean_improving_steps"],
                "oracle_ge3_frac": calib["oracle"]["frac_ge3_improving"],
                "random_gain": calib["random_true"]["mean_net_gain"],
                "min_margin": calib["min_margin_over_baselines"],
                "constants_net_gains": calib["constants_net_gains"],
                "cost": payload["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "env_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
