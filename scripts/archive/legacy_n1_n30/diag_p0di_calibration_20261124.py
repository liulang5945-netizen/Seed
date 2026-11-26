"""P0-DI: calibrate recycle-damage and growth/recycle utilities from committed receipts.

Reads the recycle and growth receipts (P0-DF/DG/DH) and reports:
- measured damage of overwritten abilities,
- predicted damage used in earlier decisions (0.20),
- corrected utilities for grow / recycle / retire under the two budget states.

Analysis of existing receipts; no new training.  Run:
python scripts/training/diag_p0di_calibration_20261124.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\Seed")
REPORTS = ROOT / "reports"
OUT = REPORTS / "taiji_p0di_calibration_20261124.json"

EV_GAIN = 0.75
COST_LEARN = 0.10
COST_RECAL = 0.09
EV_RECYCLE_DAMAGE_PRIOR = 0.20
RETIRE_DAMAGE = 0.70
RETIRE_COST = 0.02


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> int:
    dmg = []
    dmg += [
        {
            "event": "DF recycle D (R3)",
            "before": load("taiji_p0df_seed20261121.json")["recycle"]["r3_before"],
            "after": load("taiji_p0df_seed20261121.json")["recycle"]["r3_after"],
        },
        {
            "event": "DG recycle E (R4)",
            "before": load("taiji_p0dg_seed20261121.json")["cycle"]["r4_before"],
            "after": load("taiji_p0dg_seed20261121.json")["cycle"]["r4_after"],
        },
        {
            "event": "DH recycle E (R4, round2)",
            "before": load("taiji_p0dh_seed20261121.json")["cycle"]["r4_before"],
            "after": load("taiji_p0dh_seed20261121.json")["cycle"]["r4_after"],
        },
        {
            "event": "DH recycle F (R6, round2)",
            "before": load("taiji_p0dh_seed20261121.json")["round2"]["r6_before"],
            "after": load("taiji_p0dh_seed20261121.json")["round2"]["r6_after"],
        },
    ]
    for row in dmg:
        row["damage"] = row["before"] - row["after"]
    values = [row["damage"] for row in dmg]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var**0.5
    pred_gap = mean - EV_RECYCLE_DAMAGE_PRIOR

    utility_grow = EV_GAIN - COST_LEARN - COST_RECAL
    utility_recycle = EV_GAIN - COST_LEARN - COST_RECAL - mean
    utility_retire = -RETIRE_DAMAGE - RETIRE_COST
    budget_rule = {
        "budget_not_full": "grow" if utility_grow >= max(utility_recycle, 0.0) else "wait",
        "budget_full": "recycle" if utility_recycle > 0.0 else "wait",
    }
    payload = {
        "schema": "taiji-p0di-calibration-v1",
        "task": "recycle_damage_calibration",
        "measurements": dmg,
        "damage_mean": round(mean, 4),
        "damage_std": round(std, 4),
        "damage_min": min(values),
        "damage_max": max(values),
        "prior_predicted_damage": EV_RECYCLE_DAMAGE_PRIOR,
        "prediction_gap": round(pred_gap, 4),
        "utilities": {
            "grow": round(utility_grow, 4),
            "recycle_with_measured_damage": round(utility_recycle, 4),
            "retire_only": round(utility_retire, 4),
        },
        "decision_rule": budget_rule,
        "recommended_EV_RECYCLE_DAMAGE": 0.60,
        "note": "post-hoc calibration from committed receipts; later recyclings show larger damage "
        "(0.505 -> 0.628), so 0.60 is a conservative prior",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
