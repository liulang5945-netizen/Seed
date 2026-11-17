"""22.5.0 limited-exit closeout: bind the parity evidence and the preserved state.

This is a zero-training, read-only closeout. It does not change any gate, dose,
price, check exposure, or final seal. It records that the A-parent gate was not
opened because both permitted parity quantities failed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIME = ROOT / "reports/taiji_u2_22_5_0_compute_parity_20261117.json"
FLOPS = ROOT / "reports/taiji_u2_22_5_0_flops_parity_20261117.json"
PRICE = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
PARENT = ROOT / "plans/manifests/taiji_u2_22_4_accepted_parent_20261117.json"
PARTITION = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"
OUT = ROOT / "plans/manifests/taiji_u2_22_5_0_closeout_20261117.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def main() -> int:
    time_report = json.loads(TIME.read_text(encoding="utf-8"))
    flops = json.loads(FLOPS.read_text(encoding="utf-8"))
    price = json.loads(PRICE.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    closeout = {
        "schema": "taiji-u2-22-5-0-closeout-v1",
        "card": "22.5.0",
        "status": "closed_limited_exit",
        "verdict": "compute_parity_unavailable",
        "scope": "A-parent gate only; no architecture comparison, no training, no capability",
        "time_parity": {
            "quantity": time_report["parity_quantity"],
            "ratios": time_report["ratios"],
            "required_baseline_updates_at_min_dose": time_report[
                "required_baseline_updates_at_min_dose"
            ],
            "u2_updates_if_baseline_at_cap": time_report["u2_updates_if_baseline_at_cap"],
            "max_updates": time_report["max_updates_per_seed_per_architecture"],
            "effective_min_updates": time_report["effective_min_updates"],
            "report": str(TIME.relative_to(ROOT)),
            "sha256": sha(TIME),
        },
        "flops_parity": {
            "quantity": flops["parity_quantity"],
            "counts": flops["counts"],
            "ratios": flops["ratios"],
            "cap_ratio": flops["cap_ratio"],
            "common_flops_budget_exists": flops["common_flops_budget_exists"],
            "report": str(FLOPS.relative_to(ROOT)),
            "sha256": sha(FLOPS),
        },
        "first_a_cross_check": time_report["cross_checks"],
        "preserved": {
            "engineering_parent": {
                "ref": str(PARENT.relative_to(ROOT)),
                "sha256": sha(PARENT),
                "status": parent["status"],
                "authorized": parent["authorized"],
                "capability_qualifications": parent["capability_qualifications"],
            },
            "price": {
                "ref": str(PRICE.relative_to(ROOT)),
                "sha256": sha(PRICE),
                "digest": price["price_table_digest"],
            },
            "partition": {
                "recipe_digest": partition["manifest"]["data_recipe_digest"],
                "partition_digest": partition["manifest"]["partition_digest"],
            },
            "check_exposure": parent["data"]["check_exposure"],
            "final_seal": parent["data"]["final_seal"],
        },
        "not_signed": [
            "A parent",
            "B learning",
            "check2",
            "22.6",
            "22.7",
            "architecture superiority or inferiority",
            "capability",
            "training authorization",
        ],
        "handoff": {
            "current_window": "closed as limited exit",
            "higher_ceiling_path": {
                "id": "C",
                "name": "comparable-cost baseline or U2 variant",
                "requires": "new identity version, re-freeze, owner authorization",
                "not_automatic": True,
            },
            "forbidden": "do not switch metrics again, enlarge the cheaper arm, or start training to force a pass",
        },
        "cost": {
            "time_check_cpu": time_report["cost"]["cpu"],
            "flops_check_cpu": flops["cost"]["cpu"],
            "total_cpu": round(time_report["cost"]["cpu"] + flops["cost"]["cpu"], 6),
        },
        "discipline": {
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "source_pins": {
            "scripts/training/check_taiji_u2_22_5_0_parity.py": sha(
                ROOT / "scripts/training/check_taiji_u2_22_5_0_parity.py"
            ),
            "scripts/training/check_taiji_u2_22_5_0_flops.py": sha(
                ROOT / "scripts/training/check_taiji_u2_22_5_0_flops.py"
            ),
            "scripts/training/closeout_taiji_u2_22_5_0.py": sha(
                ROOT / "scripts/training/closeout_taiji_u2_22_5_0.py"
            ),
        },
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_bytes(json.dumps(closeout, ensure_ascii=False, indent=2).encode() + b"\n")
    print("closeout", sha(OUT), "status", closeout["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
