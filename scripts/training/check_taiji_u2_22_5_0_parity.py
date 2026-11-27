"""22.5.0 A-parent gate: zero-training compute-parity feasibility check.

The parity quantity is frozen to wall time before the reading. It uses the
22.4.4 price table, the 22.5.2 cap of 256 updates per seed per architecture, and
the lowest matched dose observed by the first-A manifest (24). No model forward,
optimizer step, check or final read is performed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = ROOT / "reports/taiji_u2_22_5_0_compute_parity_20261117.json"
PRICE = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
FIRST_A = ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112.json"
MAX_UPDATES = 256
EFFECTIVE_MIN_UPDATES = 24
PARITY_QUANTITY = "wall"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    price = json.loads(PRICE.read_text(encoding="utf-8"))
    first_a = json.loads(FIRST_A.read_text(encoding="utf-8"))
    components = price["price_table"]["components"]
    u2 = components["u2_reference"]["single_learning"]
    arms = {
        "u2_reference": u2,
        "transformer_sequence": components["transformer_sequence"]["single_learning"],
        "gru_sequence": components["gru_sequence"]["single_learning"],
    }
    u2_wall = float(u2["wall"])
    ratios = {
        arm: (u2_wall / float(value["wall"]) if float(value["wall"]) > 0 else None)
        for arm, value in arms.items()
        if arm != "u2_reference"
    }
    required_at_min = {
        arm: EFFECTIVE_MIN_UPDATES * ratio for arm, ratio in ratios.items() if ratio is not None
    }
    u2_at_cap = {arm: MAX_UPDATES / ratio for arm, ratio in ratios.items() if ratio is not None}
    transformer_ratio = ratios["transformer_sequence"]
    gru_ratio = ratios["gru_sequence"]
    time_parity_possible = all(value <= MAX_UPDATES for value in required_at_min.values()) or all(
        value >= EFFECTIVE_MIN_UPDATES for value in u2_at_cap.values()
    )
    verdict = "compute_parity_available" if time_parity_possible else "compute_parity_unavailable"
    reasons = []
    if not time_parity_possible:
        reasons.append(
            "matching the U2 24-update time budget requires the baseline to exceed the "
            "256-update cap"
        )
        reasons.append(
            "fitting the baseline at 256 updates leaves U2 below the observed matched dose 24"
        )
        reasons.append(
            "the first-A manifest already records compute_parity_unavailable with 1.373 vs "
            "0.1816 CPU per step and 24 vs 188 updates"
        )
    report = {
        "schema": "taiji-u2-22-5-0-compute-parity-v1",
        "card": "22.5.0",
        "phase": "zero_training_compute_parity_check",
        "verdict": verdict,
        "parity_quantity": PARITY_QUANTITY,
        "parity_quantity_frozen_before_reading": True,
        "max_updates_per_seed_per_architecture": MAX_UPDATES,
        "effective_min_updates": EFFECTIVE_MIN_UPDATES,
        "prices": arms,
        "ratios": {arm: round(value, 6) for arm, value in ratios.items()},
        "required_baseline_updates_at_min_dose": {
            arm: round(value, 4) for arm, value in required_at_min.items()
        },
        "u2_updates_if_baseline_at_cap": {arm: round(value, 4) for arm, value in u2_at_cap.items()},
        "reasons": reasons,
        "cross_checks": {
            "price_report": str(PRICE.relative_to(ROOT)),
            "price_digest": price["price_table_digest"],
            "first_a_manifest": str(FIRST_A.relative_to(ROOT)),
            "first_a_status": first_a["status"],
            "first_a_prices": {
                "native": first_a["prices"]["native"]["median_step_cpu"],
                "transformer": first_a["prices"]["transformer"]["median_step_cpu"],
                "gru": first_a["prices"]["gru"]["median_step_cpu"],
            },
            "first_a_updates": first_a["updates"],
        },
        "consequences": {
            "a_parent_gate_opened": False,
            "training_authorized": False,
            "capability_claimed": False,
            "22_5_1_unlocked": False,
            "22_6_7_unlocked": False,
            "engineering_parent_authorized": False,
        },
        "options_for_owner": [
            {
                "id": "A",
                "name": "time parity with a new dose cap",
                "effect": "would require re-freezing 22.5.2 and may land in budget_cannot_express",
            },
            {
                "id": "B",
                "name": "FLOPs parity",
                "effect": "zero-training analytic forward+backward count; allowed by plan :378; "
                "recommended next check",
            },
            {
                "id": "C",
                "name": "comparable-cost baseline or U2 variant",
                "effect": "requires a new identity version and re-freeze",
            },
            {
                "id": "D",
                "name": "close 22.5.0 as a limited exit",
                "effect": "no architecture claim, no 22.5.1/22.6/22.7 unlock",
            },
        ],
        "recommended_option": "B",
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 6),
            "wall": round(time.perf_counter() - started_wall, 6),
        },
        "discipline": {
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "parity verdict="
        + verdict
        + " transformer_ratio="
        + str(round(transformer_ratio, 4))
        + " gru_ratio="
        + str(round(gru_ratio, 4))
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
