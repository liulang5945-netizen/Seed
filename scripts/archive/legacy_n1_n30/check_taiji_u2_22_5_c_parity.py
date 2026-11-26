"""Plan C: FLOPs parity is available, but the frozen 256-update cap blocks a meaningful dose.

This is a read-only arithmetic check over the C identity manifest, the U2 FLOPs
report and the first-A dose envelope. It does not train or alter any gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

C_IDENTITY = ROOT / "plans/manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json"
FLOPS = ROOT / "reports/taiji_u2_22_5_0_flops_parity_20261117.json"
PRICE = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
FIRST_A = ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112.json"
OUT = ROOT / "plans/manifests/taiji_u2_22_5_c_parity_20261117.json"
MAX_UPDATES = 256
EFFECTIVE_MIN_UPDATES = 24


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def main() -> int:
    identity = json.loads(C_IDENTITY.read_text(encoding="utf-8"))
    flops = json.loads(FLOPS.read_text(encoding="utf-8"))
    price = json.loads(PRICE.read_text(encoding="utf-8"))
    first_a = json.loads(FIRST_A.read_text(encoding="utf-8"))
    u2_flops = flops["counts"]["u2_reference"]
    c_flops = identity["parity"]["flops"]
    ratio = u2_flops / c_flops
    cap = MAX_UPDATES / EFFECTIVE_MIN_UPDATES
    baseline_updates_at_u2_256 = math.floor(MAX_UPDATES * u2_flops / c_flops)
    u2_updates_at_baseline_256 = math.floor(MAX_UPDATES * c_flops / u2_flops)
    common_flops = MAX_UPDATES * u2_flops
    payload = {
        "schema": "taiji-u2-22-5-c-parity-v1",
        "card": "22.5.0.C",
        "identity": {
            "architecture_id": identity["architecture_id"],
            "applications": identity["architecture"]["applications"],
            "weight_tied": identity["architecture"]["weight_tied"],
            "param_total": identity["param_total"],
            "param_tensors": identity["param_tensors"],
            "param_delta_pct": identity["parity"]["param_delta_pct"],
            "flops": c_flops,
            "wall": identity["parity"]["wall"],
            "manifest": str(C_IDENTITY.relative_to(ROOT)),
            "sha256": sha(C_IDENTITY),
        },
        "u2": {
            "flops": u2_flops,
            "single_learning_cpu": price["price_table"]["components"]["u2_reference"][
                "single_learning"
            ]["cpu"],
            "single_learning_wall": price["price_table"]["components"]["u2_reference"][
                "single_learning"
            ]["wall"],
        },
        "parity": {
            "quantity": "flops",
            "ratio_u2_over_c": round(ratio, 6),
            "cap": round(cap, 6),
            "flops_parity_available": ratio <= cap,
            "time_ratio_u2_over_c": identity["parity"]["wall_ratio_to_u2"],
            "time_parity_claimed": False,
        },
        "dose_envelope": {
            "max_updates": MAX_UPDATES,
            "effective_min_updates": EFFECTIVE_MIN_UPDATES,
            "baseline_updates_at_u2_256": baseline_updates_at_u2_256,
            "u2_updates_at_baseline_256": u2_updates_at_baseline_256,
            "common_flops_per_seed_at_u2_256": common_flops,
            "current_cap_can_hold_common_flops": False,
            "reason": "at the 256 cap the FLOPs-matched U2 dose is only 24 updates, "
            "which is below the effective course minimum; at U2 256 the baseline needs "
            "2671 updates, above the 256 cap",
        },
        "proposed_refreeze": {
            "u2_updates": MAX_UPDATES,
            "baseline_updates": baseline_updates_at_u2_256,
            "common_flops_per_seed": common_flops,
            "changes": "only the baseline update cap, from 256 to the FLOPs-matched count",
            "does_not_change": [
                "U2 256-update cap",
                "check exposure",
                "final seal",
                "feedback rights",
                "data partition",
            ],
            "estimated_a_gate_cpu": round(
                3
                * MAX_UPDATES
                * price["price_table"]["components"]["u2_reference"]["single_learning"]["cpu"]
                * 1.20,
                4,
            ),
            "requires_owner_authorization": True,
        },
        "status": "flops_parity_available_dose_cap_blocks_a_gate",
        "a_parent_gate_ready": False,
        "training_authorized": False,
        "first_a_cross_check": {
            "status": first_a["status"],
            "updates": first_a["updates"],
            "prices": {
                "native": first_a["prices"]["native"]["median_step_cpu"],
                "transformer": first_a["prices"]["transformer"]["median_step_cpu"],
                "gru": first_a["prices"]["gru"]["median_step_cpu"],
            },
        },
        "source_pins": {
            "plans/manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json": sha(C_IDENTITY),
            "reports/taiji_u2_22_5_0_flops_parity_20261117.json": sha(FLOPS),
            "reports/taiji_u2_22_4_4_price_20261117.json": sha(PRICE),
        },
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "C parity ratio="
        + str(round(ratio, 4))
        + " cap="
        + str(round(cap, 4))
        + " status="
        + payload["status"]
        + " baseline_at_u2_256="
        + str(baseline_updates_at_u2_256)
        + " u2_at_baseline_256="
        + str(u2_updates_at_baseline_256)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
