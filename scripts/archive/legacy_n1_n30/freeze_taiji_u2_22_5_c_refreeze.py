"""Freeze the Plan C 22.5.0 re-freeze: FLOPs budget and baseline update cap.

Only the baseline update cap changes (256 -> the FLOPs-matched count); the U2 cap,
check exposure, final seal, feedback rights and data partition stay unchanged.
The contract is frozen but not training-authorized.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

C_PARITY = ROOT / "plans/manifests/taiji_u2_22_5_c_parity_20261117.json"
C_IDENTITY = ROOT / "plans/manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json"
PARENT_22_3 = ROOT / "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json"
PARENT_22_4 = ROOT / "plans/manifests/taiji_u2_22_4_accepted_parent_20261117.json"
PARTITION = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"
PRICE = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
OUT = ROOT / "plans/manifests/taiji_u2_22_5_0_refreeze_c_20261117.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def main() -> int:
    parity = json.loads(C_PARITY.read_text(encoding="utf-8"))
    identity = json.loads(C_IDENTITY.read_text(encoding="utf-8"))
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    u2 = parity["u2"]
    common_flops = parity["proposed_refreeze"]["common_flops_per_seed"]
    baseline_updates = parity["proposed_refreeze"]["baseline_updates"]
    contract = {
        "schema": "taiji-u2-22-5-0-refreeze-c-v1",
        "card": "22.5.0.C",
        "status": "frozen_pending_training_authorization",
        "parity_quantity": "flops",
        "u2_updates_per_seed": parity["proposed_refreeze"]["u2_updates"],
        "baseline_updates_per_seed": baseline_updates,
        "common_flops_per_seed": common_flops,
        "u2_flops_per_update": u2["flops"],
        "baseline_flops_per_update": identity["parity"]["flops"],
        "flops_ratio_u2_over_baseline": parity["parity"]["ratio_u2_over_c"],
        "flop_ratio_cap": parity["parity"]["cap"],
        "time_ratio_u2_over_baseline": parity["parity"]["time_ratio_u2_over_c"],
        "time_parity_claimed": False,
        "baseline_identity": {
            "ref": str(C_IDENTITY.relative_to(ROOT)),
            "sha256": sha(C_IDENTITY),
            "architecture_id": identity["architecture_id"],
            "applications": identity["architecture"]["applications"],
            "param_total": identity["param_total"],
            "param_tensors": identity["param_tensors"],
            "param_delta_pct": identity["parity"]["param_delta_pct"],
        },
        "parents": {
            "engineering_22_3": {
                "ref": str(PARENT_22_3.relative_to(ROOT)),
                "sha256": sha(PARENT_22_3),
            },
            "engineering_22_4": {
                "ref": str(PARENT_22_4.relative_to(ROOT)),
                "sha256": sha(PARENT_22_4),
                "authorized": json.loads(PARENT_22_4.read_text(encoding="utf-8"))["authorized"],
            },
        },
        "data": {
            "recipe_digest": partition["manifest"]["data_recipe_digest"],
            "partition_digest": partition["manifest"]["partition_digest"],
        },
        "changes_only": ["baseline update cap for the FLOPs-matched arm: 256 -> 2671 per seed"],
        "does_not_change": [
            "U2 256-update cap",
            "check exposure max 8 and allocations",
            "final seal",
            "feedback rights",
            "data partition",
            "price table for U2",
        ],
        "check_exposure": {
            "allocated_22_5": 2,
            "used": 0,
            "a_gate_would_use": 1,
            "remaining_after_a_gate": 1,
            "note": "A gate uses check1 only after explicit training authorization",
        },
        "estimated_a_gate_cpu": round(
            3 * parity["proposed_refreeze"]["u2_updates"] * u2["single_learning_cpu"] * 1.20,
            4,
        ),
        "u2_single_learning_cpu": u2["single_learning_cpu"],
        "u2_single_learning_wall": u2["single_learning_wall"],
        "baseline_single_learning_wall": identity["parity"]["wall"],
        "training_authorized": False,
        "requires_owner_authorization": True,
        "forbidden": [
            "do not change the U2 cap",
            "do not enlarge the cheaper arm beyond the FLOPs-matched count",
            "do not read check or final before authorization",
            "do not claim performance or capability from this contract",
        ],
        "source_pins": {
            "plans/manifests/taiji_u2_22_5_c_parity_20261117.json": sha(C_PARITY),
            "plans/manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json": sha(C_IDENTITY),
            "reports/taiji_u2_22_4_4_price_20261117.json": sha(PRICE),
        },
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "refreeze",
        sha(OUT),
        "u2",
        contract["u2_updates_per_seed"],
        "baseline",
        contract["baseline_updates_per_seed"],
        "flops",
        contract["common_flops_per_seed"],
        "est_cpu",
        contract["estimated_a_gate_cpu"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
