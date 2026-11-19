"""22.7b minimal growth-benefit contract skeleton (joint 24-class task) + smoke.

New card on the capacity-limited multi-attribute family (D-20261119-40:
16/32 units in the middle band, 32->64 headroom +0.1875).  Three minimal arms are
pre-registered for the first growth-benefit test:

* ``G0_growth``: warm-start 32 units, neutral birth at the midpoint -> 33 units,
  continue to the same total step budget;
* ``C1_no_growth``: same total steps, no birth;
* ``C2_preallocated``: 33 units from the start, same total steps.

Parity rule: equal total optimizer steps with the FLOPs ratio (33/32) reported and
required <= 1.05; G1/G2/G3 (random/permuted/GradMax births) and G4/G5/G6 are
listed out-of-scope for this minimal first test and need a separate ruling.

``training_authority`` stays null: ``--run`` fails closed until the owner (or the
standing authorization) records it.  Zero optimizer steps in this card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import (  # noqa: E402
    joint_label,
)
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_7b_growth_joint24_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_7b_growth_smoke_20261120.json"
CAP_CPU = 200.0
BASE = {
    "groups": 4,
    "units_per_group": 8,
    "input_dim": 8,
    "hidden_dim": 8,
    "output_dim": 24,
    "lr": 3e-3,
    "steps": 1500,
    "birth_step": 750,
    "batch": 64,
    "warmup": 100,
    "seeds": [0, 1, 2],
    "train_worlds": 256,
    "fit_worlds": 32,
}
ARMS = {
    "G0_growth": "warm 32 units, neutral birth at step 750 -> 33 units, continue",
    "C1_no_growth": "32 units, same total steps, no birth",
    "C2_preallocated": "33 units from step 0, same total steps",
}
OUT_OF_SCOPE = {
    "G1": "random content-blind gate birth",
    "G2": "permuted gate birth",
    "G3": "GradMax direction + fixed schedule",
    "G4": "replay only",
    "G5": "fixed-capacity strong optimisation",
    "G6": "preallocated final capacity snapshot",
}
EXITS = [
    "engineering_failure",
    "not_admitted",
    "budget_stop",
    "credit_insufficient",
    "retention_breach",
    "recovery_failure",
    "growth_gain_absent",
    "growth_gain_pass",
]
PINNED_FILES = (
    "scripts/training/run_s2_22_7b_growth_joint24_20261120.py",
    "scripts/training/probe_s2_22_7_joint24_20261120.py",
    "scripts/training/probe_s2_22_7_0_alpha_adapter_v3_20261120.py",
    "taiji/u2_neural_units.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.7b minimal growth benefit on joint 24-class task",
        "base": BASE,
        "task": {
            "name": "first-A joint colour x position (24 classes)",
            "adapter": "v3 raw-media palette histogram",
            "headroom_receipt": "reports/taiji_u2_22_7_joint24_20261120.json",
            "chance": round(1 / 24, 4),
        },
        "arms": ARMS,
        "out_of_scope": OUT_OF_SCOPE,
        "parity": {
            "rule": "equal total optimizer steps; FLOPs ratio (33/32) reported",
            "flops_ratio_max": 1.05,
        },
        "gates": {
            "growth_gain_min": 0.05,
            "paired_lower_bound_min": 0.0,
            "all_seeds_same_direction": True,
            "retention_drop_upper_bound_max": 0.02,
            "recovery_exact": True,
            "references_cleanup": True,
        },
        "exits_order": EXITS,
        "check_reads": 0,
        "final_reads": 0,
        "price_cpu": 200.0,
        "cap_cpu": CAP_CPU,
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-7b-growth-v1":
        raise ValueError("unsupported_growthb_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "smoke" and contract.get("status") not in {
        "designed_pending_owner",
        "frozen",
    }:
        raise ValueError("unexpected_contract_status")
    if mode == "run":
        raise ValueError("runner_not_implemented_training_authority_null")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-7b-growth-v1",
        "card": "DESIGN-TAIJI_U2_22_7B_GROWTH_JOINT24_20261120.md",
        "status": "designed_pending_owner",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 200.0,
            "optimizer_steps": BASE["steps"] * len(BASE["seeds"]) * len(ARMS),
            "arms": len(ARMS),
        },
        "cap_cpu": CAP_CPU,
        "check_exposure": {"reads": 0},
        "source_pins": {name: sha_of(ROOT / name) for name in PINNED_FILES},
        "result": {"run": False, "closed": False, "verdict": None},
    }
    payload["criteria_sha256"] = value_digest(payload["criteria"])
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"frozen": str(out), "criteria_sha256": payload["criteria_sha256"]}))
    return 0


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "smoke")
    examples = generate_examples("fit", 8)
    features = [adapter_v3_features(example) for example in examples]
    labels = [joint_label(example) for example in examples]
    labels_ok = all(0 <= label < 24 for label in labels)
    deterministic = all(
        torch.equal(left, right) for left, right in zip(features, features, strict=True)
    )
    joint24 = json.loads(
        (ROOT / "reports/taiji_u2_22_7_joint24_20261120.json").read_text(encoding="utf-8")
    )
    checks = {
        "adapter_deterministic": deterministic,
        "labels_in_range": labels_ok,
        "joint24_family_candidate": joint24["verdict"] == "joint24_family_candidate",
        "three_arms_defined": len(contract["criteria"]["arms"]) == 3,
        "out_of_scope_documented": len(contract["criteria"]["out_of_scope"]) == 6,
        "authority_null": contract["training_authority"] is None,
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-7b-growth-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "verdict": "smoke_passed" if passed else "smoke_failed",
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(SMOKE, payload)
    print(json.dumps({"verdict": payload["verdict"], "cpu": payload["cost"]["cpu"]}))
    return 0 if passed else 1


def closeout(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    smoke_receipt = json.loads(SMOKE.read_text(encoding="utf-8"))
    contract["result"] = {
        "smoke": smoke_receipt["verdict"],
        "run": False,
        "closed": False,
        "verdict": "designed_pending_owner",
    }
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"closed": "designed_pending_owner"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        return freeze(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.smoke:
        return smoke(contract)
    if args.closeout:
        return closeout(CONTRACT)
    return guards(contract, "run")


if __name__ == "__main__":
    raise SystemExit(main())
