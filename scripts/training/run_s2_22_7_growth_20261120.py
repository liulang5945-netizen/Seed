"""22.7 seven-arm growth contract skeleton + zero-training smoke.

Freezes the pre-registered 22.7 protocol (G0-G6 arms, task adapter v3, budget
parity, retention/recovery/cleanup gates, exits) and verifies the wiring without
any optimizer step.  ``training_authority`` stays null: the formal run waits for
an owner scope ruling; this card only proves the package is ready to execute.

Modes: ``--freeze`` / ``--smoke`` / ``--closeout``.  ``--run`` intentionally
refuses until the contract carries an authority and an implemented arm runner.
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
    label_index,
)
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_7_growth_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_7_growth_smoke_20261120.json"
CAP_CPU = 150.0
BASE = {
    "groups": 4,
    "units_per_group": 8,
    "input_dim": 8,
    "hidden_dim": 8,
    "output_dim": 4,
    "lr": 3e-3,
    "steps": 2000,
    "batch": 64,
    "warmup": 100,
    "seeds": [0, 1, 2],
    "train_worlds": 256,
    "fit_worlds": 32,
}
EXITS = [
    "engineering_failure",
    "not_admitted",
    "budget_stop",
    "growth_not_needed",
    "retention_breach",
    "credit_insufficient",
    "recovery_failure",
    "growth_gain_absent",
    "growth_gain_pass",
]
PINNED_FILES = (
    "scripts/training/run_s2_22_7_growth_20261120.py",
    "scripts/training/probe_s2_22_7_0_alpha_adapter_v3_20261120.py",
    "taiji/u2_neural_units.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.7 growth/consolidation/lifecycle seven-arm comparison",
        "base": BASE,
        "task": {
            "name": "first-A fit 4-way position classification",
            "adapter": "v3 raw-media palette histogram (D-20261119-34)",
            "features": 48,
            "chance": 0.25,
        },
        "arms": {
            "G0": "credit-conditioned birth (main candidate)",
            "G1": "random content-blind gate birth, same budget",
            "G2": "permuted gate birth, same budget",
            "G3": "GradMax-style direction + fixed schedule",
            "G4": "replay only, no growth",
            "G5": "fixed-capacity strong optimisation",
            "G6": "preallocated final capacity",
        },
        "budget_parity": {
            "unit": "equal optimizer steps x batch x seeds across arms",
            "steps_per_arm": BASE["steps"],
            "batch": BASE["batch"],
            "seeds": BASE["seeds"],
            "no_parameter_only_parity": True,
        },
        "structure_accounts": [
            "units",
            "effective_edges",
            "state_bytes",
            "learnable_parameters",
            "independent_credit_directions",
            "mature_units",
        ],
        "gates": {
            "growth_gain_min": 0.05,
            "world_paired_lower_bound_min": 0.0,
            "all_seeds_same_direction": True,
            "retention_drop_upper_bound_max": 0.02,
            "retire_gain_removed_fraction_min": 0.5,
            "retire_absolute_drop_min": 0.02,
            "recovery_exact": True,
            "references_cleanup": True,
        },
        "exits_order": EXITS,
        "check_reads": 0,
        "final_reads": 0,
        "price_cpu": 600.0,
        "price_source": "DESIGN-TAIJI_U2_22_7_0_GROWTH_20261120.md §4 (measured 0.00234 CPU/step)",
        "cap_cpu": CAP_CPU,
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-7-growth-v1":
        raise ValueError("unsupported_growth_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "smoke" and contract.get("status") != "designed_pending_owner":
        raise ValueError("unexpected_contract_status")
    if mode == "run":
        raise ValueError("runner_not_implemented_training_authority_null")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-7-growth-v1",
        "card": "DESIGN-TAIJI_U2_22_7_0_GROWTH_20261120.md",
        "status": "designed_pending_owner",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 600.0,
            "optimizer_steps": BASE["steps"] * len(BASE["seeds"]) * 7,
            "arms": 7,
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
    examples = generate_examples("train", 8)
    first = [adapter_v3_features(example) for example in examples]
    second = [adapter_v3_features(example) for example in examples]
    adapter_deterministic = all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    )
    labels_ok = all(label_index(example) in range(4) for example in examples)
    r4 = json.loads(
        (ROOT / "reports/taiji_u2_22_7_0_alpha_r4_20261120.json").read_text(encoding="utf-8")
    )
    budget = contract["criteria"]["budget_parity"]
    budget_parity = budget["steps_per_arm"] * budget["batch"] * len(budget["seeds"])
    checks = {
        "adapter_deterministic": adapter_deterministic,
        "labels_in_range": labels_ok,
        "r4_reference_ready": r4["verdict"] == "adapter_ready",
        "budget_parity_positive": budget_parity > 0,
        "seven_arms_defined": len(contract["criteria"]["arms"]) == 7,
        "authority_null": contract["training_authority"] is None,
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-7-growth-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "budget_parity_samples": budget_parity,
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
    import math as _math  # noqa: F401

    raise SystemExit(main())
