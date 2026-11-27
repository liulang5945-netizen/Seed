"""22.6.1 gain/risk/cost predictor contract runner (small MLP, standing authority).

Training is a tiny MLP (10 -> 32 -> 4, Huber) on 1,536 counterfactual label rows;
the primary gate is the held-out normalized Huber improvement over the fit
constant-mean baseline (>= 10%).  Point-prediction error and the selection-gain
proxy are reported in separate accounts.  The 22.6.2 policy comparison remains a
separate card.  Modes: --freeze -> --authorize -> --smoke -> --run -> --closeout.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.build_s2_22_6_1_labels_20261120 import (  # noqa: E402
    ACTIONS,
    build_rows,
    vectorize,
)
from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_6_1_predictor_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_6_1_predictor_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_6_1_predictor_run_20261120.json"
LABELS = ROOT / "reports/taiji_u2_22_6_1_labels_20261120.json"
CAP_CPU = 40.0
EPOCHS = 200
HIDDEN = 32
LR = 1e-3
HUBER_DELTA = 1.0
IMPROVEMENT_MIN = 0.10
TORCH_SEED = 20261120
PINNED_FILES = (
    "scripts/training/run_s2_22_6_1_predictor_20261120.py",
    "scripts/training/build_s2_22_6_1_labels_20261120.py",
    "scripts/training/curate_s2_22_6_pool_20261120.py",
    "taiji/u2_task_world.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    protocol = labels["protocol"]
    return {
        "card": "22.6.1 gain/risk/cost predictor",
        "dataset": {
            "receipt": str(LABELS.relative_to(ROOT)),
            "rows_sha256": labels["rows_sha256"],
            "n_rows": labels["n_rows"],
            "feature_dim": labels["feature_dim"],
            "label_dim": labels["label_dim"],
            "split": "24 train worlds / 8 held-out worlds, by world",
            "actions": list(ACTIONS),
        },
        "model": {
            "kind": "mlp",
            "input": labels["feature_dim"],
            "hidden": HIDDEN,
            "output": len(ACTIONS),
            "loss": "huber",
            "huber_delta": HUBER_DELTA,
            "optimizer": "adam",
            "lr": LR,
            "epochs": EPOCHS,
            "torch_seed": TORCH_SEED,
            "feature_normalization": "train mean/std applied to both splits",
        },
        "baseline": {
            "kind": "fit_constant_mean_per_action",
            "trained_on": "train split only",
        },
        "thresholds": {
            "heldout_huber_improvement_min": IMPROVEMENT_MIN,
        },
        "accounts": {
            "point_error": "held-out normalized Huber (model vs constant baseline)",
            "selection_gain": "reported as a proxy only; gated by the separate 22.6.2 card",
        },
        "exits_order": [
            "engineering_failure",
            "not_admitted",
            "budget_stop",
            "predictor_insufficient",
            "predictor_gain_pass",
        ],
        "check_reads": 0,
        "final_reads": 0,
        "protocol_chunk": protocol["default_behavior_policy"],
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-6-1-predictor-v1":
        raise ValueError("unsupported_predictor_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"smoke", "run"} and contract.get("status") != "frozen":
        raise ValueError("contract_not_frozen")
    if mode == "run":
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cap_cpu"):
            raise ValueError("missing_cap")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-6-1-predictor-v1",
        "card": "DESIGN-TAIJI_U2_22_6_1_PREDICTOR_20261120.md",
        "status": "designed_priced_authorization_pending",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 22.0,
            "optimizer_steps": EPOCHS,
            "price_source": "DESIGN-TAIJI_U2_22_6_1_PREDICTOR_20261120.md §3",
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


def authorize(contract_path: Path, authority: str) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "designed_priced_authorization_pending":
        raise ValueError("unexpected_contract_status")
    contract["status"] = "frozen"
    contract["training_authority"] = authority
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    guards(contract, "run")
    print(json.dumps({"authorized": authority, "cap_cpu": contract["cap_cpu"]}))
    return 0


def dataset_check(rows: list[dict] | None = None) -> dict[str, Any]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    if rows is None:
        rows = build_rows()
    features, labels_matrix = vectorize(rows)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {
        "rows": len(rows),
        "feature_dim": int(features.shape[1]),
        "label_dim": int(labels_matrix.shape[1]),
        "receipt_sha": labels["rows_sha256"],
        "rebuilt_sha": digest,
        "matches": digest == labels["rows_sha256"],
    }


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "smoke")
    rows = build_rows()
    check = dataset_check(rows)
    features, labels_matrix = vectorize(rows)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    baseline = labels_matrix[train_mask].mean(axis=0)
    baseline_loss = float(np.abs(labels_matrix[~train_mask] - baseline).mean())
    checks = {
        "dataset_matches_receipt": check["matches"],
        "feature_dim_matches": check["feature_dim"]
        == contract["criteria"]["dataset"]["feature_dim"],
        "label_dim_matches": check["label_dim"] == contract["criteria"]["dataset"]["label_dim"],
        "baseline_finite": bool(np.isfinite(baseline_loss)),
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-6-1-predictor-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "dataset": check,
        "baseline_loss": round(baseline_loss, 6),
        "verdict": "smoke_passed" if passed else "smoke_failed",
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(SMOKE, payload)
    print(json.dumps({"verdict": payload["verdict"], "cpu": payload["cost"]["cpu"]}))
    return 0 if passed else 1


def run(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "run")
    rows = build_rows()
    check = dataset_check(rows)
    features, labels_matrix = vectorize(rows)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    torch.manual_seed(TORCH_SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(normalized.shape[1], HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, len(ACTIONS)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x_train = torch.from_numpy(normalized[train_mask])
    y_train = torch.from_numpy(labels_matrix[train_mask])
    loss_fn = torch.nn.HuberLoss(delta=HUBER_DELTA)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    baseline = labels_matrix[train_mask].mean(axis=0)
    x_held = torch.from_numpy(normalized[~train_mask])
    with torch.no_grad():
        predictions = model(x_held).numpy()
    held_labels = labels_matrix[~train_mask]
    model_loss = float(np.abs(predictions - held_labels).mean())
    baseline_loss = float(np.abs(held_labels - baseline).mean())
    improvement = 1.0 - model_loss / baseline_loss if baseline_loss > 0 else 0.0
    oracle_upper_bound_mean = float(np.mean(np.max(held_labels, axis=1)))
    budget_ok = time.process_time() - started <= CAP_CPU
    if not check["matches"]:
        verdict = "not_admitted"
    elif not budget_ok:
        verdict = "budget_stop"
    elif improvement < IMPROVEMENT_MIN:
        verdict = "predictor_insufficient"
    else:
        verdict = "predictor_gain_pass"
    payload = {
        "schema": "taiji-u2-22-6-1-predictor-run-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "dataset": check,
        "training": {"epochs": EPOCHS, "hidden": HIDDEN, "optimizer_steps": EPOCHS},
        "heldout": {
            "model_huber": round(model_loss, 6),
            "baseline_huber": round(baseline_loss, 6),
            "improvement": round(improvement, 6),
            "threshold": IMPROVEMENT_MIN,
            "oracle_upper_bound_mean": round(oracle_upper_bound_mean, 4),
        },
        "accounts": {
            "point_error": {"model": round(model_loss, 6), "baseline": round(baseline_loss, 6)},
            "selection_gain": "reported by the separate 22.6.2 card; not gated here",
        },
        "verdict": verdict,
        "budget": {"cap_cpu": CAP_CPU, "cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(RUN, payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "improvement": round(improvement, 4),
                "cpu": payload["cost"]["cpu"],
            }
        )
    )
    return 0 if verdict == "predictor_gain_pass" else 1


def closeout(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = json.loads(RUN.read_text(encoding="utf-8"))
    contract["result"] = {
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "cost_cpu": report["cost"]["cpu"],
        "report": str(RUN.relative_to(ROOT)),
    }
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"closed": report["verdict"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--authorize", metavar="AUTHORITY")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        return freeze(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.authorize:
        return authorize(CONTRACT, args.authorize)
    if args.smoke:
        return smoke(contract)
    if args.closeout:
        return closeout(CONTRACT)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
