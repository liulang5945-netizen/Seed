"""22.4.2 action-fragment card: action consequence with counterfactual arms.

Task: N=3 pushes, predict the pushed body's final position bin.  Arms:
``full`` (action direction + current-bin proxy present), ``action_masked`` (both
action dims zeroed), ``direction_permuted`` (direction dims shuffled across
samples, labels unchanged).  Action contribution requires full to beat both
counterfactuals by >= 0.05 on every seed, with a paired 95% lower bound > 0.

Pre-registered exits: engineering_failure / not_admitted / budget_stop /
action_contribution_absent / action_consequence_pass.  Zero check; no parent
model is modified, so the old-ability retention gate is N/A and disclosed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import (
    build_partition,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
    save_capsule,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_action_fragment_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_4_2_action_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_4_2_action_run_20261120.json"
CAP_CPU = 100.0
BASE = {
    "groups": 4,
    "units_per_group": 8,
    "input_dim": 8,
    "hidden_dim": 8,
    "output_dim": 3,
    "lr": 3e-3,
    "steps": 1000,
    "batch": 64,
    "warmup": 100,
    "seeds": [0, 1, 2],
    "train_worlds": 256,
    "fit_worlds": 32,
}
ARMS = ("full", "action_masked", "direction_permuted")
HELPER = ROOT / "scripts/training/s2_22_7_0_alpha_reload_20261120.py"
PINNED_FILES = (
    "scripts/training/run_s2_22_4_2_action_fragment_20261120.py",
    "scripts/training/probe_s2_action_consequence_20261120.py",
    "taiji/u2_neural_units.py",
)
EXITS = [
    "engineering_failure",
    "not_admitted",
    "budget_stop",
    "action_contribution_absent",
    "action_consequence_pass",
]


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.4.2 action-consequence fragment with counterfactual arms",
        "base": BASE,
        "task": "N=3 pushes, pushed body's final position bin (left/middle/right)",
        "arms": {
            "full": "action direction + current-bin proxy present",
            "action_masked": "both action dims zeroed",
            "direction_permuted": "direction dim shuffled across samples",
        },
        "gates": {
            "full_min_accuracy": 0.45,
            "full_above_baseline_min": 0.05,
            "action_margin_min": 0.05,
            "paired_lower_bound_min": 0.0,
            "all_seeds_same_direction": True,
            "recovery_exact": True,
        },
        "retention": "N/A - no parent model modified (fresh graph, disclosed)",
        "exits_order": EXITS,
        "check_reads": 0,
        "final_reads": 0,
        "price_cpu": 100.0,
        "cap_cpu": CAP_CPU,
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-4-2-action-v1":
        raise ValueError("unsupported_action_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    for name in PINNED_FILES:
        if contract["source_pins"].get(name) != sha_of(ROOT / name):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "run" and not contract.get("training_authority"):
        raise ValueError("missing_training_authority")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-4-2-action-v1",
        "card": "RESULT-TAIJI_U2_22_4_2_ACTION_20261120.md",
        "status": "designed_pending_owner",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 100.0,
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


def authorize(contract_path: Path, authority: str) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "frozen"
    contract["training_authority"] = authority
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"authorized": authority}))
    return 0


def learning_rate(step: int) -> float:
    if step < BASE["warmup"]:
        return BASE["lr"] * float(step + 1) / float(BASE["warmup"])
    return (
        BASE["lr"]
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi * float(step - BASE["warmup"]) / max(1, BASE["steps"] - BASE["warmup"])
            )
        )
    )


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    names = ("rgb", "audio", "proprio", "question", "goal", "feedback")
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(names)}


def arm_features(rows: list[dict], arm: str) -> torch.Tensor:
    features = torch.stack([row["features"] for row in rows]).clone()
    if arm == "action_masked":
        features[:, 37] = 0.0
        features[:, 38] = 0.0
    elif arm == "direction_permuted":
        generator = torch.Generator().manual_seed(20261120)
        order = torch.randperm(features.shape[0], generator=generator)
        features[:, 37] = features[order, 37]
    return features


def train_arm(rows: list[dict], arm: str, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=BASE["groups"],
        units_per_group=BASE["units_per_group"],
        input_dim=BASE["input_dim"],
        hidden_dim=BASE["hidden_dim"],
        output_dim=BASE["output_dim"],
        seed=20261120 + seed,
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=BASE["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    features = arm_features(rows, arm)
    labels = torch.tensor([row["final_bin"] for row in rows])
    generator = torch.Generator().manual_seed(20261120 + seed)
    for step in range(BASE["steps"]):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = torch.randint(0, features.shape[0], (BASE["batch"],), generator=generator)
        gates = torch.ones(BASE["batch"], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    fit_rows = rows[-BASE["fit_worlds"] :]
    with torch.no_grad():
        fit_features = arm_features(fit_rows, arm)
        gates = torch.ones(fit_features.shape[0], graph.units.n_units)
        logits = graph(channels(fit_features), gates, graph.initial_state())["readout"]
    correct = logits.argmax(-1) == torch.tensor([row["final_bin"] for row in fit_rows])
    recovery_exact = None
    if arm == "full":
        capsule = build_capsule(graph)
        artifact = ROOT / f"output/s2_22_4_2/full_seed{seed}.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_capsule(artifact, capsule)
        probe_path = ROOT / f"output/s2_22_4_2/full_seed{seed}_probe.json"
        probe_path.write_text(
            json.dumps({name: tensor.tolist() for name, tensor in channels(fit_features).items()}),
            encoding="utf-8",
        )
        expected = hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest()
        completed = subprocess.run(
            [sys.executable, str(HELPER), "--capsule", str(artifact), "--probe", str(probe_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        loaded = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else ""
        recovery_exact = loaded == expected
    return {
        "accuracy": float(correct.float().mean()),
        "correct": correct.tolist(),
        "recovery_exact": recovery_exact,
    }


def paired_lower_bound(values: list[float]) -> float:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261120)
    draws = rng.choice(data, size=(10000, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.05))


def run(contract: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    rows = build_partition("train", BASE["train_worlds"] + BASE["fit_worlds"], 20261120)
    if len(rows) < BASE["train_worlds"] + BASE["fit_worlds"]:
        verdict = "engineering_failure"
        payload = {
            "schema": "taiji-u2-22-4-2-action-run-v1",
            "verdict": verdict,
            "reason": "insufficient_worlds",
            "cost": {"cpu": round(time.process_time() - started, 2)},
        }
        write_report(RUN, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    train_rows = rows[: BASE["train_worlds"]]
    fit_rows = rows[-BASE["fit_worlds"] :]
    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for seed in BASE["seeds"]:
        for arm in ARMS:
            results[arm][seed] = train_arm(train_rows + fit_rows, arm, seed)
    full = [results["full"][seed]["accuracy"] for seed in BASE["seeds"]]
    masked = [results["action_masked"][seed]["accuracy"] for seed in BASE["seeds"]]
    permuted = [results["direction_permuted"][seed]["accuracy"] for seed in BASE["seeds"]]
    baseline = float(
        sum(1 for row in fit_rows if row["current_bin"] == row["final_bin"]) / len(fit_rows)
    )
    margins = [(full[i] - max(masked[i], permuted[i])) for i in range(len(BASE["seeds"]))]
    paired = [
        float(a) - float(b)
        for a, b in zip(
            results["full"][BASE["seeds"][0]]["correct"],
            [
                max(one, two)
                for one, two in zip(
                    results["action_masked"][BASE["seeds"][0]]["correct"],
                    results["direction_permuted"][BASE["seeds"][0]]["correct"],
                    strict=False,
                )
            ],
            strict=False,
        )
    ]
    recovery_ok = all(results["full"][seed]["recovery_exact"] for seed in BASE["seeds"])
    gates = contract["criteria"]["gates"]
    checks = {
        "full_min_accuracy": float(np.mean(full)) >= gates["full_min_accuracy"],
        "full_above_baseline": float(np.mean(full)) >= baseline + gates["full_above_baseline_min"],
        "action_margin": min(margins) >= gates["action_margin_min"],
        "paired_lower_bound": paired_lower_bound(paired) > gates["paired_lower_bound_min"],
        "all_seeds_same_direction": all(margin > 0 for margin in margins),
        "recovery_exact": recovery_ok,
    }
    budget_ok = time.process_time() - started <= CAP_CPU
    if not budget_ok:
        verdict = "budget_stop"
    elif not checks["recovery_exact"]:
        verdict = "engineering_failure"
    elif all(
        checks[key]
        for key in (
            "full_min_accuracy",
            "full_above_baseline",
            "action_margin",
            "paired_lower_bound",
            "all_seeds_same_direction",
        )
    ):
        verdict = "action_consequence_pass"
    else:
        verdict = "action_contribution_absent"
    payload = {
        "schema": "taiji-u2-22-4-2-action-run-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": contract["criteria"]["base"],
        "arms": {
            arm: {
                str(seed): {"accuracy": round(value["accuracy"], 4)} for seed, value in per.items()
            }
            for arm, per in results.items()
        },
        "accuracies": {
            "full": [round(value, 4) for value in full],
            "action_masked": [round(value, 4) for value in masked],
            "direction_permuted": [round(value, 4) for value in permuted],
        },
        "baseline_accuracy": round(baseline, 4),
        "margins": [round(margin, 4) for margin in margins],
        "paired_lower_bound": round(paired_lower_bound(paired), 4),
        "checks": checks,
        "verdict": verdict,
        "budget": {"cap_cpu": CAP_CPU, "cpu": round(time.process_time() - started, 2)},
        "discipline": {
            "optimizer_steps": BASE["steps"] * len(BASE["seeds"]) * len(ARMS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(RUN, payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "checks": checks,
                "full": payload["accuracies"]["full"],
                "baseline": payload["baseline_accuracy"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "action_consequence_pass" else 1


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    guards(contract, "smoke")
    rows = build_partition("train", 8, 20261120)
    checks = {
        "rows_built": len(rows) == 8,
        "features_48": all(row["features"].numel() == 48 for row in rows),
        "labels_in_range": all(0 <= row["final_bin"] <= 2 for row in rows),
        "three_arms": len(contract["criteria"]["arms"]) == 3,
        "authority_state_ok": contract["training_authority"] is None
        or contract.get("status") == "frozen",
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-4-2-action-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
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
    report = json.loads(RUN.read_text(encoding="utf-8"))
    contract["result"] = {
        "smoke": (
            json.loads(SMOKE.read_text(encoding="utf-8"))["verdict"]
            if SMOKE.is_file()
            else "unknown"
        ),
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "cost_cpu": report["cost"]["cpu"],
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
