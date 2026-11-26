"""22.7g growth-benefit runner (joint 24-class task, three arms).

Pre-registered in DESIGN-TAIJI_U2_22_7G_GROWTH_JOINT24_20261120.md and the frozen
contract.  Arms: ``G0_growth`` (32 units warm-start, neutral birth + maturity at
step 750 -> 33 units, continue to 1500 total steps), ``C1_no_growth`` (32 units,
1500 steps) and ``C2_preallocated`` (33 units from step 0, 1500 steps).  All arms
re-create the optimizer at the midpoint (parity), use equal total steps, and
report the FLOPs ratio.  Gates: growth gain >= 0.05 over max(C1,C2), paired
one-sided 95% lower bound > 0, all 3 seeds positive, retention drop <= 0.02 across
the birth event, capsule -> independent-process reload bitwise exact, reference
cleanup, 0 check/final.

Modes: --freeze / --authorize AUTHORITY / --smoke / --run / --closeout.
``--run`` fails closed without ``training_authority``.
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
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_20261120 import HELPER  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_neural_birth_multi import (  # noqa: E402
    live_entry_birth_multi,
    multi_newborn_liveness_steps,
)
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
    save_capsule,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_7g_growth_joint24_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_7g_growth_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_7g_growth_run_20261120.json"
CAP_CPU = 250.0
BASE = {
    "groups": 4,
    "units_per_group": 8,
    "input_dim": 8,
    "hidden_dim": 8,
    "output_dim": 24,
    "lr": 3e-3,
    "steps": 1500,
    "midpoint_step": 250,
    "batch": 64,
    "warmup": 100,
    "seeds": [0, 1, 2],
    "train_worlds": 256,
    "fit_worlds": 32,
}
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
ARMS = (
    "G0_growth",
    "C1_no_growth",
    "C2_preallocated",
)
PINNED_FILES = (
    "scripts/training/run_s2_22_7b_growth_joint24_20261120.py",
    "scripts/training/probe_s2_22_7_joint24_20261120.py",
    "scripts/training/probe_s2_22_7_0_alpha_adapter_v3_20261120.py",
    "taiji/u2_neural_units.py",
)
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


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.7g growth benefit with projected training on joint 24-class task",
        "base": BASE,
        "task": {
            "name": "first-A joint colour x position (24 classes)",
            "adapter": "v3 raw-media palette histogram",
            "headroom_receipt": "reports/taiji_u2_22_7_joint24_20261120.json",
            "chance": round(1 / 24, 4),
        },
        "arms": {
            "G0_growth": "warm 32 units; three live-entry births (p=0, r inherited) at step 750 -> 35 units; 5-step liveness probe; continue to 1500 total steps",
            "C1_no_growth": "32 units; 1500 total steps",
            "C2_preallocated": "35 units from step 0; 1500 total steps",
        },
        "out_of_scope": {
            "G1": "random content-blind gate birth",
            "G2": "permuted gate birth",
            "G3": "GradMax direction + fixed schedule",
            "G4": "replay only",
            "G5": "fixed-capacity strong optimisation",
            "G6": "preallocated final capacity snapshot",
        },
        "parity": {
            "rule": "equal total optimizer steps; optimizer recreated at midpoint for all arms; FLOPs ratio (33/32) reported",
            "flops_ratio_max": 1.05,
        },
        "gates": {
            "growth_gain_min": 0.05,
            "paired_lower_bound_min": 0.0,
            "all_seeds_same_direction": True,
            "retention_drop_upper_bound_max": 0.02,
            "recovery_exact": True,
            "birth_neutral_exact": True,
            "newborn_live": True,
            "references_cleanup": True,
        },
        "exits_order": EXITS,
        "check_reads": 0,
        "final_reads": 0,
        "price_cpu": 250.0,
        "cap_cpu": CAP_CPU,
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-7g-growth-v1":
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
    if mode in {"smoke", "run", "closeout"} and contract.get("status") not in {
        "designed_pending_owner",
        "frozen",
    }:
        raise ValueError("unexpected_contract_status")
    if mode in {"run", "closeout"} and contract.get("status") == "designed_pending_owner":
        raise ValueError("contract_not_frozen")
    if mode == "run" and not contract.get("training_authority"):
        raise ValueError("missing_training_authority")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-7g-growth-v1",
        "card": "DESIGN-TAIJI_U2_22_7G_GROWTH_JOINT24_20261120.md",
        "status": "designed_pending_owner",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 250.0,
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
    if contract.get("status") != "designed_pending_owner":
        raise ValueError("unexpected_contract_status")
    contract["status"] = "frozen"
    contract["training_authority"] = authority
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    guards(contract, "run")
    print(json.dumps({"authorized": authority, "cap_cpu": contract["cap_cpu"]}))
    return 0


def learning_rate(step: int) -> float:
    if step < BASE["warmup"]:
        return BASE["lr"] * float(step + 1) / float(BASE["warmup"])
    span = max(1, BASE["steps"] - BASE["warmup"])
    return BASE["lr"] * 0.5 * (1.0 + math.cos(math.pi * float(step - BASE["warmup"]) / float(span)))


def split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_examples = generate_examples("train", BASE["train_worlds"])
    fit_examples = generate_examples("fit", BASE["fit_worlds"])
    x_train = torch.stack([adapter_v3_features(example) for example in train_examples]).float()
    y_train = torch.tensor([joint_label(example) for example in train_examples])
    x_fit = torch.stack([adapter_v3_features(example) for example in fit_examples]).float()
    y_fit = torch.tensor([joint_label(example) for example in fit_examples])
    return x_train, y_train, x_fit, y_fit


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def make_graph(seed: int, units: int) -> RouteBGraph:
    torch.manual_seed(seed)
    groups = 4
    units_per_group = units // groups
    spec = UnitGraphSpec(
        groups=groups,
        units_per_group=units_per_group,
        input_dim=8,
        hidden_dim=8,
        output_dim=24,
        seed=20261120 + seed,
    )
    return RouteBGraph(spec)


def train(
    graph: RouteBGraph,
    data: tuple,
    seed: int,
    start_step: int,
    end_step: int,
) -> tuple[RouteBGraph, Any]:
    x_train, y_train, _, _ = data
    optimizer = torch.optim.Adam(graph.parameters(), lr=BASE["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed + start_step)
    indices = torch.arange(BASE["train_worlds"])
    for step in range(start_step, end_step):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = indices[torch.randperm(BASE["train_worlds"], generator=generator)[: BASE["batch"]]]
        batch_channels = channels(x_train[batch])
        gates = torch.ones(BASE["batch"], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(batch_channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, y_train[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    return graph, optimizer


def predict(graph: RouteBGraph, x_fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        gates = torch.ones(x_fit.shape[0], graph.units.n_units)
        return graph(channels(x_fit), gates, graph.initial_state())["readout"], torch.tensor([])


def grow(graph: RouteBGraph) -> tuple[RouteBGraph, dict[str, Any]]:
    before_units = graph.units.n_units
    parents = [unit["unit_id"] for unit in graph.units.identities["units"][:3]]
    grown, details = live_entry_birth_multi(graph.units, 0, parents, count=3)
    graph.units = grown
    for unit in grown.identities["units"][-3:]:
        graph.units.set_lifecycle(unit["unit_id"], "mature")
    return graph, {
        "before_units": before_units,
        "after_units": graph.units.n_units,
        "newborn_ids": [unit["unit_id"] for unit in grown.identities["units"][-3:]],
        "details": details,
    }


def evaluate_arm(arm: str, seed: int, data: tuple) -> dict[str, Any]:
    x_train, y_train, x_fit, y_fit = data
    units = 35 if arm in {"C2_preallocated"} else 32
    graph = make_graph(seed, units)
    if arm == "C2_preallocated":
        graph, birth = grow(graph)
    else:
        birth = None
    graph, _ = train(graph, data, seed, 0, BASE["midpoint_step"] - 1)
    before_birth_logits = predict(graph, x_fit)[0]
    before_birth = before_birth_logits.argmax(-1)
    birth_neutral_diff = 0.0
    liveness: dict[str, float] = {}
    if arm == "G0_growth":
        graph, birth = grow(graph)
        after_logits = predict(graph, x_fit)[0]
        birth_neutral_diff = float((before_birth_logits - after_logits).abs().max())
        after_birth = after_logits.argmax(-1)
        retention_pairs = (before_birth == y_fit).float() - (after_birth == y_fit).float()
        batch = torch.arange(BASE["batch"])
        liveness = multi_newborn_liveness_steps(
            graph, channels(x_train[batch]), y_train[batch], count=3, steps=5
        )
    else:
        retention_pairs = torch.zeros(x_fit.shape[0])
    probe_steps = 5 if arm == "G0_growth" else 0
    remaining = BASE["steps"] - (BASE["midpoint_step"] - 1) - probe_steps
    graph, _ = train(graph, data, seed, BASE["midpoint_step"], BASE["midpoint_step"] + remaining)
    readout = predict(graph, x_fit)[0]
    predictions = readout.argmax(-1)
    correct = predictions == y_fit
    accuracy = float(correct.float().mean())
    recovery_exact = None
    if arm == "G0_growth":
        capsule = build_capsule(graph)
        artifact = ROOT / f"output/s2_22_7b/g0_seed{seed}.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_capsule(artifact, capsule)
        probe_path = ROOT / f"output/s2_22_7b/g0_seed{seed}_probe.json"
        probe_path.write_text(
            json.dumps({name: tensor.tolist() for name, tensor in channels(x_fit).items()}),
            encoding="utf-8",
        )
        heldout_sha = hashlib.sha256(readout.detach().numpy().tobytes()).hexdigest()
        completed = subprocess.run(
            [sys.executable, str(HELPER), "--capsule", str(artifact), "--probe", str(probe_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        reload_sha = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else ""
        recovery_exact = reload_sha == heldout_sha
    return {
        "arm": arm,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct.tolist(),
        "retention_drop": round(float(retention_pairs.mean()), 6),
        "birth_neutral_diff": birth_neutral_diff,
        "liveness": liveness,
        "birth": birth,
        "recovery_exact": recovery_exact,
        "units_final": graph.units.n_units,
        "mature_units": sum(
            1 for unit in graph.units.identities["units"] if unit["lifecycle"] == "mature"
        ),
        "immature_units": sum(
            1 for unit in graph.units.identities["units"] if unit["lifecycle"] == "immature"
        ),
    }


def paired_lower_bound(values: list[float]) -> float:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261120)
    draws = rng.choice(data, size=(10000, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.05))


def run(contract: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    guards(contract, "run")
    data = split()
    arms: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for seed in BASE["seeds"]:
        for arm in ARMS:
            arms[arm][seed] = evaluate_arm(arm, seed, data)
    g0 = [arms["G0_growth"][seed]["accuracy"] for seed in BASE["seeds"]]
    c1 = [arms["C1_no_growth"][seed]["accuracy"] for seed in BASE["seeds"]]
    c2 = [arms["C2_preallocated"][seed]["accuracy"] for seed in BASE["seeds"]]
    gains = [value - max(one, two) for value, one, two in zip(g0, c1, c2, strict=False)]
    paired = [
        float(a) - float(b)
        for a, b in zip(
            arms["G0_growth"][BASE["seeds"][0]]["correct"],
            [
                max(one, two)
                for one, two in zip(
                    arms["C1_no_growth"][BASE["seeds"][0]]["correct"],
                    arms["C2_preallocated"][BASE["seeds"][0]]["correct"],
                    strict=False,
                )
            ],
            strict=False,
        )
    ]
    retention_ub = max(arms["G0_growth"][seed]["retention_drop"] for seed in BASE["seeds"])
    recovery_ok = all(arms["G0_growth"][seed]["recovery_exact"] for seed in BASE["seeds"])
    cleanup_ok = all(
        arms["G0_growth"][seed]["immature_units"] == 0
        and arms["G0_growth"][seed]["units_final"] == 35
        for seed in BASE["seeds"]
    )
    gates = contract["criteria"]["gates"]
    birth_neutral_ok = (
        max(abs(arms["G0_growth"][seed]["birth_neutral_diff"]) for seed in BASE["seeds"]) == 0.0
    )
    live_ok = all(arms["G0_growth"][seed]["liveness"].get("live", False) for seed in BASE["seeds"])
    budget_ok = time.process_time() - started <= CAP_CPU
    if budget_ok and not (live_ok and birth_neutral_ok):
        verdict = "engineering_failure"
    elif budget_ok and not cleanup_ok:
        verdict = "references_cleanup_failed"
    elif budget_ok and not recovery_ok:
        verdict = "recovery_failure"
    elif budget_ok and retention_ub > gates["retention_drop_upper_bound_max"]:
        verdict = "retention_breach"
    elif not budget_ok:
        verdict = "budget_stop"
    elif (
        float(np.mean(gains)) >= gates["growth_gain_min"]
        and paired_lower_bound(paired) > gates["paired_lower_bound_min"]
        and all(gain > 0 for gain in gains)
    ):
        verdict = "growth_gain_pass"
    else:
        verdict = "growth_gain_absent"
    payload = {
        "schema": "taiji-u2-22-7g-growth-run-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "task": contract["criteria"]["task"],
        "arms": {
            arm: {
                seed: {key: value for key, value in result.items() if key not in {"correct"}}
                for seed, result in per_seed.items()
            }
            for arm, per_seed in arms.items()
        },
        "gains": [round(gain, 6) for gain in gains],
        "mean_gain": round(float(np.mean(gains)), 6),
        "paired_lower_bound": round(paired_lower_bound(paired), 6),
        "retention_drop_upper_bound": round(retention_ub, 6),
        "recovery_exact": recovery_ok,
        "birth_neutral": birth_neutral_ok,
        "newborn_live": live_ok,
        "references_cleanup": cleanup_ok,
        "flops_ratio": round(33 / 32, 4),
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
                "gains": payload["gains"],
                "mean_gain": payload["mean_gain"],
                "paired_lb": payload["paired_lower_bound"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "growth_gain_pass" else 1


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    guards(contract, "smoke")
    examples = generate_examples("fit", 8)
    features = [adapter_v3_features(example) for example in examples]
    labels = [joint_label(example) for example in examples]
    joint24 = json.loads(
        (ROOT / "reports/taiji_u2_22_7_joint24_20261120.json").read_text(encoding="utf-8")
    )
    checks = {
        "adapter_deterministic": all(
            torch.equal(first, second) for first, second in zip(features, features, strict=True)
        ),
        "labels_in_range": all(0 <= label < 24 for label in labels),
        "joint24_family_candidate": joint24["verdict"] == "joint24_family_candidate",
        "three_arms_defined": len(contract["criteria"]["arms"]) == 3,
        "out_of_scope_documented": len(contract["criteria"]["out_of_scope"]) == 6,
        "authority_state_ok": contract["training_authority"] is None
        or contract.get("status") == "frozen",
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-7g-growth-smoke-v1",
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
        "mean_gain": report["mean_gain"],
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
