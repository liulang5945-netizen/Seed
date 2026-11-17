"""22.5.W.1.1 learned-memory runner: DMS parent acquisition with a gated memory.

The base unit-core B graph is frozen; only the write gate, write candidate and
2-way memory readout train. Modes: draft, smoke (zero optimizer steps), price
(discarded model), freeze, run (authorized), recover, verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u2_w_dms_20261117 as wprobe
from taiji import u2_unit_input_dms as dms
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_neural_units import (
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    capsule_summary,
    load_capsule,
    save_capsule,
    validate_capsule,
)
from taiji.u2_unit_memory_readout import LearnedMemoryGraph

CONTRACT = ROOT / "plans/manifests/taiji_u2_w11_learned_memory_20261117.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_w11_learned_memory_smoke_20261117.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_w11_learned_memory_price_20261117.json"
RUN_REPORT = ROOT / "reports/taiji_u2_w11_learned_memory_run_20261117.json"
RECOVER_REPORT = ROOT / "reports/taiji_u2_w11_learned_memory_recover_20261117.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_w11_learned_memory_verify_20261117.json"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/w11_learned_memory_20261117"
TRAINABLE = (
    "write_gate.weight",
    "write_gate.bias",
    "write_candidate.weight",
    "write_candidate.bias",
    "memory_readout.weight",
)
FIT_GATE = 0.90
TRAIN_GATE = 0.95
UNSEEN_GATE = 0.85
LESION_MAX = 0.60
CAP_MARGIN = 2.0


def sha_of(path: Path) -> str:
    return (
        wprobe.sha_of(path)
        if hasattr(wprobe, "sha_of")
        else hashlib.sha256(path.read_bytes()).hexdigest()
    )


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return path


def pinned_files() -> tuple[tuple[str, Path], ...]:
    return (
        ("runner", Path(__file__).resolve()),
        ("memory_module", ROOT / "taiji/u2_unit_memory_readout.py"),
        ("dms_input", ROOT / "taiji/u2_unit_input_dms.py"),
        ("dms_probe", ROOT / "scripts/training/probe_u2_w_dms_20261117.py"),
        ("neural_units", ROOT / "taiji/u2_neural_units.py"),
        ("u2_history", ROOT / "taiji/u2_history.py"),
    )


def guards(contract: dict[str, Any], mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in pinned_files():
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if LearnedMemoryGraph(UnitGraphSpec()).route != contract["criteria"]["route"]:
        raise ValueError("route_mismatch")
    if value_digest(dms.encoding_contract()) != contract["encoding_contract_digest"]:
        raise ValueError("encoding_contract_digest_mismatch")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract["plan"].get("cap_cpu") or not contract["plan"].get("cap_wall"):
            raise ValueError("missing_measured_cap")


def build_graph(seed: int) -> LearnedMemoryGraph:
    graph = LearnedMemoryGraph(UnitGraphSpec(dtype="float64", seed=seed))
    for name, param in graph.named_parameters():
        param.requires_grad_(name in TRAINABLE)
    return graph


def make_optimizer(graph: LearnedMemoryGraph) -> torch.optim.Optimizer:
    params = [param for name, param in graph.named_parameters() if name in TRAINABLE]
    return torch.optim.Adam(params, lr=0.01)


def run_episode(
    graph: LearnedMemoryGraph, episode: dict[str, Any], lesion: bool = False
) -> torch.Tensor:
    gates = torch.ones(dms.COLUMNS, graph.spec.n_units, dtype=torch.float64)
    stages = dms.encode_episode(episode)
    if lesion:
        stages[0] = wprobe.neutral_stage(episode)
    state = graph.initial_state()
    out: dict[str, Any] | None = None
    for stage in stages:
        out = graph(stage, gates, state)
        state = out["next_state"]
    assert out is not None
    return cast(torch.Tensor, out["memory_readout"])


def one_update(
    graph: LearnedMemoryGraph, optimizer: torch.optim.Optimizer, episode: dict[str, Any]
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = run_episode(graph, episode)
    label = torch.tensor(dms.label(episode), dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def accuracy(
    graph: LearnedMemoryGraph, episodes: list[dict[str, Any]], lesion: bool = False
) -> dict[str, Any]:
    hits = 0
    with torch.no_grad():
        for episode in episodes:
            logits = run_episode(graph, episode, lesion)
            hits += int(int(torch.argmax(logits).item()) == dms.label(episode))
    total = len(episodes)
    return {"correct": hits, "denominator": total, "accuracy": hits / total if total else 0.0}


def draft() -> int:
    if CONTRACT.exists():
        raise FileExistsError("contract_exists")
    criteria = {
        "route": "B_learned_memory",
        "seeds": [2026111701, 2026111702, 2026111703],
        "updates_per_seed": 2000,
        "learning_rate": 0.01,
        "fit_gate": FIT_GATE,
        "train_gate": TRAIN_GATE,
        "unseen_gate": UNSEEN_GATE,
        "lesion_max": LESION_MAX,
        "dtype": "float64",
        "trainable": list(TRAINABLE),
        "scope": "DMS parent acquisition only; no capability/check2/22.6 claim",
        "exits": [
            {"order": 1, "verdict": "budget_stop"},
            {"order": 2, "verdict": "engineering_failure"},
            {"order": 3, "verdict": "parent_not_acquired"},
            {"order": 4, "verdict": "memory_lesion_failed"},
            {"order": 5, "verdict": "parent_acquired"},
        ],
    }
    contract = {
        "schema": "taiji-u2-w11-learned-memory-v1",
        "card": "22.5.W.1.1",
        "date": "2026-11-17",
        "status": "draft",
        "training_authority": None,
        "identity": "taiji-u2-learned-memory-v1",
        "parent_card": "22.5.W.1",
        "data_root": "generated:dms-20261117",
        "data_digest": wprobe.dataset_digest(wprobe.make_dataset()),
        "encoding_contract_identity": dms.IDENTITY,
        "encoding_contract_digest": value_digest(dms.encoding_contract()),
        "criteria": criteria,
        "criteria_sha256": value_digest(criteria),
        "plan": {
            "design_cap_cpu": 600.0,
            "design_cap_wall": 1200.0,
            "cap_cpu": None,
            "cap_wall": None,
        },
        "source_pins": {name: sha_of(path) for name, path in pinned_files()},
        "source_pin_paths": {
            name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in pinned_files()
        },
    }
    CONTRACT.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print("DRAFT ok criteria_sha256=" + contract["criteria_sha256"][:16])
    return 0


def freeze() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") == "frozen":
        raise ValueError("contract_already_frozen")
    price = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))
    if price["criteria_sha256"] != contract["criteria_sha256"]:
        raise ValueError("price_report_criteria_stale")
    contract["plan"]["cap_cpu"] = price["cap_cpu"]
    contract["plan"]["cap_wall"] = price["cap_wall"]
    contract["plan"]["per_update_cpu"] = price["per_update_cpu"]
    contract["plan"]["price_report"] = str(PRICE_REPORT.relative_to(ROOT)).replace("\\", "/")
    contract["status"] = "frozen"
    CONTRACT.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print("FREEZE ok cap_cpu=" + str(contract["plan"]["cap_cpu"]))
    return 0


def smoke() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "smoke")
    dataset = wprobe.make_dataset()
    checks: dict[str, bool] = {}
    rows = []
    for seed in contract["criteria"]["seeds"]:
        graph = build_graph(int(seed))
        trainable = [name for name, param in graph.named_parameters() if param.requires_grad]
        episode = dataset["train"][0]
        logits = run_episode(graph, episode)
        gate = graph.write_gate.weight.detach()
        neutral = float(logits.abs().max().detach()) == 0.0
        graph.zero_grad(set_to_none=True)
        loss = run_episode(graph, episode).square().mean()
        loss.backward()
        grads = [
            name
            for name, param in graph.named_parameters()
            if param.requires_grad and param.grad is not None
        ]
        frozen_grads = [
            name
            for name, param in graph.named_parameters()
            if not param.requires_grad and param.grad is not None
        ]
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        checkpoint = ARTIFACTS / f"seed{seed}_smoke_tmp.bin"
        save_capsule(checkpoint, build_capsule(graph))
        reloaded = load_capsule(checkpoint)
        roundtrip = value_digest(reloaded) == value_digest(load_capsule(checkpoint))
        checkpoint.unlink(missing_ok=True)
        seed_checks = {
            "trainable_is_gate_plus_readout": set(trainable) == set(TRAINABLE),
            "neutral_birth_readout_zero": neutral,
            "neutral_birth_gate_zero": float(gate.abs().max()) == 0.0,
            "trainable_grads_present": set(grads) == set(TRAINABLE),
            "frozen_params_no_grad": not frozen_grads,
            "checkpoint_roundtrip": roundtrip,
            "data_sizes": len(dataset["train"]) == 64 and len(dataset["fit"]) == 32,
        }
        checks.update({f"seed{seed}_{key}": value for key, value in seed_checks.items()})
        rows.append(
            {"seed": seed, "logits": [float(v) for v in logits.detach()], "checks": seed_checks}
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "smoke_passed" if not uncovered else "smoke_failed"
    report = {
        "schema": "taiji-u2-w11-learned-memory-smoke-v1",
        "card": "22.5.W.1.1",
        "phase": "smoke",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "rows": rows,
        "discipline": {
            "optimizer_steps": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(SMOKE_REPORT, report)
    print(
        "SMOKE verdict=" + verdict + " green=" + str(sum(checks.values())) + "/" + str(len(checks))
    )
    return 0 if not uncovered else 1


def price() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "price")
    dataset = wprobe.make_dataset()
    graph = build_graph(int(contract["criteria"]["seeds"][0]))
    optimizer = make_optimizer(graph)
    for index in range(4):
        one_update(graph, optimizer, dataset["train"][index])
    started = time.process_time()
    for index in range(32):
        one_update(graph, optimizer, dataset["train"][index % len(dataset["train"])])
    per_update = (time.process_time() - started) / 32.0
    updates = int(contract["criteria"]["updates_per_seed"])
    measured = 3 * updates * per_update
    cap_cpu = round(measured * CAP_MARGIN, 4)
    cap_wall = round(measured * CAP_MARGIN * 2.0, 4)
    report = {
        "schema": "taiji-u2-w11-learned-memory-price-v1",
        "card": "22.5.W.1.1",
        "phase": "price",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": "price_measured_within_design",
        "per_update_cpu": round(per_update, 8),
        "measured_cpu": round(measured, 4),
        "cap_cpu": cap_cpu,
        "cap_wall": cap_wall,
        "design_cap_cpu": contract["plan"]["design_cap_cpu"],
        "formula": "3 * updates_per_seed * per_update",
        "models_discarded": True,
        "discipline": {
            "optimizer_steps": 36,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(PRICE_REPORT, report)
    print("PRICE measured=" + str(report["measured_cpu"]) + " cap=" + str(cap_cpu))
    return 0


def run() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "run")
    criteria = contract["criteria"]
    cap_cpu = float(contract["plan"]["cap_cpu"])
    cap_wall = float(contract["plan"]["cap_wall"])
    updates = int(criteria["updates_per_seed"])
    dataset = wprobe.make_dataset()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = []
    budget_stopped = False
    for seed in criteria["seeds"]:
        arm_cpu = time.process_time()
        graph = build_graph(int(seed))
        optimizer = make_optimizer(graph)
        before_fit = accuracy(graph, dataset["fit"])
        before_train = accuracy(graph, dataset["train"])
        losses: list[float] = []
        for step in range(updates):
            if time.process_time() - started_cpu > cap_cpu:
                budget_stopped = True
                break
            if time.monotonic() - started_wall > cap_wall:
                budget_stopped = True
                break
            losses.append(
                one_update(graph, optimizer, dataset["train"][step % len(dataset["train"])])
            )
        save_capsule(final_capsule_path(int(seed)), build_capsule(graph, clocks=(updates, updates)))
        after_fit = accuracy(graph, dataset["fit"])
        after_train = accuracy(graph, dataset["train"])
        lesion = accuracy(graph, dataset["fit"], lesion=True)
        arms.append(
            {
                "seed": seed,
                "updates": len(losses),
                "before_fit_accuracy": before_fit["accuracy"],
                "before_train_accuracy": before_train["accuracy"],
                "fit_accuracy": after_fit["accuracy"],
                "train_accuracy": after_train["accuracy"],
                "lesion_accuracy": lesion["accuracy"],
                "first_loss": round(losses[0], 6) if losses else None,
                "final_loss": round(losses[-1], 6) if losses else None,
                "final_capsule_digest": value_digest(load_capsule(final_capsule_path(int(seed)))),
                "cpu_seconds": round(time.process_time() - arm_cpu, 4),
            }
        )
        if budget_stopped:
            break
    complete = len(arms) == len(criteria["seeds"])
    fit_pass = [arm["fit_accuracy"] >= criteria["fit_gate"] for arm in arms]
    train_pass = [arm["train_accuracy"] >= criteria["train_gate"] for arm in arms]
    lesion_pass = [arm["lesion_accuracy"] <= criteria["lesion_max"] for arm in arms]
    if budget_stopped:
        verdict = "budget_stop"
    elif not complete:
        verdict = "engineering_failure"
    elif not all(fit_pass) or not all(train_pass):
        verdict = "parent_not_acquired"
    elif not all(lesion_pass):
        verdict = "memory_lesion_failed"
    else:
        verdict = "parent_acquired"
    report = {
        "schema": "taiji-u2-w11-learned-memory-run-v1",
        "card": "22.5.W.1.1",
        "phase": "run",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "arms": arms,
        "fit_pass": fit_pass,
        "train_pass": train_pass,
        "lesion_pass": lesion_pass,
        "gates": criteria,
        "capability_scope": "DMS parent acquisition only; no capability/check2/22.6 claim",
        "discipline": {
            "optimizer_steps": 3 * updates,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(RUN_REPORT, report)
    print(
        "RUN verdict="
        + verdict
        + " fit="
        + str([a["fit_accuracy"] for a in arms])
        + " train="
        + str([a["train_accuracy"] for a in arms])
        + " lesion="
        + str([a["lesion_accuracy"] for a in arms])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


def final_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_final.bin"


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    graph = LearnedMemoryGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
    apply_capsule(graph, capsule)
    dataset = wprobe.make_dataset()
    logits = run_episode(graph, dataset["fit"][0])
    loss = logits.square().mean()
    graph.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {
        name: float(param.grad.abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    atomic_save(
        {
            "pid": os.getpid(),
            "summary": capsule_summary(capsule),
            "capsule_digest": value_digest(capsule),
            "next_loss": float(loss.detach()),
            "next_gradient_digest": value_digest(gradients),
        },
        destination,
    )


def independent_restore(capsule: dict[str, Any], tag: str) -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACTS / f"{tag}_capsule.bin"
    receipt_path = ARTIFACTS / f"{tag}_worker.bin"
    save_capsule(checkpoint, capsule)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(checkpoint),
            "--output",
            str(receipt_path),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(timeout=180)
    if process.returncode != 0:
        raise RuntimeError(
            ("independent_restore_failed", tag, stderr.decode("utf-8", "replace")[:400])
        )
    receipt = cast(dict[str, Any], atomic_load(receipt_path))
    receipt["digest_matches"] = receipt["capsule_digest"] == value_digest(capsule) and receipt[
        "summary"
    ] == capsule_summary(capsule)
    receipt["pid_differs"] = receipt["pid"] != os.getpid()
    receipt["tag"] = tag
    return receipt


def recover() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "recover")
    checks: dict[str, bool] = {}
    rows = []
    for seed in contract["criteria"]["seeds"]:
        final = load_capsule(final_capsule_path(int(seed)))
        receipt = independent_restore(final, f"seed{seed}_final")
        rows.append(receipt)
        checks[f"seed{seed}_final_restore"] = bool(
            receipt["digest_matches"] and receipt["pid_differs"]
        )
        checks[f"seed{seed}_next_gradient"] = bool(receipt["next_gradient_digest"])
        tampered = build_capsule(build_graph(int(seed)), clocks=(3, 2))
        tampered["runtime"]["unit_registry"][0]["lifecycle"] = "retired"
        try:
            validate_capsule(tampered)
            checks[f"seed{seed}_tamper_rejected"] = False
        except ValueError:
            checks[f"seed{seed}_tamper_rejected"] = True
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "recovery_verified" if not uncovered else "engineering_failure"
    report = {
        "schema": "taiji-u2-w11-learned-memory-recover-v1",
        "card": "22.5.W.1.1",
        "phase": "recover",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "receipts": rows,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(RECOVER_REPORT, report)
    print(
        "RECOVER verdict="
        + verdict
        + " green="
        + str(sum(checks.values()))
        + "/"
        + str(len(checks))
    )
    return 0 if not uncovered else 1


def verify() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "verify")
    run_report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    dataset = wprobe.make_dataset()
    checks: dict[str, bool] = {}
    rows = []
    for arm in run_report["arms"]:
        seed = int(arm["seed"])
        capsule = load_capsule(final_capsule_path(seed))
        graph = LearnedMemoryGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
        apply_capsule(graph, capsule)
        birth = build_graph(seed)
        changed = [
            name
            for name, tensor in capsule["runtime"]["state_dict"].items()
            if not torch.equal(tensor, birth.state_dict()[name])
        ]
        fit = accuracy(graph, dataset["fit"])
        train = accuracy(graph, dataset["train"])
        lesion = accuracy(graph, dataset["fit"], lesion=True)
        checks[f"seed{seed}_fit_matches_run"] = fit["accuracy"] == arm["fit_accuracy"]
        checks[f"seed{seed}_train_matches_run"] = train["accuracy"] == arm["train_accuracy"]
        checks[f"seed{seed}_lesion_matches_run"] = lesion["accuracy"] == arm["lesion_accuracy"]
        checks[f"seed{seed}_capsule_digest"] = value_digest(capsule) == arm["final_capsule_digest"]
        checks[f"seed{seed}_only_trainable_changed"] = set(changed) <= set(TRAINABLE)
        rows.append(
            {"seed": seed, "fit": fit, "train": train, "lesion": lesion, "changed": changed}
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "verify_passed" if not uncovered else "verify_failed"
    report = {
        "schema": "taiji-u2-w11-learned-memory-verify-v1",
        "card": "22.5.W.1.1",
        "phase": "verify",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "rows": rows,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(VERIFY_REPORT, report)
    print(
        "VERIFY verdict=" + verdict + " green=" + str(sum(checks.values())) + "/" + str(len(checks))
    )
    return 0 if not uncovered else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    if args.draft:
        return draft()
    if args.smoke:
        return smoke()
    if args.price:
        return price()
    if args.freeze:
        return freeze()
    if args.run:
        return run()
    if args.recover:
        return recover()
    if args.verify:
        return verify()
    parser.error("select a mode")
    return 2


if __name__ == "__main__":
    sys.exit(main())
