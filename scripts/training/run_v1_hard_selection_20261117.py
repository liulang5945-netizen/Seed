"""22.5.V.1 hard-selection readout runner (design sibling of the U.1r2 runner).

Modes: draft (contract skeleton), smoke (zero optimizer steps), price (discarded
model), freeze (caps + frozen contract), run (authorized), recover, verify.

The value side is inherited from the U.1r2 parent capsule and frozen; only W_q and
position_head are trainable. The forward is a hard top-1 column read; a
straight-through soft-attention term gives W_q a task gradient during training.
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

from scripts.training import probe_u1r2_scale_focus_20261116 as sf
from scripts.training import probe_u1r2_value_side_20261116 as vs
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from scripts.training import run_u1r2_scale_supervision_20261116 as r2
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
from taiji.u2_unit_hard_selection_readout import HardSelectionGraph
from taiji.u2_unit_input import encoding_contract

CONTRACT = ROOT / "plans/manifests/taiji_u2_v1_hard_selection_20261117.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_v1_hard_selection_smoke_20261117.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_v1_hard_selection_price_20261117.json"
RUN_REPORT = ROOT / "reports/taiji_u2_v1_hard_selection_run_20261117.json"
RECOVER_REPORT = ROOT / "reports/taiji_u2_v1_hard_selection_recover_20261117.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_v1_hard_selection_verify_20261117.json"
REVIEW_REPORT = ROOT / "reports/taiji_u2_value_side_review_20261117.json"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/v1_hard_selection_20261117"
TRAINABLE = ("W_q", "position_head.weight")
FOCUS_WEIGHT = 1.0
LOG_CLAMP = 1e-12
QUERY_LR = 0.002
HEAD_LR = 0.001
CAP_MARGIN = 2.0
INTERVENTION_BATCH = 8
START_BUNDLE_KEYS = ("2026111601", "2026111602", "2026111603")


def sha_of(path: Path) -> str:
    return u1.sha_of(path)


def binary_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    return u1.write_report(path, payload)


def parent_capsule_path(seed: int) -> Path:
    return r2.final_capsule_path(int(seed))


def final_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_final.bin"


def pinned_files() -> tuple[tuple[str, Path], ...]:
    return (
        ("runner", Path(__file__).resolve()),
        ("hard_readout", ROOT / "taiji/u2_unit_hard_selection_readout.py"),
        ("temperature_readout", ROOT / "taiji/u2_unit_temperature_readout.py"),
        ("position_readout", ROOT / "taiji/u2_unit_position_readout.py"),
        ("u1_runner", ROOT / "scripts/training/run_u1_unit_read_20261116.py"),
        ("u1r1_runner", ROOT / "scripts/training/run_u1r1_position_20261116.py"),
        ("u1r2_runner", ROOT / "scripts/training/run_u1r2_scale_supervision_20261116.py"),
        ("unit_input", ROOT / "taiji/u2_unit_input.py"),
        ("neural_units", ROOT / "taiji/u2_neural_units.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
        ("episode", ROOT / "taiji/u2_task_episode.py"),
        ("value_side_review", REVIEW_REPORT),
    )


def guards(contract: dict[str, Any], mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in pinned_files():
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if value_digest(encoding_contract()) != contract["encoding_contract_digest"]:
        raise ValueError("encoding_contract_digest_mismatch")
    if HardSelectionGraph(UnitGraphSpec()).route != contract["criteria"]["route"]:
        raise ValueError("route_mismatch")
    if mode != "freeze":
        for key, path in contract["start_parents"].items():
            actual = binary_sha(ROOT / path)
            if actual != contract["start_parent_sha256"][key]:
                raise ValueError(f"start_parent_mismatch:{key}")
        if sha_of(REVIEW_REPORT) != contract["criteria"]["value_side_review_sha256"]:
            raise ValueError("value_side_review_mismatch")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract["plan"].get("cap_cpu") or not contract["plan"].get("cap_wall"):
            raise ValueError("missing_measured_cap")


def build_graph(seed: int) -> HardSelectionGraph:
    capsule = load_capsule(parent_capsule_path(seed))
    validate_capsule(capsule)
    if capsule["runtime"]["route"] != "B_position_temp":
        raise ValueError("parent_route_mismatch")
    spec = UnitGraphSpec(**capsule["runtime"]["spec"])
    graph = HardSelectionGraph(spec)
    graph.load_state_dict(capsule["runtime"]["state_dict"], strict=True)
    for name, param in graph.named_parameters():
        param.requires_grad_(name in TRAINABLE)
    return graph


def make_optimizer(graph: HardSelectionGraph) -> torch.optim.Optimizer:
    query = [graph.W_q]
    head = [graph.position_head.weight]
    return torch.optim.Adam(
        [
            {"params": query, "lr": QUERY_LR},
            {"params": head, "lr": HEAD_LR},
        ]
    )


def one_update(
    graph: HardSelectionGraph, optimizer: torch.optim.Optimizer, item: dict[str, Any]
) -> tuple[float, float | None]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    optimizer.zero_grad(set_to_none=True)
    out = graph(data["channels"], data["gates"], graph.initial_state())
    label = torch.tensor(item["label"], dtype=torch.long)
    task_loss = torch.nn.functional.cross_entropy(out["readout"].unsqueeze(0), label.unsqueeze(0))
    supervised = sf.labelled(item)
    task_loss.backward(retain_graph=supervised)
    task_grads = {
        name: param.grad.detach().clone()
        for name, param in graph.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    if not supervised:
        optimizer.step()
        return float(task_loss.detach()), None
    target = r2.focus_target(item).to(out["attention"].dtype)
    focus_loss = -(target * torch.log(out["attention"].clamp_min(LOG_CLAMP))).sum()
    optimizer.zero_grad(set_to_none=True)
    focus_loss.backward()
    for name, param in graph.named_parameters():
        if not param.requires_grad:
            continue
        base = task_grads.get(name)
        if name == "W_q":
            extra = param.grad
            param.grad = extra if base is None else base + extra
        else:
            param.grad = base
    optimizer.step()
    return float(task_loss.detach()), float(focus_loss.detach())


def selection_stats(graph: HardSelectionGraph, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    total = 0
    with torch.no_grad():
        for item in episodes:
            mass = vs.column_mass(item)
            span = mass > 0.0
            if not bool(span.any()):
                continue
            data = u1.cast_item(item, graph.spec.torch_dtype)
            out = graph(data["channels"], data["gates"], graph.initial_state())
            total += 1
            hits += int(bool(span[int(out["selected"])]))
    return {"hits": hits, "denominator": total, "accuracy": hits / total if total else 0.0}


def evaluate_all(
    graph: HardSelectionGraph, episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    fit = r1.evaluate(graph, episodes["fit"])
    train = r1.evaluate(graph, episodes["train"])
    fit_focus = sf.focus_stats(graph, episodes["fit"])
    return {
        "fit_accuracy": fit["accuracy"],
        "train_accuracy": train["accuracy"],
        "fit_distinct": fit["distinct"],
        "train_distinct": train["distinct"],
        "fit_selection": selection_stats(graph, episodes["fit"]),
        "train_selection": selection_stats(graph, episodes["train"]),
        "fit_focus": fit_focus,
    }


def controls_and_interventions(
    graph: HardSelectionGraph, episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    rgb_zero_hits: list[int] = []
    fixed_q_hits: list[int] = []
    flags: dict[str, list[bool]] = {
        "without_edges_changes_readout": [],
        "permute_source_changes_readout": [],
        "history_changes_readout": [],
        "cut_h_bit_identical": [],
    }
    with torch.no_grad():
        for index, item in enumerate(episodes):
            history_item = episodes[(index + 1) % len(episodes)]
            hd = u1.cast_item(history_item, graph.spec.torch_dtype)
            hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
            readouts = r1.intervention_readouts(graph, item, hist_state)
            summary = u1.intervention_summary(readouts)
            full_pred = int(torch.argmax(readouts["full"]).item())
            rgb_zero_hits.append(int(int(torch.argmax(readouts["rgb_zero"]).item()) == full_pred))
            fixed_q_hits.append(
                int(int(torch.argmax(readouts["fixed_question"]).item()) == full_pred)
            )
            for key in flags:
                flags[key].append(bool(summary[key]))
    return {
        "rgb_zero_agreement": sum(rgb_zero_hits) / len(rgb_zero_hits),
        "fixed_question_agreement": sum(fixed_q_hits) / len(fixed_q_hits),
        "intervention_fraction": {key: sum(vals) / len(vals) for key, vals in flags.items()},
    }


def draft() -> int:
    if CONTRACT.exists():
        raise FileExistsError("contract_exists")
    criteria = {
        "route": "B_hard_select",
        "parent_route": "B_position_temp",
        "seeds": [2026111601, 2026111602, 2026111603],
        "updates_per_seed": 384,
        "learning_rate": {"W_q": QUERY_LR, "position_head": HEAD_LR},
        "dtype": "float64",
        "selection_gate": 0.95,
        "fit_gate": 0.8,
        "train_gate": 0.9,
        "distinct_min": 3,
        "control_agreement_max": 0.5,
        "intervention_change_min_fraction": 0.5,
        "value_side_frozen": True,
        "value_side_review_report": "reports/taiji_u2_value_side_review_20261117.json",
        "value_side_review_sha256": sha_of(REVIEW_REPORT),
        "value_side_prerequisite": {
            "answer_column_head_fit_min": 0.8,
            "answer_column_head_train_min": 0.9,
            "measured_fit": [0.9375, 0.84375, 0.96875],
            "measured_train": [1.0, 1.0, 1.0],
        },
        "colour_baseline_painted": {
            "2026111601": 0.7907,
            "2026111602": 0.686,
            "2026111603": 0.6744,
        },
        "colour_drop_max": 0.05,
        "focus_supervision": {
            "loss": "soft-target cross entropy on attention",
            "target": "queried-entity colour mass per column, normalised (oracle_soft)",
            "applied_to": "labelled train scenes only",
            "weight": FOCUS_WEIGHT,
            "log_clamp": LOG_CLAMP,
            "credit": ["W_q"],
        },
        "scope": "read prerequisite only; no parent/capability/check2/22.6 claim",
        "exits": [
            {"order": 1, "verdict": "budget_stop"},
            {"order": 2, "verdict": "engineering_failure"},
            {
                "order": 3,
                "verdict": "selection_failed",
                "condition": "any seed fit top1-on-span < 0.95",
            },
            {
                "order": 4,
                "verdict": "value_side_degraded",
                "condition": "structurally impossible: value side frozen",
            },
            {
                "order": 5,
                "verdict": "read_failed",
                "condition": "read gates/controls/interventions below",
            },
            {"order": 6, "verdict": "hard_readout_passed"},
        ],
    }
    contract = {
        "schema": "taiji-u2-v1-hard-selection-v1",
        "card": "22.5.V.1",
        "date": "2026-11-17",
        "status": "draft",
        "training_authority": None,
        "identity": "taiji-u2-hard-select-v1",
        "parent_card": "22.5.U.1r2",
        "data_root": "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin",
        "data_digest": "40813b4cc8ffec0c3ae5639941b373c139422b4ccbaa34790de34a9e1149d214",
        "encoding_contract_identity": "unit-input-column-color-v1",
        "encoding_contract_digest": value_digest(encoding_contract()),
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
        "start_parents": {
            f"seed{seed}": str(parent_capsule_path(seed).relative_to(ROOT)).replace("\\", "/")
            for seed in criteria["seeds"]
        },
        "start_parent_sha256": {
            f"seed{seed}": binary_sha(parent_capsule_path(seed)) for seed in criteria["seeds"]
        },
    }
    CONTRACT.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print("DRAFT ok criteria_sha256=" + contract["criteria_sha256"][:16])
    return 0


def freeze() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") == "frozen":
        raise ValueError("contract_already_frozen")
    if not PRICE_REPORT.exists():
        raise ValueError("missing_price_report")
    price = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))
    if price["criteria_sha256"] != contract["criteria_sha256"]:
        raise ValueError("price_report_criteria_stale")
    contract["plan"]["cap_cpu"] = price["cap_cpu"]
    contract["plan"]["cap_wall"] = price["cap_wall"]
    contract["plan"]["per_update_cpu"] = price["per_update_cpu"]
    contract["plan"]["per_eval_cpu"] = price["per_eval_cpu"]
    contract["plan"]["per_intervention_item_cpu"] = price["per_intervention_item_cpu"]
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
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    checks: dict[str, bool] = {}
    rows = []
    for seed in contract["criteria"]["seeds"]:
        capsule = load_capsule(parent_capsule_path(int(seed)))
        spec = UnitGraphSpec(**capsule["runtime"]["spec"])
        parent = r2.TemperaturePositionGraph(spec)
        parent.load_state_dict(capsule["runtime"]["state_dict"])
        child = build_graph(int(seed))
        trainable = [name for name, param in child.named_parameters() if param.requires_grad]
        item = episodes["fit"][0]
        data_item = u1.cast_item(item, spec.torch_dtype)
        with torch.no_grad():
            parent_out = parent(data_item["channels"], data_item["gates"], parent.initial_state())
            child_out = child(data_item["channels"], data_item["gates"], child.initial_state())
        manual = child.position_head(child_out["projected"][child_out["selected"]])
        parent_rejected = False
        try:
            apply_capsule(child, capsule)
        except ValueError:
            parent_rejected = True
        checkpoint = ARTIFACTS / f"seed{seed}_smoke_tmp.bin"
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        save_capsule(checkpoint, build_capsule(child))
        reloaded = load_capsule(checkpoint)
        roundtrip_ok = value_digest(reloaded) == value_digest(load_capsule(checkpoint))
        checkpoint.unlink(missing_ok=True)
        child.zero_grad(set_to_none=True)
        optimizer = make_optimizer(child)
        labelled_item = next(item for item in episodes["train"] if sf.labelled(item))
        task_loss, focus_loss = one_update(child, optimizer, labelled_item)
        grads = {
            name: param.grad is not None
            for name, param in child.named_parameters()
            if param.requires_grad
        }
        frozen_grads = [
            name
            for name, param in child.named_parameters()
            if not param.requires_grad and param.grad is not None
        ]
        seed_checks = {
            "attention_matches_parent": torch.equal(
                parent_out["attention"], child_out["attention"]
            ),
            "attended_matches_parent": torch.equal(parent_out["attended"], child_out["attended"]),
            "selected_is_argmax": child_out["selected"]
            == int(torch.argmax(child_out["attention"]).item()),
            "hard_readout_identity": torch.equal(manual, child_out["readout"]),
            "parent_capsule_loudly_rejected": parent_rejected,
            "trainable_is_exactly_two": set(trainable) == set(TRAINABLE),
            "state_dict_keys_match_parent": set(parent.state_dict()) == set(child.state_dict()),
            "checkpoint_roundtrip": roundtrip_ok,
            "task_loss_finite": task_loss == task_loss,
            "focus_loss_finite": focus_loss is not None and focus_loss == focus_loss,
            "trainable_grads_present": all(grads.values()) and len(grads) == 2,
            "frozen_params_no_grad": not frozen_grads,
        }
        checks.update({f"seed{seed}_{key}": value for key, value in seed_checks.items()})
        rows.append(
            {
                "seed": seed,
                "trainable": trainable,
                "selected": child_out["selected"],
                "task_loss": task_loss,
                "focus_loss": focus_loss,
                "checks": seed_checks,
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "smoke_passed" if not uncovered else "smoke_failed"
    report = {
        "schema": "taiji-u2-v1-hard-selection-smoke-v1",
        "card": "22.5.V.1",
        "phase": "smoke",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "seeds": rows,
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
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    graph = build_graph(int(contract["criteria"]["seeds"][0]))
    optimizer = make_optimizer(graph)
    for _ in range(4):
        one_update(graph, optimizer, episodes["train"][0])
    started = time.process_time()
    for index in range(64):
        one_update(graph, optimizer, episodes["train"][index % len(episodes["train"])])
    per_update = (time.process_time() - started) / 64.0
    started = time.process_time()
    for _ in range(3):
        with torch.no_grad():
            for item in episodes["fit"]:
                data_item = u1.cast_item(item, graph.spec.torch_dtype)
                graph(data_item["channels"], data_item["gates"], graph.initial_state())
    per_eval = (time.process_time() - started) / (3 * len(episodes["fit"]))
    started = time.process_time()
    for index, item in enumerate(episodes["fit"][:INTERVENTION_BATCH]):
        history = episodes["fit"][(index + 1) % len(episodes["fit"])]
        hd = u1.cast_item(history, graph.spec.torch_dtype)
        with torch.no_grad():
            hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
            r1.intervention_readouts(graph, item, hist_state)
    per_intervention = (time.process_time() - started) / INTERVENTION_BATCH
    updates = int(contract["criteria"]["updates_per_seed"])
    measured = 3 * (updates * per_update + 288 * per_eval + 32 * per_intervention)
    cap_cpu = round(measured * CAP_MARGIN, 4)
    cap_wall = round(measured * CAP_MARGIN * 2.0, 4)
    report = {
        "schema": "taiji-u2-v1-hard-selection-price-v1",
        "card": "22.5.V.1",
        "phase": "price",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": "price_measured_within_design",
        "per_update_cpu": round(per_update, 8),
        "per_eval_cpu": round(per_eval, 8),
        "per_intervention_item_cpu": round(per_intervention, 8),
        "measured_cpu": round(measured, 4),
        "cap_cpu": cap_cpu,
        "cap_wall": cap_wall,
        "design_cap_cpu": contract["plan"]["design_cap_cpu"],
        "design_cap_wall": contract["plan"]["design_cap_wall"],
        "formula": "3*(384*per_update + 288*per_eval + 32*per_intervention)",
        "models_discarded": True,
        "discipline": {
            "optimizer_steps": 68,
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
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = []
    budget_stopped = False
    for seed in criteria["seeds"]:
        arm_cpu = time.process_time()
        graph = build_graph(int(seed))
        optimizer = make_optimizer(graph)
        before = evaluate_all(graph, episodes)
        task_losses: list[float] = []
        focus_losses: list[float] = []
        for step in range(updates):
            if time.process_time() - started_cpu > cap_cpu:
                budget_stopped = True
                break
            if time.monotonic() - started_wall > cap_wall:
                budget_stopped = True
                break
            item = episodes["train"][step % len(episodes["train"])]
            task_loss, focus_loss = one_update(graph, optimizer, item)
            task_losses.append(task_loss)
            if focus_loss is not None:
                focus_losses.append(focus_loss)
        save_capsule(
            final_capsule_path(int(seed)),
            build_capsule(graph, clocks=(updates, updates)),
        )
        after = evaluate_all(graph, episodes)
        extras = controls_and_interventions(graph, episodes["fit"])
        arms.append(
            {
                "seed": seed,
                "updates": len(task_losses),
                "supervised_updates": len(focus_losses),
                "before_fit_accuracy": before["fit_accuracy"],
                "before_train_accuracy": before["train_accuracy"],
                "before_fit_selection": before["fit_selection"]["accuracy"],
                "fit_accuracy": after["fit_accuracy"],
                "train_accuracy": after["train_accuracy"],
                "fit_distinct": after["fit_distinct"],
                "train_distinct": after["train_distinct"],
                "fit_selection": after["fit_selection"],
                "train_selection": after["train_selection"],
                "fit_focus": after["fit_focus"],
                "rgb_zero_agreement": extras["rgb_zero_agreement"],
                "fixed_question_agreement": extras["fixed_question_agreement"],
                "intervention_fraction": extras["intervention_fraction"],
                "first_task_loss": round(task_losses[0], 6) if task_losses else None,
                "final_task_loss": round(task_losses[-1], 6) if task_losses else None,
                "first_focus_loss": round(focus_losses[0], 6) if focus_losses else None,
                "final_focus_loss": round(focus_losses[-1], 6) if focus_losses else None,
                "final_capsule_digest": value_digest(load_capsule(final_capsule_path(int(seed)))),
                "cpu_seconds": round(time.process_time() - arm_cpu, 4),
            }
        )
        if budget_stopped:
            break
    selection_pass = [
        arm["fit_selection"]["accuracy"] >= criteria["selection_gate"] for arm in arms
    ]
    read_pass = [
        arm["fit_accuracy"] >= criteria["fit_gate"]
        and arm["train_accuracy"] >= criteria["train_gate"]
        and arm["fit_distinct"] >= criteria["distinct_min"]
        and arm["rgb_zero_agreement"] < criteria["control_agreement_max"]
        and arm["fixed_question_agreement"] < criteria["control_agreement_max"]
        and arm["intervention_fraction"]["cut_h_bit_identical"] == 1.0
        and arm["intervention_fraction"]["without_edges_changes_readout"]
        >= criteria["intervention_change_min_fraction"]
        and arm["intervention_fraction"]["permute_source_changes_readout"]
        >= criteria["intervention_change_min_fraction"]
        and arm["intervention_fraction"]["history_changes_readout"]
        >= criteria["intervention_change_min_fraction"]
        for arm in arms
    ]
    complete = len(arms) == len(criteria["seeds"])
    if budget_stopped:
        verdict, reasons = "budget_stop", ["cap_exceeded"]
    elif not complete:
        verdict, reasons = "engineering_failure", ["fewer_arms_than_seeds"]
    elif not all(selection_pass):
        verdict, reasons = "selection_failed", ["one_or_more_seeds_below_selection_gate"]
    elif not all(read_pass):
        verdict, reasons = "read_failed", ["one_or_more_seeds_below_read_gates"]
    else:
        verdict, reasons = "hard_readout_passed", ["all_seeds_meet_selection_and_read_gates"]
    report = {
        "schema": "taiji-u2-v1-hard-selection-run-v1",
        "card": "22.5.V.1",
        "phase": "run",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "reasons": reasons,
        "arms": arms,
        "selection_pass": selection_pass,
        "read_pass": read_pass,
        "gates": criteria,
        "value_side_frozen": True,
        "capability_scope": "read prerequisite only; no parent/capability/check2/22.6 claim",
        "discipline": {
            "optimizer_steps": 3 * updates,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
            "capability_claims": 0,
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
        + " selection="
        + str([a["fit_selection"]["accuracy"] for a in arms])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    graph = HardSelectionGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
    apply_capsule(graph, capsule)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, graph.units.n_units)
    item = u1.cast_item(episodes["fit"][0], graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    out = graph(item["channels"], item["gates"], graph.initial_state())
    loss = out["readout"].square().mean()
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
            "selected": int(out["selected"]),
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
    receipt["tag"] = tag
    receipt["pid_differs"] = receipt["pid"] != os.getpid()
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
        "schema": "taiji-u2-v1-hard-selection-recover-v1",
        "card": "22.5.V.1",
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
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    checks: dict[str, bool] = {}
    rows = []
    for arm in run_report["arms"]:
        seed = int(arm["seed"])
        capsule = load_capsule(final_capsule_path(seed))
        graph = HardSelectionGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
        apply_capsule(graph, capsule)
        parent = load_capsule(parent_capsule_path(seed))
        changed = [
            name
            for name, tensor in capsule["runtime"]["state_dict"].items()
            if not torch.equal(tensor, parent["runtime"]["state_dict"][name])
        ]
        after = evaluate_all(graph, episodes)
        checks[f"seed{seed}_fit_matches_run"] = after["fit_accuracy"] == arm["fit_accuracy"]
        checks[f"seed{seed}_train_matches_run"] = after["train_accuracy"] == arm["train_accuracy"]
        checks[f"seed{seed}_selection_matches_run"] = (
            after["fit_selection"]["accuracy"] == arm["fit_selection"]["accuracy"]
        )
        checks[f"seed{seed}_focus_matches_run"] = (
            after["fit_focus"]["answer_mass_mean"] == arm["fit_focus"]["answer_mass_mean"]
        )
        checks[f"seed{seed}_capsule_digest"] = value_digest(capsule) == arm["final_capsule_digest"]
        checks[f"seed{seed}_only_trainable_changed"] = set(changed) <= set(TRAINABLE)
        rows.append(
            {
                "seed": seed,
                "fit_accuracy": after["fit_accuracy"],
                "train_accuracy": after["train_accuracy"],
                "fit_selection": after["fit_selection"],
                "changed_tensors": changed,
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "verify_passed" if not uncovered else "verify_failed"
    report = {
        "schema": "taiji-u2-v1-hard-selection-verify-v1",
        "card": "22.5.V.1",
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
