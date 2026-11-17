"""22.5.U.1 revision 2: learnable logit temperature + train-label attention supervision.

r1 located the read failure in the attention scale (trained logit range 0.025-0.203
nats against a ln(16) ceiling). Two design probes priced the fix before this runner
was written:

  * scale calibration: the range must come from the init, not from 384 Adam updates
    (required multiplier 84-513 against a reachable 3.07-3.21);
  * focus feasibility: a single W_q fitted on the labelled TRAIN scenes reaches
    focus on the unseen FIT scenes (answer mass 0.86/0.78/0.68, deranged control
    0.16/0.08/0.05), but only at a query norm 6.6-24.5x the non-saturating scale,
    where a random birth direction saturates softmax and corrupts the gradient
    direction. Hence the scale is a scalar ``log_tau`` OUTSIDE the query.

Supervision is finer than the 4-way task label (a 16-column mass distribution) and
is applied only to labelled train scenes. Its credit is restricted to W_q and
log_tau so the value-side representation stays trained by the task loss alone.

Modes: freeze (criteria digests + birth capsules), smoke (zero optimizer steps),
price (discarded model), run (authorized), recover, verify.
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
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u1r2_focus_feasibility_20261116 as ff
from scripts.training import probe_u1r2_scale_focus_20261116 as sf
from scripts.training import probe_u1r2_value_side_20261116 as vs
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
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
from taiji.u2_unit_input import encoding_contract
from taiji.u2_unit_temperature_readout import TemperaturePositionGraph

CONTRACT = ROOT / "plans/manifests/taiji_u2_u1r2_scale_supervision_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_verify_20261116.json"
RECOVER_REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_recover_20261116.json"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/u1r2_scale_supervision_20261116"
T = r1.T
QUERY_PARAMS = ("W_q", "log_tau")
FOCUS_WEIGHT = 1.0
LOG_CLAMP = 1e-12
DOSE_MARGIN = 2.0
CAP_MARGIN = 2.0
INTERVENTION_BATCH = 8
CHAIN_ATOL = 0.0


def sha_of(path: Path) -> str:
    return u1.sha_of(path)


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    return u1.write_report(path, payload)


def pinned_files() -> tuple[tuple[str, Path], ...]:
    return (
        ("runner", Path(__file__).resolve()),
        ("temperature_readout", ROOT / "taiji/u2_unit_temperature_readout.py"),
        ("position_readout", ROOT / "taiji/u2_unit_position_readout.py"),
        ("u1_runner", ROOT / "scripts/training/run_u1_unit_read_20261116.py"),
        ("unit_input", ROOT / "taiji/u2_unit_input.py"),
        ("neural_units", ROOT / "taiji/u2_neural_units.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
        ("episode", ROOT / "taiji/u2_task_episode.py"),
        ("scale_focus_probe", Path(sf.__file__).resolve()),
        ("focus_feasibility_probe", Path(ff.__file__).resolve()),
        ("value_side_probe", Path(vs.__file__).resolve()),
    )


def guards(contract: dict[str, Any], mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in pinned_files():
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if value_digest(encoding_contract()) != contract["encoding_contract_digest"]:
        raise ValueError("encoding_contract_digest_mismatch")
    criteria = contract["criteria"]
    supervision = criteria["supervision"]
    if float(supervision["weight"]) != FOCUS_WEIGHT:
        raise ValueError("supervision_weight_mismatch")
    if float(supervision["log_clamp"]) != LOG_CLAMP:
        raise ValueError("supervision_log_clamp_mismatch")
    if tuple(supervision["credit_restriction"]) != QUERY_PARAMS:
        raise ValueError("credit_restriction_mismatch")
    focus = criteria["focus_gate"]
    if float(focus["answer_mass_min"]) != sf.FOCUS_ANSWER_MASS_MIN:
        raise ValueError("focus_answer_mass_gate_mismatch")
    if float(focus["effective_columns_max"]) != sf.FOCUS_EFFECTIVE_COLUMNS_MAX:
        raise ValueError("focus_effective_columns_gate_mismatch")
    if TemperaturePositionGraph(UnitGraphSpec()).route != criteria["route"]:
        raise ValueError("route_mismatch")
    if int(criteria["updates_per_seed"]) != int(contract["plan"]["updates_per_seed"]):
        raise ValueError("updates_per_seed_disagrees_between_criteria_and_plan")
    query_lr = criteria["query_lr"]
    if float(query_lr["effective_lr"]) != float(criteria["learning_rate"]) * float(
        query_lr["multiplier"]
    ):
        raise ValueError("query_lr_effective_rate_mismatch")
    for name, path in (contract.get("design_reports") or {}).items():
        if (
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            != contract["design_report_sha256"][name]
        ):
            raise ValueError(f"design_report_mismatch:{name}")
    if mode != "freeze":
        for key, path in contract["start_bundles"].items():
            actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if actual != contract["start_bundle_sha256"][key]:
                raise ValueError(f"start_bundle_mismatch:{key}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract["plan"].get("cap_cpu") or not contract["plan"].get("cap_wall"):
            raise ValueError("missing_measured_cap")


def build_graph(seed: int, dtype: str = "float64") -> Any:
    torch.manual_seed(seed)
    return TemperaturePositionGraph(UnitGraphSpec(dtype=dtype, seed=seed))


def make_optimizer(graph: Any, contract: dict[str, Any], seed: int) -> torch.optim.Optimizer:
    criteria = contract["criteria"]
    lr = float(criteria["learning_rate"])
    query_lr = lr * float(criteria["query_lr"]["multiplier"])
    tau_lr = float(criteria["logit_temperature"]["learning_rate_by_seed"][str(seed)])
    value: list[torch.nn.Parameter] = []
    query: list[torch.nn.Parameter] = []
    temperature: list[torch.nn.Parameter] = []
    for name, param in graph.named_parameters():
        if name == "log_tau":
            temperature.append(param)
        elif name == "W_q":
            query.append(param)
        else:
            value.append(param)
    return torch.optim.Adam(
        [
            {"params": value, "lr": lr},
            {"params": query, "lr": query_lr},
            {"params": temperature, "lr": tau_lr},
        ]
    )


def focus_target(item: dict[str, Any]) -> torch.Tensor:
    return vs.forced_distribution(item, "oracle_soft")


def one_update(
    graph: Any, optimizer: torch.optim.Optimizer, item: dict[str, Any]
) -> tuple[float, float | None]:
    """Task loss on every scene; focus loss only on labelled scenes, query credit only."""
    data = u1.cast_item(item, graph.spec.torch_dtype)
    optimizer.zero_grad(set_to_none=True)
    out = graph(data["channels"], data["gates"], graph.initial_state())
    label = torch.tensor(data["label"], dtype=torch.long)
    task_loss = torch.nn.functional.cross_entropy(out["readout"].unsqueeze(0), label.unsqueeze(0))
    supervised = sf.labelled(item)
    task_loss.backward(retain_graph=supervised)
    if not supervised:
        optimizer.step()
        graph.units.project_parameters_()
        return float(task_loss.detach()), None
    task_grads = {
        name: param.grad.detach().clone()
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    target = focus_target(item).to(out["attention"].dtype)
    focus_loss = -(target * torch.log(out["attention"].clamp_min(LOG_CLAMP))).sum()
    optimizer.zero_grad(set_to_none=True)
    focus_loss.backward()
    for name, param in graph.named_parameters():
        base = task_grads.get(name)
        if name not in QUERY_PARAMS:
            param.grad = None if base is None else base.clone()
            continue
        extra = param.grad
        if extra is None:
            param.grad = None if base is None else base.clone()
        else:
            scaled = extra.detach().mul(FOCUS_WEIGHT)
            param.grad = scaled if base is None else base + scaled
    optimizer.step()
    graph.units.project_parameters_()
    return float(task_loss.detach()), float(focus_loss.detach())


def b_identity_checks(graph: Any) -> dict[str, bool]:
    return {
        "B_is_not_H_plus_head": not any(
            name.startswith("h_readout") or name in ("ell", "w_g")
            for name, _ in graph.named_parameters()
        ),
        "B_path_direct_units_position_temp": graph.route == "B_position_temp",
        "temperature_is_one_scalar": graph.log_tau.numel() == 1,
        "temperature_starts_at_zero": float(graph.log_tau.detach()) == 0.0,
    }


def chain_anchor(seed: int, item: dict[str, Any]) -> dict[str, Any]:
    """At log_tau=0 the subclass must reproduce the pinned parent bit-for-bit."""
    parent_graph = r1.build_graph(seed)
    child = build_graph(seed)
    data = u1.cast_item(item, child.spec.torch_dtype)
    with torch.no_grad():
        parent = parent_graph(data["channels"], data["gates"], parent_graph.initial_state())
        mine = child(data["channels"], data["gates"], child.initial_state())
    keys = ("readout", "attention", "attended")
    identical = {key: bool(torch.equal(parent[key], mine[key])) for key in keys}
    max_abs = max(float((parent[key] - mine[key]).abs().max()) for key in keys)
    state = parent_graph.initial_state()
    with torch.no_grad():
        parent_state = parent_graph(data["channels"], data["gates"], state)["next_state"]
        mine_state = child(data["channels"], data["gates"], child.initial_state())["next_state"]
    identical["next_state_z"] = bool(torch.equal(parent_state["z"], mine_state["z"]))
    identical["next_state_a"] = bool(torch.equal(parent_state["a"], mine_state["a"]))
    return {
        "bit_identical": all(identical.values()),
        "per_tensor": identical,
        "max_abs": max_abs,
        "atol": CHAIN_ATOL,
        "parent_route": parent_graph.route,
        "child_route": child.route,
    }


def gradient_coverage(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    coverage = r1.gradient_coverage(graph, item)
    credit = coverage["credit"]
    return {
        **coverage,
        "log_tau_credit": credit.get("log_tau", 0.0),
        "log_tau_nonzero": credit.get("log_tau", 0.0) > 0.0,
        "query_credit": {name: credit.get(name, 0.0) for name in QUERY_PARAMS},
    }


def supervision_gradient_rows(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Measure task-only, focus-only and combined gradients without stepping."""
    data = u1.cast_item(item, graph.spec.torch_dtype)
    label = torch.tensor(data["label"], dtype=torch.long)

    def grads(loss_kind: str) -> dict[str, torch.Tensor]:
        graph.zero_grad(set_to_none=True)
        out = graph(data["channels"], data["gates"], graph.initial_state())
        if loss_kind == "task":
            loss = torch.nn.functional.cross_entropy(
                out["readout"].unsqueeze(0), label.unsqueeze(0)
            )
        else:
            target = focus_target(item).to(out["attention"].dtype)
            loss = -(target * torch.log(out["attention"].clamp_min(LOG_CLAMP))).sum()
        loss.backward()
        return {
            name: param.grad.detach().clone()
            for name, param in graph.named_parameters()
            if param.grad is not None
        }

    task = grads("task")
    focus = grads("focus")
    graph.zero_grad(set_to_none=True)
    out = graph(data["channels"], data["gates"], graph.initial_state())
    task_loss = torch.nn.functional.cross_entropy(out["readout"].unsqueeze(0), label.unsqueeze(0))
    task_loss.backward(retain_graph=True)
    held = {
        name: param.grad.detach().clone()
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    target = focus_target(item).to(out["attention"].dtype)
    focus_loss = -(target * torch.log(out["attention"].clamp_min(LOG_CLAMP))).sum()
    graph.zero_grad(set_to_none=True)
    focus_loss.backward()
    combined: dict[str, torch.Tensor] = {}
    for name, param in graph.named_parameters():
        base = held.get(name)
        extra = param.grad
        if name not in QUERY_PARAMS:
            if base is not None:
                combined[name] = base.clone()
        elif extra is not None:
            scaled = extra.detach().mul(FOCUS_WEIGHT)
            combined[name] = scaled if base is None else base + scaled
        elif base is not None:
            combined[name] = base.clone()
    graph.zero_grad(set_to_none=True)
    expected = {
        name: (
            task[name]
            if name not in QUERY_PARAMS
            else task.get(name, torch.zeros_like(focus[name])) + FOCUS_WEIGHT * focus[name]
        )
        for name in set(task) | set(focus)
    }
    max_abs = max(float((combined[name] - expected[name]).abs().max()) for name in expected)
    off_query_focus = {
        name: float(value.abs().sum()) for name, value in focus.items() if name not in QUERY_PARAMS
    }
    return {
        "combined_matches_task_plus_restricted_focus": max_abs <= 1e-12,
        "max_abs": max_abs,
        "focus_credit_off_query": off_query_focus,
        "focus_credit_off_query_total": sum(off_query_focus.values()),
        "focus_query_credit": {
            name: float(focus[name].abs().sum()) for name in QUERY_PARAMS if name in focus
        },
        "task_query_credit": {
            name: float(task[name].abs().sum()) for name in QUERY_PARAMS if name in task
        },
        "parameters_in_combined": len(combined),
        "focus_parameters_not_in_task": sorted(set(focus) - set(task)),
    }


def initial_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_initial.bin"


def final_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_final.bin"


def freeze() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") == "frozen":
        raise ValueError("contract_already_frozen")
    feasibility = json.loads(ff.REPORT.read_text(encoding="utf-8"))
    calibration = json.loads(sf.REPORT.read_text(encoding="utf-8"))
    if feasibility["criteria_sha256"] != calibration["criteria_sha256"]:
        raise ValueError("design_reports_disagree_on_criteria")
    updates = int(contract["criteria"]["updates_per_seed"])
    rows = {int(row["seed"]): row for row in feasibility["seeds"]}
    targets: dict[str, float] = {}
    rates: dict[str, float] = {}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, str] = {}
    bundle_sha: dict[str, str] = {}
    for seed in contract["criteria"]["seeds"]:
        row = rows[int(seed)]
        graph = build_graph(int(seed))
        birth_norm = float(graph.W_q.detach().norm())
        fitted_norm = float(row["arms"]["lsq_train_targets"]["w_q_frobenius"])
        target = fitted_norm / birth_norm
        targets[str(seed)] = round(target, 6)
        rates[str(seed)] = round(DOSE_MARGIN * math.log(target) / updates, 9)
        path = initial_capsule_path(int(seed))
        save_capsule(path, build_capsule(graph))
        bundles[f"seed{seed}"] = str(path.relative_to(ROOT)).replace("\\", "/")
        bundle_sha[f"seed{seed}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract["source_pins"] = {name: sha_of(path) for name, path in pinned_files()}
    contract["design_reports"] = {
        "scale_calibration": str(sf.REPORT.relative_to(ROOT)).replace("\\", "/"),
        "focus_feasibility": str(ff.REPORT.relative_to(ROOT)).replace("\\", "/"),
    }
    contract["design_report_sha256"] = {
        name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for name, path in contract["design_reports"].items()
    }
    contract["criteria"]["logit_temperature"]["scale_target_by_seed"] = targets
    contract["criteria"]["logit_temperature"]["learning_rate_by_seed"] = rates
    contract["criteria"]["logit_temperature"]["dose_margin"] = DOSE_MARGIN
    contract["criteria_sha256"] = value_digest(contract["criteria"])
    contract["start_bundles"] = bundles
    contract["start_bundle_sha256"] = bundle_sha
    sf_checks = calibration["instrument_checks"]
    contract["design_anchor"] = {
        "feasibility_classification": feasibility["classification"],
        "feasibility_instrument": f"{feasibility['instrument_green']}/{feasibility['instrument_total']}",
        "calibration_verdict": calibration["verdict"],
        "calibration_instrument": f"{sum(sf_checks.values())}/{len(sf_checks)}",
        "design_reports_criteria_sha256": feasibility["criteria_sha256"],
        "design_reports_anchor": feasibility["anchored_contract"],
        "rejected_form": "larger W_q init at uniform learning rate",
        "rejection_evidence": "reports/taiji_u2_u1r2_focus_feasibility_20261116.json",
    }
    if PRICE_REPORT.exists():
        measured = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))
        if measured["criteria_sha256"] != contract["criteria_sha256"]:
            raise ValueError("price_report_criteria_stale")
        contract["plan"]["cap_cpu"] = measured["cap_cpu"]
        contract["plan"]["cap_wall"] = measured["cap_wall"]
        contract["plan"]["per_update_cpu"] = measured["per_update_cpu"]
        contract["plan"]["per_eval_cpu"] = measured["per_eval_cpu"]
        contract["plan"]["per_intervention_item_cpu"] = measured["per_intervention_item_cpu"]
        contract["plan"]["price_report"] = str(PRICE_REPORT.relative_to(ROOT)).replace("\\", "/")
        contract["design_feasibility"] = measured["design_feasibility"]
    contract["status"] = "frozen"
    CONTRACT.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "FREEZE ok seeds="
        + str(len(bundles))
        + " criteria_sha256="
        + contract["criteria_sha256"][:16]
    )
    return 0


def price() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "price")
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    seeds = contract["criteria"]["seeds"]
    train = episodes["train"]
    fit = episodes["fit"]
    samples: dict[str, list[float]] = {
        "update_epoch": [],
        "evaluation_face": [],
        "intervention_item": [],
    }
    steps = 0
    for repeat in range(3):
        seed = int(seeds[repeat % len(seeds)])
        graph = build_graph(seed)
        optimizer = make_optimizer(graph, contract, seed)
        t0 = time.process_time()
        for train_item in train:
            one_update(graph, optimizer, train_item)
        samples["update_epoch"].append((time.process_time() - t0) / len(train))
        steps += len(train)
        t0 = time.process_time()
        with torch.no_grad():
            for face_item in fit:
                r1.episode_logits(graph, face_item)
        samples["evaluation_face"].append((time.process_time() - t0) / len(fit))
        hd = u1.cast_item(fit[1], graph.spec.torch_dtype)
        t0 = time.process_time()
        with torch.no_grad():
            for index in range(INTERVENTION_BATCH):
                state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
                r1.intervention_readouts(graph, fit[index % len(fit)], state)
        samples["intervention_item"].append((time.process_time() - t0) / INTERVENTION_BATCH)
    dose = dose_measurement(contract, episodes)
    steps += dose["optimizer_steps"]
    rotation = rotation_requirement(contract, episodes)
    feasibility: dict[str, Any] = {}
    for seed in seeds:
        key = str(seed)
        measured = dose["per_seed"][key]["arms"]["r2_groups"][
            "projected_relative_displacement_over_run"
        ]
        needed = rotation[key]["needed_relative_displacement"]
        feasibility[key] = {
            "measured_projected_relative_displacement": measured,
            "needed_relative_displacement_upper_bound": needed,
            "rotation_reachable": measured >= needed,
            "scale_reachable": dose["per_seed"][key]["r2_groups_reaches_scale"],
        }
    per_update = max(samples["update_epoch"])
    per_eval = max(samples["evaluation_face"])
    per_intervention = max(samples["intervention_item"])
    updates = int(contract["criteria"]["updates_per_seed"])
    plain_evals = 2 * len(fit) + 2 * len(train) + len(train) + len(fit)
    intervention_items = len(fit)
    design_cap = float(contract["plan"]["design_cap_cpu"])
    measured_cpu = len(seeds) * (
        updates * per_update + plain_evals * per_eval + intervention_items * per_intervention
    )
    cap_cpu = measured_cpu * CAP_MARGIN
    cap_wall = cap_cpu * 2.0
    verdict = (
        "price_measured_within_design" if cap_cpu <= design_cap else "price_exceeds_design_cap"
    )
    report = {
        "schema": "taiji-u2-u1r2-price-v2",
        "card": "22.5.U.1r2",
        "phase": "price",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "samples": {key: [round(v, 8) for v in vals] for key, vals in samples.items()},
        "per_update_cpu": round(per_update, 8),
        "per_eval_cpu": round(per_eval, 8),
        "per_intervention_item_cpu": round(per_intervention, 8),
        "timing_method": (
            "batched: one full train epoch (64 updates, 48 of them supervised), one fit face "
            "(32 forwards) and 8 intervention items, each divided by its batch size"
        ),
        "timing_face": "birth graph (c=0,w=0); per_intervention_item includes the history forward",
        "seeds": len(seeds),
        "updates_per_seed": updates,
        "plain_item_evals_per_seed": plain_evals,
        "intervention_items_per_seed": intervention_items,
        "measured_cpu": round(measured_cpu, 4),
        "cap_margin": CAP_MARGIN,
        "cap_cpu": round(cap_cpu, 4),
        "cap_wall": round(cap_wall, 4),
        "design_cap_cpu": design_cap,
        "dose_measurement": dose,
        "rotation_requirement": rotation,
        "design_feasibility": feasibility,
        "low_tick_limit": (
            "process_time resolution ~0.0156s (P3.9/U.1); single-call timing quantized to 0.0 "
            "and produced a 1.5s cap lottery, so v2 times batches instead"
        ),
        "discipline": {
            "optimizer_steps": steps,
            "project_task_data_reads": 1,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "models_discarded": True,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(PRICE_REPORT, report)
    print(
        "PRICE verdict="
        + verdict
        + " cap_cpu="
        + str(report["cap_cpu"])
        + " per_update="
        + str(report["per_update_cpu"])
        + " steps="
        + str(steps)
        + " rotation_reachable="
        + str([row["rotation_reachable"] for row in feasibility.values()])
    )
    return 0


def rotation_requirement(
    contract: dict[str, Any], episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Angle the query must turn through to reach a focusing direction.

    The temperature supplies the norm, so only W_q's DIRECTION has to change. The
    minimum-norm least-squares solution is one point of a rank-deficient solution
    set, so the distance to it upper-bounds the rotation any focusing direction needs.
    """
    labelled_train = [item for item in episodes["train"] if sf.labelled(item)]
    rows: dict[str, Any] = {}
    for seed in contract["criteria"]["seeds"]:
        graph = build_graph(int(seed))
        if float(graph.log_tau.detach()) != 0.0:
            raise ValueError("rotation_requirement_needs_log_tau_zero")
        design: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for item in labelled_train:
            keys, question = ff.column_keys(graph, item)
            design.append(ff.design_row(graph, keys, question))
            targets.append(ff.target_logits(item))
        solved = ff.solve_focus(graph, design, targets)
        birth = graph.W_q.detach().reshape(-1).to(torch.float64)
        fitted = solved["w_q"].detach().reshape(-1).to(torch.float64)
        cosine = float((birth @ fitted) / (birth.norm() * fitted.norm()))
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        rows[str(seed)] = {
            "labelled_train_scenes": len(labelled_train),
            "birth_w_q_frobenius": float(birth.norm()),
            "fitted_w_q_frobenius": solved["w_q_frobenius"],
            "cosine_birth_to_fitted": cosine,
            "angle_degrees": math.degrees(angle),
            "needed_relative_displacement": 2.0 * math.sin(angle / 2.0),
            "solution_nullity": solved["unknowns"] - solved["rank"],
            "bound_note": "distance to the minimum-norm solution; the focusing set is larger",
        }
    return rows


def dose_measurement(
    contract: dict[str, Any], episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Measure Adam's actual per-step displacement on discarded models, per seed."""
    updates = int(contract["criteria"]["updates_per_seed"])
    lr = float(contract["criteria"]["learning_rate"])
    query_lr = lr * float(contract["criteria"]["query_lr"]["multiplier"])
    per_seed: dict[str, Any] = {}
    steps = 0
    for seed in contract["criteria"]["seeds"]:
        tau_lr = float(
            contract["criteria"]["logit_temperature"]["learning_rate_by_seed"][str(seed)]
        )
        arms: dict[str, Any] = {}
        for name, uniform in (("uniform_lr", True), ("r2_groups", False)):
            graph = build_graph(int(seed))
            optimizer: torch.optim.Optimizer
            if uniform:
                optimizer = torch.optim.Adam(graph.parameters(), lr=lr)
            else:
                optimizer = make_optimizer(graph, contract, int(seed))
            w_before = graph.W_q.detach().clone()
            tau_before = float(graph.log_tau.detach())
            for index in range(3):
                one_update(graph, optimizer, episodes["train"][index])
                steps += 1
            w_after = graph.W_q.detach().clone()
            relative = float((w_after - w_before).norm()) / float(w_before.norm())
            tau_delta = float(graph.log_tau.detach()) - tau_before
            arms[name] = {
                "steps": 3,
                "w_q_relative_displacement": relative,
                "w_q_relative_displacement_per_step": relative / 3,
                "log_tau_displacement": tau_delta,
                "log_tau_displacement_per_step": tau_delta / 3,
                "projected_relative_displacement_over_run": relative / 3 * updates,
                "projected_log_tau_over_run": tau_delta / 3 * updates,
                "log_tau_rate": lr if uniform else tau_lr,
                "w_q_rate": lr if uniform else query_lr,
            }
        target = float(contract["criteria"]["logit_temperature"]["scale_target_by_seed"][str(seed)])
        needed = math.log(target)
        per_seed[str(seed)] = {
            "arms": arms,
            "log_tau_needed": needed,
            "r2_groups_reaches_scale": arms["r2_groups"]["projected_log_tau_over_run"] >= needed,
            "uniform_lr_reaches_scale": arms["uniform_lr"]["projected_log_tau_over_run"] >= needed,
        }
    return {
        "optimizer_steps": steps,
        "per_seed": per_seed,
        "note": "3-step displacement extrapolated linearly; Adam's per-coordinate step is bounded by lr",
    }


def smoke() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "smoke")
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    from taiji.u2_unit_input import color_mass, derive_label, scene_entity

    oracle_ok = 0
    oracle_total = 0
    for split in ("train", "fit"):
        for example in data[split]:
            mass = color_mass(example["public"]["observations"]["rgb"])
            got = derive_label(mass, scene_entity(example["public"]["request"]))
            oracle_total += 1
            oracle_ok += int(got == example["target"].decode("utf-8"))
    train_ids = {item["id"] for item in episodes["train"]}
    fit_ids = {item["id"] for item in episodes["fit"]}
    supervised_ids = {item["id"] for item in episodes["train"] if sf.labelled(item)}
    checks: dict[str, bool] = {
        "encoding_oracle_consistent": oracle_ok == oracle_total,
        "supervision_ids_subset_of_train": supervised_ids <= train_ids,
        "supervision_ids_disjoint_from_fit": not (supervised_ids & fit_ids),
        "supervision_denominator_positive": len(supervised_ids) > 0,
        "labelled_fit_denominator_positive": sum(1 for i in episodes["fit"] if sf.labelled(i)) > 0,
        "train_fit_ids_disjoint": not (train_ids & fit_ids),
    }
    seeds_report = []
    for seed in contract["criteria"]["seeds"]:
        graph = build_graph(int(seed))
        item = episodes["fit"][0]
        history_item = episodes["fit"][1]
        hd = u1.cast_item(history_item, graph.spec.torch_dtype)
        with torch.no_grad():
            hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
        anchor = chain_anchor(int(seed), item)
        identity = b_identity_checks(graph)
        probe = u1.mechanism_probe(graph)
        summary = u1.intervention_summary(r1.intervention_readouts(probe, item, hist_state))
        birth = r1.neutral_birth_check(graph, item)
        coverage = gradient_coverage(graph, item)
        supervised_item = next(i for i in episodes["train"] if sf.labelled(i))
        restriction = supervision_gradient_rows(graph, supervised_item)
        birth_focus = sf.focus_stats(graph, episodes["fit"])
        oracle_focus = sf.focus_stats(graph, episodes["fit"], "oracle_soft")
        seed_checks = {
            **identity,
            "chain_bit_identical_at_log_tau_zero": anchor["bit_identical"],
            "without_edges_changes_readout": summary["without_edges_changes_readout"],
            "permute_source_changes_readout": summary["permute_source_changes_readout"],
            "history_changes_readout": summary["history_changes_readout"],
            "cut_h_bit_identical": summary["cut_h_bit_identical"],
            "h_perturb_bit_identical": summary["h_perturb_bit_identical"],
            "neutral_birth_exact": birth["neutral_birth_exact"],
            "new_unit_r_credit_path_wired": birth["new_unit_r_credit_path_wired"],
            "all_parameters_have_gradient": coverage["all_parameters_have_gradient"],
            "named_nonzero": coverage["named_nonzero"],
            "log_tau_nonzero": coverage["log_tau_nonzero"],
            "combined_matches_task_plus_restricted_focus": restriction[
                "combined_matches_task_plus_restricted_focus"
            ],
            "focus_credit_is_subset_of_task_credit": not restriction[
                "focus_parameters_not_in_task"
            ],
            "focus_ruler_false_at_birth": not birth_focus["focus_met"],
            "focus_ruler_true_under_oracle": oracle_focus["focus_met"],
        }
        checks.update({f"seed{seed}_{key}": value for key, value in seed_checks.items()})
        seeds_report.append(
            {
                "seed": seed,
                "chain_anchor": anchor,
                "identity": identity,
                "interventions": summary,
                "neutral_birth": birth,
                "gradient_coverage": {
                    "all": coverage["all_parameters_have_gradient"],
                    "named_nonzero": coverage["named_nonzero"],
                    "log_tau_credit": coverage["log_tau_credit"],
                    "query_credit": coverage["query_credit"],
                },
                "supervision_restriction": restriction,
                "focus_birth": birth_focus,
                "focus_oracle": oracle_focus,
                "checks": seed_checks,
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "smoke_passed" if not uncovered else "smoke_failed"
    report = {
        "schema": "taiji-u2-u1r2-smoke-v1",
        "card": "22.5.U.1r2",
        "phase": "smoke",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "encoding_oracle": {"ok": oracle_ok, "total": oracle_total},
        "supervision_face": {
            "train_scenes": len(train_ids),
            "fit_scenes": len(fit_ids),
            "supervised_scenes": len(supervised_ids),
            "labelled_fit_scenes": sum(1 for i in episodes["fit"] if sf.labelled(i)),
        },
        "seeds": seeds_report,
        "discipline": {
            "optimizer_steps": 0,
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
    write_report(SMOKE_REPORT, report)
    print(
        "SMOKE verdict="
        + verdict
        + " green="
        + str(sum(checks.values()))
        + "/"
        + str(len(checks))
        + " oracle="
        + str(oracle_ok)
        + "/"
        + str(oracle_total)
    )
    return 0 if not uncovered else 1


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
        save_capsule(initial_capsule_path(int(seed)), build_capsule(graph))
        optimizer = make_optimizer(graph, contract, int(seed))
        before_fit = r1.evaluate(graph, episodes["fit"])
        before_train = r1.evaluate(graph, episodes["train"])
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
        after_fit = r1.evaluate(graph, episodes["fit"])
        after_train = r1.evaluate(graph, episodes["train"])
        fit_focus = sf.focus_stats(graph, episodes["fit"])
        train_focus = sf.focus_stats(graph, episodes["train"])
        rgb_zero_hits = []
        fixed_q_hits = []
        intervention_flags: dict[str, list[bool]] = {
            "without_edges_changes_readout": [],
            "permute_source_changes_readout": [],
            "history_changes_readout": [],
            "cut_h_bit_identical": [],
        }
        with torch.no_grad():
            for index, item in enumerate(episodes["fit"]):
                history_item = episodes["fit"][(index + 1) % len(episodes["fit"])]
                hd = u1.cast_item(history_item, graph.spec.torch_dtype)
                hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
                readouts = r1.intervention_readouts(graph, item, hist_state)
                summary = u1.intervention_summary(readouts)
                full_pred = int(torch.argmax(readouts["full"]).item())
                rgb_zero_hits.append(
                    int(int(torch.argmax(readouts["rgb_zero"]).item()) == full_pred)
                )
                fixed_q_hits.append(
                    int(int(torch.argmax(readouts["fixed_question"]).item()) == full_pred)
                )
                for key in intervention_flags:
                    intervention_flags[key].append(bool(summary[key]))
        log_tau = float(graph.log_tau.detach())
        save_capsule(
            final_capsule_path(int(seed)),
            build_capsule(
                graph,
                optimizer_state={"state_entries": len(optimizer.state)},
                clocks=(updates, updates),
            ),
        )
        arms.append(
            {
                "seed": seed,
                "updates": len(task_losses),
                "supervised_updates": len(focus_losses),
                "before_fit_accuracy": before_fit["accuracy"],
                "before_train_accuracy": before_train["accuracy"],
                "fit_accuracy": after_fit["accuracy"],
                "train_accuracy": after_train["accuracy"],
                "fit_distinct": after_fit["distinct"],
                "train_distinct": after_train["distinct"],
                "fit_hits": after_fit["hits"],
                "rgb_zero_agreement": sum(rgb_zero_hits) / len(rgb_zero_hits),
                "fixed_question_agreement": sum(fixed_q_hits) / len(fixed_q_hits),
                "intervention_fraction": {
                    key: sum(vals) / len(vals) for key, vals in intervention_flags.items()
                },
                "fit_focus": fit_focus,
                "train_focus": train_focus,
                "log_tau_final": log_tau,
                "logit_scale_final": math.exp(log_tau),
                "logit_scale_target": float(
                    criteria["logit_temperature"]["scale_target_by_seed"][str(seed)]
                ),
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
    fit_gate = float(criteria["fit_gate"])
    train_gate = float(criteria["train_gate"])
    distinct_min = int(criteria["distinct_min"])
    control_max = float(criteria["control_agreement_max"])
    change_min = float(criteria["intervention_change_min_fraction"])
    read_pass = [
        arm["fit_accuracy"] >= fit_gate
        and arm["train_accuracy"] >= train_gate
        and arm["fit_distinct"] >= distinct_min
        and arm["rgb_zero_agreement"] < control_max
        and arm["fixed_question_agreement"] < control_max
        and arm["intervention_fraction"]["cut_h_bit_identical"] == 1.0
        and arm["intervention_fraction"]["without_edges_changes_readout"] >= change_min
        and arm["intervention_fraction"]["permute_source_changes_readout"] >= change_min
        and arm["intervention_fraction"]["history_changes_readout"] >= change_min
        for arm in arms
    ]
    focus_pass = [arm["fit_focus"]["focus_met"] for arm in arms]
    complete = len(arms) == len(criteria["seeds"])
    if budget_stopped:
        verdict, reasons = "budget_stop", ["cap_exceeded"]
    elif not complete:
        verdict, reasons = "engineering_failure", ["fewer_arms_than_seeds"]
    elif not all(focus_pass):
        verdict, reasons = "focus_not_met", ["one_or_more_seeds_below_focus_gate"]
    elif not all(read_pass):
        verdict, reasons = "focus_met_value_failed", ["focus_met_but_read_gates_below"]
    else:
        verdict, reasons = "read_prerequisite_passed", ["all_seeds_meet_focus_and_read_gates"]
    report = {
        "schema": "taiji-u2-u1r2-run-v1",
        "card": "22.5.U.1r2",
        "phase": "run",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "reasons": reasons,
        "arms": arms,
        "read_pass": read_pass,
        "focus_pass": focus_pass,
        "gates": {
            "fit": fit_gate,
            "train": train_gate,
            "distinct_min": distinct_min,
            "control_max": control_max,
            "intervention_change_min": change_min,
            "focus": criteria["focus_gate"],
        },
        "capability_scope": "unit-core B read prerequisite only; no correction/retention/parent/capability claim",
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
            "supervision_face": "train_labelled_only",
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
        + " focus="
        + str([a["fit_focus"]["answer_mass_mean"] for a in arms])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    spec = capsule["runtime"]["spec"]
    graph = TemperaturePositionGraph(UnitGraphSpec(**spec))
    apply_capsule(graph, capsule)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, graph.units.n_units)
    item = u1.cast_item(episodes["fit"][0], graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    loss = graph(item["channels"], item["gates"], graph.initial_state())["readout"].square().mean()
    loss.backward()
    gradients = {
        name: float(p.grad.abs().sum())
        for name, p in graph.named_parameters()
        if p.grad is not None
    }
    atomic_save(
        {
            "pid": os.getpid(),
            "summary": capsule_summary(capsule),
            "capsule_digest": value_digest(capsule),
            "next_loss": float(loss.detach()),
            "next_gradient_digest": value_digest(gradients),
            "log_tau_gradient": gradients.get("log_tau", 0.0),
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
        initial = load_capsule(initial_capsule_path(int(seed)))
        receipt = independent_restore(initial, f"seed{seed}_initial")
        rows.append(receipt)
        checks[f"seed{seed}_initial_restore"] = bool(
            receipt["digest_matches"] and receipt["pid_differs"]
        )
        final = load_capsule(final_capsule_path(int(seed)))
        final_receipt = independent_restore(final, f"seed{seed}_final")
        rows.append(final_receipt)
        checks[f"seed{seed}_final_restore"] = bool(
            final_receipt["digest_matches"] and final_receipt["pid_differs"]
        )
        checks[f"seed{seed}_final_next_gradient"] = bool(final_receipt["next_gradient_digest"])
        checks[f"seed{seed}_log_tau_gradient_present"] = final_receipt["log_tau_gradient"] != 0.0
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
        "schema": "taiji-u2-u1r2-recover-v1",
        "card": "22.5.U.1r2",
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
        graph = TemperaturePositionGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
        apply_capsule(graph, capsule)
        fit = r1.evaluate(graph, episodes["fit"])
        train = r1.evaluate(graph, episodes["train"])
        fit_focus = sf.focus_stats(graph, episodes["fit"])
        checks[f"seed{seed}_fit_matches_run"] = fit["accuracy"] == arm["fit_accuracy"]
        checks[f"seed{seed}_train_matches_run"] = train["accuracy"] == arm["train_accuracy"]
        checks[f"seed{seed}_focus_matches_run"] = (
            fit_focus["answer_mass_mean"] == arm["fit_focus"]["answer_mass_mean"]
            and fit_focus["focus_met"] == arm["fit_focus"]["focus_met"]
        )
        checks[f"seed{seed}_log_tau_matches_run"] = (
            float(graph.log_tau.detach()) == arm["log_tau_final"]
        )
        checks[f"seed{seed}_capsule_digest"] = value_digest(capsule) == arm["final_capsule_digest"]
        checks[f"seed{seed}_B_identity"] = all(b_identity_checks(graph).values())
        rows.append(
            {
                "seed": seed,
                "fit_accuracy": fit["accuracy"],
                "train_accuracy": train["accuracy"],
                "fit_distinct": fit["distinct"],
                "fit_answer_mass": fit_focus["answer_mass_mean"],
                "log_tau": float(graph.log_tau.detach()),
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "verify_passed" if not uncovered else "verify_failed"
    report = {
        "schema": "taiji-u2-u1r2-verify-v1",
        "card": "22.5.U.1r2",
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
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    if args.freeze:
        return freeze()
    if args.smoke:
        return smoke()
    if args.price:
        return price()
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
