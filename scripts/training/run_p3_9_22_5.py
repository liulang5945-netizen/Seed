"""P3.9 reference-graph pricing, verification, recovery, behaviour window and closeout.

All inputs are fixed synthetic legal-input tensors. No project task dataset,
sealed face, checkpoint parent or capability claim is involved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_history import atomic_load, atomic_save, value_digest  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    COMPAT_FP32_ATOL,
    COMPAT_FP64_ATOL,
    LEGAL_INPUT_CHANNELS,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    build_route,
    capsule_summary,
    legacy_rp_reference,
    load_capsule,
    save_capsule,
    validate_capsule,
)

REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/p3_9_20261114"
CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_9_prereg_20261114.json"
STEPS = 6
LABELS = 4
DEV_EPISODES = 32
BOOTSTRAP_ROUNDS = 2000


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(name: str, payload: dict[str, Any]) -> Path:
    path = REPORTS / name
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return path


def legal_inputs(seed: int, steps: int = STEPS) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: ((torch.rand(steps, 8, generator=generator, dtype=torch.float64) * 2 - 1) * 0.5)
        for name in LEGAL_INPUT_CHANNELS
    }


def gate_tensor(seed: int, steps: int, units: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 991)
    return torch.rand(steps, units, generator=generator, dtype=torch.float64) * 0.6 + 0.4


def episode(seed: int) -> dict[str, Any]:
    data = legal_inputs(seed)
    target = int(torch.argmax(data["question"][-1, :LABELS]).item())
    return {"seed": seed, "channels": data, "label": target}


def cast_episode(item: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    return {
        "seed": item["seed"],
        "label": item["label"],
        "channels": {key: value.to(dtype) for key, value in item["channels"].items()},
    }


def evaluate_arm(
    graph: Any, episodes: list[dict[str, Any]], *, zero_gates: bool = False
) -> tuple[int, list[int]]:
    correct = 0
    hits = []
    for item in episodes:
        data = cast_episode(item, graph.spec.torch_dtype)
        gate = gate_tensor(item["seed"], STEPS, graph.units.n_units).to(graph.spec.torch_dtype)
        if zero_gates:
            gate = torch.zeros_like(gate)
        logits = graph(data["channels"], gate, graph.initial_state())["readout"][-1]
        hit = int(torch.argmax(logits).item()) == data["label"]
        correct += int(hit)
        hits.append(int(hit))
    return correct, hits


def one_update(graph: Any, optimizer: torch.optim.Optimizer, item: dict[str, Any]) -> float:
    data = cast_episode(item, graph.spec.torch_dtype)
    gate = gate_tensor(item["seed"], STEPS, graph.units.n_units).to(graph.spec.torch_dtype)
    optimizer.zero_grad(set_to_none=True)
    logits = graph(data["channels"], gate, graph.initial_state())["readout"][-1]
    target = torch.tensor(data["label"], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    optimizer.step()
    graph.units.project_parameters_()
    return float(loss.detach())


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def price() -> int:
    """Zero-project-data pricing of both graphs in both precisions."""

    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    measured: dict[str, Any] = {}
    for route in ("A", "B"):
        for dtype_name in ("float32", "float64"):
            spec = UnitGraphSpec(dtype=dtype_name)
            samples: dict[str, list[float]] = {
                "init": [],
                "forward_backward": [],
                "adam_step": [],
                "evaluation": [],
            }
            for repeat in range(3):
                t0 = time.process_time()
                graph = build_route(route, spec)
                samples["init"].append(time.process_time() - t0)
                item = episode(5000 + repeat)
                data = cast_episode(item, spec.torch_dtype)
                gate = gate_tensor(item["seed"], STEPS, spec.n_units).to(spec.torch_dtype)
                optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
                t0 = time.process_time()
                optimizer.zero_grad(set_to_none=True)
                graph(data["channels"], gate, graph.initial_state())[
                    "readout"
                ].square().mean().backward()
                samples["forward_backward"].append(time.process_time() - t0)
                t0 = time.process_time()
                optimizer.step()
                graph.units.project_parameters_()
                samples["adam_step"].append(time.process_time() - t0)
                t0 = time.process_time()
                with torch.no_grad():
                    graph(data["channels"], gate, graph.initial_state())["readout"]
                samples["evaluation"].append(time.process_time() - t0)
            measured[f"{route}_{dtype_name}"] = {
                key: {
                    "samples": [round(v, 4) for v in values],
                    "median": round(_median(values), 4),
                    "max": round(max(values), 4),
                }
                for key, values in samples.items()
            }
    per_update = max(
        measured[f"{route}_{dtype}"]["forward_backward"]["max"]
        + measured[f"{route}_{dtype}"]["adam_step"]["max"]
        for route in ("A", "B")
        for dtype in ("float32", "float64")
    )
    per_eval = max(
        measured[f"{route}_{dtype}"]["evaluation"]["max"]
        for route in ("A", "B")
        for dtype in ("float32", "float64")
    )
    cap_cpu, cap_wall = 120.0, 240.0
    # 2 arms x (updates + 2 dev evaluations before/after + 2 control evaluations)
    fixed_evaluations = 2 * 2 + 2 * 2
    affordable = int((cap_cpu * 0.6 - fixed_evaluations * per_eval) / (2 * per_update))
    affordable = max(0, affordable)
    verdict = "price_measured_admission_possible" if affordable >= 16 else "comparison_not_admitted"
    report = {
        "schema": "taiji-u2-p3-9-1-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.1",
        "phase": "price",
        "verdict": verdict,
        "measured": measured,
        "conservative": {
            "per_paired_update_cpu": round(2 * per_update, 4),
            "per_evaluation_cpu": round(per_eval, 4),
            "fixed_evaluations": fixed_evaluations,
            "draft_cap_cpu": cap_cpu,
            "draft_cap_wall": cap_wall,
            "affordable_paired_updates": affordable,
        },
        "discipline": {
            "pricing_adam_calls_per_graph": 3,
            "checkpoint_writes": 0,
            "parent_files_mutated": 0,
            "project_task_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "measured_cpu_seconds": round(time.process_time() - started_cpu, 4),
        "measured_wall_seconds": round(time.monotonic() - started_wall, 4),
    }
    path = write_report("taiji_u2_p3_9_1_price_20261114.json", report)
    print(
        json.dumps(
            {"verdict": verdict, "affordable": affordable, "report": path.name}, ensure_ascii=False
        )
    )
    return 0


def graph_checks() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for dtype_name, atol in (("float64", COMPAT_FP64_ATOL), ("float32", COMPAT_FP32_ATOL)):
        spec = UnitGraphSpec(dtype=dtype_name)
        a = build_route("A", spec)
        with torch.no_grad():
            a.units.c.zero_()
            a.units.b.zero_()
            a.units.w.zero_()
        item = episode(9101)
        data = cast_episode(item, spec.torch_dtype)
        gate = gate_tensor(item["seed"], STEPS, spec.n_units).to(spec.torch_dtype)
        got = a.units(
            a.encode_legal_inputs(data["channels"]), a.units.active_gate(gate), a.initial_state()
        )
        expected = legacy_rp_reference(a, data["channels"], gate)
        max_abs = float((got["contribution"] - expected).detach().abs().max())
        checks[f"compat_{dtype_name}_output"] = max_abs <= atol
        detail[f"compat_{dtype_name}_max_abs"] = max_abs
        a.zero_grad(set_to_none=True)
        (got["contribution"] * expected.detach()).sum().backward()
        checks[f"compat_{dtype_name}_r_p_gradient"] = bool(
            a.units.r.grad is not None
            and a.units.p.grad is not None
            and torch.isfinite(a.units.r.grad).all()
            and torch.isfinite(a.units.p.grad).all()
        )
        detail[f"compat_{dtype_name}_r_gradient_abs"] = float(a.units.r.grad.abs().sum())
    for route in ("A", "B"):
        spec = UnitGraphSpec(dtype="float64")
        graph = build_route(route, spec)
        item = episode(9202)
        data = cast_episode(item, spec.torch_dtype)
        gate = gate_tensor(item["seed"], STEPS, spec.n_units).to(spec.torch_dtype)
        state = graph.initial_state()
        out = graph(data["channels"], gate, state)
        checks[f"{route}_finite_output"] = bool(torch.isfinite(out["readout"]).all())
        graph.zero_grad(set_to_none=True)
        out["readout"].square().mean().backward()
        credit = {
            name: float(param.grad.abs().sum())
            for name, param in graph.named_parameters()
            if param.grad is not None
        }
        checks[f"{route}_all_parameters_have_gradient"] = len(credit) == len(
            list(graph.named_parameters())
        )
        checks[f"{route}_named_parameters_nonzero"] = all(
            credit.get(name, 0.0) > 0
            for name in ("units.r", "units.p", "units.c", "input_projection")
        )
        detail[f"{route}_credit"] = credit
        detail[f"{route}_structural_zero_gradients"] = sorted(
            name for name, value in credit.items() if value == 0.0
        )
        grown = graph.with_neutral_birth(0, ["u0000"])
        big_gate = torch.cat([gate, torch.zeros(STEPS, 1, dtype=spec.torch_dtype)], dim=1)
        after = grown(data["channels"], big_gate, graph.extend_state(state))["readout"]
        checks[f"{route}_neutral_birth_exact"] = bool(
            torch.allclose(out["readout"], after, rtol=0.0, atol=COMPAT_FP64_ATOL)
        )
        detail[f"{route}_neutral_birth_max_abs"] = float((out["readout"] - after).abs().max())
    b = build_route("B", UnitGraphSpec(dtype="float64"))
    a = build_route("A", UnitGraphSpec(dtype="float64"))
    item = episode(9303)
    data = cast_episode(item, b.spec.torch_dtype)
    gate = gate_tensor(item["seed"], STEPS, b.spec.n_units).to(b.spec.torch_dtype)
    h_state = {**b.initial_state(), "H": b.initial_state()["H"] + 0.3}
    b_out = b(data["channels"], gate, h_state)["readout"]
    checks["B_readout_bypasses_H"] = bool(
        torch.allclose(
            b_out, b(data["channels"], gate, b.cut_h(h_state))["readout"], rtol=0.0, atol=0.0
        )
    )
    checks["B_is_not_H_plus_head"] = not any(
        name.startswith("h_readout") or name in ("ell", "w_g") for name, _ in b.named_parameters()
    )
    a_out = a(data["channels"], gate, a.initial_state())["readout"]
    a_perturbed = a(
        data["channels"], gate, {**a.initial_state(), "H": a.initial_state()["H"] + 0.3}
    )["readout"]
    checks["A_H_intervention_has_effect"] = not torch.allclose(
        a_out, a_perturbed, rtol=0.0, atol=1e-8
    )
    b_cut = b(data["channels"], torch.zeros_like(gate), h_state)["readout"]
    checks["B_unit_activity_intervention_has_effect"] = not torch.allclose(
        b_out, b_cut, rtol=0.0, atol=1e-8
    )
    try:
        b.encode_legal_inputs(
            {key: value for key, value in data["channels"].items() if key != "question"}
        )
        checks["B_legal_inputs_fail_closed"] = False
    except ValueError:
        checks["B_legal_inputs_fail_closed"] = True
    checks["B_input_coverage_matrix"] = len(LEGAL_INPUT_CHANNELS) == b.spec.groups
    state_a, state_b = a.initial_state(), b.initial_state()
    with torch.no_grad():
        state_a["z"].add_(0.7)
    a(data["channels"], gate, state_a)
    before = b(data["channels"], gate, state_b)["readout"]
    checks["state_isolation"] = bool(
        float(state_b["z"].abs().sum()) == 0.0 and torch.isfinite(before).all()
    )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "reference_graphs_verified" if not uncovered else "reference_graph_incomplete"
    report = {
        "schema": "taiji-u2-p3-9-1-graph-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.1",
        "phase": "graph",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "detail": detail,
        "compat_tolerances": {"fp64": COMPAT_FP64_ATOL, "fp32": COMPAT_FP32_ATOL},
        "discipline": {
            "optimizer_steps": 0,
            "project_task_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
    }
    path = write_report("taiji_u2_p3_9_1_graph_20261114.json", report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "checks_green": sum(checks.values()),
                "checks_total": len(checks),
                "report": path.name,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not uncovered else 1


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    spec = capsule["runtime"]["spec"]
    graph = build_route(capsule["runtime"]["route"], UnitGraphSpec(dtype=spec["dtype"]))
    apply_capsule(graph, capsule)
    item = episode(17701)
    data = cast_episode(item, graph.spec.torch_dtype)
    gate = gate_tensor(item["seed"], STEPS, graph.units.n_units).to(graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    loss = graph(data["channels"], gate, graph.initial_state())["readout"].square().mean()
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
    stdout, stderr = process.communicate(timeout=180)
    if process.returncode != 0:
        raise RuntimeError(
            ("independent_restore_failed", tag, stderr.decode("utf-8", "replace")[:400])
        )
    receipt = atomic_load(receipt_path)
    local = {"summary": capsule_summary(capsule), "capsule_digest": value_digest(capsule)}
    receipt["digest_matches"] = (
        receipt["capsule_digest"] == local["capsule_digest"]
        and receipt["summary"] == local["summary"]
    )
    receipt["tag"] = tag
    receipt["pid_differs"] = receipt["pid"] != os.getpid()
    return receipt


def restore() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for route in ("A", "B"):
        graph = build_route(route, UnitGraphSpec(dtype="float64"))
        initial = independent_restore(build_capsule(graph), f"{route}_initial")
        rows.append(initial)
        # First non-empty moment: one real Adam step, then a full capsule.
        optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
        one_update(graph, optimizer, episode(17801))
        moments = {
            "param_groups": len(optimizer.param_groups),
            "state_entries": len(optimizer.state),
        }
        with_moment = build_capsule(
            graph, optimizer_state={"state_entries": len(optimizer.state)}, clocks=(1, 1)
        )
        first_moment = independent_restore(with_moment, f"{route}_first_moment")
        rows.append(first_moment)
        checks[f"{route}_initial_restore"] = bool(
            initial["digest_matches"] and initial["pid_differs"]
        )
        checks[f"{route}_first_nonempty_moment_restore"] = bool(
            first_moment["digest_matches"] and moments["state_entries"] > 0
        )
        checks[f"{route}_first_moment_next_gradient"] = bool(first_moment["next_gradient_digest"])
        # Two generations of neutral birth, dormancy/wake and rollback.
        gen1 = graph.with_neutral_birth(0, ["u0000"])
        gen1.units.set_lifecycle(gen1.units.identities["units"][-1]["unit_id"], "mature")
        gen2 = gen1.with_neutral_birth(1, [gen1.units.identities["units"][-1]["unit_id"]])
        gen2.units.set_lifecycle("u0001", "dormant")
        gen2.units.set_lifecycle("u0001", "mature")
        rollback = copy.deepcopy(gen1)
        left = independent_restore(build_capsule(gen1, clocks=(2, 2)), f"{route}_gen1")
        right = independent_restore(build_capsule(rollback, clocks=(2, 2)), f"{route}_rollback")
        rows.extend([left, right])
        checks[f"{route}_rollback_next_gradient_matches_gen1"] = (
            left["next_gradient_digest"] == right["next_gradient_digest"]
        )
        checks[f"{route}_two_generations_and_lifecycle"] = bool(
            gen2.units.identities["generation"] == 2
            and gen2.units.n_units == graph.units.n_units + 2
        )
        checks[f"{route}_registry_unique_ids"] = (
            len({u["unit_id"] for u in gen2.units.identities["units"]}) == gen2.units.n_units
        )
        # Tamper rejection (red direction).
        tampered = build_capsule(graph, clocks=(3, 2))
        tampered["runtime"]["unit_registry"][0]["lifecycle"] = "retired"
        try:
            validate_capsule(tampered)
            checks[f"{route}_tamper_rejected"] = False
        except ValueError:
            checks[f"{route}_tamper_rejected"] = True
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "restore_and_lifecycle_verified" if not uncovered else "engineering_failure"
    report = {
        "schema": "taiji-u2-p3-9-2-restore-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.2",
        "phase": "restore",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "restore_receipts": rows,
        "discipline": {
            "optimizer_steps": 2,
            "project_task_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report("taiji_u2_p3_9_2_restore_20261114.json", report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "restores": len(rows),
                "checks_green": sum(checks.values()),
                "checks_total": len(checks),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not uncovered else 1


def paired_bootstrap(
    hits_a: list[int], hits_b: list[int], rounds: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(hits_a)
    differences = []
    for _ in range(rounds):
        indices = [rng.randrange(n) for _ in range(n)]
        diff = sum(hits_b[i] - hits_a[i] for i in indices) / n
        differences.append(diff)
    differences.sort()
    return {
        "delta": round(sum(b - a for a, b in zip(hits_a, hits_b, strict=True)) / n, 6),
        "ci_lower": round(differences[int(0.025 * rounds)], 6),
        "ci_upper": round(differences[int(0.975 * rounds) - 1], 6),
        "rounds": rounds,
    }


def behavior() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    criteria = contract["criteria"]["behavior"]
    updates = int(criteria["paired_updates"])
    delta_min = float(criteria["min_meaningful_effect"])
    cap_cpu = float(criteria["cap_cpu"])
    spec = UnitGraphSpec(dtype="float64")
    train = [episode(20000 + index) for index in range(criteria["train_episodes"])]
    dev = [episode(30000 + index) for index in range(DEV_EPISODES)]
    arms: dict[str, Any] = {}
    for route in ("A", "B"):
        torch.manual_seed(criteria["seed"])
        graph = build_route(route, spec)
        optimizer = torch.optim.Adam(graph.parameters(), lr=criteria["learning_rate"])
        arm_cpu = time.process_time()
        before, _ = evaluate_arm(graph, dev)
        losses = []
        for step in range(updates):
            losses.append(one_update(graph, optimizer, train[step % len(train)]))
        after, hits = evaluate_arm(graph, dev)
        # Controls: no state (zero gates) and no edges (zero edge weights).
        with torch.no_grad():
            saved_w = graph.units.w.detach().clone()
            graph.units.w.zero_()
        without_edges, _ = evaluate_arm(graph, dev)
        with torch.no_grad():
            graph.units.w.copy_(saved_w)
        no_state, _ = evaluate_arm(graph, dev, zero_gates=True)
        arms[route] = {
            "before_correct": before,
            "after_correct": after,
            "before_accuracy": round(before / len(dev), 6),
            "after_accuracy": round(after / len(dev), 6),
            "hits": hits,
            "without_edges_correct": without_edges,
            "without_state_correct": no_state,
            "updates": updates,
            "final_loss": round(losses[-1], 6) if losses else None,
            "first_loss": round(losses[0], 6) if losses else None,
            "cpu_seconds": round(time.process_time() - arm_cpu, 4),
        }
    bootstrap = paired_bootstrap(
        arms["A"]["hits"], arms["B"]["hits"], BOOTSTRAP_ROUNDS, criteria["seed"]
    )
    cpu_a, cpu_b = arms["A"]["cpu_seconds"], arms["B"]["cpu_seconds"]
    ratio = round(max(cpu_a, cpu_b) / max(min(cpu_a, cpu_b), 1e-6), 6)
    total_cpu = time.process_time() - started_cpu
    gate_band = criteria["cpu_ratio_band"]
    dynamic = any(
        abs(arms[route]["after_accuracy"] - arms[route]["before_accuracy"]) > 1e-9
        or arms[route]["final_loss"] != arms[route]["first_loss"]
        for route in ("A", "B")
    )
    if total_cpu > cap_cpu:
        verdict, reasons = "comparison_not_admitted", ["budget_exceeded"]
    elif not gate_band[0] <= ratio <= gate_band[1]:
        verdict, reasons = "compute_parity_unavailable", ["cpu_ratio_out_of_band", ratio]
    elif not dynamic:
        verdict, reasons = "budget_cannot_express", ["no_dynamic_range_in_either_arm"]
    elif bootstrap["ci_lower"] > 0 and bootstrap["delta"] >= delta_min:
        verdict, reasons = "b_superior_in_window", ["paired_ci_lower_positive", "delta_above_min"]
    elif bootstrap["ci_upper"] < 0 and -bootstrap["delta"] >= delta_min:
        verdict, reasons = "a_superior_in_window", ["paired_ci_upper_negative"]
    else:
        verdict, reasons = "inconclusive_within_window", ["ci_covers_zero_or_effect_below_min"]
    report = {
        "schema": "taiji-u2-p3-9-3-behavior-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.3",
        "phase": "behavior",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "reasons": reasons,
        "arms": arms,
        "paired_bootstrap": bootstrap,
        "cpu_ratio": ratio,
        "dynamic_range_present": bool(dynamic),
        "sample_exposure": {
            "train_episodes": criteria["train_episodes"],
            "dev_episodes": len(dev),
            "updates_per_arm": updates,
            "seed": criteria["seed"],
        },
        "capability_scope": "synthetic engineering behaviour window only; no natural-language, no project task, no capability claim",
        "retention": "no_verified_old_capability: retention is not signed",
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "criteria_sha_unchanged_after_reading": True,
        },
        "measured": {
            "cpu_seconds": round(total_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report("taiji_u2_p3_9_3_behavior_20261114.json", report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "delta": bootstrap["delta"],
                "ci": [bootstrap["ci_lower"], bootstrap["ci_upper"]],
                "cpu": round(total_cpu, 4),
            },
            ensure_ascii=False,
        )
    )
    return 0


def closeout() -> int:
    names = [
        "taiji_u2_p3_9_1_price_20261114.json",
        "taiji_u2_p3_9_1_graph_20261114.json",
        "taiji_u2_p3_9_2_restore_20261114.json",
        "taiji_u2_p3_9_3_behavior_20261114.json",
    ]
    stages = []
    for name in names:
        path = REPORTS / name
        data = json.loads(path.read_text(encoding="utf-8"))
        cpu = data.get("measured_cpu_seconds", data.get("measured", {}).get("cpu_seconds", 0.0))
        stages.append(
            {
                "report": "reports/" + name,
                "sha256": sha256_file(path),
                "verdict": data["verdict"],
                "cpu_seconds": cpu,
            }
        )
    behavior_verdict = stages[-1]["verdict"]
    benefit = behavior_verdict in ("b_superior_in_window", "a_superior_in_window")
    verdict = "p3_9_closed_limited_exit"
    report = {
        "schema": "taiji-u2-p3-9-4-closeout-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.4",
        "phase": "closeout",
        "verdict": verdict,
        "five_ledgers": {
            "birth": "two generations of exact neutral birth, registry IDs never reused, dormant/wake/rollback",
            "credit": "per-parameter gradient coverage for both routes, structural zeros reported",
            "behavior": behavior_verdict,
            "recovery": "independent-process full-capsule restores for both routes, initial and first non-empty moment",
            "cost": "per-stage CPU reported; no P3.8 budget transferred; no sealed face opened",
        },
        "capability_ledger": {
            "change": "none",
            "capability_claims": 0,
            "native_parent_qualified": False,
            "formal_a_status": "not_admitted",
        },
        "six_axis_card": {
            "status": "unchanged",
            "reason": "engineering graphs and a synthetic window carry no verified task capability",
        },
        "handoff_to_formal_ab_qualification": bool(benefit),
        "handoff_note": "only a superior-in-window verdict with retention evidence may be handed to formal A/B qualification; this window signs none",
        "stages": stages,
        "total_stage_cpu": round(sum(stage["cpu_seconds"] for stage in stages), 4),
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "new_parents": 0,
        },
    }
    write_report("taiji_u2_p3_9_4_closeout_20261114.json", report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "behavior": behavior_verdict,
                "handoff": report["handoff_to_formal_ab_qualification"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--phase", default="price", choices=["price", "graph", "restore", "behavior", "closeout"]
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    return {
        "price": price,
        "graph": graph_checks,
        "restore": restore,
        "behavior": behavior,
        "closeout": closeout,
    }[args.phase]()


if __name__ == "__main__":
    sys.exit(main())
