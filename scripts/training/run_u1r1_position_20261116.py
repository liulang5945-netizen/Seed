"""22.5.U.1 revision 1: position-aware readout on unit-core B.

Replaces U.1's mean-pool readout with question-conditioned column attention
(taiji.u2_unit_position_readout.PositionRouteBGraph). The input encoding contract
(unit-input-column-color-v1) is frozen unchanged. Targets the U.1 located gap:
mean-pool washing of column-position binding + RGB credit starvation.

Modes: smoke (zero optimizer steps), price (discarded model), run (authorized),
recover (independent readback), verify (re-derive from saved capsules).
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

from scripts.training import run_u1_unit_read_20261116 as u1
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_neural_units import (
    COMPAT_FP64_ATOL,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    capsule_summary,
    load_capsule,
    save_capsule,
    validate_capsule,
)
from taiji.u2_unit_input import encoding_contract
from taiji.u2_unit_position_readout import PositionRouteBGraph

CONTRACT = ROOT / "plans/manifests/taiji_u2_u1r1_position_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_u1r1_position_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_u1r1_position_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_u1r1_position_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_u1r1_position_verify_20261116.json"
RECOVER_REPORT = ROOT / "reports/taiji_u2_u1r1_position_recover_20261116.json"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/u1r1_position_20261116"
T = 16
LABELS = u1.LABELS


def sha_of(path: Path) -> str:
    return u1.sha_of(path)


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    return u1.write_report(path, payload)


def guards(contract: dict[str, Any], mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("position_readout", ROOT / "taiji/u2_unit_position_readout.py"),
        ("u1_runner", ROOT / "scripts/training/run_u1_unit_read_20261116.py"),
        ("unit_input", ROOT / "taiji/u2_unit_input.py"),
        ("neural_units", ROOT / "taiji/u2_neural_units.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
        ("episode", ROOT / "taiji/u2_task_episode.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if value_digest(encoding_contract()) != contract["encoding_contract_digest"]:
        raise ValueError("encoding_contract_digest_mismatch")
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
    return PositionRouteBGraph(UnitGraphSpec(dtype=dtype, seed=seed))


def episode_logits(graph: Any, item: dict[str, Any]) -> torch.Tensor:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    out = graph(data["channels"], data["gates"], graph.initial_state())
    return cast(torch.Tensor, out["readout"])  # [4], already attended


def evaluate(graph: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    hits = []
    preds = []
    with torch.no_grad():
        for item in episodes:
            pred = int(torch.argmax(episode_logits(graph, item)).item())
            preds.append(pred)
            hits.append(int(pred == item["label"]))
    return {
        "correct": sum(hits),
        "denominator": len(episodes),
        "accuracy": sum(hits) / len(episodes),
        "distinct": len(set(preds)),
        "hits": hits,
        "preds": preds,
    }


def one_update(graph: Any, optimizer: torch.optim.Optimizer, item: dict[str, Any]) -> float:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    optimizer.zero_grad(set_to_none=True)
    logits = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    target = torch.tensor(data["label"], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    optimizer.step()
    graph.units.project_parameters_()
    return float(loss.detach())


def intervention_readouts(
    graph: Any, item: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    full = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    saved_w = graph.units.w.detach().clone()
    with torch.no_grad():
        graph.units.w.zero_()
    without_edges = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    with torch.no_grad():
        graph.units.w.copy_(saved_w)
    perm = u1._derangement(graph.units.n_units)
    saved_mask = graph.units.edge_mask.detach().clone()
    with torch.no_grad():
        graph.units.w.copy_(saved_w[:, perm])
        graph.units.edge_mask.copy_(saved_mask[:, perm])
    permute_source = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    with torch.no_grad():
        graph.units.w.copy_(saved_w)
        graph.units.edge_mask.copy_(saved_mask)
    with_history = graph(data["channels"], data["gates"], history)["readout"]
    h_state = {**graph.initial_state(), "H": graph.initial_state()["H"] + 0.3}
    cut_h = graph(data["channels"], data["gates"], graph.cut_h(h_state))["readout"]
    h_perturbed = graph(data["channels"], data["gates"], h_state)["readout"]
    rgb_zero = {**data["channels"], "rgb": torch.zeros_like(data["channels"]["rgb"])}
    rgb_zero_logits = graph(rgb_zero, data["gates"], graph.initial_state())["readout"]
    fixed_q = torch.zeros_like(data["channels"]["question"])
    fixed_q[:, 0] = 1.0
    fixed_q[:, 6] = 1.0
    fixed_q[:, 7] = 1.0
    fixed_question = {**data["channels"], "question": fixed_q}
    fixed_q_logits = graph(fixed_question, data["gates"], graph.initial_state())["readout"]
    return {
        "full": full.detach(),
        "without_edges": without_edges.detach(),
        "permute_source": permute_source.detach(),
        "fresh": full.detach(),
        "with_history": with_history.detach(),
        "cut_h": cut_h.detach(),
        "h_perturbed": h_perturbed.detach(),
        "rgb_zero": rgb_zero_logits.detach(),
        "fixed_question": fixed_q_logits.detach(),
    }


def b_identity_checks(graph: Any) -> dict[str, bool]:
    return {
        "B_is_not_H_plus_head": not any(
            name.startswith("h_readout") or name in ("ell", "w_g")
            for name, _ in graph.named_parameters()
        ),
        "B_path_direct_units_position": graph.route == "B_position",
    }


def neutral_birth_check(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    state = graph.initial_state()
    before = graph(data["channels"], data["gates"], state)["readout"]
    grown = graph.with_neutral_birth(0, ["u0000"])
    big_gate = torch.cat([data["gates"], torch.zeros(T, 1, dtype=graph.spec.torch_dtype)], dim=1)
    after = grown(data["channels"], big_gate, graph.extend_state(state))["readout"]
    exact = bool(torch.allclose(before, after, rtol=0.0, atol=COMPAT_FP64_ATOL))
    max_abs = float((before.detach() - after.detach()).abs().max())
    grown.units.set_lifecycle(grown.units.identities["units"][-1]["unit_id"], "mature")
    with torch.no_grad():
        grown.units.p[:, -1].fill_(0.1)
    big_active = torch.cat([data["gates"], torch.ones(T, 1, dtype=graph.spec.torch_dtype)], dim=1)
    grown.zero_grad(set_to_none=True)
    logits = grown(data["channels"], big_active, grown.initial_state())["readout"]
    target = torch.tensor(data["label"], dtype=torch.long)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)).backward()
    new_r_grad = float(grown.units.r.grad[-1].abs().sum())
    return {
        "neutral_birth_exact": exact,
        "neutral_birth_max_abs": max_abs,
        "new_unit_r_gradient_after_credit": new_r_grad,
        "new_unit_r_credit_path_wired": new_r_grad > 1e-9,
        "generation": grown.units.identities["generation"],
    }


def gradient_coverage(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    logits = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    logits.square().mean().backward()
    credit = {
        name: float(param.grad.abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    return {
        "all_parameters_have_gradient": len(credit) == len(list(graph.named_parameters())),
        "named_nonzero": all(
            credit.get(name, 0.0) > 0
            for name in (
                "units.r",
                "units.p",
                "units.c",
                "input_projection",
                "W_q",
                "position_head.weight",
            )
        ),
        "credit": credit,
        "rgb_encoder_credit": credit.get("input_encoders.rgb.weight", 0.0),
        "question_encoder_credit": credit.get("input_encoders.question.weight", 0.0),
    }


def initial_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_initial.bin"


def final_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_final.bin"


def price() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "price")
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    item = episodes["fit"][0]
    samples: dict[str, list[float]] = {"forward_backward": [], "adam_step": [], "evaluation": []}
    for repeat in range(3):
        graph = build_graph(
            contract["criteria"]["seeds"][repeat % len(contract["criteria"]["seeds"])]
        )
        optimizer = torch.optim.Adam(graph.parameters(), lr=contract["criteria"]["learning_rate"])
        t0 = time.process_time()
        one_update(graph, optimizer, item)
        samples["forward_backward"].append(time.process_time() - t0)
        t0 = time.process_time()
        optimizer.step()
        graph.units.project_parameters_()
        samples["adam_step"].append(time.process_time() - t0)
        t0 = time.process_time()
        with torch.no_grad():
            episode_logits(graph, item)
        samples["evaluation"].append(time.process_time() - t0)
    per_update = max(samples["forward_backward"]) + max(samples["adam_step"])
    per_eval = max(samples["evaluation"])
    seeds = len(contract["criteria"]["seeds"])
    updates = int(contract["criteria"]["updates_per_seed"])
    fixed_evals_per_seed = 2 + 2 + 9
    design_cap = float(contract["plan"]["design_cap_cpu"])
    cap_cpu = seeds * (updates * per_update + fixed_evals_per_seed * per_eval)
    cap_wall = cap_cpu * 2.0
    verdict = (
        "price_measured_within_design" if cap_cpu <= design_cap else "price_exceeds_design_cap"
    )
    report = {
        "schema": "taiji-u2-u1r1-price-v1",
        "card": "22.5.U.1r1",
        "phase": "price",
        "verdict": verdict,
        "samples": {key: [round(v, 5) for v in vals] for key, vals in samples.items()},
        "per_update_cpu": round(per_update, 5),
        "per_eval_cpu": round(per_eval, 5),
        "seeds": seeds,
        "updates_per_seed": updates,
        "fixed_evals_per_seed": fixed_evals_per_seed,
        "cap_cpu": round(cap_cpu, 4),
        "cap_wall": round(cap_wall, 4),
        "design_cap_cpu": design_cap,
        "low_tick_limit": "process_time resolution ~0.0156s (P3.9/U.1)",
        "discipline": {"optimizer_steps": 3, "project_task_data_reads": 1, "check_reads": 0},
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(PRICE_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "cap_cpu": report["cap_cpu"],
                "per_update": report["per_update_cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


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
    seeds_report = []
    checks: dict[str, bool] = {"encoding_oracle_consistent": oracle_ok == oracle_total}
    for seed in contract["criteria"]["seeds"]:
        graph = build_graph(seed)
        item = episodes["fit"][0]
        history_item = episodes["fit"][1]
        hd = u1.cast_item(history_item, graph.spec.torch_dtype)
        hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
        identity = b_identity_checks(graph)
        probe = u1.mechanism_probe(graph)
        summary = u1.intervention_summary(intervention_readouts(probe, item, hist_state))
        birth = neutral_birth_check(graph, item)
        coverage = gradient_coverage(graph, item)
        seed_checks = {
            **identity,
            "without_edges_changes_readout": summary["without_edges_changes_readout"],
            "permute_source_changes_readout": summary["permute_source_changes_readout"],
            "history_changes_readout": summary["history_changes_readout"],
            "cut_h_bit_identical": summary["cut_h_bit_identical"],
            "h_perturb_bit_identical": summary["h_perturb_bit_identical"],
            "neutral_birth_exact": birth["neutral_birth_exact"],
            "new_unit_r_credit_path_wired": birth["new_unit_r_credit_path_wired"],
            "all_parameters_have_gradient": coverage["all_parameters_have_gradient"],
            "named_nonzero": coverage["named_nonzero"],
        }
        checks.update({f"seed{seed}_{key}": value for key, value in seed_checks.items()})
        seeds_report.append(
            {
                "seed": seed,
                "identity": identity,
                "interventions": summary,
                "neutral_birth": birth,
                "gradient_coverage": {
                    "all": coverage["all_parameters_have_gradient"],
                    "named_nonzero": coverage["named_nonzero"],
                    "rgb_encoder_credit": coverage["rgb_encoder_credit"],
                    "question_encoder_credit": coverage["question_encoder_credit"],
                },
                "checks": seed_checks,
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "smoke_passed" if not uncovered else "smoke_failed"
    report = {
        "schema": "taiji-u2-u1r1-smoke-v1",
        "card": "22.5.U.1r1",
        "phase": "smoke",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "checks": checks,
        "uncovered": uncovered,
        "encoding_oracle": {"ok": oracle_ok, "total": oracle_total},
        "seeds": seeds_report,
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
        json.dumps(
            {
                "verdict": verdict,
                "green": sum(checks.values()),
                "total": len(checks),
                "oracle": f"{oracle_ok}/{oracle_total}",
            },
            ensure_ascii=False,
        )
    )
    return 0 if not uncovered else 1


def run() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "run")
    criteria = contract["criteria"]
    cap_cpu = float(contract["plan"]["cap_cpu"])
    updates = int(criteria["updates_per_seed"])
    lr = float(criteria["learning_rate"])
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = []
    budget_stopped = False
    for seed in criteria["seeds"]:
        arm_cpu = time.process_time()
        graph = build_graph(seed)
        save_capsule(initial_capsule_path(seed), build_capsule(graph))
        optimizer = torch.optim.Adam(graph.parameters(), lr=lr)
        before_fit = evaluate(graph, episodes["fit"])
        before_train = evaluate(graph, episodes["train"])
        losses = []
        for step in range(updates):
            if time.process_time() - started_cpu > cap_cpu:
                budget_stopped = True
                break
            losses.append(
                one_update(graph, optimizer, episodes["train"][step % len(episodes["train"])])
            )
        after_fit = evaluate(graph, episodes["fit"])
        after_train = evaluate(graph, episodes["train"])
        rgb_zero_hits = []
        fixed_q_hits = []
        intervention_flags: dict[str, list[bool]] = {
            "without_edges_changes_readout": [],
            "permute_source_changes_readout": [],
            "history_changes_readout": [],
            "cut_h_bit_identical": [],
        }
        attn_entropy = []
        with torch.no_grad():
            for index, item in enumerate(episodes["fit"]):
                history_item = episodes["fit"][(index + 1) % len(episodes["fit"])]
                hd = u1.cast_item(history_item, graph.spec.torch_dtype)
                hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
                readouts = intervention_readouts(graph, item, hist_state)
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
                d = u1.cast_item(item, graph.spec.torch_dtype)
                attn = graph(d["channels"], d["gates"], graph.initial_state())["attention"]
                attn_entropy.append(float(-(attn * torch.log(attn + 1e-12)).sum()))
        save_capsule(
            final_capsule_path(seed),
            build_capsule(
                graph,
                optimizer_state={"state_entries": len(optimizer.state)},
                clocks=(updates, updates),
            ),
        )
        arms.append(
            {
                "seed": seed,
                "updates": len(losses),
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
                "attention_entropy_mean": round(sum(attn_entropy) / len(attn_entropy), 6),
                "first_loss": round(losses[0], 6) if losses else None,
                "final_loss": round(losses[-1], 6) if losses else None,
                "final_capsule_digest": value_digest(load_capsule(final_capsule_path(seed))),
                "cpu_seconds": round(time.process_time() - arm_cpu, 4),
            }
        )
        if budget_stopped:
            break
    fit_gate = float(criteria["fit_gate"])
    train_gate = float(criteria["train_gate"])
    distinct_min = int(criteria["distinct_min"])
    control_max = float(criteria["control_agreement_max"])
    seed_pass = [
        arm["fit_accuracy"] >= fit_gate
        and arm["train_accuracy"] >= train_gate
        and arm["fit_distinct"] >= distinct_min
        and arm["rgb_zero_agreement"] < control_max
        and arm["fixed_question_agreement"] < control_max
        and arm["intervention_fraction"]["cut_h_bit_identical"] == 1.0
        and arm["intervention_fraction"]["without_edges_changes_readout"] >= 0.5
        and arm["intervention_fraction"]["permute_source_changes_readout"] >= 0.5
        and arm["intervention_fraction"]["history_changes_readout"] >= 0.5
        for arm in arms
    ]
    all_pass = len(seed_pass) == len(criteria["seeds"]) and all(seed_pass)
    if budget_stopped:
        verdict, reasons = "budget_stop", ["cap_cpu_exceeded"]
    elif all_pass:
        verdict, reasons = "read_prerequisite_passed", ["all_seeds_meet_read_gates"]
    else:
        verdict, reasons = "read_prerequisite_failed", ["one_or_more_seeds_below_gate"]
    report = {
        "schema": "taiji-u2-u1r1-run-v1",
        "card": "22.5.U.1r1",
        "phase": "run",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "reasons": reasons,
        "arms": arms,
        "seed_pass": seed_pass,
        "gates": {
            "fit": fit_gate,
            "train": train_gate,
            "distinct_min": distinct_min,
            "control_max": control_max,
        },
        "capability_scope": "unit-core B read prerequisite only; no correction/retention/parent/capability claim",
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "fit": [a["fit_accuracy"] for a in arms],
                "train": [a["train_accuracy"] for a in arms],
                "cpu": report["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    spec = capsule["runtime"]["spec"]
    graph = PositionRouteBGraph(UnitGraphSpec(**spec))
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
        initial = load_capsule(initial_capsule_path(seed))
        receipt = independent_restore(initial, f"seed{seed}_initial")
        rows.append(receipt)
        checks[f"seed{seed}_initial_restore"] = bool(
            receipt["digest_matches"] and receipt["pid_differs"]
        )
        final = load_capsule(final_capsule_path(seed))
        final_receipt = independent_restore(final, f"seed{seed}_final")
        rows.append(final_receipt)
        checks[f"seed{seed}_final_restore"] = bool(
            final_receipt["digest_matches"] and final_receipt["pid_differs"]
        )
        checks[f"seed{seed}_final_next_gradient"] = bool(final_receipt["next_gradient_digest"])
        tampered = build_capsule(build_graph(seed), clocks=(3, 2))
        tampered["runtime"]["unit_registry"][0]["lifecycle"] = "retired"
        try:
            validate_capsule(tampered)
            checks[f"seed{seed}_tamper_rejected"] = False
        except ValueError:
            checks[f"seed{seed}_tamper_rejected"] = True
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "recovery_verified" if not uncovered else "engineering_failure"
    report = {
        "schema": "taiji-u2-u1r1-recover-v1",
        "card": "22.5.U.1r1",
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
        json.dumps(
            {"verdict": verdict, "green": sum(checks.values()), "total": len(checks)},
            ensure_ascii=False,
        )
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
        seed = arm["seed"]
        capsule = load_capsule(final_capsule_path(seed))
        graph = PositionRouteBGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
        apply_capsule(graph, capsule)
        fit = evaluate(graph, episodes["fit"])
        train = evaluate(graph, episodes["train"])
        checks[f"seed{seed}_fit_matches_run"] = fit["accuracy"] == arm["fit_accuracy"]
        checks[f"seed{seed}_train_matches_run"] = train["accuracy"] == arm["train_accuracy"]
        checks[f"seed{seed}_capsule_digest"] = value_digest(capsule) == arm["final_capsule_digest"]
        checks[f"seed{seed}_B_identity"] = all(b_identity_checks(graph).values())
        rows.append(
            {
                "seed": seed,
                "fit_accuracy": fit["accuracy"],
                "train_accuracy": train["accuracy"],
                "fit_distinct": fit["distinct"],
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "verify_passed" if not uncovered else "verify_failed"
    report = {
        "schema": "taiji-u2-u1r1-verify-v1",
        "card": "22.5.U.1r1",
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
        json.dumps(
            {"verdict": verdict, "green": sum(checks.values()), "total": len(checks)},
            ensure_ascii=False,
        )
    )
    return 0 if not uncovered else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
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
