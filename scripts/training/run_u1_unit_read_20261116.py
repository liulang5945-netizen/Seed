"""22.5.U.1: unit-native read foundation on unit-graph core B.

The byte-world joint gate was globally closed on the compatible embedding A
(U2Subject): read -> update scoping -> position attention -> value encoding each
failed independently (CLOSEOUT-TAIJI_U2_BYTE_WORLD_JOINT_GATE_20261116). The
strategic turn (D-20261116-17) moves to unit-native core B, where queries read
unit state directly without the fixed H mediator.

U.1 is the READ prerequisite of the joint gate on B. It asks one question: can
direct unit read bind query -> colour -> position on the byte-world single-query
read task that U2Subject itself could not do? The input encoding contract
(taiji.u2_unit_input, unit-input-column-color-v1) supplies faithful per-column
colour evidence (validated 96/96 against the public oracle), isolating the unit
binding/read question from the encoder-loss question that killed the A line.

Structure: RouteBGraph (12 units / 6 groups, direct query over gated z/a, H is
scratch). Train all graph parameters with CE on the 4-way position label; the
encoding is fixed external sensory evidence. Emission readout is the graph's own
direct unit query (no H). Neutral birth, four causal interventions (full / block
messages / permute source / remove history) plus the B identity cut_h, and two
task controls (RGB zeroed / fixed question) are measured.

Gates (read prerequisite, 3 seeds): fit read >= 0.80 (32), train read >= 0.90
(64), distinct predictions >= 3, control agreement < 0.5, interventions change
the readout while cut_h is bit-identical, neutral birth exact, independent-process
recovery bit-exact. A positive U.1 signs the read prerequisite only; correction
and retention (U.2) and the formal A/B comparison (U.3) are separately designed.

Modes: smoke (zero optimizer steps), price (discarded model), run (authorized),
recover (independent readback), verify (re-derive from saved capsules).
"""

from __future__ import annotations

import argparse
import copy
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

from scripts.training.first_a_support_22_5 import LABELS
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_neural_units import (
    COMPAT_FP64_ATOL,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    build_route,
    capsule_summary,
    load_capsule,
    save_capsule,
    validate_capsule,
)
from taiji.u2_unit_input import (
    encode_scene,
    encoding_contract,
    scene_entity,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_u1_unit_read_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_u1_unit_read_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_u1_unit_read_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_u1_unit_read_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_u1_unit_read_verify_20261116.json"
ARTIFACTS = ROOT / "data/experiments/taiji_u2/u1_unit_read_20261116"
T = 16
POSITIONS = LABELS[:3]


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    return path


def guards(contract: dict[str, Any], mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
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


def load_data(contract: dict[str, Any]) -> dict[str, Any]:
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    return cast(dict[str, Any], data)


def make_gates(n_units: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(T, n_units, dtype=dtype)


def prepare_episodes(data: dict[str, Any], n_units: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"train": [], "fit": []}
    for split in ("train", "fit"):
        for example in data[split]:
            channels = encode_scene(example["public"], torch.float64)
            entity = scene_entity(example["public"]["request"])
            label = LABELS.index(example["target"].decode("utf-8"))
            out[split].append(
                {
                    "id": example["id"],
                    "channels": channels,
                    "label": label,
                    "entity": entity,
                    "gates": make_gates(n_units, torch.float64),
                }
            )
    return out


def cast_item(item: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    return {
        **item,
        "channels": {key: value.to(dtype) for key, value in item["channels"].items()},
        "gates": item["gates"].to(dtype),
    }


def episode_logits(graph: Any, item: dict[str, Any]) -> torch.Tensor:
    data = cast_item(item, graph.spec.torch_dtype)
    out = graph(data["channels"], data["gates"], graph.initial_state())
    return cast(torch.Tensor, out["readout"].mean(dim=0))


def evaluate(graph: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    hits = []
    preds = []
    with torch.no_grad():
        for item in episodes:
            logits = episode_logits(graph, item)
            pred = int(torch.argmax(logits).item())
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
    data = cast_item(item, graph.spec.torch_dtype)
    optimizer.zero_grad(set_to_none=True)
    logits = graph(data["channels"], data["gates"], graph.initial_state())["readout"].mean(dim=0)
    target = torch.tensor(data["label"], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    optimizer.step()
    graph.units.project_parameters_()
    return float(loss.detach())


def _derangement(n: int) -> list[int]:
    return [(i + 1) % n for i in range(n)]


def intervention_readouts(
    graph: Any, item: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    """Four causal interventions + B identity cut_h + two task controls.

    Returns episode logits for each condition. ``history`` is the next_state from
    a prior episode, used for the remove-history intervention.
    """

    data = cast_item(item, graph.spec.torch_dtype)
    full = graph(data["channels"], data["gates"], graph.initial_state())["readout"].mean(dim=0)

    # block cross-unit messages: zero edge weights.
    saved_w = graph.units.w.detach().clone()
    with torch.no_grad():
        graph.units.w.zero_()
    without_edges = graph(data["channels"], data["gates"], graph.initial_state())["readout"].mean(
        dim=0
    )
    with torch.no_grad():
        graph.units.w.copy_(saved_w)

    # permute source units: rewire the source end of every edge by a derangement.
    perm = _derangement(graph.units.n_units)
    saved_mask = graph.units.edge_mask.detach().clone()
    with torch.no_grad():
        graph.units.w.copy_(saved_w[:, perm])
        graph.units.edge_mask.copy_(saved_mask[:, perm])
    permute_source = graph(data["channels"], data["gates"], graph.initial_state())["readout"].mean(
        dim=0
    )
    with torch.no_grad():
        graph.units.w.copy_(saved_w)
        graph.units.edge_mask.copy_(saved_mask)

    # remove history state (keep params): target from fresh state vs prior state.
    fresh = full
    with_history = graph(data["channels"], data["gates"], history)["readout"].mean(dim=0)

    # B identity: H is scratch, perturbing it must not change the readout.
    h_state = {**graph.initial_state(), "H": graph.initial_state()["H"] + 0.3}
    cut_h = graph(data["channels"], data["gates"], graph.cut_h(h_state))["readout"].mean(dim=0)
    h_perturbed = graph(data["channels"], data["gates"], h_state)["readout"].mean(dim=0)

    # task control: RGB zeroed (cannot locate).
    rgb_zero = {**data["channels"], "rgb": torch.zeros_like(data["channels"]["rgb"])}
    rgb_zero_logits = graph(rgb_zero, data["gates"], graph.initial_state())["readout"].mean(dim=0)

    # task control: fixed question (entity 0 for every scene; cannot select).
    fixed_q = torch.zeros_like(data["channels"]["question"])
    fixed_q[:, 0] = 1.0
    fixed_q[:, 6] = 1.0
    fixed_q[:, 7] = 1.0
    fixed_question = {**data["channels"], "question": fixed_q}
    fixed_q_logits = graph(fixed_question, data["gates"], graph.initial_state())["readout"].mean(
        dim=0
    )

    return {
        "full": full.detach(),
        "without_edges": without_edges.detach(),
        "permute_source": permute_source.detach(),
        "fresh": fresh.detach(),
        "with_history": with_history.detach(),
        "cut_h": cut_h.detach(),
        "h_perturbed": h_perturbed.detach(),
        "rgb_zero": rgb_zero_logits.detach(),
        "fixed_question": fixed_q_logits.detach(),
    }


def intervention_summary(readouts: dict[str, Any]) -> dict[str, Any]:
    full = readouts["full"]
    full_pred = int(torch.argmax(full).item())

    def changed(other: torch.Tensor) -> bool:
        return not torch.allclose(full, other, rtol=0.0, atol=1e-8)

    def agreement(other: torch.Tensor) -> float:
        return float(int(torch.argmax(other).item()) == full_pred)

    return {
        "without_edges_changes_readout": changed(readouts["without_edges"]),
        "permute_source_changes_readout": changed(readouts["permute_source"]),
        "history_changes_readout": changed(readouts["with_history"]),
        "cut_h_bit_identical": bool(torch.allclose(full, readouts["cut_h"], rtol=0.0, atol=0.0)),
        "h_perturb_bit_identical": bool(
            torch.allclose(full, readouts["h_perturbed"], rtol=0.0, atol=0.0)
        ),
        "rgb_zero_agreement": agreement(readouts["rgb_zero"]),
        "fixed_question_agreement": agreement(readouts["fixed_question"]),
        "max_abs_without_edges": float((full - readouts["without_edges"]).abs().max()),
        "max_abs_permute_source": float((full - readouts["permute_source"]).abs().max()),
        "max_abs_history": float((full - readouts["with_history"]).abs().max()),
    }


def mechanism_probe(graph: Any) -> Any:
    """Untrained graphs have c=0,w=0 (no memory/edges); open both to prove the
    intervention mechanisms are causally wired without any optimizer step."""

    probe = copy.deepcopy(graph)
    with torch.no_grad():
        probe.units.c.fill_(0.5)
        probe.units.w.copy_(0.5 * probe.units.edge_mask)
    return probe


def b_identity_checks(graph: Any) -> dict[str, bool]:
    return {
        "B_is_not_H_plus_head": not any(
            name.startswith("h_readout") or name in ("ell", "w_g")
            for name, _ in graph.named_parameters()
        ),
        "B_path_direct_units": graph.route == "B",
    }


def neutral_birth_check(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    data = cast_item(item, graph.spec.torch_dtype)
    state = graph.initial_state()
    before = graph(data["channels"], data["gates"], state)["readout"]
    grown = graph.with_neutral_birth(0, ["u0000"])
    big_gate = torch.cat([data["gates"], torch.zeros(T, 1, dtype=graph.spec.torch_dtype)], dim=1)
    after = grown(data["channels"], big_gate, graph.extend_state(state))["readout"]
    exact = bool(torch.allclose(before, after, rtol=0.0, atol=COMPAT_FP64_ATOL))
    max_abs = float((before.detach() - after.detach()).abs().max())
    # Birth credit path: at the exact neutral point p[:,new]=0 and the unit is
    # immature (gated off), so r gradient is structurally zero (spec section 5:
    # birth-point nabla p/c/b may be zero). Once the unit is matured and its
    # output column p carries credit, task gradient must reach its input row r.
    grown.units.set_lifecycle(grown.units.identities["units"][-1]["unit_id"], "mature")
    with torch.no_grad():
        grown.units.p[:, -1].fill_(0.1)
    big_gate_active = torch.cat(
        [data["gates"], torch.ones(T, 1, dtype=graph.spec.torch_dtype)], dim=1
    )
    grown.zero_grad(set_to_none=True)
    logits = grown(data["channels"], big_gate_active, grown.initial_state())["readout"].mean(dim=0)
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
    data = cast_item(item, graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    logits = graph(data["channels"], data["gates"], graph.initial_state())["readout"].mean(dim=0)
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
                "query",
                "readout.weight",
            )
        ),
        "credit": credit,
    }


def build_graph(seed: int, dtype: str = "float64") -> Any:
    return build_route("B", UnitGraphSpec(dtype=dtype, seed=seed))


def initial_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_initial.bin"


def final_capsule_path(seed: int) -> Path:
    return ARTIFACTS / f"seed{seed}_final.bin"


# --------------------------------------------------------------------------- price


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def price() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "price")
    data = load_data(contract)
    episodes = prepare_episodes(data, UnitGraphSpec().n_units)
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
    # per seed: updates + fit/train eval before/after + 9 intervention/control evals
    fixed_evals_per_seed = 2 + 2 + 9
    design_cap = float(contract["plan"]["design_cap_cpu"])
    cap_cpu = seeds * (updates * per_update + fixed_evals_per_seed * per_eval)
    cap_wall = cap_cpu * 2.0
    verdict = (
        "price_measured_within_design" if cap_cpu <= design_cap else "price_exceeds_design_cap"
    )
    report = {
        "schema": "taiji-u2-u1-price-v1",
        "card": "22.5.U.1",
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
        "low_tick_limit": "process_time resolution ~0.0156s; low-CPU readings carry this limit (P3.9)",
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


# --------------------------------------------------------------------------- smoke


def smoke() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "smoke")
    data = load_data(contract)
    episodes = prepare_episodes(data, UnitGraphSpec().n_units)
    # encoding oracle consistency on every train+fit scene.
    oracle_ok = 0
    oracle_total = 0
    for split in ("train", "fit"):
        for example in data[split]:
            from taiji.u2_unit_input import color_mass, derive_label

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
        hist_state = graph(
            cast_item(history_item, graph.spec.torch_dtype)["channels"],
            cast_item(history_item, graph.spec.torch_dtype)["gates"],
            graph.initial_state(),
        )["next_state"]
        identity = b_identity_checks(graph)
        probe = mechanism_probe(graph)
        readouts = intervention_readouts(probe, item, hist_state)
        summary = intervention_summary(readouts)
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
                },
                "checks": seed_checks,
            }
        )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "smoke_passed" if not uncovered else "smoke_failed"
    report = {
        "schema": "taiji-u2-u1-smoke-v1",
        "card": "22.5.U.1",
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


# --------------------------------------------------------------------------- run


def run() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "run")
    criteria = contract["criteria"]
    cap_cpu = float(contract["plan"]["cap_cpu"])
    updates = int(criteria["updates_per_seed"])
    lr = float(criteria["learning_rate"])
    data = load_data(contract)
    episodes = prepare_episodes(data, UnitGraphSpec().n_units)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    arms = []
    budget_stopped = False
    for seed in contract["criteria"]["seeds"]:
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
        # controls + interventions on the trained graph (agreement over fit).
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
                hd = cast_item(history_item, graph.spec.torch_dtype)
                hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
                readouts = intervention_readouts(graph, item, hist_state)
                summary = intervention_summary(readouts)
                full_pred = int(torch.argmax(readouts["full"]).item())
                rgb_zero_hits.append(
                    int(int(torch.argmax(readouts["rgb_zero"]).item()) == full_pred)
                )
                fixed_q_hits.append(
                    int(int(torch.argmax(readouts["fixed_question"]).item()) == full_pred)
                )
                for key in intervention_flags:
                    intervention_flags[key].append(bool(summary[key]))
        save_capsule(
            final_capsule_path(seed),
            build_capsule(
                graph,
                optimizer_state={"state_entries": len(optimizer.state)},
                clocks=(updates, updates),
            ),
        )
        arm = {
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
            "first_loss": round(losses[0], 6) if losses else None,
            "final_loss": round(losses[-1], 6) if losses else None,
            "final_capsule_digest": value_digest(load_capsule(final_capsule_path(seed))),
            "cpu_seconds": round(time.process_time() - arm_cpu, 4),
        }
        arms.append(arm)
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
    all_pass = len(seed_pass) == len(contract["criteria"]["seeds"]) and all(seed_pass)
    if budget_stopped:
        verdict, reasons = "budget_stop", ["cap_cpu_exceeded"]
    elif all_pass:
        verdict, reasons = "read_prerequisite_passed", ["all_seeds_meet_read_gates"]
    else:
        verdict, reasons = "read_prerequisite_failed", ["one_or_more_seeds_below_gate"]
    report = {
        "schema": "taiji-u2-u1-run-v1",
        "card": "22.5.U.1",
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
                "cpu": report["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


# --------------------------------------------------------------------------- recover


def worker(capsule_path: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(capsule_path)
    spec = capsule["runtime"]["spec"]
    graph = build_route(capsule["runtime"]["route"], UnitGraphSpec(**spec))
    apply_capsule(graph, capsule)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = load_data(contract)
    episodes = prepare_episodes(data, graph.units.n_units)
    item = episodes["fit"][0]
    d = cast_item(item, graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    loss = (
        graph(d["channels"], d["gates"], graph.initial_state())["readout"]
        .mean(dim=0)
        .square()
        .mean()
    )
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
    local_digest = value_digest(capsule)
    receipt["digest_matches"] = receipt["capsule_digest"] == local_digest and receipt[
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
        # tamper rejection (red direction).
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
        "schema": "taiji-u2-u1-recover-v1",
        "card": "22.5.U.1",
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
    write_report(ROOT / "reports/taiji_u2_u1_unit_read_recover_20261116.json", report)
    print(
        json.dumps(
            {"verdict": verdict, "green": sum(checks.values()), "total": len(checks)},
            ensure_ascii=False,
        )
    )
    return 0 if not uncovered else 1


# --------------------------------------------------------------------------- verify


def verify() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    guards(contract, "verify")
    run_report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = load_data(contract)
    episodes = prepare_episodes(data, UnitGraphSpec().n_units)
    checks: dict[str, bool] = {}
    rows = []
    for arm in run_report["arms"]:
        seed = arm["seed"]
        capsule = load_capsule(final_capsule_path(seed))
        graph = build_route("B", UnitGraphSpec(**capsule["runtime"]["spec"]))
        apply_capsule(graph, capsule)
        fit = evaluate(graph, episodes["fit"])
        train = evaluate(graph, episodes["train"])
        checks[f"seed{seed}_fit_matches_run"] = fit["accuracy"] == arm["fit_accuracy"]
        checks[f"seed{seed}_train_matches_run"] = train["accuracy"] == arm["train_accuracy"]
        checks[f"seed{seed}_capsule_digest"] = value_digest(capsule) == arm["final_capsule_digest"]
        identity = b_identity_checks(graph)
        checks[f"seed{seed}_B_identity"] = all(identity.values())
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
        "schema": "taiji-u2-u1-verify-v1",
        "card": "22.5.U.1",
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
    parser.error("select a mode: --smoke --price --run --recover --verify")
    return 2


if __name__ == "__main__":
    sys.exit(main())
