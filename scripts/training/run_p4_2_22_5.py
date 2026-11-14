"""P4.2: carry the column pathway into the subject's byte emission organ.

P4.1 trained the body's own rgb encoder to name a column's palette colour and
reached fit 0.906 through a 59-parameter side readout. That readout is not the
subject's language organ, so the card showed the encoder can carry the
information but not that the body can answer through its byte head.

This card has two stages and one single served output.

  Stage 1  the body's own object read learns to expose the answer. The encoder
           stays frozen at its reconstructed P4.1 B-arm weights; every other
           parameter plus a declared 4-way training-only scaffold on the read
           context is trained. The scaffold never appears at evaluation.
  Stage 2  the byte head learns to say what the read exposes. Every parameter is
           frozen; only ``joint_readout``, ``byte_head`` and ``copy_gate`` are
           trained, on contexts cached from the frozen body. The cache is exact:
           it holds the arguments the body itself passed to the emission
           operator, and a contract check replays them at the start.

Success is a complete response round through the subject's own byte emission:
exact answer bytes, EOS and valid UTF-8, counted over every fit round. The P4.1
checkpoints were not preserved, so ``reconstruct`` rebuilds arm B from the frozen
P4.1 contract and accepts the weights only if all three published dose
signatures reproduce exactly.

Modes: ``census`` (zero training), ``price`` (in-memory timing), ``reconstruct``
(training), ``run`` (training).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional

import taiji.u2_subject as subject_module
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import evaluate, make_model, output, prepare
from taiji.u2_history import assert_values_equal, atomic_load, value_digest
from taiji.u2_observation_columns import column_coordinates, column_values

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_2_byte_path_20261115.json"
P4_1_CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_1_column_supervision_20261114.json"
P4_1_REPORT = ROOT / "reports/taiji_u2_p4_1_column_supervision_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p4_2_price_20261115.json"
CENSUS_REPORT = ROOT / "reports/taiji_u2_p4_2_census_20261115.json"
RECON_REPORT = ROOT / "reports/taiji_u2_p4_2_reconstruction_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p4_2_byte_path_20261115.json"
LABELS = ("左边", "中间", "右边", "未知")
SIGNATURE = (
    "fit_correct",
    "train_correct",
    "colour_accuracy_fit",
    "colour_accuracy_train",
    "centroid_column_hits_fit",
    "centroid_column_total_fit",
    "abstain_on_unknown_fit",
    "abstain_on_other_fit",
    "probe_painted_fit",
    "probe_overall_fit",
)


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"reconstruct", "run"}:
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


def column_public(example: dict) -> dict:
    public = dict(example["public"])
    observations = dict(public["observations"])
    coordinates = dict(public["coordinates"])
    observations["rgb"] = column_values(observations["rgb"])
    coordinates["rgb"] = column_coordinates(coordinates["rgb"])
    public["observations"] = observations
    public["coordinates"] = coordinates
    return public


def column_example(example: dict) -> dict:
    return {**example, "public": column_public(example)}


def build(contract: dict, encoder_state: dict | None = None):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    if encoder_state is not None:
        model.encoder["rgb"].load_state_dict(encoder_state)
    return model


def non_encoder_digest(model) -> str:
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("encoder.rgb.")
    }
    return value_digest(state)


def prepared(model, example: dict):
    with torch.no_grad():
        return prepare(model, column_public(example), example["id"])


def byte_loss(model, state, target: bytes):
    terms = []
    for symbol in [*target, 256]:
        probability, _, details = output(model, state)
        terms.append(-probability[symbol])
        with torch.no_grad():
            state = model.accept_byte(state, symbol, details)
    return torch.stack(terms).mean()


def round_evaluate(model, examples: list[dict], guard=None) -> dict:
    return evaluate(model, [column_example(example) for example in examples], budget_guard=guard)


def read_context(model, state):
    """Run the body's real read and return the input of its joint readout."""
    captured = {}

    def capture(module, inputs, result):
        captured.setdefault("x", inputs[0])
        return None

    hook = model.joint_readout.register_forward_hook(capture)
    try:
        model.branch_context(state, model.language_goal(state), state["prediction_action"])
    finally:
        hook.remove()
    return captured["x"]


def set_requires_grad(model, names: set[str]) -> list[str]:
    found = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
        if name in names:
            found.append(name)
    if sorted(found) != sorted(names):
        raise ValueError("trainable_set_mismatch")
    return sorted(found)


def optimizer_for(parameters, spec: dict):
    return torch.optim.Adam(
        parameters,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )


def batch_at(examples: list[dict], batch: int, update: int) -> list[dict]:
    return [examples[((update - 1) * batch + offset) % len(examples)] for offset in range(batch)]


def scaffold_loss(model, scaffold, batch: list[dict]) -> torch.Tensor:
    logits = []
    for example in batch:
        state = prepare(model, column_public(example), example["id"])
        logits.append(scaffold(read_context(model, state)).mean(0))
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    return functional.cross_entropy(torch.stack(logits), labels)


def scaffold_accuracy(model, scaffold, examples: list[dict]) -> dict:
    with torch.no_grad():
        hits = 0
        for example in examples:
            state = prepared(model, example)
            predicted = int(scaffold(read_context(model, state)).mean(0).argmax())
            hits += int(LABELS[predicted] == example["target"].decode())
    return {"correct": hits, "denominator": len(examples), "accuracy": hits / len(examples)}


def save_checkpoint(path: Path, payload: dict) -> str:
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    restored = torch.load(path, weights_only=False)
    assert_values_equal(payload, restored, f"checkpoint.{path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_positions(model, examples: list[dict]) -> list[dict]:
    """Cache exactly what the body passed to the emission operator, per prefix.

    The body is frozen at this point, so these arguments are constants. The
    trainable Stage-2 modules (joint readout, byte head, copy gate) are not in
    the cache; they are recomputed from ``x`` and ``previous``.
    """
    positions: list[dict] = []
    real_emission = subject_module.emission
    captured: dict = {}

    def wrapper(state, log_alpha, log_beta, lam, gamma, generator, ids, valid, mask):
        captured["args"] = {
            "cursor": {k: v.detach().clone() for k, v in state.items()},
            "alpha": log_alpha.detach().clone(),
            "beta": log_beta.detach().clone(),
            "ids": ids.detach().clone(),
            "valid": valid.detach().clone(),
            "mask": mask.detach().clone(),
        }
        return real_emission(state, log_alpha, log_beta, lam, gamma, generator, ids, valid, mask)

    def capture(module, inputs, result):
        captured.setdefault("x", inputs[0].detach().clone())
        return None

    subject_module.emission = wrapper
    hook = model.joint_readout.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for example in examples:
                state = prepare(model, column_public(example), example["id"])
                for position, symbol in enumerate([*example["target"], 256]):
                    captured.clear()
                    goal = model.language_goal(state)
                    _, _, details = model.output(state, goal, state["prediction_action"])
                    positions.append(
                        {
                            "example": example["id"],
                            "position": position,
                            "x": captured["x"],
                            "goal": goal.detach().clone(),
                            "previous": int(state["previous_byte"]),
                            "symbol": symbol,
                            **captured["args"],
                        }
                    )
                    state = model.accept_byte(state, symbol, details)
    finally:
        hook.remove()
        subject_module.emission = real_emission
    return positions


def emission_loss(model, items: list[dict]) -> torch.Tensor:
    terms = []
    for item in items:
        context = model.joint_readout(item["x"])
        previous_value = model.previous_embedding(item["previous"])[None].expand(
            context.shape[0], -1
        )
        generator = model.byte_head(torch.cat([context, previous_value], -1)).log_softmax(-1)
        gate = model.copy_gate(
            torch.cat(
                [
                    context,
                    item["goal"][None].expand(context.shape[0], -1),
                    previous_value,
                ],
                -1,
            )
        ).sigmoid()
        lam, gamma = gate.unbind(-1)
        out = subject_module.emission(
            item["cursor"],
            item["alpha"],
            item["beta"],
            lam,
            gamma,
            generator,
            item["ids"],
            item["valid"],
            item["mask"],
        )
        terms.append(-out["log_output"][item["symbol"]].mean())
    return torch.stack(terms).mean()


def replayed_log_probability(model, item: dict) -> float:
    """Contract check: the cached arguments reproduce the body's own log-prob."""
    with torch.no_grad():
        return float(emission_loss(model, [item]))


def p4_1_reconstruct(contract: dict) -> dict:
    """Rebuild P4.1's B arm and accept it only against the published signature."""
    from scripts.training import run_p4_1_22_5 as p4_1

    frozen = json.loads(P4_1_CONTRACT.read_text(encoding="utf-8"))
    p4_1.guards(frozen, "probe")
    data = atomic_load(Path(frozen["data_root"]))
    assert value_digest(data) == frozen["data_digest"]
    published = json.loads(P4_1_REPORT.read_text(encoding="utf-8-sig"))
    root = Path(contract["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    doses = frozen["dose_points"]
    weight = float(frozen["aux_weight"])
    p4_1.build(frozen, "A")
    model, module, scaffold, optimizer = p4_1.build(frozen, "B")
    rows = []
    for update in range(1, max(doses) + 1):
        if time.process_time() > contract["cpu_cap"]:
            raise BudgetStop("p4_2_reconstruction_budget_stop")
        batch = p4_1.batch_at(data, frozen["order"], frozen["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss = p4_1.objective(model, module, scaffold, batch, weight)
        loss.backward()
        optimizer.step()
        if update not in doses:
            continue
        started = time.process_time()
        measured = p4_1.probe(model, data, 1.0)
        train = p4_1.accuracy(model, module, data["train"])
        fit = p4_1.accuracy(model, module, data["fit"])
        bundle = {
            "module": module.state_dict(),
            "encoder": {
                name: value.detach().clone()
                for name, value in model.encoder["rgb"].state_dict().items()
            },
            "scaffold": scaffold.state_dict(),
        }
        path = root / f"reconstruct-B-{update:06d}.pt"
        digest = save_checkpoint(path, bundle)
        rows.append(
            {
                "update": update,
                "fit_correct": fit["correct"],
                "train_correct": train["correct"],
                "colour_accuracy_fit": fit["colour_accuracy"],
                "colour_accuracy_train": train["colour_accuracy"],
                "centroid_column_hits_fit": fit["centroid_column_hits"],
                "centroid_column_total_fit": fit["centroid_column_total"],
                "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                "abstain_on_other_fit": fit["abstain_on_other"],
                "probe_painted_fit": measured["fit"]["painted"],
                "probe_overall_fit": measured["fit"]["overall"],
                "cpu": round(time.process_time() - started, 6),
                "bundle": str(path),
                "bundle_sha256": digest,
            }
        )
    published_rows = {row["update"]: published["arms"]["B"]["doses"] for row in rows}
    mismatches = []
    for row in rows:
        expected = next(
            item for item in published_rows[row["update"]] if item["update"] == row["update"]
        )
        for field in SIGNATURE:
            if expected.get(field) != row[field]:
                mismatches.append(
                    {
                        "update": row["update"],
                        "field": field,
                        "expected": expected.get(field),
                        "reconstructed": row[field],
                    }
                )
    report = {
        "schema": "taiji-u2-p4-2-reconstruction-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "source": "scripts/training/run_p4_1_22_5.py arm B, frozen contract unchanged",
        "doses": rows,
        "signature_fields": list(SIGNATURE),
        "mismatches": mismatches,
        "verdict": "p4_1_arm_b_reconstructed" if not mismatches else "p4_1_reconstruction_mismatch",
        "discipline": {"optimizer_calls": max(doses), "check_reads": 0, "final_reads": 0},
    }
    write_report(RECON_REPORT, report)
    if mismatches:
        raise ValueError("p4_1_reconstruction_mismatch")
    return report


def gradient_support(contract: dict, example: dict) -> tuple[list[str], list[str]]:
    """Byte-side support and the upstream support that produced the cached state."""
    model = build(contract)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    state = prepare(model, column_public(example), example["id"])
    total = sum(leaf.square().sum() for leaf in tensor_leaves(state))
    total.backward()
    upstream = sorted(name for name, p in model.named_parameters() if p.grad is not None)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = build(contract)
    state = prepared(model, example)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    byte_loss(model, state, example["target"]).backward()
    byte_side = sorted(name for name, p in model.named_parameters() if p.grad is not None)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return byte_side, upstream


def tensor_leaves(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.is_floating_point() and value.requires_grad else []
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in tensor_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in tensor_leaves(item)]
    return []


def census(contract: dict) -> int:
    guards(contract, "census")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    example = column_example(data["train"][0])
    byte_side, upstream = gradient_support(contract, example)
    emission = sorted(set(byte_side) - set(upstream))
    overlap = sorted(set(byte_side) & set(upstream))
    model = build(contract)
    report = {
        "schema": "taiji-u2-p4-2-census-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "scope": (
            "byte-side support with the prepared state held constant, minus every parameter "
            "that moves that state; no optimizer, no step"
        ),
        "emission_parameters": emission,
        "emission_parameter_count": len(emission),
        "emission_scalar_count": int(
            sum(p.numel() for n, p in model.named_parameters() if n in set(emission))
        ),
        "byte_side_support": byte_side,
        "upstream_support": upstream,
        "excluded_overlap": overlap,
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {"optimizer_calls": 0, "checkpoints": 0, "ledger_writes": 0},
    }
    write_report(CENSUS_REPORT, report)
    print(json.dumps({"emission_parameters": len(emission)}))
    return 0


def price(contract: dict) -> int:
    guards(contract, "price")
    from scripts.training import run_p4_1_22_5 as p4_1

    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    spec = contract["optimizer"]

    prepare_samples = []
    model = build(contract)
    for index in range(3):
        started = time.process_time()
        prepared(model, data["train"][index])
        prepare_samples.append(time.process_time() - started)
    per_prepare = statistics.median(prepare_samples)

    scaffold = nn.Linear(24, 4)
    for parameter in model.parameters():
        parameter.requires_grad_(not parameter_name(model, parameter).startswith("encoder.rgb."))
    optimizer = optimizer_for(list(model.parameters()) + list(scaffold.parameters()), spec)
    scaffold_samples = []
    for update in range(1, 4):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = scaffold_loss(model, scaffold, batch_at(data["train"], contract["batch"], update))
        loss.backward()
        optimizer.step()
        scaffold_samples.append(time.process_time() - started)
    per_scaffold = statistics.median(scaffold_samples)

    model = build(contract)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    set_requires_grad(model, set(contract["emission_parameters"]))
    cached = cache_positions(model, data["train"][: contract["batch"]])
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    emission_samples = []
    for update in range(1, 4):
        items = cached[((update - 1) * 8) % max(1, len(cached) - 8) :][:8]
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        emission_loss(model, items).backward()
        optimizer.step()
        emission_samples.append(time.process_time() - started)
    per_emission = statistics.median(emission_samples)

    model = build(contract)
    started = time.process_time()
    cache_positions(model, data["train"][:16])
    per_cache_position = (time.process_time() - started) / (16 * 7)

    model = build(contract)
    single = prepared(model, data["train"][0])
    optimizer = optimizer_for(list(model.parameters()), spec)
    single_samples = []
    for _ in range(3):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        byte_loss(model, single, data["train"][0]["target"]).backward()
        optimizer.step()
        single_samples.append(time.process_time() - started)
    per_full_single = statistics.median(single_samples)

    started = time.process_time()
    measured = round_evaluate(model, data["fit"])
    eval_cpu = time.process_time() - started

    frozen = json.loads(P4_1_CONTRACT.read_text(encoding="utf-8"))
    recon_model, recon_module, recon_scaffold, recon_optimizer = p4_1.build(frozen, "B")
    recon_samples = []
    for update in range(1, 4):
        started = time.process_time()
        recon_optimizer.zero_grad(set_to_none=True)
        loss = p4_1.objective(
            recon_model,
            recon_module,
            recon_scaffold,
            p4_1.batch_at(data, frozen["order"], frozen["batch"], update),
            float(frozen["aux_weight"]),
        )
        loss.backward()
        recon_optimizer.step()
        recon_samples.append(time.process_time() - started)
    per_recon = statistics.median(recon_samples)
    started = time.process_time()
    p4_1.probe(recon_model, data, 1.0)
    probe_cpu = time.process_time() - started

    doses = contract["scaffold_dose_points"]
    emission_doses = contract["byte_dose_points"]
    recon_cpu = max(doses) * per_recon + len(doses) * probe_cpu + 2.0
    scaffold_cpu = max(doses) * per_scaffold
    cache_cpu = (len(data["train"]) + len(data["fit"])) * 7 * per_cache_position
    byte_cpu = max(emission_doses) * per_emission
    eval_cpu_total = 2 * (len(doses) + len(emission_doses) + 1) * eval_cpu
    feedback_cpu = (
        contract["feedback_errors_allowance"]
        * contract["feedback_updates"]
        * (per_prepare + per_full_single)
        + 3 * eval_cpu
    )
    estimate = recon_cpu + scaffold_cpu + cache_cpu + byte_cpu + eval_cpu_total + feedback_cpu + 8.0
    cap = float(int(estimate * 1.15 + 0.999))
    report = {
        "schema": "taiji-u2-p4-2-price-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "measurements": {
            "per_prepare_cpu": round(per_prepare, 6),
            "per_scaffold_update_cpu": round(per_scaffold, 6),
            "per_byte_update_cpu": round(per_emission, 6),
            "per_cache_position_cpu": round(per_cache_position, 6),
            "per_full_single_update_cpu": round(per_full_single, 6),
            "per_fit_round_evaluate_cpu": round(eval_cpu, 6),
            "per_p4_1_reconstruction_update_cpu": round(per_recon, 6),
            "per_p4_1_probe_cpu": round(probe_cpu, 6),
            "prepare_samples": prepare_samples,
            "scaffold_samples": scaffold_samples,
            "byte_samples": emission_samples,
        },
        "estimate": {
            "reconstruction_cpu": round(recon_cpu, 6),
            "scaffold_curve_cpu": round(scaffold_cpu, 6),
            "cache_cpu": round(cache_cpu, 6),
            "byte_curve_cpu": round(byte_cpu, 6),
            "evaluation_cpu": round(eval_cpu_total, 6),
            "feedback_cpu_allowance": round(feedback_cpu, 6),
            "guard_and_margin_cpu": 8.0,
            "cap_cpu": cap,
        },
        "readings": {"fit_accuracy_at_price": measured["accuracy"]},
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "optimizer_calls": 12,
            "checkpoints": 0,
            "ledger_writes": 0,
            "check_reads": 0,
        },
        "note": "in-memory timing steps; nothing is saved apart from this receipt",
    }
    write_report(PRICE_REPORT, report)
    print(json.dumps({"cap_cpu": cap, "scaffold": round(scaffold_cpu, 3)}))
    return 0


def parameter_name(model, parameter) -> str:
    for name, value in model.named_parameters():
        if value is parameter:
            return name
    raise ValueError("unknown_parameter")


def body_log_probability(model, example: dict, position: int, symbol: int) -> float:
    """The body's own log-probability for one byte, through the full pipeline."""
    with torch.no_grad():
        state = prepare(model, column_public(example), example["id"])
        for index, target in enumerate([*example["target"], 256]):
            probability, _, details = output(model, state)
            if index == position:
                return float(probability[symbol])
            state = model.accept_byte(state, target, details)
    raise ValueError("position_out_of_range")


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    root = Path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("p4_2_run_already_started")

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p4_2_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p4_2_wall_stop")

    reconstruction = p4_1_reconstruct(contract)
    bundle_path = root / f"reconstruct-B-{max(contract['dose_points']):06d}.pt"
    bundle = torch.load(bundle_path, weights_only=False)
    model = build(contract, bundle["encoder"])
    init_encoder = value_digest(bundle["encoder"])

    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = nn.Linear(24, 4)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    phase1_names = set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    save_checkpoint(
        root / "initial.pt",
        {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "scaffold": {k: v.detach().clone() for k, v in scaffold.state_dict().items()},
            "encoder_sha256": init_encoder,
            "revision": model.parameter_revision,
        },
    )
    stage1_rows = []
    stage1_status = "max_dose"
    for update in range(1, max(contract["scaffold_dose_points"]) + 1):
        budget()
        optimizer.zero_grad(set_to_none=True)
        loss = scaffold_loss(model, scaffold, batch_at(data["train"], contract["batch"], update))
        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        ):
            raise ValueError("nonfinite_p4_2_gradient")
        optimizer.step()
        model.parameter_revision += 1
        if update not in contract["scaffold_dose_points"]:
            continue
        train = scaffold_accuracy(model, scaffold, data["train"][:32])
        fit = scaffold_accuracy(model, scaffold, data["fit"])
        path = root / f"phase1-{update:06d}.pt"
        digest = save_checkpoint(
            path,
            {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "scaffold": {k: v.detach().clone() for k, v in scaffold.state_dict().items()},
                "update": update,
                "revision": model.parameter_revision,
            },
        )
        stage1_rows.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "fit_accuracy": fit["accuracy"],
                "fit_correct": fit["correct"],
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
            }
        )
        if train["accuracy"] >= 0.95 and fit["accuracy"] >= 0.95:
            stage1_status = "scaffold_converged"
            break
        if update == contract["scaffold_dose_points"][1] and all(
            row["fit_accuracy"] <= 0.35 for row in stage1_rows
        ):
            stage1_status = "scaffold_cannot_read_answer"
            break

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase2_names = set_requires_grad(model, set(contract["emission_parameters"]))
    positions = cache_positions(model, data["train"])
    checks = []
    for item in positions[:: max(1, len(positions) // 3)][:3]:
        example = next(e for e in data["train"] if e["id"] == item["example"])
        body = body_log_probability(model, example, item["position"], item["symbol"])
        replay = -float(emission_loss(model, [item]))
        checks.append(
            {
                "example": item["example"],
                "position": item["position"],
                "body_log_probability": body,
                "replayed": replay,
                "equal": abs(body - replay) <= 1e-6,
            }
        )
    if not all(check["equal"] for check in checks):
        raise ValueError("p4_2_cache_replay_mismatch")

    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    stage2_rows = []
    for update in range(1, max(contract["byte_dose_points"]) + 1):
        budget()
        items = [
            positions[((update - 1) * contract["batch"] + offset) % len(positions)]
            for offset in range(contract["batch"])
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = emission_loss(model, items)
        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
            if p.requires_grad
        ):
            raise ValueError("nonfinite_p4_2_emission_gradient")
        optimizer.step()
        model.parameter_revision += 1
        if update not in contract["byte_dose_points"]:
            continue
        train = round_evaluate(model, data["train"][:32], budget_guard=budget)
        fit = round_evaluate(model, data["fit"], budget_guard=budget)
        path = root / f"phase2-{update:06d}.pt"
        digest = save_checkpoint(
            path,
            {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "update": update,
                "revision": model.parameter_revision,
            },
        )
        stage2_rows.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "train_correct": train["correct"],
                "fit_accuracy": fit["accuracy"],
                "fit_correct": fit["correct"],
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
            }
        )

    first = round_evaluate(model, data["fit"], budget_guard=budget)
    errors = [row for row in first["rows"] if not row["correct"]]
    control = copy.deepcopy(model)
    control_before = value_digest(control.state_dict())
    control_second = round_evaluate(control, data["fit"], budget_guard=budget)
    control_after = value_digest(control.state_dict())
    learned = copy.deepcopy(model)
    feedback_optimizer = optimizer_for(learned.parameters(), contract["feedback_optimizer"])
    feedback_updates = 0
    for row in errors:
        example = next(e for e in data["fit"] if e["id"] == row["id"])
        for _ in range(contract["feedback_updates"]):
            budget()
            feedback_optimizer.zero_grad(set_to_none=True)
            state = prepare(learned, column_public(example), example["id"])
            byte_loss(learned, state, example["target"]).backward()
            feedback_optimizer.step()
            learned.parameter_revision += 1
            feedback_updates += 1
    learned_second = round_evaluate(learned, data["fit"], budget_guard=budget)
    learned_hits = {row["id"]: row["correct"] for row in learned_second["rows"]}
    corrected = sum(learned_hits[row["id"]] for row in errors)
    correction = corrected / len(errors) if len(errors) >= 4 else None
    gain = learned_second["accuracy"] - control_second["accuracy"] if len(errors) >= 4 else None
    stable = [row["id"] for row in first["rows"] if row["correct"] and not learned_hits[row["id"]]]
    learned_path = root / "feedback-learned.pt"
    learned_digest = save_checkpoint(
        learned_path,
        {
            "model": {k: v.detach().clone() for k, v in learned.state_dict().items()},
            "update": "feedback",
            "revision": learned.parameter_revision,
        },
    )

    final_path = root / "final.pt"
    final_digest = save_checkpoint(
        final_path,
        {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "update": max(contract["byte_dose_points"]),
            "revision": model.parameter_revision,
        },
    )
    restored = build(contract)
    restored.load_state_dict(torch.load(final_path, weights_only=False)["model"])
    recovered = round_evaluate(restored, data["fit"], budget_guard=budget)
    recovery_equal = recovered["rows"] == first["rows"]

    final_fit = first["accuracy"]
    main_gate = final_fit >= contract["criteria"]["main_gate"]
    gate_rows = {
        "full_answer_round_fit": "passed" if main_gate else "failed",
        "correction": (
            "passed"
            if correction is not None and correction >= contract["criteria"]["correction_gate"]
            else "failed" if correction is not None else "not_identifiable_by_design"
        ),
        "feedback_gain": (
            "passed"
            if gain is not None and gain >= contract["criteria"]["feedback_gain_gate"]
            else "failed" if gain is not None else "not_identifiable_by_design"
        ),
        "recovery": "passed" if recovery_equal else "failed",
        "world_paired_95_lower": "not_in_scope_single_seed",
        "three_seed_direction": "not_in_scope_single_seed",
        "a_retention": "not_applicable_no_A_parent",
        "check2": "not_read",
    }
    if (
        main_gate
        and gate_rows["correction"] != "failed"
        and gate_rows["feedback_gain"] != "failed"
        and recovery_equal
    ):
        verdict = "byte_path_carries_answer"
    elif main_gate:
        verdict = "byte_path_carries_answer_gates_incomplete"
    elif stage2_rows and stage2_rows[-1]["fit_accuracy"] - stage2_rows[0]["fit_accuracy"] >= 0.10:
        verdict = "byte_path_learning_below_gate"
    else:
        verdict = "byte_path_blocks"
    report = {
        "schema": "taiji-u2-p4-2-byte-path-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "phase": "column_pathway_through_byte_emission",
        "verdict": verdict,
        "reconstruction": {
            "verdict": reconstruction["verdict"],
            "doses": reconstruction["doses"],
            "mismatches": reconstruction["mismatches"],
            "encoder_sha256": init_encoder,
        },
        "stage1": {
            "status": stage1_status,
            "doses": stage1_rows,
            "trainable_parameters": len(phase1_names),
            "scaffold_parameters": sum(p.numel() for p in scaffold.parameters()),
            "scaffold_role": "training-only 4-way head on the read context; never at evaluation",
        },
        "stage2": {
            "doses": stage2_rows,
            "trainable_parameters": phase2_names,
            "cache_positions": len(positions),
            "cache_replay_checks": checks,
        },
        "feedback": {
            "first_pass_correct": 32 - len(errors),
            "first_pass_errors": len(errors),
            "control_second_pass": control_second["accuracy"],
            "learned_second_pass": learned_second["accuracy"],
            "correction_rate": correction,
            "feedback_gain": gain,
            "feedback_updates": feedback_updates,
            "control_updates": 0,
            "control_state_unchanged": control_before == control_after,
            "learned_checkpoint": str(learned_path),
            "learned_sha256": learned_digest,
            "stability_losses": len(stable),
        },
        "recovery": {"equal_in_process": recovery_equal, "final_sha256": final_digest},
        "gate_table": gate_rows,
        "criteria": contract["criteria"],
        "optimizer": contract["optimizer"],
        "data_digest": contract["data_digest"],
        "init_encoder_sha256": init_encoder,
        "cost": {
            "total_cpu": round(time.process_time() - cpu_start, 6),
            "total_wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": (
            "development fit only; the served output is the subject's byte head; "
            "a parent candidate, not an accepted learning parent"
        ),
    }
    write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "fit": final_fit,
                "stage1": stage1_status,
                "cost": report["cost"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("census", "price", "reconstruct", "run"), default="census"
    )
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "census":
        return census(contract)
    if args.mode == "price":
        return price(contract)
    if args.mode == "reconstruct":
        guards(contract, "reconstruct")
        p4_1_reconstruct(contract)
        return 0
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
