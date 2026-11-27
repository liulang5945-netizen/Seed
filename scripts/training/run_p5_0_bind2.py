"""P5.0/P5.1: pre-registered three-arm pilot for object-centric binding v2.

The column branch closed as ``read_still_blind``: the encoder carries colour, the
byte head is not the wall, and the three object slots never form a distinct,
correct-sized set. This runner implements the design card's pilot only:

  * A: full binding v2 (identity anchor + two in-binding steps);
  * B: one in-binding step (anchor kept);
  * C: two steps with the anchor disabled at query time.

The pilot is a liveness and checkpoint gate, not a capability measurement. Arm A
alone decides stop/go; B and C are attribution arms. Nothing here writes a
capability or parent claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training import run_p4_3_22_5 as p4_3
from scripts.training import run_p4_4_22_5 as p4_4
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import prepare
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import U2Config, U2Subject

CONTRACT = ROOT / "plans/manifests/taiji_u2_bind_2_pilot_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_bind2_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_bind2_pilot_20261115.json"
COLOURS = 6
NONE = 6
SLOTS = 3
PERMUTATIONS = list(itertools.permutations(range(SLOTS)))
NEW_TENSORS = (
    "slot_anchor",
    "slot_norm.bias",
    "slot_norm.weight",
    "slot_update.first.bias",
    "slot_update.first.weight",
    "slot_update.last.bias",
    "slot_update.last.weight",
)


class BudgetStop(RuntimeError):
    pass


def root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("p4_3_runner", ROOT / "scripts/training/run_p4_3_22_5.py"),
        ("p4_4_runner", ROOT / "scripts/training/run_p4_4_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("design_manifest", ROOT / "plans/manifests/taiji_u2_bind_2_design_20261115.json"),
    ):
        if pins.get(name) != p4_2.sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


def build_v2(contract: dict, encoder: dict | None = None):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = U2Subject(replace(U2Config.small(), seed=seed, binding="v2"))
    model = model.metadata_candidate().request_candidate()
    if encoder is not None:
        model.encoder["rgb"].load_state_dict(encoder)
    return model


def label_tensors(examples: list[dict], labels: dict):
    colours = torch.full((len(examples), SLOTS), NONE, dtype=torch.long)
    positions = torch.zeros((len(examples), SLOTS), dtype=torch.long)
    mask = torch.zeros((len(examples), SLOTS))
    for index, example in enumerate(examples):
        for slot, (colour, where) in enumerate(labels[example["id"]][:SLOTS]):
            colours[index, slot] = colour
            positions[index, slot] = where
            mask[index, slot] = 1.0
    return colours, positions, mask


def scaffold_loss(model, scaffold, batch: list[dict], labels: dict) -> torch.Tensor:
    colour_rows, position_rows = [], []
    for example in batch:
        state = prepare(model, p4_2.column_public(example), example["id"])
        colour, position = scaffold(model.object_z(state))
        colour_rows.append(colour)
        position_rows.append(position)
    colours = torch.stack(colour_rows)
    positions = torch.stack(position_rows)
    target_colour, target_position, mask = label_tensors(batch, labels)
    best = None
    for permutation in PERMUTATIONS:
        colour_loss = functional.cross_entropy(
            colours[:, permutation, :].reshape(-1, COLOURS + 1),
            target_colour.reshape(-1),
            reduction="none",
        ).view(len(batch), SLOTS)
        position_loss = functional.cross_entropy(
            positions[:, permutation, :].reshape(-1, 3),
            target_position.reshape(-1),
            reduction="none",
        ).view(len(batch), SLOTS)
        loss = (colour_loss.sum(1) + (position_loss * mask).sum(1)) / (
            SLOTS + mask.sum(1)
        ).clamp_min(1)
        best = loss if best is None else torch.minimum(best, loss)
    return best.mean()


def predicted_pairs(colour_logits: torch.Tensor, position_logits: torch.Tensor):
    pairs = []
    for slot in range(colour_logits.shape[0]):
        colour = int(colour_logits[slot].argmax())
        if colour == NONE:
            continue
        pairs.append((colour, int(position_logits[slot].argmax())))
    return pairs


def scaffold_accuracy(model, scaffold, examples: list[dict], labels: dict) -> dict:
    pair_hits = pair_total = set_hits = duplicate_scenes = count_hits = 0
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepared(model, example)
            colour, position = scaffold(model.object_z(state))
            predicted = predicted_pairs(colour, position)
            truth = labels[example["id"]]
            pair_hits += len(set(predicted) & set(truth))
            pair_total += len(truth)
            set_hits += int(sorted(predicted) == sorted(truth))
            duplicate_scenes += int(len(predicted) != len({colour for colour, _ in predicted}))
            count_hits += int(len(predicted) == len(truth))
    return {
        "pair_accuracy": pair_hits / pair_total if pair_total else 1.0,
        "set_accuracy": set_hits / len(examples),
        "pair_hits": pair_hits,
        "pair_total": pair_total,
        "set_correct": set_hits,
        "denominator": len(examples),
        "duplicate_scenes": duplicate_scenes,
        "duplicate_rate": duplicate_scenes / len(examples),
        "count_accuracy": count_hits / len(examples),
    }


def digest_outputs(model, scaffold, examples: list[dict], labels: dict) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepared(model, example)
            z = model.object_z(state)
            colour, position = scaffold(z)
            digest.update(z.numpy().tobytes())
            digest.update(colour.numpy().tobytes())
            digest.update(position.numpy().tobytes())
            digest.update(repr(labels[example["id"]]).encode("utf-8"))
    return digest.hexdigest()


def binding_diagnostics(model, examples: list[dict], labels: dict) -> dict:
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = model.binding

    def wrapper(state, tokens, coordinates):
        result = original(state, tokens, coordinates)
        if result is not None:
            captured.append((result[0].detach().clone(), result[1].detach().clone()))
        return result

    model.binding = wrapper
    scene_rows = []
    all_content = []
    all_mass = []
    try:
        with torch.no_grad():
            for example in examples:
                captured.clear()
                p4_2.prepared(model, example)
                if not captured:
                    continue
                content, mass = captured[-1]
                content = content.mean(0)
                mass = mass.mean(0)
                all_content.append(content)
                all_mass.append(mass)
                if len(labels[example["id"]]) == 3:
                    scene_rows.append(
                        {
                            "id": example["id"],
                            "mass": [round(float(value), 6) for value in mass],
                            "alive": int((mass >= 0.05).sum()),
                        }
                    )
    finally:
        model.binding = original
    if not all_content:
        return {"scenes": 0}
    stacked = torch.stack(all_content)
    unit = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = unit @ unit.transpose(-1, -2)
    off_diagonal = cosine[:, ~torch.eye(SLOTS, dtype=torch.bool)]
    mass = torch.stack(all_mass)
    return {
        "scenes": len(all_content),
        "three_object_scenes": len(scene_rows),
        "scene_fraction_alive": (
            sum(row["alive"] >= 2 for row in scene_rows) / len(scene_rows) if scene_rows else 0.0
        ),
        "mean_pairwise_cosine": float(off_diagonal.mean()),
        "max_pairwise_cosine": float(off_diagonal.max()),
        "mean_slot_mass": [round(float(value), 6) for value in mass.mean(0)],
        "scenes_detail": scene_rows[:8],
    }


def recover(path: Path, contract: dict) -> str:
    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data_root"]))
    labels = p4_4.label_cache(data)
    checkpoint = torch.load(path, weights_only=False)
    model = build_v2(contract)
    model.load_state_dict(checkpoint["model"])
    model._binding_steps = checkpoint["arm"]["steps"]
    model._binding_anchor = checkpoint["arm"]["anchor"]
    scaffold = p4_3.ObjectScaffold()
    scaffold.load_state_dict(checkpoint["scaffold"])
    return digest_outputs(model, scaffold, data["fit"][:3], labels)


def run_arm(
    contract: dict,
    arm: dict,
    data: dict,
    encoder: dict,
    labels: dict,
    root: Path,
    budget,
) -> dict:
    model = build_v2(contract, encoder)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    names = p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    model._binding_steps = arm["steps"]
    model._binding_anchor = arm["anchor"]
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = p4_3.ObjectScaffold()
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    losses: list[float] = []
    for update in range(1, int(contract["dose"]) + 1):
        budget()
        batch = p4_2.batch_at(data["train"], int(contract["batch"]), update)
        optimizer.zero_grad(set_to_none=True)
        loss = scaffold_loss(model, scaffold, batch, labels)
        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in [*model.parameters(), *scaffold.parameters()]
        ):
            raise ValueError("nonfinite_bind2_gradient")
        optimizer.step()
        model.parameter_revision += 1
        losses.append(float(loss.detach()))
    path = root / f"arm-{arm['name']}-{int(contract['dose']):06d}.pt"
    digest = p4_2.save_checkpoint(
        path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "scaffold": {
                key: value.detach().clone() for key, value in scaffold.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "update": int(contract["dose"]),
            "revision": model.parameter_revision,
            "arm": dict(arm),
        },
    )
    train = scaffold_accuracy(model, scaffold, data["train"][:32], labels)
    fit = scaffold_accuracy(model, scaffold, data["fit"], labels)
    diagnostics = binding_diagnostics(model, data["fit"], labels)
    in_process = digest_outputs(model, scaffold, data["fit"][:3], labels)
    independent = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    recovered = json.loads(independent.stdout.strip().splitlines()[-1])["digest"]
    new_grads = {
        name: (
            None
            if dict(model.named_parameters())[name].grad is None
            else bool(torch.isfinite(dict(model.named_parameters())[name].grad).all())
        )
        for name in NEW_TENSORS
    }
    return {
        "name": arm["name"],
        "steps": arm["steps"],
        "anchor": arm["anchor"],
        "update": int(contract["dose"]),
        "trainable_parameters": len(names),
        "loss_first50": statistics.fmean(losses[:50]),
        "loss_last50": statistics.fmean(losses[-50:]),
        "train": train,
        "fit": fit,
        "binding": diagnostics,
        "new_tensor_grads": new_grads,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "recovery": {
            "in_process_digest": in_process,
            "independent_digest": recovered,
            "identical": in_process == recovered,
        },
    }


def pilot_gates(arm: dict, contract: dict) -> dict:
    criteria = contract["criteria"]
    new_grads = arm["new_tensor_grads"]
    return {
        "finite_new_tensor_grads": all(value is True for value in new_grads.values()),
        "checkpoint_reload_identical": arm["recovery"]["identical"],
        "loss_decreases": arm["loss_last50"] < arm["loss_first50"],
        "two_slots_alive": arm["binding"].get("scene_fraction_alive", 0.0)
        >= criteria["slot_mass_scene_fraction"],
        "content_not_identical": arm["binding"].get("mean_pairwise_cosine", 1.0)
        <= criteria["slot_content_cosine_max"],
    }


def graph_receipt(contract: dict) -> dict:
    seed = int(contract["candidate"]["seed"])
    v1 = U2Subject(U2Config(seed=seed))
    v2 = U2Subject(replace(U2Config(seed=seed), binding="v2"))
    p1 = dict(v1.named_parameters())
    p2 = dict(v2.named_parameters())
    shared = sorted(set(p1) & set(p2))
    digest = hashlib.sha256()
    for name in shared:
        digest.update(name.encode("utf-8"))
        digest.update(p1[name].detach().numpy().tobytes())
    return {
        "v1_scalars": sum(value.numel() for value in p1.values()),
        "v1_tensors": len(p1),
        "v2_scalars": sum(value.numel() for value in p2.values()),
        "v2_tensors": len(p2),
        "state_fields": len(v2.state_registry()),
        "new_tensors": sorted(set(p2) - set(p1)),
        "shared_bit_identical": all(torch.equal(p1[name], p2[name]) for name in shared),
        "shared_parameter_digest": digest.hexdigest(),
    }


def initial_gate(contract: dict, data: dict, encoder: dict, labels: dict, root: Path) -> dict:
    model = build_v2(contract, encoder)
    scaffold = p4_3.ObjectScaffold()
    path = root / "initial.pt"
    digest = p4_2.save_checkpoint(
        path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "scaffold": {
                key: value.detach().clone() for key, value in scaffold.state_dict().items()
            },
            "update": 0,
            "revision": model.parameter_revision,
            "arm": {"name": "initial", "steps": 2, "anchor": True},
        },
    )
    in_process = digest_outputs(model, scaffold, data["fit"][:3], labels)
    independent = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    recovered = json.loads(independent.stdout.strip().splitlines()[-1])["digest"]
    optimizer = p4_2.optimizer_for(model.parameters(), contract["optimizer"])
    batch = data["train"][: int(contract["batch"])]
    optimizer.zero_grad(set_to_none=True)
    scaffold_loss(model, scaffold, batch, labels).backward()
    optimizer.step()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    in_optimizer = {
        names[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in names
    }
    step_path = root / "engineering-step.pt"
    step_digest = p4_2.save_checkpoint(
        step_path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "scaffold": {
                key: value.detach().clone() for key, value in scaffold.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "update": 1,
            "revision": model.parameter_revision + 1,
            "arm": {"name": "engineering", "steps": 2, "anchor": True},
        },
    )
    saved = torch.load(step_path, weights_only=False)
    rebuilt = build_v2(contract)
    rebuilt.load_state_dict(saved["model"])
    reloaded_optimizer = p4_2.optimizer_for(rebuilt.parameters(), contract["optimizer"])
    reloaded_optimizer.load_state_dict(saved["optimizer"])
    return {
        "initial_checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "initial_sha256": digest,
        "initial_recovery_identical": in_process == recovered,
        "engineering_step_checkpoint": str(step_path.relative_to(ROOT)).replace("\\", "/"),
        "engineering_step_sha256": step_digest,
        "engineering_model_reload_identical": all(
            torch.equal(saved["model"][key], value) for key, value in rebuilt.state_dict().items()
        ),
        "engineering_optimizer_state_entries": len(reloaded_optimizer.state),
        "new_tensors_in_optimizer": sorted(set(NEW_TENSORS) & in_optimizer),
    }


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    label_started = time.process_time()
    labels = p4_4.label_cache(data)
    label_cpu = time.process_time() - label_started
    encoder = p4_3.load_p4_1_bundle(contract)
    model = build_v2(contract, encoder)
    model._binding_steps = 2
    model._binding_anchor = True
    scaffold = p4_3.ObjectScaffold()
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    batch = data["train"][: int(contract["batch"])]
    model.zero_grad(set_to_none=True)
    scaffold_loss(model, scaffold, batch, labels).backward()
    optimizer.zero_grad(set_to_none=True)
    samples = []
    for _ in range(5):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        scaffold_loss(model, scaffold, batch, labels).backward()
        samples.append(time.process_time() - started)
    step_cpu = statistics.median(samples)
    started = time.process_time()
    with torch.no_grad():
        scaffold_accuracy(model, scaffold, data["fit"], labels)
    eval_cpu = time.process_time() - started
    dose = int(contract["dose"])
    pilot_cpu = 3 * dose * step_cpu + 3 * eval_cpu + 20.0
    stage1_cpu = 2 * 1844 * step_cpu + 2 * eval_cpu + 60.0
    payload = {
        "schema": "taiji-u2-bind2-price-v1",
        "card": "22.5.P5.1",
        "date": "2026-11-15",
        "label_cache_cpu": round(label_cpu, 6),
        "step_cpu_samples": [round(value, 6) for value in samples],
        "step_cpu_median": round(step_cpu, 6),
        "fit_eval_cpu": round(eval_cpu, 6),
        "pilot_estimate_cpu": round(pilot_cpu, 6),
        "stage1_estimate_cpu": round(stage1_cpu, 6),
        "pilot_cap_cpu": contract["cpu_cap"],
        "stage1_design_cap_cpu": contract["stage1_cpu_cap"],
        "pricing_over_budget": bool(
            pilot_cpu > contract["cpu_cap"] or stage1_cpu > contract["stage1_cpu_cap"]
        ),
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "training": 0,
            "optimizer_step": 0,
            "check_final_reads": 0,
            "capability_claims": 0,
        },
    }
    write_report(PRICE_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    labels = p4_4.label_cache(data)
    encoder = p4_3.load_p4_1_bundle(contract)
    root = root_path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("bind2_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("bind2_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("bind2_wall_stop")

    receipt = graph_receipt(contract)
    initial = initial_gate(contract, data, encoder, labels, root)
    arms = []
    status = "pilot_complete"
    try:
        for arm in contract["arms"]:
            arms.append(run_arm(contract, arm, data, encoder, labels, root, budget))
    except BudgetStop:
        status = "budget_stop"
    gates = pilot_gates(arms[0], contract) if arms else {}
    if status == "budget_stop" or not arms:
        verdict = "budget_stop"
    elif all(gates.values()):
        verdict = "pilot_passed"
    else:
        verdict = "binding_pilot_collapsed"
    payload = {
        "schema": "taiji-u2-bind2-pilot-v1",
        "card": "22.5.P5.1",
        "date": "2026-11-15",
        "status": status,
        "verdict": verdict,
        "candidate": contract["candidate"],
        "graph": receipt,
        "inheritance": {
            "parent_bundle": contract["p4_1_bundle"],
            "parent_bundle_sha256": contract["p4_1_bundle_sha256"],
            "encoder_sha256": hashlib.sha256(
                encoder["weight"].detach().numpy().tobytes()
                + encoder["bias"].detach().numpy().tobytes()
            ).hexdigest(),
            "shared_v1_tensors_reproduced": receipt["shared_bit_identical"],
        },
        "initial_gate": initial,
        "arms": arms,
        "pilot_gates": gates,
        "criteria": contract["criteria"],
        "data_digest": contract["data_digest"],
        "parent_bundle_sha256": contract["p4_1_bundle_sha256"],
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "scope": "pilot only: liveness, checkpoint recovery and slot-mass diagnostics",
    }
    write_report(RUN_REPORT, payload)
    print(json.dumps({"verdict": verdict, "status": status, "arms": len(arms)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if args.recover is not None:
        print(json.dumps({"digest": recover(args.recover, contract)}, ensure_ascii=False))
        return 0
    if args.price:
        return price(contract)
    return run(contract)


if __name__ == "__main__":
    sys.exit(main())
