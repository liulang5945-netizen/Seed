"""P4.4: slot competition on the object read.

P4.3 taught the read per-slot colour/position but the exact object set stayed at
zero: duplicate colours (6/32), wrong slot counts (20/32), all three colours
present in only 3/32. This card keeps the encoder, data, observation layout and
the byte head, and changes only the read scaffold:

  * labels use the world's exact positions for the colours the image actually
    shows (visible subset, 0-3 pairs), removing the mean-column bin noise;
  * every slot gets an existence gate (1 for a real object, 0 for padding);
  * a soft duplicate-colour penalty discourages two slots from claiming the
    same object.

Verdicts are frozen before any reading; see the manifest.
"""

from __future__ import annotations

import argparse
import itertools
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

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training import run_p4_3_22_5 as p4_3
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import RGB_COLORS, prepare
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_episode import position_word
from taiji.u2_task_world import PhysicalWorld

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_4_slot_competition_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p4_4_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p4_4_slot_competition_20261115.json"
LABELS = ("左边", "中间", "右边", "未知")
COLOURS = 6
NONE = 6
SLOTS = 3
PERMUTATIONS = list(itertools.permutations(range(SLOTS)))


class BudgetStop(RuntimeError):
    pass


class ObjectScaffoldV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.colour = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, COLOURS + 1))
        self.position = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 3))
        self.existence = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, z: torch.Tensor):
        z = z.mean(0)
        return self.colour(z), self.position(z), self.existence(z).squeeze(-1)


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("p4_3_runner", ROOT / "scripts/training/run_p4_3_22_5.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
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


def world_pairs(example: dict) -> list[tuple[int, int]]:
    audit = example["audit"]
    world = PhysicalWorld.reset(audit["seed"], {"body_count": audit["objects"]})
    world.elapse("public-scene")
    pairs = []
    for body in world.snapshot()["bodies"]:
        colour = RGB_COLORS.index(tuple(body["color"]))
        where = LABELS.index(position_word(body["position"][0]))
        pairs.append((colour, where))
    return pairs


def label_cache(data: dict) -> dict:
    labels = {}
    for split in ("train", "fit"):
        for example in data[split]:
            visible = {colour for colour, _ in p4_3.visible_pairs(example)}
            labels[example["id"]] = [pair for pair in world_pairs(example) if pair[0] in visible]
    return labels


def label_tensors(examples: list[dict], labels: dict):
    colours = torch.full((len(examples), SLOTS), NONE, dtype=torch.long)
    positions = torch.zeros((len(examples), SLOTS), dtype=torch.long)
    position_mask = torch.zeros((len(examples), SLOTS))
    existence = torch.zeros((len(examples), SLOTS))
    for index, example in enumerate(examples):
        for slot, (colour, where) in enumerate(labels[example["id"]][:SLOTS]):
            colours[index, slot] = colour
            positions[index, slot] = where
            position_mask[index, slot] = 1.0
            existence[index, slot] = 1.0
    return colours, positions, position_mask, existence


def duplicate_penalty(colour_logits: torch.Tensor) -> torch.Tensor:
    probabilities = colour_logits.softmax(-1)[..., :COLOURS]
    total = 0.0
    for first, second in itertools.combinations(range(SLOTS), 2):
        total = total + (probabilities[:, first] * probabilities[:, second]).sum(-1)
    return total


def scaffold_loss(model, scaffold, batch: list[dict], labels: dict) -> torch.Tensor:
    colour_rows, position_rows, existence_rows = [], [], []
    for example in batch:
        state = prepare(model, p4_2.column_public(example), example["id"])
        colour, position, existence = scaffold(model.object_z(state))
        colour_rows.append(colour)
        position_rows.append(position)
        existence_rows.append(existence)
    colours = torch.stack(colour_rows)
    positions = torch.stack(position_rows)
    existence = torch.stack(existence_rows)
    target_colour, target_position, mask, target_existence = label_tensors(batch, labels)
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
    existence_loss = functional.binary_cross_entropy_with_logits(
        existence, target_existence, reduction="none"
    ).mean(1)
    duplicate = duplicate_penalty(colours)
    return (best + 0.5 * existence_loss + 0.5 * duplicate).mean()


def predicted_pairs(colour_logits: torch.Tensor, position_logits: torch.Tensor, existence):
    pairs = []
    for slot in range(colour_logits.shape[0]):
        if float(existence[slot].sigmoid()) <= 0.5:
            continue
        colour = int(colour_logits[slot].argmax())
        if colour == NONE:
            continue
        pairs.append((colour, int(position_logits[slot].argmax())))
    return pairs


def scaffold_accuracy(model, scaffold, examples: list[dict], labels: dict) -> dict:
    pair_hits = pair_total = set_hits = 0
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepared(model, example)
            colour, position, existence = scaffold(model.object_z(state))
            predicted = predicted_pairs(colour, position, existence)
            truth = labels[example["id"]]
            pair_hits += len(set(predicted) & set(truth))
            pair_total += len(truth)
            set_hits += int(sorted(predicted) == sorted(truth))
    return {
        "pair_accuracy": pair_hits / pair_total if pair_total else 1.0,
        "set_accuracy": set_hits / len(examples),
        "pair_hits": pair_hits,
        "pair_total": pair_total,
        "set_correct": set_hits,
        "denominator": len(examples),
    }


def stage1(contract: dict, data: dict, encoder: dict, labels: dict, root: Path, budget) -> tuple:
    model = p4_2.build(contract, encoder)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    names = p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = ObjectScaffoldV2()
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    rows = []
    status = "max_dose"
    for update in range(1, max(contract["scaffold_dose_points"]) + 1):
        budget()
        batch = p4_2.batch_at(data["train"], contract["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss = scaffold_loss(model, scaffold, batch, labels)
        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        ):
            raise ValueError("nonfinite_p4_4_gradient")
        optimizer.step()
        model.parameter_revision += 1
        if update not in contract["scaffold_dose_points"]:
            continue
        train = scaffold_accuracy(model, scaffold, data["train"][:32], labels)
        fit = scaffold_accuracy(model, scaffold, data["fit"], labels)
        path = root / f"read-{update:06d}.pt"
        digest = p4_2.save_checkpoint(
            path,
            {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "scaffold": {k: v.detach().clone() for k, v in scaffold.state_dict().items()},
                "update": update,
                "revision": model.parameter_revision,
            },
        )
        rows.append(
            {
                "update": update,
                "train_pair_accuracy": train["pair_accuracy"],
                "train_set_accuracy": train["set_accuracy"],
                "fit_pair_accuracy": fit["pair_accuracy"],
                "fit_set_accuracy": fit["set_accuracy"],
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
            }
        )
        criteria = contract["criteria"]
        if (
            train["pair_accuracy"] >= criteria["read_pair_gate"]
            and fit["pair_accuracy"] >= criteria["read_pair_gate"]
            and fit["set_accuracy"] >= criteria["read_set_gate"]
        ):
            status = "read_taught"
            break
        if update == contract["scaffold_dose_points"][1] and all(
            row["fit_pair_accuracy"] <= criteria["read_flat"] for row in rows
        ):
            status = "read_still_blind"
            break
    return model, scaffold, names, rows, status


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    labels = label_cache(data)
    encoder = p4_3.load_p4_1_bundle(contract)
    root = Path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("p4_4_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p4_4_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p4_4_wall_stop")

    initial = p4_2.build(contract, encoder)
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = ObjectScaffoldV2()
    p4_2.save_checkpoint(
        root / "initial.pt",
        {
            "model": {k: v.detach().clone() for k, v in initial.state_dict().items()},
            "scaffold": {k: v.detach().clone() for k, v in scaffold.state_dict().items()},
            "revision": initial.parameter_revision,
        },
    )
    model, scaffold, read_names, read_rows, read_status = stage1(
        contract, data, encoder, labels, root, budget
    )
    taught = read_status == "read_taught"
    feedback: dict = {}
    byte_rows: list[dict] = []
    byte_names: list[str] = []
    checks: list[dict] = []
    final = None
    recovery_equal = None
    final_digest = None
    positions_count = 0
    if taught:
        byte_names, positions, checks, byte_rows = p4_3.stage2(contract, data, model, root, budget)
        positions_count = len(positions)
        feedback = p4_3.feedback_round(contract, data, model, budget)
        learned = feedback.pop("learned_model")
        learned_path = root / "feedback-learned.pt"
        feedback["learned_checkpoint"] = str(learned_path)
        feedback["learned_sha256"] = p4_2.save_checkpoint(
            learned_path,
            {
                "model": {k: v.detach().clone() for k, v in learned.state_dict().items()},
                "update": "feedback",
                "revision": learned.parameter_revision,
            },
        )
        final = p4_2.round_evaluate(model, data["fit"], budget_guard=budget)
        final_path = root / "final.pt"
        final_digest = p4_2.save_checkpoint(
            final_path,
            {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "update": max(contract["byte_dose_points"]),
                "revision": model.parameter_revision,
            },
        )
        restored = p4_2.build(contract)
        restored.load_state_dict(torch.load(final_path, weights_only=False)["model"])
        recovered = p4_2.round_evaluate(restored, data["fit"], budget_guard=budget)
        recovery_equal = recovered["rows"] == final["rows"]
    criteria = contract["criteria"]
    read_final = read_rows[-1] if read_rows else {"fit_pair_accuracy": 0.0, "fit_set_accuracy": 0.0}
    main_gate = bool(final and final["accuracy"] >= criteria["main_gate"])
    joint_ok = False
    if feedback:
        correction = feedback["correction_rate"]
        gain = feedback["feedback_gain"]
        joint_ok = (
            (correction is None or correction >= criteria["correction_gate"])
            and (gain is None or gain >= criteria["feedback_gain_gate"])
            and feedback["stability_losses"] <= feedback["stability_ceiling"]
        )
    if read_status == "read_still_blind":
        verdict = "read_still_blind"
    elif read_status != "read_taught":
        verdict = "read_partial"
    elif main_gate and joint_ok and recovery_equal:
        verdict = "read_taught_byte_path_carries_answer"
    else:
        verdict = "read_taught_byte_path_below_gate"

    def mark(value, blind="not_run_read_blind"):
        if not taught:
            return blind
        if value is None:
            return "not_identifiable_by_design"
        return "passed" if value else "failed"

    gate_table = {
        "read_pair": mark(read_final["fit_pair_accuracy"] >= criteria["read_pair_gate"], "failed"),
        "read_set": mark(read_final["fit_set_accuracy"] >= criteria["read_set_gate"], "failed"),
        "full_answer_round_fit": mark(main_gate),
        "correction": mark(
            None
            if not feedback or feedback.get("correction_rate") is None
            else feedback["correction_rate"] >= criteria["correction_gate"]
        ),
        "feedback_gain": mark(
            None
            if not feedback or feedback.get("feedback_gain") is None
            else feedback["feedback_gain"] >= criteria["feedback_gain_gate"]
        ),
        "stability": mark(
            None if not feedback else feedback["stability_losses"] <= feedback["stability_ceiling"]
        ),
        "recovery": mark(recovery_equal),
        "world_paired_95_lower": "not_in_scope_single_seed",
        "three_seed_direction": "not_in_scope_single_seed",
        "a_retention": "not_applicable_no_A_parent",
        "check2": "not_read",
    }
    report = {
        "schema": "taiji-u2-p4-4-slot-competition-v1",
        "card": "22.5.P4.4",
        "date": "2026-11-15",
        "verdict": verdict,
        "read": {
            "status": read_status,
            "doses": read_rows,
            "trainable_parameters": len(read_names),
            "scaffold_parameters": sum(p.numel() for p in scaffold.parameters()),
            "labels": "world positions for the colours the image shows",
        },
        "byte": {
            "doses": byte_rows,
            "trainable_parameters": byte_names,
            "cache_positions": positions_count,
            "cache_replay_checks": checks,
        },
        "feedback": feedback,
        "recovery": {"equal_in_process": recovery_equal, "final_sha256": final_digest},
        "gate_table": gate_table,
        "criteria": criteria,
        "criteria_sha256": contract["criteria_sha256"],
        "data_digest": contract["data_digest"],
        "init_bundle_sha256": contract["p4_1_bundle_sha256"],
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
        "scope": "development fit only; the served output is the subject's byte head",
    }
    write_report(RUN_REPORT, report)
    print(json.dumps({"verdict": verdict, "read": read_status, "cost": report["cost"]}))
    return 0


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    started = time.process_time()
    labels = label_cache(data)
    label_cpu = time.process_time() - started
    encoder = p4_3.load_p4_1_bundle(contract)
    model = p4_2.build(contract, encoder)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    scaffold = ObjectScaffoldV2()
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    samples = []
    for update in range(1, 4):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        scaffold_loss(
            model,
            scaffold,
            p4_2.batch_at(data["train"], contract["batch"], update),
            labels,
        ).backward()
        optimizer.step()
        samples.append(time.process_time() - started)
    per_read = statistics.median(samples)
    model = p4_2.build(contract, encoder)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    p4_2.set_requires_grad(model, set(contract["emission_parameters"]))
    cached = p4_2.cache_positions(model, data["train"][: contract["batch"]])
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    byte_samples = []
    for update in range(1, 4):
        items = cached[((update - 1) * 8) % max(1, len(cached) - 8) :][:8]
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        p4_2.emission_loss(model, items).backward()
        optimizer.step()
        byte_samples.append(time.process_time() - started)
    per_byte = statistics.median(byte_samples)
    started = time.process_time()
    p4_2.round_evaluate(model, data["fit"])
    per_eval = time.process_time() - started
    read_cpu = max(contract["scaffold_dose_points"]) * per_read
    byte_cpu = max(contract["byte_dose_points"]) * per_byte
    eval_cpu = (
        2 * (len(contract["scaffold_dose_points"]) + len(contract["byte_dose_points"])) * per_eval
    )
    estimate = label_cpu + read_cpu + byte_cpu + eval_cpu + 40.0 + 6.0
    cap = float(int(estimate * 1.15 + 0.999))
    report = {
        "schema": "taiji-u2-p4-4-price-v1",
        "card": "22.5.P4.4",
        "date": "2026-11-15",
        "measurements": {
            "label_cache_cpu": round(label_cpu, 6),
            "per_read_update_cpu": round(per_read, 6),
            "per_byte_update_cpu": round(per_byte, 6),
            "per_fit_round_evaluate_cpu": round(per_eval, 6),
            "read_samples": samples,
            "byte_samples": byte_samples,
        },
        "estimate": {
            "read_curve_cpu": round(read_cpu, 6),
            "byte_curve_cpu": round(byte_cpu, 6),
            "evaluation_cpu": round(eval_cpu, 6),
            "feedback_reserve_cpu": 40.0,
            "margin_cpu": 6.0,
            "cap_cpu": cap,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {"optimizer_calls": 9, "checkpoints": 0, "check_reads": 0},
        "note": "in-memory timing steps; nothing is saved apart from this receipt",
    }
    write_report(PRICE_REPORT, report)
    print(json.dumps({"cap_cpu": cap, "read": round(read_cpu, 3)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("price", "run"), default="price")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "price":
        return price(contract)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
