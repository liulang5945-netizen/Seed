"""P4.3: teach the body's object read with dense object-level supervision.

P4.2 showed the byte head is not the blocker: with a four-way answer signal the
read stayed at chance (fit 0.3125 to 0.34375) and the byte head followed (0.25).
P4.1 showed the encoder can carry colour when it is given identity labels per
column. This card applies the same rule one layer down: every visible object
gets a colour (six palette colours plus none) and a position (left, middle,
right) label, through a training-only per-slot head on the body's own object
representation (``object_z``). The served output is still the subject's byte
head, exactly as in P4.2.

Stage 0 loads the pinned P4.1 B-arm encoder from the P4.2 run root (sha256
pinned), so the lost weights are not rebuilt again.

Verdicts are frozen before any reading; see the manifest for the criteria.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import prepare
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_3_object_read_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p4_3_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p4_3_object_read_20261115.json"
COLOURS = 6
NONE = 6
SLOTS = 3
PERMUTATIONS = list(itertools.permutations(range(SLOTS)))


class BudgetStop(RuntimeError):
    pass


class ObjectScaffold(nn.Module):
    """Training-only per-slot head on the body's object representation."""

    def __init__(self) -> None:
        super().__init__()
        self.colour = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, COLOURS + 1))
        self.position = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, z: torch.Tensor):
        z = z.mean(0)
        return self.colour(z), self.position(z)


def remove_scaffold(model, scaffold) -> None:
    for parameter in scaffold.parameters():
        parameter.grad = None


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
    ):
        if pins.get(name) != p4_2.sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"run"}:
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


def visible_pairs(example: dict) -> list[tuple[int, int]]:
    """Colour and position bin of every visible palette object, from the image."""
    labels = p4_1.column_labels(example)
    pairs = []
    for colour in range(COLOURS):
        columns = (labels == colour).nonzero().flatten()
        if columns.numel() == 0:
            continue
        x = 2 * (columns.float().mean() + 0.5) / 16 - 1
        where = 0 if x < -1 / 3 else 2 if x > 1 / 3 else 1
        pairs.append((int(colour), int(where)))
    return pairs


def label_tensors(examples: list[dict]):
    colours = torch.full((len(examples), SLOTS), NONE, dtype=torch.long)
    positions = torch.zeros((len(examples), SLOTS), dtype=torch.long)
    position_mask = torch.zeros((len(examples), SLOTS))
    for index, example in enumerate(examples):
        for slot, (colour, where) in enumerate(visible_pairs(example)[:SLOTS]):
            colours[index, slot] = colour
            positions[index, slot] = where
            position_mask[index, slot] = 1.0
    return colours, positions, position_mask


def scaffold_loss(model, scaffold, batch: list[dict]) -> torch.Tensor:
    colour_rows, position_rows = [], []
    for example in batch:
        state = prepare(model, p4_2.column_public(example), example["id"])
        colour, position = scaffold(model.object_z(state))
        colour_rows.append(colour)
        position_rows.append(position)
    colours = torch.stack(colour_rows)
    positions = torch.stack(position_rows)
    target_colour, target_position, mask = label_tensors(batch)
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


def scaffold_accuracy(model, scaffold, examples: list[dict]) -> dict:
    pair_hits = pair_total = set_hits = 0
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepared(model, example)
            colour, position = scaffold(model.object_z(state))
            predicted = predicted_pairs(colour, position)
            truth = visible_pairs(example)
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


def load_p4_1_bundle(contract: dict) -> dict:
    bundle = Path(contract["p4_1_bundle"])
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != contract["p4_1_bundle_sha256"]:
        raise ValueError("p4_1_bundle_mismatch")
    return torch.load(bundle, weights_only=False)["encoder"]


def stage1(contract: dict, data: dict, encoder: dict, root: Path, budget) -> tuple:
    model = p4_2.build(contract, encoder)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    names = p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = ObjectScaffold()
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
        loss = scaffold_loss(model, scaffold, batch)
        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        ):
            raise ValueError("nonfinite_p4_3_gradient")
        optimizer.step()
        model.parameter_revision += 1
        if update not in contract["scaffold_dose_points"]:
            continue
        train = scaffold_accuracy(model, scaffold, data["train"][:32])
        fit = scaffold_accuracy(model, scaffold, data["fit"])
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


def stage2(contract: dict, data: dict, model, root: Path, budget) -> tuple:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = p4_2.set_requires_grad(model, set(contract["emission_parameters"]))
    positions = p4_2.cache_positions(model, data["train"])
    checks = []
    for item in positions[:: max(1, len(positions) // 3)][:3]:
        example = next(e for e in data["train"] if e["id"] == item["example"])
        body = p4_2.body_log_probability(model, example, item["position"], item["symbol"])
        replay = -float(p4_2.emission_loss(model, [item]).detach())
        checks.append(
            {
                "example": item["example"],
                "position": item["position"],
                "equal": abs(body - replay) <= 1e-6,
            }
        )
    if not all(check["equal"] for check in checks):
        raise ValueError("p4_3_cache_replay_mismatch")
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    rows = []
    for update in range(1, max(contract["byte_dose_points"]) + 1):
        budget()
        items = [
            positions[((update - 1) * contract["batch"] + offset) % len(positions)]
            for offset in range(contract["batch"])
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = p4_2.emission_loss(model, items)
        loss.backward()
        optimizer.step()
        model.parameter_revision += 1
        if update not in contract["byte_dose_points"]:
            continue
        train = p4_2.round_evaluate(model, data["train"][:32], budget_guard=budget)
        fit = p4_2.round_evaluate(model, data["fit"], budget_guard=budget)
        path = root / f"byte-{update:06d}.pt"
        digest = p4_2.save_checkpoint(
            path,
            {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "update": update,
                "revision": model.parameter_revision,
            },
        )
        rows.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "fit_accuracy": fit["accuracy"],
                "fit_correct": fit["correct"],
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
            }
        )
    return names, positions, checks, rows


def feedback_round(contract: dict, data: dict, model, budget) -> dict:
    first = p4_2.round_evaluate(model, data["fit"], budget_guard=budget)
    errors = [row for row in first["rows"] if not row["correct"]]
    control = copy.deepcopy(model)
    control_before = value_digest(control.state_dict())
    control_second = p4_2.round_evaluate(control, data["fit"], budget_guard=budget)
    control_after = value_digest(control.state_dict())
    learned = copy.deepcopy(model)
    optimizer = p4_2.optimizer_for(learned.parameters(), contract["feedback_optimizer"])
    updates = 0
    for row in errors:
        example = next(e for e in data["fit"] if e["id"] == row["id"])
        for _ in range(contract["feedback_updates"]):
            budget()
            optimizer.zero_grad(set_to_none=True)
            state = prepare(learned, p4_2.column_public(example), example["id"])
            p4_2.byte_loss(learned, state, example["target"]).backward()
            optimizer.step()
            learned.parameter_revision += 1
            updates += 1
    learned_second = p4_2.round_evaluate(learned, data["fit"], budget_guard=budget)
    hits = {row["id"]: row["correct"] for row in learned_second["rows"]}
    corrected = sum(hits[row["id"]] for row in errors)
    first_correct = [row["id"] for row in first["rows"] if row["correct"]]
    stability_losses = sum(not hits[key] for key in first_correct)
    correction = corrected / len(errors) if len(errors) >= 4 else None
    gain = learned_second["accuracy"] - control_second["accuracy"] if len(errors) >= 4 else None
    return {
        "first_pass_correct": 32 - len(errors),
        "first_pass_errors": len(errors),
        "control_second_pass": control_second["accuracy"],
        "learned_second_pass": learned_second["accuracy"],
        "correction_rate": correction,
        "feedback_gain": gain,
        "feedback_updates": updates,
        "control_updates": 0,
        "control_state_unchanged": control_before == control_after,
        "stability_losses": stability_losses,
        "stability_ceiling": max(1, int(round(0.1 * len(first_correct)))),
        "learned_model": learned,
    }


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    encoder = load_p4_1_bundle(contract)
    root = Path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("p4_3_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p4_3_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p4_3_wall_stop")

    initial = p4_2.build(contract, encoder)
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = ObjectScaffold()
    p4_2.save_checkpoint(
        root / "initial.pt",
        {
            "model": {k: v.detach().clone() for k, v in initial.state_dict().items()},
            "scaffold": {k: v.detach().clone() for k, v in scaffold.state_dict().items()},
            "revision": initial.parameter_revision,
        },
    )
    model, scaffold, read_names, read_rows, read_status = stage1(
        contract, data, encoder, root, budget
    )
    taught = read_status == "read_taught"
    feedback: dict = {}
    stage2_rows: list[dict] = []
    stage2_names: list[str] = []
    checks: list[dict] = []
    final = None
    recovery_equal = None
    final_digest = None
    positions_count = 0
    if taught:
        stage2_names, positions, checks, stage2_rows = stage2(contract, data, model, root, budget)
        positions_count = len(positions)
        feedback = feedback_round(contract, data, model, budget)
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

    def mark(value: bool | None, blind: str = "not_run_read_blind") -> str:
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
        "schema": "taiji-u2-p4-3-object-read-v1",
        "card": "22.5.P4.3",
        "date": "2026-11-15",
        "verdict": verdict,
        "read": {
            "status": read_status,
            "doses": read_rows,
            "trainable_parameters": len(read_names),
            "scaffold_parameters": sum(p.numel() for p in scaffold.parameters()),
            "baselines": contract["read_baselines"],
        },
        "byte": {
            "doses": stage2_rows,
            "trainable_parameters": stage2_names,
            "cache_positions": positions_count,
            "cache_replay_checks": checks,
        },
        "feedback": feedback,
        "recovery": {
            "equal_in_process": recovery_equal,
            "final_sha256": final_digest,
        },
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
                "read": read_status,
                "cost": report["cost"],
            }
        )
    )
    return 0


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    encoder = load_p4_1_bundle(contract)
    model = p4_2.build(contract, encoder)
    scaffold = ObjectScaffold()
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    samples = []
    for update in range(1, 4):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        scaffold_loss(
            model, scaffold, p4_2.batch_at(data["train"], contract["batch"], update)
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
    started = time.process_time()
    p4_2.cache_positions(model, data["train"][:16])
    per_cache_position = (time.process_time() - started) / (16 * 7)
    read_cpu = max(contract["scaffold_dose_points"]) * per_read
    byte_cpu = max(contract["byte_dose_points"]) * per_byte
    cache_cpu = (len(data["train"]) + len(data["fit"])) * 7 * per_cache_position
    eval_cpu = (
        2 * (len(contract["scaffold_dose_points"]) + len(contract["byte_dose_points"])) * per_eval
    )
    feedback_cpu = 40.0
    estimate = read_cpu + byte_cpu + cache_cpu + eval_cpu + feedback_cpu + 6.0
    cap = float(int(estimate * 1.15 + 0.999))
    report = {
        "schema": "taiji-u2-p4-3-price-v1",
        "card": "22.5.P4.3",
        "date": "2026-11-15",
        "measurements": {
            "per_read_update_cpu": round(per_read, 6),
            "per_byte_update_cpu": round(per_byte, 6),
            "per_cache_position_cpu": round(per_cache_position, 6),
            "per_fit_round_evaluate_cpu": round(per_eval, 6),
            "read_samples": samples,
            "byte_samples": byte_samples,
        },
        "estimate": {
            "read_curve_cpu": round(read_cpu, 6),
            "byte_curve_cpu": round(byte_cpu, 6),
            "cache_cpu": round(cache_cpu, 6),
            "evaluation_cpu": round(eval_cpu, 6),
            "feedback_reserve_cpu": feedback_cpu,
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
