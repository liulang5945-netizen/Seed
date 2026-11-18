"""E5 correction-vs-retention dose response, with the old ability actually earned.

A.5 measured "old ability preserved" on a subject that had already been trained on
the A task for 600 updates (fit 0.5 at 200, 1.0 at 400).  At birth the A task is
0/32, so a probe started from birth has nothing to damage and its ``damage``
column is meaningless.  An earlier version of this probe did exactly that and
printed ``fit_before 0.0``; that reading is discarded, not repaired.

Phases, all read-only over the frozen A.5 contract:
  1. build the subject and the codes exactly as ``run_a5_22_5.run`` does;
  2. train the A task for ``steps`` updates and report the learning curve;
  3. snapshot that bundle -- this is the "old ability";
  4. for each dose: restore the snapshot, accumulate ``dose`` record updates per
     first-pass error, then measure correction on task B and damage on A fit.

A.5's one-at-a-time protocol retested after a single update and rolled back on
failure, so every one of its 24 updates was undone and the retention gate could
not fail.  Here the rollback is removed (that is the point of the dose sweep) and
damage is measured on a subject that demonstrably has the ability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_dose_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
DOSES = (8, 32)
STEPS = 600
CURVE = (400, 600)
CORRECTION_GATE = 0.80
DAMAGE_GATE = 0.02


def template_function(contract: dict):
    count = len(contract["task_b"]["templates"])

    def template_of(index: int) -> int:
        return index % count

    return template_of


def load_contract() -> tuple[dict, dict]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    return contract, data


def assemble(contract: dict, data: dict):
    """The same order as run_a5_22_5.run: subject, head, prior, codes, trainable."""
    model = a5.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)
    a5.set_trainable(model, contract)
    return model, codes


def a_fit(model, data, codes) -> list[dict]:
    return [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]


def state_of(model) -> dict:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def max_drift(model, saved: dict) -> float:
    current = model.state_dict()
    return max(float((current[key] - saved[key]).abs().max()) for key in saved)


def train_a(model, contract, data, codes, steps: int, curve: tuple[int, ...], budget) -> dict:
    """A.5's A-training loop, verbatim in structure, with the curve kept."""
    positions = a2.cache_positions(model, data["train"], codes)
    order = list(range(len(positions)))
    batch_size = int(contract["batch"])
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    required = set(contract["emission_parameters"])
    first_loss = None
    last_loss = None
    points: dict[str, dict] = {}
    cursor = 0
    for update in range(1, steps + 1):
        budget()
        batch = [positions[order[(cursor + offset) % len(order)]] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(order)
        step_loss, _ = a5.train_step(model, optimizer, batch, required)
        if update == 1:
            first_loss = round(step_loss, 6)
        last_loss = round(step_loss, 6)
        if update in curve:
            budget()
            rows = a_fit(model, data, codes)
            points[str(update)] = {
                "fit_correct": sum(int(row["correct"]) for row in rows),
                "fit_accuracy": a5.accuracy(rows),
            }
            print("a_curve", update, points[str(update)], flush=True)
    return {
        "steps": steps,
        "curve": points,
        "loss_first": first_loss,
        "loss_last": last_loss,
    }


def feedback_dose(model, contract, items, errors, codes, template_of, dose, budget) -> int:
    """Accumulate ``dose`` record updates per first-pass error; no rollback.

    ``errors`` carries the original item indices so the template offset stays the
    one that produced the wrong answer.
    """
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    updates = 0
    for index in errors:
        item = items[index]
        public = a5.p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        template = template_of(index)
        for _ in range(dose):
            budget()
            optimizer.zero_grad(set_to_none=True)
            loss = a5.record_loss(
                model,
                public,
                item["example"]["id"],
                item["records"][template],
                item["target"],
                code,
            )
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            updates += 1
    return updates


def smoke(contract: dict, data: dict) -> int:
    """No optimizer step: prove the assembly, the snapshot and the two gradients."""
    started = time.process_time()
    torch.set_num_threads(1)
    model, codes = assemble(contract, data)
    items, skipped = a5.task_b_examples(data, contract)
    template_of = template_function(contract)
    birth = a_fit(model, data, codes)
    first = a5.b_pass(model, items, codes, template_of)
    errors = [index for index, row in enumerate(first) if not row["correct"]]
    positions = a2.cache_positions(model, data["train"], codes)
    saved = state_of(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.full_like(parameter, 0.02))
    perturbed = max_drift(model, saved)
    model.load_state_dict(saved)
    restored = max_drift(model, saved)

    batch = positions[: int(contract["batch"])]
    loss_a = a2.emission_loss(model, batch, use_code=True)
    loss_a.backward()
    a_grad = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }

    model.load_state_dict(saved)
    index = errors[0]
    item = items[index]
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    optimizer.zero_grad(set_to_none=True)
    loss_b = a5.record_loss(
        model,
        a5.p4_2.column_public(item["example"]),
        item["example"]["id"],
        item["records"][template_of(index)],
        item["target"],
        codes[item["example"]["id"]],
    )
    loss_b.backward()
    b_grad = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    payload: dict[str, Any] = {
        "schema": "taiji-u2-e5-dose-smoke-v1",
        "card": "E5",
        "birth": {
            "a_fit_correct": sum(int(row["correct"]) for row in birth),
            "a_fit": a5.accuracy(birth),
            "b_first_pass_correct": sum(int(row["correct"]) for row in first),
            "b_first_pass": a5.accuracy(first),
            "errors": len(errors),
            "skipped": len(skipped),
        },
        "assembly": {
            "positions": len(positions),
            "batch": int(contract["batch"]),
            "templates": len(contract["task_b"]["templates"]),
            "trainable": sum(1 for p in model.parameters() if p.requires_grad),
        },
        "snapshot": {
            "perturbation_max": round(perturbed, 6),
            "restore_max_abs": round(restored, 6),
        },
        "gradients": {
            "a_tensors": len(a_grad),
            "a_nonzero": sum(1 for value in a_grad.values() if value > 0),
            "a_loss": round(float(loss_a.detach()), 6),
            "b_tensors": len(b_grad),
            "b_nonzero": sum(1 for value in b_grad.values() if value > 0),
            "b_loss": round(float(loss_b.detach()), 6),
        },
        "cpu": round(time.process_time() - started, 4),
        "optimizer_steps": 0,
    }
    failures: list[str] = []
    if payload["snapshot"]["restore_max_abs"] != 0.0:
        failures.append("snapshot_restore_not_exact")
    if payload["birth"]["errors"] != len(items):
        failures.append("first_pass_not_all_wrong")
    if payload["gradients"]["a_nonzero"] == 0:
        failures.append("a_gradient_absent")
    if payload["gradients"]["b_nonzero"] == 0:
        failures.append("feedback_gradient_absent")
    payload["failures"] = failures
    payload["verdict"] = "smoke_passed" if not failures else "smoke_failed"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("smoke_cpu", payload["cpu"], "verdict", payload["verdict"])
    return 0 if not failures else 1


def price(contract: dict, data: dict, steps: int, doses: tuple[int, ...]) -> int:
    """Measure the four primitives once each, then extrapolate the whole sweep."""
    torch.set_num_threads(1)
    started = time.process_time()
    model, codes = assemble(contract, data)
    items, _ = a5.task_b_examples(data, contract)
    template_of = template_function(contract)
    positions = a2.cache_positions(model, data["train"], codes)
    batch = positions[: int(contract["batch"])]
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    required = set(contract["emission_parameters"])

    marker = time.process_time()
    for _ in range(3):
        a5.train_step(model, optimizer, batch, required)
    a_step = (time.process_time() - marker) / 3

    marker = time.process_time()
    a5.b_pass(model, items[:2], codes, template_of)
    b_round = (time.process_time() - marker) / 2

    marker = time.process_time()
    for example in data["fit"][:2]:
        a5.a_round(model, example, codes[example["id"]])
    a_round = (time.process_time() - marker) / 2

    marker = time.process_time()
    feedback_optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    item = items[0]
    feedback_optimizer.zero_grad(set_to_none=True)
    loss = a5.record_loss(
        model,
        a5.p4_2.column_public(item["example"]),
        item["example"]["id"],
        item["records"][0],
        item["target"],
        codes[item["example"]["id"]],
    )
    loss.backward()
    feedback_optimizer.step()
    feedback_step = time.process_time() - marker

    errors = len(items)
    total_updates = errors * sum(doses)
    a_training = steps * a_step
    feedback = total_updates * feedback_step
    a_eval = len(data["fit"]) * (len(doses) + len(CURVE) + 1) * a_round
    b_eval = errors * (len(doses) + 1) * b_round
    subtotal = a_training + feedback + a_eval + b_eval
    contingency = float(contract["plan"]["contingency"])
    cap = subtotal * (1.0 + contingency)
    payload: dict[str, Any] = {
        "schema": "taiji-u2-e5-dose-price-v1",
        "card": "E5",
        "unit_price_cpu": {
            "a_train_step": round(a_step, 6),
            "feedback_step": round(feedback_step, 6),
            "a_round": round(a_round, 6),
            "b_round": round(b_round, 6),
        },
        "counts": {
            "steps": steps,
            "errors": errors,
            "doses": list(doses),
            "feedback_updates": total_updates,
            "a_evals": len(doses) + len(CURVE) + 1,
            "b_evals": len(doses) + 1,
        },
        "estimate_cpu": {
            "a_training": round(a_training, 4),
            "feedback": round(feedback, 4),
            "a_eval": round(a_eval, 4),
            "b_eval": round(b_eval, 4),
            "subtotal": round(subtotal, 4),
            "contingency": contingency,
            "cap_cpu": round(cap, 3),
            "cap_wall": round(cap * 2.0, 3),
        },
        "one_shot_adam_steps": 4,
        "instrument_cpu": round(time.process_time() - started, 4),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run(
    contract: dict,
    data: dict,
    steps: int,
    doses: tuple[int, ...],
    cap_cpu: float,
    out: Path,
) -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    wall_start = time.perf_counter()

    def budget() -> None:
        if time.process_time() - started > cap_cpu:
            raise SystemExit("e5_dose_budget_stop")

    model, codes = assemble(contract, data)
    items, skipped = a5.task_b_examples(data, contract)
    template_of = template_function(contract)
    a_training = train_a(model, contract, data, codes, steps, CURVE, budget)
    saved = state_of(model)
    old_rows = a_fit(model, data, codes)
    old_accuracy = a5.accuracy(old_rows)
    first = a5.b_pass(model, items, codes, template_of)
    errors = [index for index, row in enumerate(first) if not row["correct"]]
    rows = []
    for dose in doses:
        budget()
        dose_start = time.process_time()
        model.load_state_dict(saved)
        updates = feedback_dose(model, contract, items, errors, codes, template_of, dose, budget)
        corrected = 0
        for index in errors:
            item = items[index]
            row = a5.b_round(
                model,
                item,
                codes[item["example"]["id"]],
                template_of(index),
            )
            corrected += int(row["correct"])
        after_accuracy = a5.accuracy(a_fit(model, data, codes))
        rows.append(
            {
                "dose": dose,
                "errors": len(errors),
                "updates": updates,
                "corrected": corrected,
                "correction_rate": round(corrected / max(len(errors), 1), 4),
                "old_ability": round(old_accuracy, 4),
                "old_ability_after": round(after_accuracy, 4),
                "damage": round(old_accuracy - after_accuracy, 4),
                "cpu": round(time.process_time() - dose_start, 2),
            }
        )
        print("dose", json.dumps(rows[-1], ensure_ascii=False), flush=True)
    solvable = [
        row
        for row in rows
        if row["correction_rate"] >= CORRECTION_GATE and row["damage"] <= DAMAGE_GATE
    ]
    payload: dict[str, Any] = {
        "schema": "taiji-u2-e5-dose-v2",
        "card": "E5",
        "premise": {
            "old_ability": round(old_accuracy, 4),
            "old_ability_gate": 0.80,
            "b_first_pass_correct": sum(int(row["correct"]) for row in first),
            "errors": len(errors),
            "skipped": len(skipped),
            "a_training": a_training,
        },
        "doses": rows,
        "criteria": {"correction_gate": CORRECTION_GATE, "damage_gate": DAMAGE_GATE},
        "verdict": ("correction_and_retention_reachable" if solvable else "no_dose_satisfies_both"),
        "cpu": round(time.process_time() - started, 2),
        "wall": round(time.perf_counter() - wall_start, 2),
    }
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print("verdict", payload["verdict"], "cpu", payload["cpu"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E5 correction-vs-retention dose sweep")
    parser.add_argument("--smoke", action="store_true", help="no optimizer step")
    parser.add_argument("--price", action="store_true", help="measure unit prices")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--doses", type=int, nargs="*", default=list(DOSES))
    parser.add_argument("--cap", type=float, default=600.0)
    parser.add_argument("--tag", default="", help="suffix for a split run's report")
    args = parser.parse_args(argv)
    contract, data = load_contract()
    if args.smoke:
        return smoke(contract, data)
    if args.price:
        return price(contract, data, args.steps, tuple(args.doses))
    out = OUT if not args.tag else OUT.with_name(OUT.stem + f"_{args.tag}" + OUT.suffix)
    if out.exists():
        raise FileExistsError(out)
    return run(contract, data, args.steps, tuple(args.doses), args.cap, out)


if __name__ == "__main__":
    raise SystemExit(main())
