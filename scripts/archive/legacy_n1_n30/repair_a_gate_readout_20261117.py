"""Fit-only readout repair for the A gate: mask EOS at the first emitted byte.

The smoke showed a first-byte EOS bias: teacher-forced content probabilities were
high but EOS won the first argmax, so inference stopped empty. This runner trains
a fresh U2 arm on train only with EOS masked at byte zero, then evaluates train and
fit with the same constrained decoder. No check is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    accept,
    generate_examples,
    load_learner,
    make_model,
    output,
    prepare,
    save_learner,
)
from taiji.u2_history import value_digest  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_a_readout_repair_20261117.json"
SEED = 20261110
UPDATES = 512
BATCH = 2
LR = 0.003
CPU_CAP = 4000.0
WALL_CAP = 8000.0


def task_loss_masked(model, examples, first_weight: float = 1.0):
    losses = []
    for example in examples:
        state = prepare(model, example["public"], example["id"])
        terms = []
        for index, symbol in enumerate([*example["target"], 256]):
            probability, _, details = output(model, state)
            if index == 0:
                masked = probability.clone()
                masked[256] = -torch.inf
                probability = masked - torch.logsumexp(masked, -1)
            weight = first_weight if index == 0 else 1.0
            terms.append(-probability[symbol] * weight)
            state = accept(model, state, symbol, details)
        losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean()


def predict_masked(model, public, identity):
    state = prepare(model, public, identity)
    data = bytearray()
    for _ in range(65):
        probability, _, details = output(model, state)
        if not data:
            probability = probability.clone()
            probability[256] = -torch.inf
        symbol = int(probability.detach().argmax())
        state = accept(model, state, symbol, details)
        if symbol == 256:
            return bytes(data), True
        data.append(symbol)
    return bytes(data), False


def evaluate_masked(model, examples):
    correct = 0
    rows = []
    with torch.no_grad():
        for example in examples:
            content, eos = predict_masked(model, example["public"], example["id"])
            ok = content == example["target"]
            correct += int(ok)
            rows.append(
                {
                    "id": example["id"],
                    "target": example["target"].decode(),
                    "output": content.decode("utf-8", errors="replace"),
                    "correct": ok,
                    "eos": eos,
                }
            )
    return {"correct": correct, "denominator": len(examples), "rows": rows}


def gradients(model, batch, first_weight: float = 1.0):
    model.zero_grad(set_to_none=True)
    loss = task_loss_masked(model, batch, first_weight)
    loss.backward()
    measured = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return float(loss.detach()), measured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-weight", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    first_weight = args.first_weight
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"a_readout_repair_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    train = generate_examples("train", 64)
    fit = generate_examples("fit", 32)
    price = generate_examples("price", 8)
    model = make_model("native", SEED)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(SEED + 20261209)
    ).tolist()
    initial = directory / "initial.bin"
    save_learner(initial, model, optimizer, "native", 0, 0, order, value_digest(train))
    restored, _, _ = load_learner(initial)
    first_loss, first_grads = gradients(model, price[:2], first_weight)
    second_loss, second_grads = gradients(restored, price[:2], first_weight)
    if first_loss != second_loss:
        raise ValueError("readout_repair_checkpoint_loss_mismatch")
    for name, value in first_grads.items():
        other = second_grads[name]
        if (value is None) != (other is None) or (
            value is not None and not torch.equal(value, other)
        ):
            raise ValueError(f"readout_repair_checkpoint_gradient_mismatch:{name}")
    history = []
    cursor = 0
    for step in range(1, UPDATES + 1):
        indices = [order[(cursor + offset) % len(order)] for offset in range(BATCH)]
        batch = [train[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss_masked(model, batch, first_weight)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cursor = (cursor + BATCH) % len(order)
        if step in {256, 512}:
            fit_result = evaluate_masked(model, fit)
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach()), 6),
                    "fit_correct": fit_result["correct"],
                    "fit_denominator": fit_result["denominator"],
                }
            )
    train_result = evaluate_masked(model, train)
    fit_result = evaluate_masked(model, fit)
    final = directory / "final.bin"
    save_learner(
        final, model, optimizer, "native", UPDATES, UPDATES * BATCH, order, value_digest(train)
    )
    recovered, _, _ = load_learner(final)
    recovered_fit = evaluate_masked(recovered, fit)
    recovery_equal = value_digest(fit_result["rows"]) == value_digest(recovered_fit["rows"])
    report = {
        "schema": "taiji-u2-22-5-a-readout-repair-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "status": "fit_readout_signal" if fit_result["correct"] > 0 else "fit_readout_no_signal",
        "method": (
            "mask EOS at the first emitted byte during training and decoding; "
            f"first-token loss weight {first_weight}"
        ),
        "conditions": {
            "seed": SEED,
            "updates": UPDATES,
            "batch": BATCH,
            "lr": LR,
            "first_weight": first_weight,
            "train_examples": 64,
            "fit_examples": 32,
            "check_reads": 0,
        },
        "preflight": {"initial_loss": first_loss, "recovered_loss": second_loss},
        "history": history,
        "train": {
            "correct": train_result["correct"],
            "denominator": train_result["denominator"],
            "accuracy": train_result["correct"] / train_result["denominator"],
        },
        "fit": {
            "correct": fit_result["correct"],
            "denominator": fit_result["denominator"],
            "accuracy": fit_result["correct"] / fit_result["denominator"],
            "examples": fit_result["rows"][:8],
        },
        "recovery_equal": recovery_equal,
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
        },
        "check_exposure": {"used": 1, "remaining": 1, "unchanged": True},
        "discipline": {"optimizer_steps": UPDATES, "check_reads": 0, "final_reads": 0},
        "checkpoint": str(final.relative_to(ROOT)),
    }
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "readout repair",
        report["status"],
        "train",
        train_result["correct"],
        "/",
        train_result["denominator"],
        "fit",
        fit_result["correct"],
        "/",
        fit_result["denominator"],
        "cost",
        report["cost"]["cpu"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
