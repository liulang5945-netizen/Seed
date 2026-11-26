"""Throughput baseline for the pixel-salience arm: updates per hour and early fit.

The 1e6 example-update figure came from a full-batch standalone head, a different
unit from the gate's sequential updates.  This run measures the real unit: 200
updates at batch 2 on the frozen fixture, with fit read every 50 updates, a
checkpoint at the end, and a reload check.  One seed, no check read.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    evaluate,
    generate_examples,
    load_learner,
    save_learner,
    task_loss,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_throughput_20261117.json"
SEED = 20261110
UPDATES = 200
BATCH = 2
LR = 0.001
MARKS = (0, 50, 100, 200)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"pixel_throughput_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    config = replace(U2Config.small(), seed=SEED, readout="pixel-salience")
    model = U2Subject(config).metadata_candidate().request_candidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    train = generate_examples("train", 64)
    fit = generate_examples("fit", 32)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + 20261209))
    order = order.tolist()

    marks = []
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    cursor = 0
    for step in range(UPDATES + 1):
        if step in MARKS:
            result = evaluate(model, fit)
            marks.append(
                {
                    "update": step,
                    "fit_correct": result["correct"],
                    "fit_denominator": result["denominator"],
                    "fit_accuracy": result["accuracy"],
                    "cpu": round(time.process_time() - started_cpu, 4),
                    "wall": round(time.perf_counter() - started_wall, 4),
                }
            )
        if step == UPDATES:
            break
        indices = [order[(cursor + offset) % len(order)] for offset in range(BATCH)]
        batch = [train[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, batch)
        loss.backward()
        optimizer.step()
        cursor = (cursor + BATCH) % len(order)

    total_cpu = time.process_time() - started_cpu
    total_wall = time.perf_counter() - started_wall
    final = directory / "final.bin"
    save_learner(
        final, model, optimizer, "native", UPDATES, UPDATES * BATCH, order, value_digest(train)
    )
    restored, _, _ = load_learner(final)
    before = evaluate(model, fit)
    after = evaluate(restored, fit)
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-throughput-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "conditions": {
            "seed": SEED,
            "updates": UPDATES,
            "batch": BATCH,
            "lr": LR,
            "train_examples": len(train),
            "fit_examples": len(fit),
            "readout": "pixel-salience",
        },
        "marks": marks,
        "throughput": {
            "updates_per_cpu_hour": round(UPDATES / max(total_cpu, 1e-9) * 3600, 1),
            "cpu_per_update": round(total_cpu / UPDATES, 4),
            "cpu_total": round(total_cpu, 4),
            "wall_total": round(total_wall, 4),
        },
        "recovery_equal": value_digest(before["rows"]) == value_digest(after["rows"]),
        "discipline": {"u2_optimizer_steps": UPDATES, "check_reads": 0, "final_reads": 0},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "fit="
        + json.dumps([round(mark["fit_accuracy"], 4) for mark in marks])
        + " cpu_per_update="
        + str(payload["throughput"]["cpu_per_update"])
        + " updates_per_cpu_hour="
        + str(payload["throughput"]["updates_per_cpu_hour"])
        + " recovery="
        + str(payload["recovery_equal"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
