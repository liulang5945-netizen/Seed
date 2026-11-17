"""One training segment: fixed updates, one evaluation at the end, one checkpoint.

Follows the corrected cost model: training is ~0.2 CPU/example, evaluation was the
real cost, so the segment evaluates eight fit worlds once and fits the delta probe
on sixty-four world pairs.  Resumable through the bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    LABELS,
    generate_examples,
    load_learner,
    prepare,
    save_learner,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    FIRST_WEIGHT,
    SEED,
    delta_spread,
    evaluate,
    masked_loss,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402


def delta_rows(model, partition: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, labels = [], []
    with torch.no_grad():
        for example in generate_examples(partition, count):
            state = prepare(model, example["public"], example["id"])
            rows.append(state["pixel_delta"][0])
            labels.append(LABELS.index(example["target"].decode()))
    return torch.stack(rows), torch.tensor(labels)


def probe(model) -> dict:
    train_x, train_y = delta_rows(model, "train", 64)
    fit_x, fit_y = delta_rows(model, "fit", 64)
    torch.manual_seed(SEED)
    linear = torch.nn.Linear(train_x.shape[1], len(LABELS))
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(linear(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {
            "train": round(float((linear(train_x).argmax(-1) == train_y).float().mean()), 4),
            "fit": round(float((linear(fit_x).argmax(-1) == fit_y).float().mean()), 4),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"a_pixel_segment_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        model, optimizer, payload = load_learner(args.resume)
    else:
        config = replace(U2Config.small(), seed=SEED, readout="pixel-salience")
        model = U2Subject(config).metadata_candidate().request_candidate()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        payload = {"order": torch.randperm(64, generator=torch.Generator().manual_seed(9)).tolist()}
    train = generate_examples("train", 64)
    order = (
        payload["order"] if isinstance(payload, dict) and "order" in payload else list(range(64))
    )
    order = [int(i) for i in order]
    started = time.process_time()
    cursor = int(payload.get("cursor", 0)) if isinstance(payload, dict) else 0
    for _ in range(args.updates):
        indices = [order[(cursor + offset) % len(order)] for offset in range(args.batch)]
        batch = [train[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        masked_loss(model, batch, FIRST_WEIGHT).backward()
        optimizer.step()
        cursor = (cursor + args.batch) % len(order)
    fit = generate_examples("fit", 8)
    result = evaluate(model, fit)
    checkpoint = directory / "segment.bin"
    save_learner(
        checkpoint,
        model,
        optimizer,
        "native",
        args.updates,
        args.updates * args.batch,
        order,
        "seg",
    )
    report = {
        "schema": "taiji-u2-22-5-a-pixel-segment-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "conditions": {"updates": args.updates, "batch": args.batch, "lr": args.lr},
        "fit_correct": result["correct"],
        "fit_denominator": result["denominator"],
        "outputs": [row["output"] for row in result["rows"]],
        "delta_spread": round(delta_spread(model, fit), 6),
        "probe": probe(model),
        "cost": {"cpu": round(time.process_time() - started, 4)},
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
    }
    out = directory / "report.json"
    out.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
