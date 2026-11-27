"""Pixel arm under the repaired emission protocol: EOS masked at byte zero.

The 200-update baseline ended with an empty answer on every fit world: the first
byte was EOS.  Masking EOS at byte zero during training *and* decoding is the
repair D-37 validated, and the first byte carries a weight because it decides the
whole word.  The readout delta is also monitored across samples: if it stays a
constant vector, the units are learning a bias rather than reading the scene.

300 updates, one seed, no check read.
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
    accept,
    generate_examples,
    load_learner,
    output,
    prepare,
    save_learner,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_protocol_20261117.json"
SEED = 20261110
UPDATES = 300
BATCH = 2
LR = 0.001
FIRST_WEIGHT = 2.0
MARKS = (0, 100, 200, 300)


def masked_loss(
    model, examples, first_weight: float, mask_eos: bool = True, byte_weight=None
) -> torch.Tensor:
    losses = []
    for example in examples:
        state = prepare(model, example["public"], example["id"])
        terms = []
        for index, symbol in enumerate([*example["target"], 256]):
            probability, _, details = output(model, state)
            if index == 0 and mask_eos:
                masked = probability.clone()
                masked[256] = -torch.inf
                probability = masked - torch.logsumexp(masked, -1)
            weight = first_weight if index == 0 else 1.0
            terms.append(-probability[symbol] * weight)
            state = accept(model, state, symbol, details)
        scale = 1.0 if not byte_weight else byte_weight.get(int(example["target"][0]), 1.0)
        losses.append(torch.stack(terms).mean() * scale)
    return torch.stack(losses).mean()


def decode(model, public, identity) -> tuple[bytes, bool]:
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


def delta_spread(model, examples) -> float:
    rows = []
    with torch.no_grad():
        for example in examples:
            state = prepare(model, example["public"], example["id"])
            rows.append(state["pixel_delta"][0])
    stack = torch.stack(rows)
    return float(stack.std(0).mean())


def evaluate(model, examples) -> dict:
    correct = 0
    rows = []
    with torch.no_grad():
        for example in examples:
            content, eos = decode(model, example["public"], example["id"])
            ok = bool(eos and content == example["target"])
            correct += int(ok)
            rows.append({"output": content.decode("utf-8", "replace"), "correct": ok})
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "rows": rows,
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"a_pixel_protocol_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    config = replace(U2Config.small(), seed=SEED, readout="pixel-salience")
    model = U2Subject(config).metadata_candidate().request_candidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    train = generate_examples("train", 64)
    fit = generate_examples("fit", 32)
    spread_set = generate_examples("fit", 8)
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(SEED + 20261209)
    ).tolist()

    marks = []
    started_cpu = time.process_time()
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
                    "delta_spread": round(delta_spread(model, spread_set), 6),
                    "outputs": [row["output"] for row in result["rows"]],
                }
            )
        if step == UPDATES:
            break
        indices = [order[(cursor + offset) % len(order)] for offset in range(BATCH)]
        batch = [train[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        masked_loss(model, batch, FIRST_WEIGHT).backward()
        optimizer.step()
        cursor = (cursor + BATCH) % len(order)

    final = directory / "final.bin"
    save_learner(
        final, model, optimizer, "native", UPDATES, UPDATES * BATCH, order, value_digest(train)
    )
    restored, _, _ = load_learner(final)
    recovery = value_digest(evaluate(model, fit)) == value_digest(evaluate(restored, fit))
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-protocol-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "conditions": {
            "seed": SEED,
            "updates": UPDATES,
            "batch": BATCH,
            "lr": LR,
            "first_weight": FIRST_WEIGHT,
            "eos_masked_at_byte_zero": True,
            "readout": "pixel-salience",
        },
        "marks": marks,
        "recovery_equal": recovery,
        "cost": {"cpu": round(time.process_time() - started_cpu, 4)},
        "discipline": {"u2_optimizer_steps": UPDATES, "check_reads": 0, "final_reads": 0},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "fit="
        + json.dumps([round(mark["fit_accuracy"], 4) for mark in marks])
        + " spread="
        + json.dumps([mark["delta_spread"] for mark in marks])
        + " sample="
        + json.dumps(marks[-1]["outputs"][:4], ensure_ascii=False)
        + " recovery="
        + str(recovery)
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
