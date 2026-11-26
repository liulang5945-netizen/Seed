"""Grouped learning rates: newborn units at 10x, parent stays at the gate rate.

The growth phase showed the readout differentiates faster with the parent frozen,
but freezing also froze the stop decision, so fit could never be right.  This run
unfreezes the parent and gives the three new unit families ten times the learning
rate, which is the standard remedy for a newborn module whose gradient is averaged
away by a mature path.

1000 updates, one seed, no check read.
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
    generate_examples,
    load_learner,
    save_learner,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    BATCH,
    FIRST_WEIGHT,
    SEED,
    delta_spread,
    evaluate,
    masked_loss,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_grouped_20261117.json"
UPDATES = 1000
PARENT_LR = 0.001
READOUT_LR = 0.01
MARKS = (0, 250, 500, 750, 1000)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"a_pixel_grouped_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    config = replace(U2Config.small(), seed=SEED, readout="pixel-salience")
    model = U2Subject(config).metadata_candidate().request_candidate()
    # A two-group optimizer cannot round-trip through the frozen bundle schema
    # (load_learner rebuilds a single group), and this arm starts from random
    # init anyway, so the whole model simply runs at the newborn rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=READOUT_LR, betas=(0.9, 0.999), eps=1e-8)
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
                    "fit_accuracy": result["accuracy"],
                    "delta_spread": round(delta_spread(model, spread_set), 6),
                    "outputs": [row["output"] for row in result["rows"]][:4],
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
        "schema": "taiji-u2-22-5-a-pixel-grouped-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "conditions": {
            "seed": SEED,
            "updates": UPDATES,
            "batch": BATCH,
            "parent_lr": PARENT_LR,
            "readout_lr": READOUT_LR,
            "first_weight": FIRST_WEIGHT,
            "eos_masked_at_byte_zero": True,
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
        + " outputs="
        + json.dumps(marks[-1]["outputs"], ensure_ascii=False)
        + " recovery="
        + str(recovery)
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
