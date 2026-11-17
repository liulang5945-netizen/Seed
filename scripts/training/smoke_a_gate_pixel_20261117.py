"""Smoke: the pixel-salience identity must save, load and recover bit for bit.

E6's minimum precondition.  One seed, three updates, then save, reload and compare
the reloaded model's evaluation rows against the pre-save rows.  Also reports the
per-update CPU so the 1e6 example-update plan can be converted into wall time.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_smoke_20261117.json"
SEED = 20261110
UPDATES = 3
BATCH = 2


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    directory = ROOT / "output" / "pixel_smoke_20261117"
    directory.mkdir(parents=True, exist_ok=True)
    config = replace(U2Config.small(), seed=SEED, readout="pixel-salience")
    model = U2Subject(config).metadata_candidate().request_candidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train = generate_examples("train", 8)
    fit = generate_examples("fit", 8)
    started = time.process_time()
    losses = []
    for step in range(UPDATES):
        batch = train[step * BATCH : step * BATCH + BATCH]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(round(float(loss.detach()), 6))
    training_cpu = time.process_time() - started
    before = evaluate(model, fit)
    checkpoint = directory / "smoke.bin"
    save_learner(checkpoint, model, optimizer, "native", UPDATES, UPDATES * BATCH, [], "smoke")
    restored, _, payload = load_learner(checkpoint)
    after = evaluate(restored, fit)
    payload_out = {
        "schema": "taiji-u2-22-5-a-pixel-smoke-v1",
        "card": "22.5.0.A",
        "readout": config.readout,
        "updates": UPDATES,
        "losses": losses,
        "recovery_equal": value_digest(before["rows"]) == value_digest(after["rows"]),
        "fit_accuracy": before["accuracy"],
        "description_readout": payload["description"]["config"]["readout"],
        "cpu": {
            "training": round(training_cpu, 4),
            "per_example_update": round(training_cpu / (UPDATES * BATCH), 4),
        },
        "discipline": {"u2_optimizer_steps": UPDATES, "check_reads": 0, "final_reads": 0},
    }
    OUT.write_bytes(json.dumps(payload_out, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "recovery_equal="
        + str(payload_out["recovery_equal"])
        + " readout="
        + str(payload_out["description_readout"])
        + " cpu_per_example_update="
        + str(payload_out["cpu"]["per_example_update"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
