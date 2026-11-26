"""Linear probe on the readout delta: does it carry the answer, or the emission fails?

The 10x-lr arm moved delta's cross-sample spread to the same order as its norm but
the emitted word stayed mostly constant.  This probe freezes the model, collects
the 8-dim delta for train and fit worlds, and fits a linear four-class readout on
those vectors alone.  High probe accuracy means the delta encodes the position and
the failure sits in the emission path; chance means the operator is not reading.
"""

from __future__ import annotations

import glob
import json
import sys
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
)

OUT = ROOT / "reports/taiji_u2_22_5_a_delta_probe_20261117.json"
TRAIN_WORLDS = 256
FIT_WORLDS = 128
STEPS = 400


def collect(model, partition: str, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    examples = generate_examples(partition, count)
    rows, labels = [], []
    with torch.no_grad():
        for example in examples:
            state = prepare(model, example["public"], example["id"])
            rows.append(state["pixel_delta"][0])
            labels.append(LABELS.index(example["target"].decode()))
    return torch.stack(rows), torch.tensor(labels)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    checkpoint = sorted(glob.glob(str(ROOT / "output/a_pixel_grouped_*/final.bin")))[-1]
    model, _, _ = load_learner(Path(checkpoint))
    train_x, train_y = collect(model, "train", TRAIN_WORLDS)
    fit_x, fit_y = collect(model, "fit", FIT_WORLDS)

    torch.manual_seed(20261110)
    probe = torch.nn.Linear(train_x.shape[1], len(LABELS))
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_x).argmax(-1) == train_y).float().mean())
        fit_logits = probe(fit_x)
        fit_accuracy = float((fit_logits.argmax(-1) == fit_y).float().mean())
        means = torch.stack([fit_x[fit_y == index].mean(0) for index in range(len(LABELS))])
        spread = float(fit_x.std(0).mean())
    table = [[0] * len(LABELS) for _ in range(len(LABELS))]
    for truth, guess in zip(fit_y.tolist(), fit_logits.argmax(-1).tolist(), strict=True):
        table[truth][guess] += 1
    payload = {
        "schema": "taiji-u2-22-5-a-delta-probe-v1",
        "card": "22.5.0.A",
        "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
        "probe": {
            "train_accuracy": round(train_accuracy, 4),
            "fit_accuracy": round(fit_accuracy, 4),
            "fit_confusion": table,
            "chance": 0.25,
        },
        "delta": {
            "per_class_means": [[round(float(v), 4) for v in row] for row in means.tolist()],
            "cross_sample_std": round(spread, 4),
        },
        "verdict": (
            "delta_carries_position" if fit_accuracy >= 0.70 else "delta_does_not_carry_position"
        ),
        "conditions": {"train_worlds": TRAIN_WORLDS, "fit_worlds": FIT_WORLDS, "steps": STEPS},
        "discipline": {"u2_optimizer_steps": 0, "probe_steps": STEPS, "check_reads": 0},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "fit_probe="
        + str(round(fit_accuracy, 4))
        + " train_probe="
        + str(round(train_accuracy, 4))
        + " verdict="
        + payload["verdict"]
        + " delta_std="
        + str(payload["delta"]["cross_sample_std"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
