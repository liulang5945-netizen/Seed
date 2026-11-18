"""Where does the middle class go: readout delta or emission?

The trained arm reaches 24/32 on fit with every miss on the middle class, while a
linear probe on the same delta reaches 0.906.  If the probe classifies middle
correctly on the same worlds the emission gets wrong, the delta already separates
the class and the defect is in the emission path; if the probe also loses it, the
delta is at fault.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (
    LABELS,
    generate_examples,
    load_learner,
)  # noqa: E402
from scripts.training.run_a_gate_pixel_20261117 import decode  # noqa: E402

OUT = ROOT / "reports/taiji_u2_middle_diagnosis_20261118.json"
NAMES = ("左边", "中间", "右边", "未知")


def collect(model, partition: str, count: int):
    rows, labels, outputs = [], [], []
    with torch.no_grad():
        for example in generate_examples(partition, count):
            state = __import__(
                "scripts.training.first_a_support_22_5", fromlist=["prepare"]
            ).prepare(model, example["public"], example["id"])
            rows.append(state["pixel_delta"][0])
            labels.append(LABELS.index(example["target"].decode()))
            content, _ = decode(model, example["public"], example["id"])
            outputs.append(content.decode("utf-8", "replace"))
    return torch.stack(rows), torch.tensor(labels), outputs


def confusion(truth: torch.Tensor, guess: torch.Tensor) -> list[list[int]]:
    table = [[0] * 4 for _ in range(4)]
    for row, column in zip(truth.tolist(), guess.tolist(), strict=True):
        table[row][column] += 1
    return table


def main() -> int:
    checkpoint = sorted(glob.glob(str(ROOT / "output/a_pixel_parallel_*/segment.bin")))[-1]
    model, _, _ = load_learner(Path(checkpoint))
    train_x, train_y, _ = collect(model, "train", 64)
    fit_x, fit_y, outputs = collect(model, "fit", 32)

    torch.manual_seed(20261110)
    linear = torch.nn.Linear(train_x.shape[1], len(LABELS))
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(linear(train_x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        predicted = linear(fit_x).argmax(-1)
    emitted = torch.tensor([NAMES.index(text) if text in NAMES else -1 for text in outputs])
    valid = emitted >= 0

    payload = {
        "schema": "taiji-u2-middle-diagnosis-v1",
        "card": "22.5.0.A",
        "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
        "emission_accuracy": round(float((emitted[valid] == fit_y[valid]).float().mean()), 4),
        "probe_accuracy": round(float((predicted == fit_y).float().mean()), 4),
        "emission_confusion": confusion(fit_y[valid], emitted[valid]),
        "probe_confusion": confusion(fit_y, predicted),
        "unparsed_outputs": int((~valid).sum()),
        "outputs": outputs,
        "verdict": (
            "delta_separates_emission_fails"
            if float((predicted == fit_y).float().mean())
            > float((emitted[valid] == fit_y[valid]).float().mean()) + 0.10
            else "delta_also_loses_middle"
        ),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "verdict",
                    "emission_accuracy",
                    "probe_accuracy",
                    "emission_confusion",
                    "probe_confusion",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
