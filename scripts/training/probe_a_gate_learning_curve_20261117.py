"""Learning curve for the pixel-salience readout: where does fit cross 0.80?

The pricing step used 4.6M example-updates, but that number came from "6000
full-batch steps to convergence" and the convergence step count varied twelvefold
across seeds (500 to 6000).  A number with that variance is a poor basis for a
training size, so this probe measures the curve directly: train once per seed,
evaluate the three-body fit at fixed example-update budgets, and report the first
budget where the median crosses 0.80.

Reuses the cached features and the operator from the pixel-read probe.  No U2
optimizer step, no check read, no final read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.probe_a_gate_ceiling_20261117 import _features  # noqa: E402
from scripts.training.probe_a_gate_pixel_read_20261117 import (  # noqa: E402
    PixelSalienceHead,
    _decode,
    _score,
)
from scripts.training.probe_a_query_bind_head_20261117 import (  # noqa: E402
    HEAD_SEEDS,
    _confusion,
)

REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_learning_curve_20261117.json"
BUDGETS = (100_000, 300_000, 1_000_000, 3_000_000)
WORLDS = 768
LR = 0.01


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows = _features()
    train_image, train_onehot = _decode(rows["train12"])
    train_label = rows["train12"]["label"]
    splits = {}
    for key in ("fit3", "fit12"):
        image, onehot = _decode(rows[key])
        splits[key] = {"image": image, "onehot": onehot, "label": rows[key]["label"]}

    per_seed = []
    for seed in HEAD_SEEDS:
        torch.manual_seed(seed)
        head = PixelSalienceHead()
        optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
        next_budget = 0
        marks = []
        steps = 0
        while next_budget < len(BUDGETS):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(head(train_image, train_onehot), train_label)
            loss.backward()
            optimizer.step()
            steps += 1
            seen = steps * WORLDS
            while next_budget < len(BUDGETS) and seen >= BUDGETS[next_budget]:
                head.eval()
                with torch.no_grad():
                    logits = head(train_image, train_onehot)
                row = {
                    "example_updates": BUDGETS[next_budget],
                    "steps": steps,
                    "train_accuracy": (logits.argmax(-1) == train_label).float().mean().item(),
                    "train_confusion": _confusion(logits.argmax(-1), train_label),
                    "loss": float(loss.detach()),
                }
                for key, split in splits.items():
                    accuracy, confusion = _score(
                        head, split["image"], split["onehot"], split["label"]
                    )
                    row[f"{key}_accuracy"] = accuracy
                    row[f"{key}_confusion"] = confusion
                marks.append(row)
                next_budget += 1
                head.train()
        per_seed.append({"seed": seed, "marks": marks, "steps": steps})

    def median(values: list[float]) -> float:
        return sorted(values)[len(values) // 2]

    curve = []
    for index, budget in enumerate(BUDGETS):
        entries = [row["marks"][index] for row in per_seed]
        curve.append(
            {
                "example_updates": budget,
                "median_train_accuracy": median([e["train_accuracy"] for e in entries]),
                "median_fit3_accuracy": median([e["fit3_accuracy"] for e in entries]),
                "median_fit12_accuracy": median([e["fit12_accuracy"] for e in entries]),
                "fit3_per_seed": [e["fit3_accuracy"] for e in entries],
                "train_per_seed": [e["train_accuracy"] for e in entries],
            }
        )
    crossing = next(
        (row["example_updates"] for row in curve if row["median_fit3_accuracy"] >= 0.80), None
    )
    payload = {
        "schema": "taiji-u2-22-5-a-gate-learning-curve-v1",
        "card": "22.5.0.A",
        "verdict": "crossing_found" if crossing else "no_crossing_inside_budgets",
        "first_budget_with_median_fit3_at_least_0.80": crossing,
        "budgets": list(BUDGETS),
        "curve": curve,
        "per_seed": per_seed,
        "conditions": {
            "train_split": "train12 (768 worlds, 1-2 bodies)",
            "readout": "PixelSalienceHead (256 pixel tokens, multiplicative match, linear moment read)",
            "adam": {"lr": LR, "weight_decay": 0.0001},
            "seeds": list(HEAD_SEEDS),
            "worlds_per_step": WORLDS,
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": sum(row["steps"] for row in per_seed),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
        },
        "source_pins": {
            "scripts/training/probe_a_gate_learning_curve_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "curve verdict="
        + payload["verdict"]
        + " crossing="
        + str(crossing)
        + " fit3="
        + json.dumps([round(row["median_fit3_accuracy"], 4) for row in curve])
        + " train="
        + json.dumps([round(row["median_train_accuracy"], 4) for row in curve])
        + " cost="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
