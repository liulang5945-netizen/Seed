"""22.7.0-α adapter v3 + 2-class task face probe (pre-registered fallback).

Per D-20261119-33, if the 4-way task face stays below the probe threshold, the
alpha task face switches to a simpler 2-class decision on the SAME media features:
left vs right only (中间/未知 excluded).  This separates "adapter has no signal"
from "4-way label boundary too hard".  Threshold: linear probe fit accuracy
>= 0.70 required before r4.  Diagnostic only: no graph, zero check/final.
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

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_adapter_v3_2class_20261120.json"
LABELS = ("左边", "中间", "右边", "未知")
BINARY = {"左边": 0, "右边": 1}
THRESHOLD = 0.70


def select(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for example in examples:
        target = example["target"].decode("utf-8")
        if target not in BINARY:
            continue
        features.append(adapter_v3_features(example))
        labels.append(BINARY[target])
    return torch.stack(features).float(), torch.tensor(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    x_train, y_train = select(generate_examples("train", 256))
    x_fit, y_fit = select(generate_examples("fit", 32))
    torch.manual_seed(20261120)
    probe = torch.nn.Linear(48, 2)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(probe(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(x_train).argmax(-1) == y_train).float().mean())
        fit_accuracy = float((probe(x_fit).argmax(-1) == y_fit).float().mean())
    verdict = (
        "adapter_v3_2class_ready" if fit_accuracy >= THRESHOLD else "adapter_v3_2class_insufficient"
    )
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-adapter-v3-2class-v1",
        "card": "alpha 2-class task-face probe on adapter v3 features",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "task": "left vs right (middle/unknown excluded)",
            "features": 48,
            "train_examples": int(x_train.shape[0]),
            "fit_examples": int(x_fit.shape[0]),
            "probe": "linear softmax, 500 Adam steps",
            "threshold_fit_accuracy": THRESHOLD,
            "chance": 0.5,
        },
        "train_accuracy": round(train_accuracy, 4),
        "fit_accuracy": round(fit_accuracy, 4),
        "verdict": verdict,
        "discipline": {"graph_optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if verdict == "adapter_v3_2class_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
