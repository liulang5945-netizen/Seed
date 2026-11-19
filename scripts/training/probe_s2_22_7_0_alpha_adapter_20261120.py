"""Zero-graph diagnostic: is the alpha statistical adapter informative at all?

Trains a linear softmax probe on the exact 48-dim adapter features (256 first-A
train worlds -> 32 fit worlds) to separate "features carry no label signal" from
"unit-graph optimisation is weak".  Diagnostic only: no unit graph, no growth,
zero check/final.
"""

from __future__ import annotations

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
from scripts.training.probe_s2_22_7_0_alpha_20261120 import build_windows  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_adapter_probe_20261120.json"


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    train_channels, train_labels = build_windows(generate_examples("train", 256))
    fit_channels, fit_labels = build_windows(generate_examples("fit", 32))
    x_train = torch.cat([train_channels[name] for name in train_channels], dim=1)
    x_fit = torch.cat([fit_channels[name] for name in fit_channels], dim=1)
    torch.manual_seed(20261120)
    probe = torch.nn.Linear(x_train.shape[1], 4)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(probe(x_train), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(x_train).argmax(-1) == train_labels).float().mean())
        fit_accuracy = float((probe(x_fit).argmax(-1) == fit_labels).float().mean())
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-adapter-probe-v1",
        "card": "alpha adapter informativeness diagnostic (linear softmax probe)",
        "created_utc": datetime.now(UTC).isoformat(),
        "features": int(x_train.shape[1]),
        "train_accuracy": round(train_accuracy, 4),
        "fit_accuracy": round(fit_accuracy, 4),
        "chance": 0.25,
        "verdict": (
            "adapter_signal_insufficient"
            if max(train_accuracy, fit_accuracy) < 0.5
            else "adapter_signal_present"
        ),
        "discipline": {"graph_optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(OUT, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
