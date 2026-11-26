"""First-token head feasibility on frozen U2 states: train/fit only, no check.

Extracts context, read, and H features from the U2 first-A preparation, then fits
a small linear four-class head on train and evaluates on fit. No U2 weights are
updated and no check or final is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    generate_examples,
    make_model,
    prepare,
)

REPORT = ROOT / "reports/taiji_u2_22_5_a_first_token_head_20261117.json"
LABELS = {"左边": 0, "中间": 1, "右边": 2, "未知": 3}
SEED = 20261110


def extract(model, examples):
    rows = {"context": [], "read": [], "h": [], "context_h": []}
    labels = []
    with torch.no_grad():
        for example in examples:
            state = prepare(model, example["public"], example["id"])
            goal = model.language_goal(state)
            action = state["prediction_action"]
            context = model.branch_context(state, goal, action).reshape(-1)
            read = model.read_h(state, goal).reshape(-1)
            h = state["H"].reshape(-1)
            rows["context"].append(context)
            rows["read"].append(read)
            rows["h"].append(h)
            rows["context_h"].append(torch.cat([context, h]))
            labels.append(LABELS[example["target"].decode()])
    return {name: torch.stack(values) for name, values in rows.items()}, torch.tensor(labels)


def fit_head(features, labels, seed: int):
    generator = torch.Generator().manual_seed(seed)
    mean = features.mean(0)
    std = features.std(0).clamp_min(1e-6)
    normalized = (features - mean) / std
    head = nn.Linear(features.shape[1], 4)
    nn.init.normal_(head.weight, std=0.01, generator=generator)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-3)
    head.train()
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(head(normalized), labels)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        predicted = head(normalized).argmax(-1)
    return head, mean, std, (predicted == labels).float().mean().item()


def main() -> int:
    model = make_model("native", SEED)
    train_examples = generate_examples("train", 64)
    fit_examples = generate_examples("fit", 32)
    train_features, train_labels = extract(model, train_examples)
    fit_features, fit_labels = extract(model, fit_examples)
    summary = {}
    for name in train_features:
        results = []
        for seed in (20261110, 20261111, 20261112):
            head, mean, std, train_accuracy = fit_head(train_features[name], train_labels, seed)
            with torch.no_grad():
                normalized = (fit_features[name] - mean) / std
                fit_accuracy = (head(normalized).argmax(-1) == fit_labels).float().mean().item()
            results.append(
                {"seed": seed, "train_accuracy": train_accuracy, "fit_accuracy": fit_accuracy}
            )
        summary[name] = {
            "dimension": int(train_features[name].shape[1]),
            "results": results,
            "max_fit_accuracy": max(row["fit_accuracy"] for row in results),
            "median_fit_accuracy": sorted(row["fit_accuracy"] for row in results)[1],
        }
    best = max(summary, key=lambda key: summary[key]["max_fit_accuracy"])
    report = {
        "schema": "taiji-u2-22-5-a-first-token-head-probe-v1",
        "card": "22.5.0.A",
        "verdict": (
            "head_feasible" if summary[best]["max_fit_accuracy"] >= 0.8 else "head_not_feasible"
        ),
        "best_feature": best,
        "summary": summary,
        "train_examples": len(train_examples),
        "fit_examples": len(fit_examples),
        "discipline": {
            "optimizer_steps_u2": 0,
            "head_optimizer_steps": 500 * 3 * len(summary),
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "first-token head verdict="
        + report["verdict"]
        + " best="
        + best
        + " max_fit="
        + str(round(summary[best]["max_fit_accuracy"], 4))
        + " median_fit="
        + str(round(summary[best]["median_fit_accuracy"], 4))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
