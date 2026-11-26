"""First-token head probe over U2 encoder tokens: query-conditioned attention.

The state-level context/read probes were not linearly separable. This probe uses
the U2 encoder tokens (rgb/audio/proprio/outcome) and the request goal, then fits
a small attention selector on train and evaluates on fit. No check or final.
"""

from __future__ import annotations

import json
import math
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

REPORT = ROOT / "reports/taiji_u2_22_5_a_first_token_attention_20261117.json"
LABELS = {"左边": 0, "中间": 1, "右边": 2, "未知": 3}
SEED = 20261110


class AttentionHead(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, 4)

    def forward(self, tokens: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        query = self.query(goal)[:, None, :]
        keys = self.key(tokens)
        values = self.value(tokens)
        scores = torch.einsum("bqd,bkd->bqk", query, keys) / math.sqrt(tokens.shape[-1])
        weights = scores.softmax(-1)
        pooled = torch.einsum("bqk,bkd->bqd", weights, values)[:, 0]
        return self.output(pooled)


def extract(model, examples):
    tokens = []
    goals = []
    labels = []
    with torch.no_grad():
        for example in examples:
            state = prepare(model, example["public"], example["id"])
            goal = model.language_goal(state)
            encoded = model.encode(
                example["public"]["observations"],
                example["public"]["coordinates"],
                example["public"]["outcome"],
            )
            token = torch.cat(
                [encoded["rgb"], encoded["audio"], encoded["proprio"], encoded["outcome"]], 0
            )
            tokens.append(token)
            goals.append(goal)
            labels.append(LABELS[example["target"].decode()])
    return torch.stack(tokens), torch.stack(goals), torch.tensor(labels)


def train_head(kind: str, tokens, goals, labels, seed: int):
    torch.manual_seed(seed)
    if kind == "linear_flat":
        head = nn.Linear(tokens.shape[1] * tokens.shape[2], 4)
        inputs = tokens.flatten(1)
    elif kind == "linear_mean":
        head = nn.Linear(tokens.shape[2], 4)
        inputs = tokens.mean(1)
    elif kind == "attention":
        head = AttentionHead(tokens.shape[2])
        inputs = None
    else:
        raise ValueError(kind)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        logits = head(tokens, goals) if kind == "attention" else head(inputs)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        logits = head(tokens, goals) if kind == "attention" else head(inputs)
    return head, (logits.argmax(-1) == labels).float().mean().item()


def main() -> int:
    model = make_model("native", SEED)
    train_tokens, train_goals, train_labels = extract(model, generate_examples("train", 64))
    fit_tokens, fit_goals, fit_labels = extract(model, generate_examples("fit", 32))
    summary = {}
    for kind in ("linear_flat", "linear_mean", "attention"):
        results = []
        for seed in (20261110, 20261111, 20261112):
            head, train_accuracy = train_head(kind, train_tokens, train_goals, train_labels, seed)
            with torch.no_grad():
                logits = (
                    head(fit_tokens, fit_goals)
                    if kind == "attention"
                    else head(
                        fit_tokens.flatten(1) if kind == "linear_flat" else fit_tokens.mean(1)
                    )
                )
            fit_accuracy = (logits.argmax(-1) == fit_labels).float().mean().item()
            results.append(
                {"seed": seed, "train_accuracy": train_accuracy, "fit_accuracy": fit_accuracy}
            )
        summary[kind] = {
            "results": results,
            "max_fit_accuracy": max(row["fit_accuracy"] for row in results),
            "median_fit_accuracy": sorted(row["fit_accuracy"] for row in results)[1],
        }
    best = max(summary, key=lambda key: summary[key]["max_fit_accuracy"])
    report = {
        "schema": "taiji-u2-22-5-a-first-token-attention-probe-v1",
        "card": "22.5.0.A",
        "verdict": (
            "head_feasible" if summary[best]["max_fit_accuracy"] >= 0.8 else "head_not_feasible"
        ),
        "best_head": best,
        "summary": summary,
        "token_shape": list(train_tokens.shape),
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": 500 * 3 * len(summary),
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "first-token attention verdict="
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
