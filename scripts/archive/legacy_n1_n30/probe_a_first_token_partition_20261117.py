"""First-token attention head on the 22.4.3 train/fit partitions.

The 64/32 first-A preflight was too small for a selection head. This probe uses the
768 train and 192 fit worlds, extracts frozen U2 encoder tokens plus the request
goal, and fits a query-conditioned attention classifier. No check or final.
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

from scripts.training.first_a_support_22_5 import make_model, prepare  # noqa: E402
from taiji.u2_task_episode import NAMES, locate_public, position_word  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

PARTITION = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"
REPORT = ROOT / "reports/taiji_u2_22_5_a_first_token_partition_20261117.json"
LABELS = {"左边": 0, "中间": 1, "右边": 2, "未知": 3}
GOAL = "依据当前可核验观测，用自然中文回答请求。".encode()
RGB_COLORS = (
    (220, 45, 45),
    (45, 90, 220),
    (45, 190, 80),
    (230, 190, 40),
    (170, 60, 200),
    (40, 190, 195),
)
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
        pooled = torch.einsum("bqk,bkd->bqd", scores.softmax(-1), values)[:, 0]
        return self.output(pooled)


def _seeds(split: str) -> list[int]:
    payload = json.loads(PARTITION.read_text(encoding="utf-8"))
    return [int(seed) for seed in payload["manifest"]["splits"][split]["seeds"]]


def _examples(seeds: list[int]) -> list[dict]:
    examples = []
    for index, seed in enumerate(seeds):
        world = PhysicalWorld.reset(seed, {"body_count": 3})
        result = world.elapse(f"a-partition-{index}")
        bodies = world.snapshot()["bodies"]
        names = [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in bodies]
        if index % 4 == 0:
            absent = [name for name in NAMES if name not in names]
            entity = absent[seed % len(absent)]
            target = "未知"
        else:
            entity = None
            target = None
            start = (index // 4) % len(bodies)
            order = list(range(len(bodies)))
            order = order[start:] + order[:start]
            for body_index in order:
                local = locate_public(result["raw_media"]["rgb"], names[body_index])
                word = "未知" if local is None else position_word(local)
                actual = position_word(bodies[body_index]["position"][0])
                if word == actual:
                    entity, target = names[body_index], actual
                    break
            if entity is None:
                absent = [name for name in NAMES if name not in names]
                entity = absent[seed % len(absent)]
                target = "未知"
        examples.append(
            {
                "id": f"partition-{index}",
                "public": {
                    "goal": GOAL,
                    "request": f"{entity}现在在哪里？".encode(),
                    "observations": result["observations"],
                    "coordinates": result["coordinates"],
                    "outcome": result["execution_outcome"],
                    "action": torch.zeros(15),
                    "t_start": result["t_start"],
                    "t_end": result["t_end"],
                },
                "target": target.encode(),
            }
        )
    return examples


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
    if kind == "linear_mean":
        head = nn.Linear(tokens.shape[2], 4)
        inputs = tokens.mean(1)
    else:
        head = AttentionHead(tokens.shape[2])
        inputs = None
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
    train_examples = _examples(_seeds("train"))
    fit_examples = _examples(_seeds("development-fit"))
    train_tokens, train_goals, train_labels = extract(model, train_examples)
    fit_tokens, fit_goals, fit_labels = extract(model, fit_examples)
    summary = {}
    for kind in ("linear_mean", "attention"):
        results = []
        for seed in (20261110, 20261111, 20261112):
            head, train_accuracy = train_head(kind, train_tokens, train_goals, train_labels, seed)
            with torch.no_grad():
                logits = (
                    head(fit_tokens, fit_goals) if kind == "attention" else head(fit_tokens.mean(1))
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
        "schema": "taiji-u2-22-5-a-first-token-partition-probe-v1",
        "card": "22.5.0.A",
        "verdict": (
            "head_feasible" if summary[best]["max_fit_accuracy"] >= 0.8 else "head_not_feasible"
        ),
        "best_head": best,
        "summary": summary,
        "train_examples": len(train_examples),
        "fit_examples": len(fit_examples),
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
        "first-token partition verdict="
        + report["verdict"]
        + " best="
        + best
        + " max_fit="
        + str(round(summary[best]["max_fit_accuracy"], 4))
        + " median_fit="
        + str(round(summary[best]["median_fit_accuracy"], 4))
        + " train_n="
        + str(len(train_examples))
        + " fit_n="
        + str(len(fit_examples))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
