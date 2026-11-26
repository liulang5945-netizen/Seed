"""23.5: equal-compute GRU/Transformer baselines against the token-attention read (0 check).

Same Q2/Q3 data and the same 23.0b frozen parent for the U2 arm; the GRU and Transformer
arms see the same 3 object tokens (8 dims) and query (8 dims).  All arms train 1,500 steps
on identical batches and are compared on the fit split; parameters, analytic FLOPs and
measured CPU per arm are recorded so the comparison can be audited for near-equality.
Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    AttentionRead,
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_5_baselines_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 32
LR = 3e-3
STEPS = 1500
NONINFERIOR_MARGIN = 0.05
RETENTION_MAX = 0.02


def data() -> dict[str, dict[str, torch.Tensor]]:
    def build(partition: str) -> dict[str, torch.Tensor]:
        rows = [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]
        tokens, queries = [], []
        for row in rows:
            token, query = tokens_for(row)
            tokens.append(token)
            queries.append(query)
        return {
            "tokens": torch.stack(tokens).float(),
            "queries": torch.stack(queries).float(),
            "labels": torch.tensor([label_for(row) for row in rows]),
            "kinds": torch.tensor([row["kind"] == "q2" for row in rows]),
        }

    return {"train": build("train"), "fit": build("fit")}


class GruBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
        self.head = nn.Sequential(nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(tokens)
        return self.head(torch.cat([hidden[-1], query], dim=-1))


class TransformerBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Linear(8, 16)
        self.query = nn.Linear(8, 16)
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(16, 24), nn.Tanh(), nn.Linear(24, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        sequence = torch.cat([self.token(tokens), self.query(query).unsqueeze(1)], dim=1)
        encoded = self.encoder(sequence)
        return self.head(encoded.mean(dim=1))


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def train_module(module, batch, seed, tokens_override=None) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 83)
    tokens = batch["tokens"] if tokens_override is None else tokens_override
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, tokens.shape[0], (BATCH,), generator=generator)
        loss = loss_fn(module(tokens[index], batch["queries"][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(module, batch, tokens_override=None) -> float:
    tokens = batch["tokens"] if tokens_override is None else tokens_override
    with torch.no_grad():
        logits = module(tokens, batch["queries"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def analytic_flops(module: nn.Module, tokens_per_sample: int = 3, feature: int = 8) -> float:
    linear = sum(
        parameter.numel()
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
        for parameter in layer.parameters()
    )
    return 2.0 * linear * tokens_per_sample


def run_seed(seed: int, base: dict) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    train, fit = base["train"], base["fit"]
    cross_train = unit_activity(graph, train["tokens"]).detach()
    cross_fit = unit_activity(graph, fit["tokens"]).detach()
    u2 = AttentionRead(cross_train.shape[-1])
    gru = GruBaseline()
    transformer = TransformerBaseline()
    records = {}
    timings = {}
    for name, module, tokens_train, tokens_fit in (
        ("u2", u2, cross_train, cross_fit),
        ("gru", gru, None, None),
        ("transformer", transformer, None, None),
    ):
        beginning = time.process_time()
        train_module(module, train, seed, tokens_override=tokens_train)
        timings[name] = time.process_time() - beginning
        accuracy = evaluate(module, fit, tokens_override=tokens_fit)
        records[name] = {
            "fit": accuracy,
            "params": count_params(module),
            "flops": analytic_flops(module),
            "cpu": round(timings[name], 2),
        }
    retention = {
        "baseline": baseline,
        "after": {
            "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
            "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
            "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
        },
    }
    retention["damage"] = {
        skill: max(0.0, baseline[skill] - retention["after"][skill]) for skill in baseline
    }
    return {"seed": seed, "arms": records, "retention": retention}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    base = data()
    rows = [run_seed(seed, base) for seed in SEEDS]
    u2 = [row["arms"]["u2"]["fit"] for row in rows]
    gru = [row["arms"]["gru"]["fit"] for row in rows]
    transformer = [row["arms"]["transformer"]["fit"] for row in rows]
    strongest = [max(g, t) for g, t in zip(gru, transformer, strict=True)]
    gaps = [u - s for u, s in zip(u2, strongest, strict=True)]
    mean_gap = sum(gaps) / len(gaps)
    cpu_ratio = sum(row["arms"]["gru"]["cpu"] for row in rows) / sum(
        row["arms"]["u2"]["cpu"] for row in rows
    )
    flop_ratio = sum(row["arms"]["gru"]["flops"] for row in rows) / sum(
        row["arms"]["u2"]["flops"] for row in rows
    )
    tf_cpu_ratio = sum(row["arms"]["transformer"]["cpu"] for row in rows) / sum(
        row["arms"]["u2"]["cpu"] for row in rows
    )
    gates = {
        "noninferior": mean_gap >= -NONINFERIOR_MARGIN,
        "near_equal_compute_gru": abs(cpu_ratio - 1.0) <= 0.10,
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = (
        "baseline_comparison_verified" if all(gates.values()) else "baseline_comparison_partial"
    )
    payload = {
        "schema": "taiji-u2-23-5-baselines-v1",
        "card": "23.5 equal-compute GRU/Transformer baselines",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch": BATCH,
            "noninferior_margin": NONINFERIOR_MARGIN,
        },
        "rows": rows,
        "u2_fit": u2,
        "gru_fit": gru,
        "transformer_fit": transformer,
        "gap_vs_strongest": gaps,
        "mean_gap": mean_gap,
        "cpu_ratio_gru_over_u2": cpu_ratio,
        "flop_ratio_gru_over_u2_estimated_linear_only": flop_ratio,
        "cpu_ratio_transformer_over_u2": tf_cpu_ratio,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "u2": u2,
                "gru": gru,
                "tf": transformer,
                "gaps": gaps,
                "cpu_ratio": cpu_ratio,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "baseline_comparison_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
