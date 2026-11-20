"""22.8S2b: correction isolation - frozen shared units, independent correct/old heads (0 check).

Base: train appearance on the shared 30-unit RouteB for three sessions (S2 protocol).
Then freeze every base parameter, add ``head_old`` and ``head_correct`` over the frozen
direct query, train head_old on the uncorrected subset, train head_correct on the
corrected subset with flipped labels, and measure K1 (correction gain), K2 (old ability
damage) and K3 (frozen units have no gradients while the heads do).  Cap 300 CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_multihead import MultiHeadGraph  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_8b_isolation_20261120.json"
CAP_CPU = 300.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 3
ROUNDS = 32
STEPS_PER_ROUND = 8
BATCH = 32
LR = 3e-3
WARMUP = 50
OUTPUT = 24
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def data() -> dict[str, torch.Tensor]:
    train = generate_examples("train", 256)
    fit = generate_examples("fit", 32)
    return {
        "x": torch.stack([adapter_v3_features(item) for item in train]).float(),
        "y": torch.tensor([joint_label(item) for item in train]),
        "x_fit": torch.stack([adapter_v3_features(item) for item in fit]).float(),
        "y_fit": torch.tensor([joint_label(item) for item in fit]),
    }


def build(seed: int) -> MultiHeadGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=seed
    )
    return MultiHeadGraph(spec, heads=2)


def direct(graph: MultiHeadGraph, features: torch.Tensor) -> torch.Tensor:
    gates = torch.ones(features.shape[0], graph.units.n_units)
    return graph(channels(features), gates, graph.initial_state(), head=0)["direct_query"]


def train_base(graph: MultiHeadGraph, bundle: dict, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed + 3)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    step = 0
    for _session in range(SESSIONS):
        for _round in range(ROUNDS):
            index = torch.randint(0, bundle["x"].shape[0], (BATCH,), generator=generator)
            logits = graph(
                channels(bundle["x"][index]),
                torch.ones(BATCH, graph.units.n_units),
                graph.initial_state(),
                head=0,
            )["readout"]
            loss = loss_fn(logits, bundle["y"][index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            graph.units.project_parameters_()
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, SESSIONS * ROUNDS)
            step += 1


def freeze_base(graph: MultiHeadGraph) -> None:
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def train_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        logits = head(features[index])
        loss = loss_fn(logits, labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = head(features)
    return float((logits.argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict:
    bundle = data()
    graph = build(seed)
    train_base(graph, bundle, seed)
    generator = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():
        train_direct = direct(graph, bundle["x"])
        fit_direct = direct(graph, bundle["x_fit"])
    freeze_base(graph)
    head_old = nn.Linear(train_direct.shape[1], OUTPUT)
    head_correct = nn.Linear(train_direct.shape[1], OUTPUT)
    old_mask = (bundle["y"] % 4) != 0
    new_mask = (bundle["y"] % 4) == 0
    flipped = (bundle["y"] + 12) % OUTPUT
    train_head(head_old, train_direct[old_mask], bundle["y"][old_mask], 200, generator)
    head_old.eval()
    head_correct.eval()
    old_before = accuracy(
        head_old,
        fit_direct[(bundle["y_fit"] % 4) != 0],
        bundle["y_fit"][(bundle["y_fit"] % 4) != 0],
    )
    corrected_before = accuracy(
        head_correct,
        fit_direct[(bundle["y_fit"] % 4) == 0],
        (bundle["y_fit"][(bundle["y_fit"] % 4) == 0] + 12) % OUTPUT,
    )
    train_head(
        head_correct,
        train_direct[new_mask],
        flipped[new_mask],
        200,
        generator,
    )
    head_correct.eval()
    head_old.eval()
    corrected_after = accuracy(
        head_correct,
        fit_direct[(bundle["y_fit"] % 4) == 0],
        (bundle["y_fit"][(bundle["y_fit"] % 4) == 0] + 12) % OUTPUT,
    )
    old_after = accuracy(
        head_old,
        fit_direct[(bundle["y_fit"] % 4) != 0],
        bundle["y_fit"][(bundle["y_fit"] % 4) != 0],
    )
    units_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in graph.parameters()
        if parameter.grad is not None
    )
    heads_grad = any(parameter.grad is not None for parameter in head_old.parameters()) or any(
        parameter.grad is not None for parameter in head_correct.parameters()
    )
    capsule = build_capsule(graph)
    with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
        path = save_capsule(Path(directory) / "after.json", capsule)
        loaded = load_capsule(path)
    fresh = build(seed)
    apply_capsule(fresh, loaded)
    digest_a = {key: value.sum().item() for key, value in graph.state_dict().items()}
    digest_b = {key: value.sum().item() for key, value in fresh.state_dict().items()}
    recovery_diff = max(abs(digest_a[key] - digest_b[key]) for key in digest_a)
    return {
        "seed": seed,
        "old_before": old_before,
        "old_after": old_after,
        "retention_damage": max(0.0, old_before - old_after),
        "corrected_before": corrected_before,
        "corrected_after": corrected_after,
        "correction_gain": corrected_after - corrected_before,
        "frozen_units_grad_sum": units_grad,
        "heads_have_grad": heads_grad,
        "recovery_diff": recovery_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    rows = [run_seed(seed) for seed in SEEDS]
    k1 = all(row["correction_gain"] >= 0.05 for row in rows)
    k2 = all(row["retention_damage"] <= 0.02 for row in rows)
    k3 = all(
        row["frozen_units_grad_sum"] == 0.0
        and row["heads_have_grad"]
        and row["recovery_diff"] <= 1e-10
        for row in rows
    )
    gates = {"k1_gain": k1, "k2_retention": k2, "k3_engineering": k3}
    if all(gates.values()):
        verdict = "correction_isolation_verified"
    elif k1 and k2:
        verdict = "correction_isolation_partial"
    else:
        verdict = "correction_isolation_absent"
    payload = {
        "schema": "taiji-u2-22-8b-isolation-v1",
        "card": "22.8S2b correction isolation with frozen shared units and independent heads",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": SESSIONS,
            "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "correction_steps": 200,
            "old_steps": 200,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": len(SEEDS) * (SESSIONS * ROUNDS + 400),
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "correction_isolation_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
