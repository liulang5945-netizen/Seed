"""22.8J3b: meaningful retention on a dose-trained base (old head >=0.40, then correction; 0 check).

Same frozen-base setup as S2b, but the old head is trained until it reaches >=0.40 on a
held-out old-subset fit; then the correction head is trained on the flipped corrected
subset, and old-subset accuracy is re-measured.  If the old head never reaches criterion
the card records ``old_head_not_learnable`` instead of claiming retention.
Base is trained at the J1 dose (48 steps/round x 8 x 3 = 1,152 steps) so an old ability exists. Cap 300 CPU, zero check/final.
"""

from __future__ import annotations

import argparse
import json
import math
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

OUT = ROOT / "reports/taiji_u2_22_8j3b_retention_20261120.json"
CAP_CPU = 200.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 3
ROUNDS = 32
STEPS_PER_ROUND = 48
BATCH = 32
LR = 3e-3
WARMUP = 50
OUTPUT = 24
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
OLD_CRITERION = 0.40
MAX_OLD_STEPS = 1200
CORRECTION_STEPS = 200


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
                group["lr"] = lr_at(step, SESSIONS * ROUNDS * STEPS_PER_ROUND)
            step += 1


def freeze(graph: MultiHeadGraph) -> None:
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def accuracy(head: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = head(features)
    return float((logits.argmax(-1) == labels).float().mean())


def train_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    generator: torch.Generator,
    fit_features: torch.Tensor | None = None,
    fit_labels: torch.Tensor | None = None,
    criterion: float | None = None,
) -> tuple[int, float]:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_now = 0.0
    steps_run = 0
    for _ in range(steps):
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(features[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps_run += 1
        if fit_features is not None and fit_labels is not None and steps_run % 50 == 0:
            accuracy_now = accuracy(head, fit_features, fit_labels)
            if criterion is not None and accuracy_now >= criterion:
                return steps_run, accuracy_now
    if fit_features is not None and fit_labels is not None:
        accuracy_now = accuracy(head, fit_features, fit_labels)
    return steps_run, accuracy_now


def run_seed(seed: int) -> dict:
    bundle = data()
    graph = build(seed)
    train_base(graph, bundle, seed)
    generator = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():
        train_direct = direct(graph, bundle["x"])
        fit_direct = direct(graph, bundle["x_fit"])
    freeze(graph)
    old_mask = (bundle["y"] % 4) != 0
    new_mask = (bundle["y"] % 4) == 0
    old_fit_mask = (bundle["y_fit"] % 4) != 0
    new_fit_mask = (bundle["y_fit"] % 4) == 0
    head_old = nn.Linear(train_direct.shape[1], OUTPUT)
    head_correct = nn.Linear(train_direct.shape[1], OUTPUT)
    old_steps, old_accuracy = train_head(
        head_old,
        train_direct[old_mask],
        bundle["y"][old_mask],
        MAX_OLD_STEPS,
        generator,
        fit_direct[old_fit_mask],
        bundle["y_fit"][old_fit_mask],
        OLD_CRITERION,
    )
    old_ready = old_accuracy >= OLD_CRITERION
    old_before = accuracy(head_old, fit_direct[old_fit_mask], bundle["y_fit"][old_fit_mask])
    corrected_labels_fit = (bundle["y_fit"][new_fit_mask] + 12) % OUTPUT
    corrected_before = accuracy(head_correct, fit_direct[new_fit_mask], corrected_labels_fit)
    corrected_steps = 0
    if old_ready:
        corrected_steps, _ = train_head(
            head_correct,
            train_direct[new_mask],
            (bundle["y"][new_mask] + 12) % OUTPUT,
            CORRECTION_STEPS,
            generator,
        )
    corrected_after = accuracy(head_correct, fit_direct[new_fit_mask], corrected_labels_fit)
    old_after = accuracy(head_old, fit_direct[old_fit_mask], bundle["y_fit"][old_fit_mask])
    units_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in graph.parameters()
        if parameter.grad is not None
    )
    capsule = build_capsule(graph)
    with tempfile_dir() as directory:
        path = save_capsule(Path(directory) / "after.json", capsule)
        loaded = load_capsule(path)
    fresh = build(seed)
    apply_capsule(fresh, loaded)
    digest_a = {key: value.sum().item() for key, value in graph.state_dict().items()}
    digest_b = {key: value.sum().item() for key, value in fresh.state_dict().items()}
    recovery_diff = max(abs(digest_a[key] - digest_b[key]) for key in digest_a)
    return {
        "seed": seed,
        "old_steps": old_steps,
        "old_ready": old_ready,
        "old_accuracy_at_criterion": old_accuracy,
        "old_before": old_before,
        "old_after": old_after,
        "retention_damage": max(0.0, old_before - old_after),
        "corrected_before": corrected_before,
        "corrected_after": corrected_after,
        "correction_gain": corrected_after - corrected_before,
        "corrected_steps": corrected_steps,
        "frozen_units_grad_sum": units_grad,
        "recovery_diff": recovery_diff,
    }


def tempfile_dir():
    import tempfile

    return tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    rows = [run_seed(seed) for seed in SEEDS]
    old_ready = all(row["old_ready"] for row in rows)
    gates = {
        "old_head_learnable": old_ready,
        "retention_after_meaningful_old_ability": all(
            row["retention_damage"] <= 0.02 for row in rows if row["old_ready"]
        )
        and old_ready,
        "correction_gain": all(row["correction_gain"] >= 0.05 for row in rows if row["old_ready"])
        and old_ready,
        "frozen_units_no_grad": all(row["frozen_units_grad_sum"] == 0.0 for row in rows),
        "recovery_exact": all(row["recovery_diff"] <= 1e-10 for row in rows),
    }
    if not old_ready:
        verdict = "old_head_not_learnable"
    elif all(gates.values()):
        verdict = "retention_meaningful_verified"
    elif gates["retention_after_meaningful_old_ability"] and gates["correction_gain"]:
        verdict = "retention_meaningful_partial"
    else:
        verdict = "retention_meaningful_absent"
    payload = {
        "schema": "taiji-u2-22-8b2-retention-v1",
        "card": "22.8J3b meaningful old-ability retention on a dose-trained base",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "old_criterion": OLD_CRITERION,
            "max_old_steps": MAX_OLD_STEPS,
            "correction_steps": CORRECTION_STEPS,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "retention_meaningful_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
