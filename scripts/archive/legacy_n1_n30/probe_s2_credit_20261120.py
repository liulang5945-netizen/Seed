"""Feedback-credit localization probe: route a correction step to responsible units.

Trains a 30-unit joint24 graph, then repeats a small correction batch (fit items
misclassified by the base model): one arm applies the gradient update only to the
top-k units by gradient importance (``p`` column + ``r`` row norms), the other
applies the SAME update to k random units, both from identical weights.  The credit
arm should improve the corrected items more than the random arm if feedback is
being routed to responsible modules.  3 seeds; zero check; cap 120 CPU.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_credit_20261120.json"
CAP_CPU = 120.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
BASE_STEPS = 900
CORRECTION_STEPS = 5
BATCH = 16
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
TOP_K = 5
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def make_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6,
            units_per_group=5,
            input_dim=8,
            hidden_dim=8,
            output_dim=24,
            seed=20261120 + seed,
        )
    )


def train_base(graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    for step in range(BASE_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, BASE_STEPS)
        batch = torch.randint(0, features.shape[0], (BATCH * 4,), generator=generator)
        gates = torch.ones(batch.numel(), graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def predict(graph: RouteBGraph, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        return graph(channels(features), gates, graph.initial_state())["readout"].argmax(-1)


def unit_importance(
    graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    gates = torch.ones(features.shape[0], graph.units.n_units)
    readout = graph(channels(features), gates, graph.initial_state())["readout"]
    loss = torch.nn.functional.cross_entropy(readout, labels)
    graph.zero_grad(set_to_none=True)
    loss.backward()
    p_grad = graph.units.p.grad.detach().norm(dim=0)
    r_grad = graph.units.r.grad.detach().norm(dim=1)
    return p_grad + r_grad


def restrict_gradients(graph: RouteBGraph, keep: list[int]) -> None:
    mask = torch.zeros(graph.units.n_units)
    mask[keep] = 1.0
    for name, parameter in graph.units.named_parameters():
        if parameter.grad is None:
            continue
        if name == "p":
            parameter.grad.mul_(mask[None, :])
        elif name == "r":
            parameter.grad.mul_(mask[:, None])
        elif name in {"c", "eta", "b"}:
            parameter.grad.mul_(mask)
        elif name == "w":
            parameter.grad.mul_(mask[:, None] * mask[None, :])
        else:
            parameter.grad.zero_()


def correction(
    graph: RouteBGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    seed: int,
) -> float:
    optimizer = torch.optim.Adam(graph.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(20261120 + seed + 5)
    for _ in range(CORRECTION_STEPS):
        if mode == "credit":
            importance = unit_importance(graph, features, labels)
            keep = torch.topk(importance, TOP_K).indices.tolist()
        else:
            keep = torch.randperm(graph.units.n_units, generator=generator)[:TOP_K].tolist()
        gates = torch.ones(features.shape[0], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features), gates, graph.initial_state())["readout"]
        loss = torch.nn.functional.cross_entropy(readout, labels)
        loss.backward()
        restrict_gradients(graph, keep)
        optimizer.step()
        graph.units.project_parameters_()
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    return float((logits.argmax(-1) == labels).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    train_features = torch.stack([adapter_v3_features(ex) for ex in train_examples]).float()
    train_labels = torch.tensor([joint_label(ex) for ex in train_examples])
    fit_features = torch.stack([adapter_v3_features(ex) for ex in fit_examples]).float()
    fit_labels = torch.tensor([joint_label(ex) for ex in fit_examples])

    results = []
    for seed in SEEDS:
        base = make_graph(seed)
        train_base(base, train_features, train_labels, seed)
        predictions = predict(base, fit_features)
        errors = torch.nonzero(predictions != fit_labels).flatten()
        if errors.numel() < 4:
            results.append({"seed": seed, "errors": int(errors.numel()), "skipped": True})
            continue
        batch = errors[: min(BATCH, errors.numel())]
        correction_features = fit_features[batch]
        correction_labels = fit_labels[batch]
        state = {name: value.detach().clone() for name, value in base.state_dict().items()}
        credit = make_graph(seed)
        credit.load_state_dict(state)
        random_arm = make_graph(seed)
        random_arm.load_state_dict(state)
        baseline_accuracy = float(
            (predict(credit, correction_features) == correction_labels).float().mean()
        )
        credit_accuracy = correction(credit, correction_features, correction_labels, "credit", seed)
        random_accuracy = correction(
            random_arm, correction_features, correction_labels, "random", seed
        )
        results.append(
            {
                "seed": seed,
                "errors": int(errors.numel()),
                "baseline": round(baseline_accuracy, 4),
                "credit": round(credit_accuracy, 4),
                "random": round(random_accuracy, 4),
                "credit_gain": round(credit_accuracy - baseline_accuracy, 4),
                "random_gain": round(random_accuracy - baseline_accuracy, 4),
            }
        )
    usable = [result for result in results if not result.get("skipped")]
    gains = [result["credit_gain"] for result in usable]
    random_gains = [result["random_gain"] for result in usable]
    margins = [result["credit"] - result["random"] for result in usable]
    gates = {
        "credit_beats_random_0_03": bool(np.mean(margins) >= 0.03),
        "all_seeds_positive": all(margin > 0 for margin in margins),
        "credit_gain_positive": bool(np.mean(gains) > 0),
    }
    verdict = "feedback_credit_localized" if all(gates.values()) else "feedback_credit_absent"
    payload = {
        "schema": "taiji-u2-credit-v1",
        "card": "feedback credit localization to responsible units",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "base_steps": BASE_STEPS,
            "correction_steps": CORRECTION_STEPS,
            "top_k": TOP_K,
            "seeds": list(SEEDS),
            "greedy": "importance = ||grad p[:,i]|| + ||grad r[i]||",
        },
        "results": results,
        "mean_credit_gain": round(float(np.mean(gains)) if gains else 0.0, 4),
        "mean_random_gain": round(float(np.mean(random_gains)) if random_gains else 0.0, 4),
        "mean_margin": round(float(np.mean(margins)) if margins else 0.0, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": BASE_STEPS * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "feedback_credit_localized" else 1


if __name__ == "__main__":
    raise SystemExit(main())
