"""22.6.4b uncertainty calibration evidence probe (pre-registered thresholds).

Pre-registered in the docstring before running:
* joint24, groups=6, 30 units, 3-seed ensemble, 1,200 steps;
* eval split: 256 fit worlds, first 128 for temperature calibration, last 128
  for testing (disjoint);
* verdicts: ``calibration_improved`` if baseline ECE > 0.10 and calibrated ECE
  <= 0.5*baseline; ``calibration_already_adequate`` if baseline ECE <= 0.10 and
  calibrated ECE <= 0.12; otherwise ``calibration_failed``.

Also reports Brier scores and a 10-bin reliability table.  Zero check; cap 150.
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

OUT = ROOT / "reports/taiji_u2_22_6_4b_calibration_20261120.json"
CAP_CPU = 150.0
TRAIN_WORLDS = 256
EVAL_WORLDS = 256
STEPS = 1500
SEEDS = (0, 1, 2)
BATCH = 64
LR = 3e-3
WARMUP = 100
BINS = 10
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def train_model(features: torch.Tensor, labels: torch.Tensor, seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120 + seed
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    return graph


def ensemble_logits(models: list[RouteBGraph], features: torch.Tensor) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        gates = torch.ones(features.shape[0], models[0].units.n_units)
        for graph in models:
            outputs.append(graph(channels(features), gates, graph.initial_state())["readout"])
    return torch.stack(outputs).mean(dim=0).numpy()


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def reliability(probabilities: np.ndarray, labels: np.ndarray) -> list[dict]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    table = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum() == 0:
            continue
        table.append(
            {
                "bin": [round(float(low), 2), round(float(high), 2)],
                "count": int(mask.sum()),
                "confidence": round(float(confidence[mask].mean()), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
            }
        )
    return table


def ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    value = 0.0
    edges = np.linspace(0.0, 1.0, BINS + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum() == 0:
            continue
        value += float(mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(24)[labels]
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.linspace(0.1, 5.0, 200)
    best, best_nll = 1.0, 1e9
    for temperature in grid:
        probabilities = softmax(logits, float(temperature))
        picked = probabilities[np.arange(len(labels)), labels]
        nll = float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_nll, best = nll, float(temperature)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    eval_examples = generate_examples("fit", EVAL_WORLDS)
    train_features = torch.stack(
        [adapter_v3_features(example) for example in train_examples]
    ).float()
    train_labels = torch.tensor([joint_label(example) for example in train_examples])
    eval_features = torch.stack([adapter_v3_features(example) for example in eval_examples]).float()
    eval_labels = np.array([joint_label(example) for example in eval_examples])
    models = [train_model(train_features, train_labels, seed) for seed in SEEDS]
    logits = ensemble_logits(models, eval_features)
    half = EVAL_WORLDS // 2
    calibration_logits, calibration_labels = logits[:half], eval_labels[:half]
    test_logits, test_labels = logits[half:], eval_labels[half:]
    temperature = fit_temperature(calibration_logits, calibration_labels)
    base = softmax(test_logits, 1.0)
    calibrated = softmax(test_logits, temperature)
    ece_before = ece(base, test_labels)
    ece_after = ece(calibrated, test_labels)
    accuracy = float((calibrated.argmax(axis=1) == test_labels).mean())
    verdict = "calibration_failed"
    if ece_before > 0.10 and ece_after <= 0.5 * ece_before:
        verdict = "calibration_improved"
    elif ece_before <= 0.10 and ece_after <= 0.12:
        verdict = "calibration_already_adequate"
    payload = {
        "schema": "taiji-u2-22-6-4b-calibration-v1",
        "card": "22.6.4b uncertainty calibration evidence probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "train_worlds": TRAIN_WORLDS,
            "eval_worlds": EVAL_WORLDS,
            "calibration_items": half,
            "test_items": EVAL_WORLDS - half,
            "ensemble_members": len(SEEDS),
            "nexus": "joint24, groups=6, 30 units",
            "thresholds": "improved: before>0.10 and after<=0.5*before; adequate: before<=0.10 and after<=0.12",
        },
        "temperature": round(temperature, 4),
        "test_accuracy": round(accuracy, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "brier_before": round(brier(base, test_labels), 4),
        "brier_after": round(brier(calibrated, test_labels), 4),
        "reliability_after": reliability(calibrated, test_labels),
        "verdict": verdict,
        "discipline": {"optimizer_steps": STEPS * len(SEEDS), "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "ece_before": payload["ece_before"],
                "ece_after": payload["ece_after"],
                "temperature": payload["temperature"],
                "accuracy": payload["test_accuracy"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict != "calibration_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
