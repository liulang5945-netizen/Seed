"""22.7.0-α interface feasibility probe (unit graph on first-A fit, small training).

Maps first-A fit worlds to the six legal channels of ``RouteBGraph`` with a fixed,
deterministic statistics adapter, attaches the existing 4-way position label,
trains a short run (500 steps, 3 seeds) and checks:

* gradients flow at initialisation;
* training loss drops by >= 20%;
* held-out fit accuracy >= 0.5 (chance 0.25);
* capsule save -> independent-process reload reproduces the held-out logits
  bitwise;
* zero check/final reads, cap 60 CPU.

This is an INTERFACE milestone only: no growth, no capability signing, 22.7 stays
locked.  Exits: engineering_failure / adapter_insufficient / budget_stop /
adapter_ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402

LABELS = ("左边", "中间", "右边", "未知")  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    LEGAL_INPUT_CHANNELS,
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_20261120.json"
HELPER = ROOT / "scripts/training/s2_22_7_0_alpha_reload_20261120.py"
CAP_CPU = 60.0
STEPS = 500
SEEDS = (0, 1, 2)
TRAIN_WORLDS = 24
HELDOUT_WORLDS = 8


def pool_features(tensor: torch.Tensor, width: int = 8) -> torch.Tensor:
    flat = tensor.reshape(-1).float()
    if flat.numel() >= width:
        return flat[:width]
    return torch.cat([flat, torch.zeros(width - flat.numel())])


def bytes_features(raw: bytes, width: int = 8) -> torch.Tensor:
    values = torch.zeros(width)
    for index, byte in enumerate(raw):
        values[index % width] += float(byte) / 255.0
    return values / max(1.0, float(len(raw)))


def channel_features(example: dict) -> torch.Tensor:
    """Return [6, 8] legal-channel features for one first-A example."""
    public = example["public"]
    observations = public["observations"]
    channels = {
        "rgb": pool_features(observations["rgb"].mean(0)),
        "audio": pool_features(observations["audio"].mean(0)),
        "proprio": pool_features(observations["proprio"].mean(0)),
        "question": bytes_features(public["request"]),
        "goal": bytes_features(public["goal"]),
        "feedback": pool_features(public["outcome"]),
    }
    return torch.stack([channels[name] for name in LEGAL_INPUT_CHANNELS])


def label_index(example: dict) -> int:
    word = example["target"].decode("utf-8")
    return LABELS.index(word)


def build_windows(examples: list[dict]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features = torch.stack([channel_features(example) for example in examples])  # [B,6,8]
    channels = {name: features[:, index, :] for index, name in enumerate(LEGAL_INPUT_CHANNELS)}
    labels = torch.tensor([label_index(example) for example in examples])
    return channels, labels


def train_once(seed: int) -> dict:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=2, units_per_group=4, input_dim=8, hidden_dim=8, output_dim=4, seed=20261120 + seed
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    fit = generate_examples("fit", 32)
    train_examples = fit[:TRAIN_WORLDS]
    heldout_examples = fit[TRAIN_WORLDS : TRAIN_WORLDS + HELDOUT_WORLDS]
    train_channels, train_labels = build_windows(train_examples)
    heldout_channels, heldout_labels = build_windows(heldout_examples)
    gates = torch.ones(TRAIN_WORLDS, graph.units.n_units)
    initial_loss = None
    gradient_norm = None
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        readout = graph(train_channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, train_labels)
        if step == 0:
            initial_loss = float(loss.detach())
            loss.backward()
            gradient_norm = float(
                sum(
                    parameter.grad.detach().pow(2).sum()
                    for parameter in graph.parameters()
                    if parameter.grad is not None
                ).sqrt()
            )
            optimizer.step()
            continue
        loss.backward()
        optimizer.step()
    final_loss = float(loss.detach())
    with torch.no_grad():
        heldout_gates = torch.ones(HELDOUT_WORLDS, graph.units.n_units)
        heldout_readout = graph(heldout_channels, heldout_gates, graph.initial_state())["readout"]
    accuracy = float((heldout_readout.argmax(-1) == heldout_labels).float().mean())
    capsule = build_capsule(graph)
    artifact = ROOT / f"output/s2_22_7_0/alpha_capsule_seed{seed}.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_capsule(artifact, capsule)
    probe_path = ROOT / f"output/s2_22_7_0/alpha_heldout_seed{seed}.json"
    probe_path.write_text(
        json.dumps({name: tensor.tolist() for name, tensor in heldout_channels.items()}),
        encoding="utf-8",
    )
    return {
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": (initial_loss - final_loss) / initial_loss if initial_loss else 0.0,
        "gradient_norm": gradient_norm,
        "heldout_accuracy": accuracy,
        "capsule": str(artifact.relative_to(ROOT)),
        "probe": str(probe_path.relative_to(ROOT)),
        "heldout_logits_sha": hashlib.sha256(
            heldout_readout.detach().numpy().tobytes()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    runs = []
    for seed in SEEDS:
        run = train_once(seed)
        completed = subprocess.run(
            [
                sys.executable,
                str(HELPER),
                "--capsule",
                str(ROOT / run["capsule"]),
                "--probe",
                str(ROOT / run["probe"]),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        run["reload_ok"] = completed.returncode == 0
        run["reload_logits_sha"] = (
            completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else ""
        )
        run["reload_exact"] = run["reload_logits_sha"] == run["heldout_logits_sha"]
        runs.append(run)
    checks = {
        "gradients_flow": all((run["gradient_norm"] or 0.0) > 0 for run in runs),
        "loss_drops": all(run["loss_drop"] >= 0.20 for run in runs),
        "heldout_above_chance": all(run["heldout_accuracy"] >= 0.5 for run in runs),
        "reload_exact": all(run["reload_exact"] for run in runs),
    }
    budget_ok = time.process_time() - started <= CAP_CPU
    passed = all(checks.values()) and budget_ok
    verdict = (
        "budget_stop" if not budget_ok else "adapter_ready" if passed else "adapter_insufficient"
    )
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-v1",
        "card": "22.7.0-alpha interface feasibility (unit graph on first-A fit)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "task": "first-A fit world 4-way position classification",
            "train_worlds": TRAIN_WORLDS,
            "heldout_worlds": HELDOUT_WORLDS,
            "steps": STEPS,
            "seeds": list(SEEDS),
            "chance": 0.25,
            "exits": [
                "engineering_failure",
                "adapter_insufficient",
                "budget_stop",
                "adapter_ready",
            ],
            "cap_cpu": CAP_CPU,
            "check_reads": 0,
            "final_reads": 0,
        },
        "runs": runs,
        "checks": checks,
        "verdict": verdict,
        "scope": "interface milestone only; no growth, no capability signing, 22.7 locked",
        "discipline": {"optimizer_steps": STEPS * len(SEEDS), "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "checks": checks, "runs": runs, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "adapter_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
