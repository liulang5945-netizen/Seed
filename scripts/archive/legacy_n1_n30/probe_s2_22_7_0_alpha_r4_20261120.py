"""22.7.0-α r4: unit graph learns the adapter-v3 features (pre-registered in D-33).

Adapter v3 (raw-media palette histogram with the corrected pixel x-axis) passes
the informativeness gate: 4-way fit accuracy 0.9062 (linear probe).  r4 uses the
same unit-graph configuration as r3 (32 units, lr 3e-3, 2,000 steps, batch 64,
cosine warmup 100) on those 48-dim features.  Success: gradients flow, loss drops
>= 20%, fit accuracy >= 0.5 on >= 2/3 seeds, capsule -> independent-process
reload bitwise exact, 0 check/final, cap 150 CPU.  22.7 stays locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scripts.training.probe_s2_22_7_0_alpha_20261120 import HELPER  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
    label_index,
)
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_r4_20261120.json"
CAP_CPU = 150.0
STEPS = 2000
SEEDS = (0, 1, 2)
TRAIN_WORLDS = 256
BATCH = 64
WARMUP = 100
LR = 3e-3
GROUPS = 4
UNITS_PER_GROUP = 8
HELDOUT_WORLDS = 32


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def build_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", HELDOUT_WORLDS)
    x_train = torch.stack([adapter_v3_features(example) for example in train_examples]).float()
    y_train = torch.tensor([label_index(example) for example in train_examples])
    x_fit = torch.stack([adapter_v3_features(example) for example in fit_examples]).float()
    y_fit = torch.tensor([label_index(example) for example in fit_examples])
    return x_train, y_train, x_fit, y_fit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    x_train, y_train, x_fit, y_fit = build_split()
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        spec = UnitGraphSpec(
            groups=GROUPS,
            units_per_group=UNITS_PER_GROUP,
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            seed=20261120 + seed,
        )
        graph = RouteBGraph(spec)
        optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
        loss_fn = torch.nn.CrossEntropyLoss()
        initial_loss = None
        gradient_norm = None
        generator = torch.Generator().manual_seed(20261120 + seed)
        indices = torch.arange(TRAIN_WORLDS)
        for step in range(STEPS):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step)
            batch = indices[torch.randperm(TRAIN_WORLDS, generator=generator)[:BATCH]]
            channels = {
                name: x_train[batch, index * 8 : (index + 1) * 8]
                for index, name in enumerate(
                    ("rgb", "audio", "proprio", "question", "goal", "feedback")
                )
            }
            gates = torch.ones(BATCH, graph.units.n_units)
            optimizer.zero_grad(set_to_none=True)
            readout = graph(channels, gates, graph.initial_state())["readout"]
            loss = loss_fn(readout, y_train[batch])
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
            channels = {
                name: x_fit[:, index * 8 : (index + 1) * 8]
                for index, name in enumerate(
                    ("rgb", "audio", "proprio", "question", "goal", "feedback")
                )
            }
            heldout_gates = torch.ones(HELDOUT_WORLDS, graph.units.n_units)
            heldout_readout = graph(channels, heldout_gates, graph.initial_state())["readout"]
        accuracy = float((heldout_readout.argmax(-1) == y_fit).float().mean())
        capsule = build_capsule(graph)
        artifact = ROOT / f"output/s2_22_7_0/alpha_r4_capsule_seed{seed}.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_capsule(artifact, capsule)
        probe_path = ROOT / f"output/s2_22_7_0/alpha_r4_heldout_seed{seed}.json"
        probe_path.write_text(
            json.dumps(
                {
                    name: x_fit[:, index * 8 : (index + 1) * 8].tolist()
                    for index, name in enumerate(
                        ("rgb", "audio", "proprio", "question", "goal", "feedback")
                    )
                }
            ),
            encoding="utf-8",
        )
        heldout_sha = hashlib.sha256(heldout_readout.detach().numpy().tobytes()).hexdigest()
        completed = subprocess.run(
            [sys.executable, str(HELPER), "--capsule", str(artifact), "--probe", str(probe_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        reload_sha = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 else ""
        runs.append(
            {
                "seed": seed,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_drop": (initial_loss - final_loss) / initial_loss if initial_loss else 0.0,
                "gradient_norm": gradient_norm,
                "heldout_accuracy": accuracy,
                "reload_exact": reload_sha == heldout_sha,
            }
        )
    checks = {
        "gradients_flow": all((run["gradient_norm"] or 0.0) > 0 for run in runs),
        "loss_drops": all(run["loss_drop"] >= 0.20 for run in runs),
        "heldout_above_chance": sum(1 for run in runs if run["heldout_accuracy"] >= 0.5) >= 2,
        "reload_exact": all(run["reload_exact"] for run in runs),
    }
    budget_ok = time.process_time() - started <= CAP_CPU
    passed = all(checks.values()) and budget_ok
    verdict = (
        "budget_stop" if not budget_ok else "adapter_ready" if passed else "adapter_insufficient"
    )
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-r4-v1",
        "card": "22.7.0-alpha r4 interface feasibility (adapter v3 features)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "adapter": "v3 raw-media palette histogram, corrected x-axis",
            "adapter_probe_fit": 0.9062,
            "groups": GROUPS,
            "units_per_group": UNITS_PER_GROUP,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "warmup": WARMUP,
            "train_worlds": TRAIN_WORLDS,
            "heldout_worlds": HELDOUT_WORLDS,
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
