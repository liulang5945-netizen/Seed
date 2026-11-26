"""22.7.0-α r2 interface probe: train on first-A *train* split, evaluate on fit.

r1 (train=fit[:24], heldout=fit[24:]) exposed a design defect: the unit graph
memorised 24 worlds but generalised at chance.  r2 pre-registers the proper
split: fit on the diverse first-A train worlds (256, sampled mini-batches) and
evaluate on the 32 fit worlds.  Same adapter/unit graph/steps and thresholds:
gradients flow, loss drop >= 20%, fit accuracy >= 0.5 with at least 2/3 seeds,
capsule -> independent-process reload bitwise exact, cap 60 CPU, 0 check/final.

This remains an INTERFACE milestone only; 22.7 stays locked, no capability
signing.  Exits: engineering_failure / adapter_insufficient / budget_stop /
adapter_ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_20261120 import (  # noqa: E402
    HELPER,
    build_windows,
)
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_r2_20261120.json"
CAP_CPU = 60.0
STEPS = 500
SEEDS = (0, 1, 2)
TRAIN_WORLDS = 256
BATCH = 24
HELDOUT_WORLDS = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    train_channels, train_labels = build_windows(train_examples)
    heldout_examples = generate_examples("fit", HELDOUT_WORLDS)
    heldout_channels, heldout_labels = build_windows(heldout_examples)
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        spec = UnitGraphSpec(
            groups=2,
            units_per_group=4,
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            seed=20261120 + seed,
        )
        graph = RouteBGraph(spec)
        optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        initial_loss = None
        gradient_norm = None
        generator = torch.Generator().manual_seed(20261120 + seed)
        indices = torch.arange(TRAIN_WORLDS)
        for step in range(STEPS):
            batch = indices[torch.randperm(TRAIN_WORLDS, generator=generator)[:BATCH]]
            channels = {name: tensor[batch] for name, tensor in train_channels.items()}
            labels = train_labels[batch]
            gates = torch.ones(BATCH, graph.units.n_units)
            optimizer.zero_grad(set_to_none=True)
            readout = graph(channels, gates, graph.initial_state())["readout"]
            loss = loss_fn(readout, labels)
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
        final_loss = float(loss.detach()) if STEPS else float("nan")
        with torch.no_grad():
            heldout_gates = torch.ones(HELDOUT_WORLDS, graph.units.n_units)
            heldout_readout = graph(heldout_channels, heldout_gates, graph.initial_state())[
                "readout"
            ]
        accuracy = float((heldout_readout.argmax(-1) == heldout_labels).float().mean())
        capsule = build_capsule(graph)
        artifact = ROOT / f"output/s2_22_7_0/alpha_r2_capsule_seed{seed}.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_capsule(artifact, capsule)
        probe_path = ROOT / f"output/s2_22_7_0/alpha_r2_heldout_seed{seed}.json"
        probe_path.write_text(
            json.dumps({name: tensor.tolist() for name, tensor in heldout_channels.items()}),
            encoding="utf-8",
        )
        heldout_sha = hashlib.sha256(heldout_readout.detach().numpy().tobytes()).hexdigest()
        completed = __import__("subprocess").run(
            [
                sys.executable,
                str(HELPER),
                "--capsule",
                str(artifact),
                "--probe",
                str(probe_path),
            ],
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
        "schema": "taiji-u2-22-7-0-alpha-r2-v1",
        "card": "22.7.0-alpha r2 interface feasibility (train split -> fit evaluation)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "r1_defect": "trained on fit[:24] and generalised at chance",
            "fit_split": "first-A train worlds",
            "train_worlds": TRAIN_WORLDS,
            "batch": BATCH,
            "eval_split": "first-A fit worlds",
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
