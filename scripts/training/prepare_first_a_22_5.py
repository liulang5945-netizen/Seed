"""Freeze first-A data, compute-matched doses and exact checkpoint preflight."""

# ruff: noqa: E402 -- include startup in the reported wall interval
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import (
    generate_examples,
    load_learner,
    make_model,
    save_learner,
    task_loss,
)
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import assert_values_equal, atomic_save, value_digest

MANIFEST = ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112.json"
REPORT = ROOT / "reports/taiji_u2_first_A_preflight_20261112.json"


def gradients(model, batch):
    model.zero_grad(set_to_none=True)
    loss = task_loss(model, batch)
    loss.backward()
    measured = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    assert all(g is None or bool(torch.isfinite(g).all()) for g in measured.values())
    return float(loss.detach()), measured


def prepare():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("first_A_preflight_already_frozen")
    torch.set_num_threads(1)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(get_external_path(f"data/experiments/taiji_u2/first_a/{run_id}"))
    dataset = {
        part: generate_examples(part, n) for part, n in [("price", 8), ("train", 64), ("fit", 32)]
    }
    data_digest = value_digest(dataset)
    atomic_save(dataset, run_root / "data.bin")
    order = torch.randperm(64, generator=torch.Generator().manual_seed(20261209)).tolist()
    checks, prices, models = {}, {}, {}
    with StepGuard() as guard:
        for architecture in ("native", "gru", "transformer"):
            model = make_model(architecture)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            directory = run_root / architecture
            directory.mkdir()
            parent_parameters = value_digest(model.state_dict())
            save_learner(
                directory / "initial.bin", model, optimizer, architecture, 0, 0, order, data_digest
            )
            restored, _, _ = load_learner(directory / "initial.bin")
            loss, first = gradients(model, dataset["price"][:2])
            other_loss, second = gradients(restored, dataset["price"][:2])
            assert loss == other_loss
            assert_values_equal(first, second, "preflight.next_batch_gradients")
            samples = []
            for block in range(8):
                start = time.process_time()
                for inner in range(4):
                    offset = (block * 4 + inner) * 2 % 8
                    gradients(model, dataset["price"][offset : offset + 2])
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                shadow = parameter - 0.001 * parameter.grad / (
                                    parameter.grad.abs() + 1e-8
                                )
                                assert bool(torch.isfinite(shadow).all())
                samples.append((time.process_time() - start) / 4)
            price = statistics.median(samples)
            assert price > 0
            assert value_digest(model.state_dict()) == parent_parameters
            checks[architecture] = {
                "checkpoint": str(directory / "initial.bin"),
                "parameter_sha256": parent_parameters,
                "parameters": sum(p.numel() for p in model.parameters()),
                "tensors": len(list(model.parameters())),
                "next_loss": loss,
                "gradient_tensors": len(first),
                "next_batch_equal": True,
            }
            prices[architecture] = {
                "median_step_cpu": price,
                "block_mean_cpu": samples,
                "repeat_blocks": 8,
                "inner_repeats": 4,
                "scope": "forward, backward, local first-Adam algebra; no parameter update",
            }
            models[architecture] = model
    budget = 0.95 * min(row["median_step_cpu"] * 256 for row in prices.values())
    doses = {name: int(budget / row["median_step_cpu"]) for name, row in prices.items()}
    feasible = all(32 <= steps <= 256 for steps in doses.values())
    planned = {name: steps * prices[name]["median_step_cpu"] for name, steps in doses.items()}
    if max(planned.values()) - min(planned.values()) > 0.1 * max(planned.values()):
        feasible = False
    sources = {
        p: hashlib.sha256((ROOT / p).read_text(encoding="utf-8-sig").encode()).hexdigest()
        for p in [
            "scripts/training/first_a_support_22_5.py",
            "scripts/training/task_baselines_22_4_a.py",
            "scripts/training/prepare_first_a_22_5.py",
            "taiji/u2_subject.py",
            "taiji/u2_request_state.py",
            "taiji/u2_events.py",
            "taiji/u2_history.py",
            "taiji/u2_operators.py",
            "taiji/u2_record_metadata.py",
            "taiji/u2_task_world.py",
            "taiji/u2_task_sensors.py",
            "taiji/u2_task_physics.py",
        ]
    }
    manifest = {
        "schema": "taiji-first-A-execution-v1",
        "status": "frozen" if feasible else "compute_parity_unavailable",
        "run_id": run_id,
        "run_root": str(run_root),
        "data_digest": data_digest,
        "order": order,
        "source_pins": sources,
        "models": checks,
        "prices": prices,
        "updates": doses,
        "planned_training_cpu": planned,
        "train_cpu_budget": budget,
        "cpu_cap": 1200,
        "wall_cap": 2400,
        "seed": 20261110,
        "optimizer": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "batch": 2,
        "scope": "single-scene current-position byte QA; train objects1/2, fit objects3, no check/final",
        "signal_gate": {"fit_min": 0.5, "gain_min": 0.15},
        "formal_A_parent_gate_unchanged": True,
        "training_authority": "owner selected necessary A/B controls and budget then first bounded learning",
        "cost_scope": "match training CPU; evaluation and total costs separately disclosed, no architecture lifecycle superiority claim",
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    report = {
        "schema": "taiji-first-A-preflight-v1",
        "verdict": "passed_first_A_preflight" if feasible else "compute_parity_unavailable",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "data_digest": data_digest,
        "partition_counts": {k: len(v) for k, v in dataset.items()},
        "models": checks,
        "prices": prices,
        "updates": doses,
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "updates": doses,
                "prices": prices,
                "run_root": str(run_root),
                "cpu_seconds": report["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    prepare()
