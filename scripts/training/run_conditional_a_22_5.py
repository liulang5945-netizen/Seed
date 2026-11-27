"""Frozen objective intervention from P1 research parents; no capability parent assumed."""

# ruff: noqa: E402 -- capture startup before tensor imports
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.conditional_a_support_22_5 import objective, paired_example
from scripts.training.first_a_support_22_5 import (
    evaluate,
    generate_examples,
    load_learner,
    save_learner,
    task_loss,
)
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

MANIFEST = ROOT / "plans/manifests/taiji_u2_conditional_A_execution_20261112_v2.json"
REPORT = ROOT / "reports/taiji_u2_conditional_A_20261112.json"


def finite_grad(model):
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError("nonfinite_conditional_A_gradient")


def dependency(model, examples):
    with torch.no_grad():
        gaps = [
            float(task_loss(model, [paired_example(e)]) - task_loss(model, [e])) for e in examples
        ]
    return sum(gaps) / len(gaps)


def prepare():
    torch.set_num_threads(1)
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    p1 = json.loads(
        (ROOT / "reports/taiji_u2_first_A_pilot_20261112.json").read_text(encoding="utf-8")
    )
    original = json.loads(
        (ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112_v3.json").read_text(
            encoding="utf-8"
        )
    )
    old = atomic_load(Path(original["run_root"]) / "data.bin")
    data = {
        "train": old["train"],
        "price": old["price"],
        "fit": generate_examples("fit", 160)[128:160],
    }
    for example in data["train"] + data["price"] + data["fit"]:
        paired_example(example)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(get_external_path(f"data/experiments/taiji_u2/conditional_a/{run_id}"))
    digest = value_digest(data)
    atomic_save(data, root / "data.bin")
    arms = {}
    with StepGuard() as guard:
        for architecture in ("native", "transformer"):
            for mode in ("token", "conditional"):
                arm = architecture + "-" + mode
                directory = root / arm
                directory.mkdir()
                model, _, _ = load_learner(p1["models"][architecture]["checkpoint"])
                opt = torch.optim.Adam(
                    model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
                )
                initial = value_digest(model.state_dict())
                save_learner(
                    directory / "initial.bin",
                    model,
                    opt,
                    architecture,
                    0,
                    0,
                    original["order"],
                    digest,
                )
                times = []
                for index in range(8):
                    start = time.process_time()
                    opt.zero_grad(set_to_none=True)
                    loss = objective(
                        model, data["price"][(index % 4) * 2 : (index % 4) * 2 + 2], mode
                    )
                    loss.backward()
                    finite_grad(model)
                    times.append(time.process_time() - start)
                assert initial == value_digest(model.state_dict())
                step = statistics.median(times)
                arms[arm] = {
                    "architecture": architecture,
                    "mode": mode,
                    "initial": str(directory / "initial.bin"),
                    "median_step_cpu": step,
                    "price_samples": times,
                    "updates": min(256, max(1, int(32 / step))),
                    "parent_parameter_digest": initial,
                }
    paths = [
        "scripts/training/conditional_a_support_22_5.py",
        "scripts/training/run_conditional_a_22_5.py",
        "scripts/training/first_a_support_22_5.py",
        "scripts/training/task_baselines_22_4_a.py",
    ]
    c = {
        "schema": "taiji-conditional-A-execution-v1",
        "run_id": run_id,
        "run_root": str(root),
        "data_digest": digest,
        "arms": arms,
        "order": original["order"],
        "cpu_cap": 600,
        "wall_cap": 1200,
        "training_cpu_per_arm": 32,
        "source_pins": {
            p: hashlib.sha256((ROOT / p).read_text(encoding="utf-8-sig").encode()).hexdigest()
            for p in paths
        },
        "fit_recipe_indices": [128, 160],
        "scope": "post-P1 objective intervention, development fit only",
        "optimizer_reset": "same fresh Adam within each architecture pair",
        "lambda": 10,
        "margin": 0.1,
        "main_gate": {"accuracy": 0.5, "gain_over_token": 0.15},
        "optimizer_steps_before_run": guard.calls,
        "preparation_cpu_seconds": time.process_time(),
        "formal_A_parent_changed": False,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    MANIFEST.write_text(json.dumps(c, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "stage": "prepared",
                "updates": {k: v["updates"] for k, v in arms.items()},
                "cpu_seconds": time.process_time(),
                "accounted_cpu_seconds": time.process_time() + c["preparation_cpu_seconds"],
            }
        ),
        flush=True,
    )


def run():
    torch.set_num_threads(1)
    c = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = Path(c["run_root"])
    for p, h in c["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / p).read_text(encoding="utf-8-sig").encode()).hexdigest() == h
        ), p
    if REPORT.exists() or (root / "started.bin").exists():
        raise FileExistsError("conditional_A_already_started")
    data = atomic_load(root / "data.bin")
    assert value_digest(data) == c["data_digest"]
    results: dict[str, Any] = {}
    calls = 0

    def guard():
        if (
            time.process_time() + c["preparation_cpu_seconds"] > c["cpu_cap"]
            or time.monotonic() - STARTED > c["wall_cap"]
        ):
            raise RuntimeError("conditional_A_budget_stop")

    with writer_lease(root / "train.lock"):
        atomic_save({"run_id": c["run_id"]}, root / "started.bin")
        try:
            for arm, recipe in c["arms"].items():
                model, opt, _ = load_learner(recipe["initial"])
                cursor = 0
                directory = root / arm
                initial = evaluate(model, data["fit"], budget_guard=guard)
                gap_before = dependency(model, data["fit"][:8])
                training_cpu = 0.0
                for update in range(1, recipe["updates"] + 1):
                    guard()
                    batch = [data["train"][c["order"][(cursor + i) % 64]] for i in range(2)]
                    claim = directory / f"claim-{update:06}.bin"
                    if claim.exists():
                        raise FileExistsError(claim)
                    atomic_save({"update": update, "cursor": cursor}, claim)
                    start = time.process_time()
                    opt.zero_grad(set_to_none=True)
                    loss = objective(model, batch, recipe["mode"])
                    loss.backward()
                    finite_grad(model)
                    opt.step()
                    calls += 1
                    model.parameter_revision += 1
                    training_cpu += time.process_time() - start
                    cursor += 2
                    save_learner(
                        directory / f"update-{update:06}.bin",
                        model,
                        opt,
                        recipe["architecture"],
                        update,
                        cursor,
                        c["order"],
                        c["data_digest"],
                    )
                    if update == 1 or update % 32 == 0 or update == recipe["updates"]:
                        print(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "update": update,
                                    "planned": recipe["updates"],
                                    "loss": float(loss.detach()),
                                }
                            ),
                            flush=True,
                        )
                final = evaluate(model, data["fit"], budget_guard=guard)
                gap_after = dependency(model, data["fit"][:8])
                results[arm] = {
                    "initial": initial,
                    "final": final,
                    "updates": recipe["updates"],
                    "training_cpu": training_cpu,
                    "pair_margin_before": gap_before,
                    "pair_margin_after": gap_after,
                    "checkpoint": str(directory / f"update-{recipe['updates']:06}.bin"),
                }
                atomic_save(results, root / "results.bin")
            verdict = "completed_conditional_A_intervention"
        except Exception as error:
            verdict = "budget_stop" if "budget_stop" in str(error) else "engineering_failure"
            results["error"] = f"{type(error).__name__}: {error}"
    report = {
        "schema": "taiji-conditional-A-results-v1",
        "verdict": verdict,
        "arms": results,
        "optimizer_steps": calls,
        "cpu_seconds": time.process_time(),
        "accounted_cpu_seconds": time.process_time() + c["preparation_cpu_seconds"],
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "scope": c["scope"],
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    if report["accounted_cpu_seconds"] > c["cpu_cap"] or report["wall_seconds"] > c["wall_cap"]:
        report["verdict"] = "budget_stop"
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "optimizer_steps": calls,
                "accuracy": {k: v["final"]["accuracy"] for k, v in results.items() if k != "error"},
                "cpu_seconds": report["cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    else:
        run()
