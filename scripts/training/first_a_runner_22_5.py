"""Single-writer compute-matched first-A pilot with durable step checkpoints."""

# ruff: noqa: E402 -- account for startup before tensor imports
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import evaluate, load_learner, save_learner, task_loss
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

MANIFEST = ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112_v3.json"
REPORT = ROOT / "reports/taiji_u2_first_A_pilot_20261112.json"


class BudgetStop(RuntimeError):
    pass


def append_ledger(path, row):
    raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run():
    torch.set_num_threads(1)
    c = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if c["status"] != "frozen":
        raise ValueError("first_A_not_admitted")
    for path, pin in c["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / path).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        ), path
    root = Path(c["run_root"])
    data = atomic_load(root / "data.bin")
    assert value_digest(data) == c["data_digest"]
    ledger = root / "training.ndjson"
    if ledger.exists() or REPORT.exists():
        raise FileExistsError("pilot_run_already_started")
    state = {
        "schema": "taiji-first-A-run-state-v1",
        "status": "running",
        "run_id": c["run_id"],
        "models": {},
    }
    results = {}
    calls = 0

    def budget_guard():
        if time.process_time() + 150 > c["cpu_cap"] or time.monotonic() - STARTED > c["wall_cap"]:
            raise BudgetStop("first_A_total_budget_stop")

    with writer_lease(root / "training.lock"):
        atomic_save(state, root / "run_state.bin")
        try:
            for architecture in ("native", "gru", "transformer"):
                budget_guard()
                model, optimizer, loaded = load_learner(c["models"][architecture]["checkpoint"])
                directory = root / architecture
                initial = evaluate(model, data["fit"], budget_guard=budget_guard)
                write_report(directory / "initial_evaluation.json", initial)
                order = c["order"]
                cursor = 0
                training_cpu = 0.0
                losses = []
                updates = []
                count = c["updates"][architecture]
                for update in range(1, count + 1):
                    budget_guard()
                    indices = [order[(cursor + i) % 64] for i in range(2)]
                    batch = [data["train"][i] for i in indices]
                    claim = directory / f"claim-{update:06}.bin"
                    if claim.exists():
                        raise FileExistsError(claim)
                    atomic_save(
                        {
                            "architecture": architecture,
                            "update": update,
                            "parent_update": update - 1,
                            "cursor": cursor,
                            "data_digest": c["data_digest"],
                        },
                        claim,
                    )
                    append_ledger(
                        ledger,
                        {
                            "event": "open",
                            "architecture": architecture,
                            "update": update,
                            "cursor": cursor,
                        },
                    )
                    started = time.process_time()
                    optimizer.zero_grad(set_to_none=True)
                    loss = task_loss(model, batch)
                    loss.backward()
                    named = dict(model.named_parameters())
                    if not all(
                        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in named.values()
                    ):
                        raise ValueError("nonfinite_first_A_gradient")
                    watched = [
                        name
                        for name in named
                        if name.endswith("encoder.rgb.weight")
                        or name.endswith("text_input.weight")
                        or name in {"native_w", "byte_head.weight"}
                    ]
                    before = {name: named[name].detach().clone() for name in watched}
                    norms = {
                        name: None if named[name].grad is None else float(named[name].grad.norm())
                        for name in watched
                    }
                    optimizer.step()
                    calls += 1
                    model.parameter_revision += 1
                    elapsed = time.process_time() - started
                    training_cpu += elapsed
                    if not all(bool(torch.isfinite(p).all()) for p in named.values()):
                        raise ValueError("nonfinite_first_A_parameter")
                    cursor += 2
                    checkpoint = directory / f"update-{update:06}.bin"
                    digest = save_learner(
                        checkpoint,
                        model,
                        optimizer,
                        architecture,
                        update,
                        cursor,
                        order,
                        c["data_digest"],
                    )
                    row = {
                        "event": "accepted",
                        "architecture": architecture,
                        "update": update,
                        "cursor": cursor,
                        "loss": float(loss.detach()),
                        "cpu_seconds": elapsed,
                        "checkpoint": str(checkpoint),
                        "checkpoint_digest": digest,
                        "gradient_norms": norms,
                        "parameter_change": {
                            name: float((named[name].detach() - before[name]).abs().max())
                            for name in watched
                        },
                    }
                    append_ledger(ledger, row)
                    updates.append(row)
                    losses.append(row["loss"])
                    state["models"][architecture] = {
                        "update": update,
                        "cursor": cursor,
                        "checkpoint": str(checkpoint),
                    }
                    atomic_save(state, root / "run_state.bin")
                    if update == 1 or update % 16 == 0 or update == count:
                        print(
                            json.dumps(
                                {
                                    "architecture": architecture,
                                    "update": update,
                                    "planned": count,
                                    "loss": row["loss"],
                                    "training_cpu": training_cpu,
                                }
                            ),
                            flush=True,
                        )
                final = evaluate(model, data["fit"], budget_guard=budget_guard)
                write_report(directory / "final_evaluation.json", final)
                gain = final["accuracy"] - initial["accuracy"]
                results[architecture] = {
                    "updates": count,
                    "initial": initial,
                    "final": final,
                    "gain": gain,
                    "training_cpu_seconds": training_cpu,
                    "planned_training_cpu": c["planned_training_cpu"][architecture],
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "learning_signal": final["accuracy"] >= 0.5 and gain >= 0.15,
                    "checkpoint": state["models"][architecture]["checkpoint"],
                    "first_update": updates[0],
                    "last_update": updates[-1],
                }
                state["models"][architecture]["status"] = "complete"
                atomic_save(state, root / "run_state.bin")
            costs = [r["training_cpu_seconds"] for r in results.values()]
            parity = (max(costs) - min(costs)) / max(costs) <= 0.10
            state["status"] = "complete"
            verdict = (
                "pilot_learning_signal"
                if results["native"]["learning_signal"]
                else "pilot_task_gain_absent"
            )
        except Exception as error:
            state["status"] = (
                "budget_stop" if isinstance(error, BudgetStop) else "engineering_failure"
            )
            state["error"] = f"{type(error).__name__}: {error}"
            verdict = state["status"]
            parity = False
        atomic_save(state, root / "run_state.bin")
    cpu = time.process_time() + 150
    if cpu > c["cpu_cap"]:
        verdict = "budget_stop"
    report = {
        "schema": "taiji-first-A-pilot-v1",
        "verdict": verdict,
        "run_id": c["run_id"],
        "models": results,
        "optimizer_step_calls": calls,
        "run_state": state,
        "training_cpu_matched_within10pct": parity,
        "cpu_seconds_worker": time.process_time(),
        "preparation_reserve_cpu": 150,
        "accounted_cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "cpu_cap": c["cpu_cap"],
        "check_reads": 0,
        "final_reads": 0,
        "qualification": "development fit pilot, one seed and episodic current position; no formal A parent or lifecycle architecture superiority",
        "capability_qualifications": [],
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
        "manifest_sha256": hashlib.sha256(
            MANIFEST.read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "optimizer_steps": calls,
                "training_cpu_matched": parity,
                "scores": {k: v["final"]["accuracy"] for k, v in results.items()},
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
