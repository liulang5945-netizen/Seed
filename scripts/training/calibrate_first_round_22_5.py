"""One frozen first-round calibration dose with complete learner recovery."""

# ruff: noqa: E402 -- include startup in the program budget
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_round_baseline_22_5 import (
    FirstRoundBaseline,
    generate,
    loss_and_logits,
    make_batch,
)
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_first_round_calibration_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_first_round_calibration_20261113.json"
STATS: dict[str, Any] = {"optimizer_steps": 0, "selected_update": 0}


def sources():
    names = (
        "scripts/training/first_round_baseline_22_5.py",
        "scripts/training/calibrate_first_round_22_5.py",
        "scripts/training/paired_world_22_5.py",
        "scripts/training/check_report_run_conditions.py",
        "taiji/u2_operators.py",
        "taiji/utf8_state.py",
        "taiji/u2_history.py",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
        for name in names
    }


def optimizer_for(model):
    return torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )


def payload_for(model, optimizer, update, data_digest, episode_ids, training_cpu):
    return {
        "schema": "taiji-u2-first-round-calibration-learner-v1",
        "seed": model.seed,
        "parameters": detached_tree(model.state_dict()),
        "optimizer": detached_tree(optimizer.state_dict()),
        "update": update,
        "examples_seen": 180 * update,
        "data_order": list(episode_ids),
        "data_digest": data_digest,
        "rng": torch.get_rng_state(),
        "source_pins": sources(),
        "training": model.training,
        "training_cpu": training_cpu,
        "conditions": {
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "torch": str(torch.__version__),
        },
    }


def restore(payload, data_digest):
    assert payload["schema"] == "taiji-u2-first-round-calibration-learner-v1"
    assert payload["source_pins"] == sources() and payload["data_digest"] == data_digest
    assert payload["examples_seen"] == 180 * payload["update"]
    assert payload["conditions"] == {
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
    }
    model = FirstRoundBaseline(payload["seed"])
    model.load_state_dict(payload["parameters"])
    model.train(payload["training"])
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(detached_tree(payload["optimizer"]))
    torch.set_rng_state(payload["rng"])
    return model, optimizer


def measure(model, batch):
    loss, logp = loss_and_logits(model, batch)
    values = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return {
        "loss": loss.detach(),
        "logp": logp.detach(),
        "gradients": [None if value is None else value.detach() for value in values],
    }


def persist(path, payload, model, optimizer, batch):
    assert payload["data_order"] == batch["episode_ids"]
    atomic_save(payload, path)
    restored, other = restore(atomic_load(path), payload["data_digest"])
    assert_values_equal(model.state_dict(), restored.state_dict(), "calibration.parameters")
    assert_values_equal(optimizer.state_dict(), other.state_dict(), "calibration.moments")
    assert_values_equal(
        measure(model, batch), measure(restored, batch), "calibration.next-gradient"
    )


def worker(checkpoint, destination):
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batch = make_batch(data["train"])
    with StepGuard() as blocked:
        payload = atomic_load(checkpoint)
        assert payload["data_order"] == batch["episode_ids"]
        model, optimizer = restore(payload, contract["data_digest"])
        result = {
            "measurement": measure(model, batch),
            "generated": generate(model, batch),
            "optimizer_digest": value_digest(optimizer.state_dict()),
        }
    atomic_save(
        {
            "result": result,
            "pid": os.getpid(),
            "optimizer_steps": blocked.calls,
            "cpu_seconds": time.process_time(),
        },
        destination,
    )


def run():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["source_pins"] == sources()
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batches = {name: make_batch(data[name]) for name in ("train", "dev")}
    assert all(len(batch["episode_ids"]) == 180 for batch in batches.values())
    model = FirstRoundBaseline(contract["seed"])
    assert sum(parameter.numel() for parameter in model.parameters()) == contract["parameters"]
    optimizer = optimizer_for(model)
    root = Path(
        get_external_path(f"data/experiments/taiji_u2/first_round_calibration/{time.time_ns()}")
    )
    STATS["run_root"] = str(root)
    batch = batches["train"]

    def guard():
        assert sources() == contract["source_pins"]
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("first_round_calibration_budget_stop")

    selected = payload_for(model, optimizer, 0, contract["data_digest"], batch["episode_ids"], 0.0)
    persist(root / "update-000000.bin", selected, model, optimizer, batch)
    initial = {name: generate(model, value, guard) for name, value in batches.items()}
    training_cpu, overshoot = 0.0, None
    first_credit = {}
    for update in range(1, contract["max_updates"] + 1):
        guard()
        before = (
            {name: value.detach().clone() for name, value in model.named_parameters()}
            if update == 1
            else {}
        )
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_and_logits(model, batch)
        loss.backward()
        assert all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        optimizer.step()
        STATS["optimizer_steps"] += 1
        training_cpu += time.process_time() - started
        guard()
        if update == 1:
            first_credit = {
                name: {
                    "gradient_norm": float(parameter.grad.norm()),
                    "max_parameter_move": float((parameter.detach() - before[name]).abs().max()),
                }
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        current = payload_for(
            model, optimizer, update, contract["data_digest"], batch["episode_ids"], training_cpu
        )
        if training_cpu > contract["training_cpu_cap"]:
            atomic_save(current, root / "overshoot.bin")
            overshoot = {"update": update, "training_cpu": training_cpu}
            model, optimizer = restore(selected, contract["data_digest"])
            break
        selected = current
        STATS["selected_update"] = update
        if update == 1 or update % 50 == 0:
            persist(root / f"update-{update:06}.bin", selected, model, optimizer, batch)
        if update == 1 or update % 100 == 0:
            print(
                json.dumps(
                    {
                        "update": update,
                        "training_cpu": training_cpu,
                        "loss": float(loss.detach()),
                        "cpu_seconds": time.process_time(),
                    }
                ),
                flush=True,
            )
    assert selected["update"] > 0
    checkpoint = root / "selected.bin"
    persist(checkpoint, selected, model, optimizer, batch)
    final = {name: generate(model, value, guard) for name, value in batches.items()}
    controls = {}
    for kind in ("zero_rgb", "empty_question"):
        changed = dict(batches["dev"])
        if kind == "zero_rgb":
            changed["images"] = torch.zeros_like(changed["images"])
        else:
            changed["question_valid"] = torch.zeros_like(changed["question_valid"])
        controls[kind] = generate(model, changed, guard)
        assert controls[kind]["accuracy"] <= 0.5
    expected = {
        "measurement": measure(model, batch),
        "generated": final["train"],
        "optimizer_digest": value_digest(optimizer.state_dict()),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            str(checkpoint),
            "--output",
            str(root / "worker.bin"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while process.poll() is None:
            guard()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    if process.returncode:
        raise RuntimeError(stderr[-6000:] + stdout[-1000:])
    recovered = atomic_load(root / "worker.bin")
    assert recovered["pid"] != os.getpid() and recovered["optimizer_steps"] == 0
    assert_values_equal(expected, recovered["result"], "calibration.independent_recovery")
    cpu = time.process_time() + recovered["cpu_seconds"]
    if cpu > contract["cpu_cap"]:
        raise RuntimeError("first_round_calibration_budget_stop")
    passed = (
        final["train"]["accuracy"] >= contract["gates"]["train"]
        and final["dev"]["accuracy"] >= contract["gates"]["dev"]
        and min(item["accuracy"] for item in final["dev"]["by_question"].values())
        >= contract["gates"]["each_question"]
    )
    report = {
        "schema": "taiji-u2-first-round-calibration-v1",
        "verdict": "first_round_calibrated" if passed else "first_round_not_calibrated",
        "initial": initial,
        "final": final,
        "controls": controls,
        "optimizer_steps": STATS["optimizer_steps"],
        "selected_updates": selected["update"],
        "selected_examples_seen": selected["examples_seen"],
        "selected_training_cpu": selected["training_cpu"],
        "spent_training_cpu": training_cpu,
        "overshoot": overshoot,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "first_update_credit": first_credit,
        "independent_recovery_gradient_tensors": len(expected["measurement"]["gradients"]),
        "independent_recovery_exact": True,
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_root": str(root),
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "run_conditions": {
            "device": "cpu",
            "threads": 1,
            "torch": torch.__version__,
            "environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "train": final["train"]["accuracy"],
                "dev": final["dev"]["accuracy"],
                "updates": selected["update"],
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        try:
            run()
        except Exception:
            error = traceback.format_exc()
            failure = {
                "schema": "taiji-u2-first-round-calibration-v1",
                "verdict": "budget_stop" if "budget_stop" in error else "engineering_failure",
                "error": error,
                **STATS,
                "cpu_seconds": time.process_time(),
                "wall_seconds": time.monotonic() - STARTED,
                "check_reads": 0,
                "final_reads": 0,
                "capability_qualifications": [],
                "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
            }
            if not REPORT.exists():
                write_report(REPORT, failure)
            raise
