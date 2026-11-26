"""Same content-supervised reference with train-objective-accepted parameter steps."""

# ruff: noqa: E402 -- include import cost
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

from scripts.training import calibrate_content_credit_22_5 as content
from scripts.training import calibrate_first_round_22_5 as original
from scripts.training.check_report_run_conditions import write_report
from scripts.training.credit_update_22_5 import accepted_adam_step
from scripts.training.first_round_baseline_22_5 import generate, make_batch
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_stable_credit_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_stable_credit_20261113.json"
STATS: dict[str, Any] = {"optimizer_steps": 0, "accepted_parameter_updates": 0}


def sources():
    pins = content.sources()
    for filename in (
        "scripts/training/calibrate_stable_credit_22_5.py",
        "scripts/training/credit_update_22_5.py",
    ):
        pins[filename] = hashlib.sha256(
            (ROOT / filename).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return pins


def snapshot(model, head, optimizer, attempt, contract, batch, labels, weights, training_cpu):
    result = content.snapshot(
        model, head, optimizer, attempt, contract, batch, labels, weights, training_cpu
    )
    result.update(
        update_rule_sources=sources(),
        accepted_parameter_updates=STATS["accepted_parameter_updates"],
        optimizer_calls=STATS["optimizer_steps"],
    )
    return result


def restore(payload, contract, batch):
    assert payload["update_rule_sources"] == sources()
    assert (
        0
        <= payload["accepted_parameter_updates"]
        <= payload["optimizer_calls"]
        <= payload["update"]
    )
    return content.restore(payload, contract, batch)


def worker(checkpoint, destination):
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batch = make_batch(data["train"])
    _, classes, weights = content.labels_for(batch)
    with StepGuard() as blocked:
        model, head, optimizer = restore(atomic_load(checkpoint), contract, batch)
        result = {
            "measurement": content.measure(model, head, batch, classes, weights),
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
    controls = {}
    for key, item in contract["controls"].items():
        assert hashlib.sha256(Path(item["report"]).read_bytes()).hexdigest() == item["sha256"]
        controls[key] = json.loads(Path(item["report"]).read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batches = {name: make_batch(data[name]) for name in ("train", "dev")}
    batch = batches["train"]
    labels, classes, weights = content.labels_for(batch)
    checkpoint0 = Path(contract["initial_checkpoint"])
    assert (
        hashlib.sha256(checkpoint0.read_bytes()).hexdigest()
        == contract["initial_checkpoint_sha256"]
    )
    model, optimizer = original.restore(atomic_load(checkpoint0), contract["data_digest"])
    head = content.head_for(contract["head_seed"], len(labels))
    optimizer.add_param_group({"params": list(head.parameters())})
    root = Path(get_external_path(f"data/experiments/taiji_u2/stable_credit/{time.time_ns()}"))
    STATS["run_root"] = str(root)

    def guard():
        assert sources() == contract["source_pins"]
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("stable_credit_budget_stop")

    def persist(path, payload):
        assert payload["update_rule_sources"] == sources()
        content.persist(path, payload, model, head, optimizer, contract, batch, classes, weights)

    selected = snapshot(model, head, optimizer, 0, contract, batch, labels, weights, 0.0)
    persist(root / "update-000000.bin", selected)
    training_cpu, overshoot = 0.0, None
    decisions = []
    for attempt in range(1, contract["max_attempts"] + 1):
        guard()
        started = time.process_time()
        result = accepted_adam_step(
            [*model.parameters(), *head.parameters()],
            optimizer,
            lambda model=model, head=head: content.terms(model, head, batch, classes, weights)[
                "total"
            ],
            STATS,
            scales=tuple(contract["scales"]),
            armijo=contract["armijo"],
            tolerance=contract["tolerance"],
        )
        training_cpu += time.process_time() - started
        if result["accepted"]:
            STATS["accepted_parameter_updates"] += 1
        decisions.append({"attempt": attempt, **result, "training_cpu": training_cpu})
        guard()
        current = snapshot(
            model, head, optimizer, attempt, contract, batch, labels, weights, training_cpu
        )
        if training_cpu > contract["training_cpu_cap"]:
            atomic_save(current, root / "overshoot.bin")
            overshoot = {"attempt": attempt, "training_cpu": training_cpu}
            model, head, optimizer = restore(selected, contract, batch)
            break
        selected = current
        if attempt == 1 or attempt % 50 == 0:
            persist(root / f"attempt-{attempt:06}.bin", selected)
        if attempt == 1 or attempt % 100 == 0:
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "accepted_updates": STATS["accepted_parameter_updates"],
                        "objective": result["after"],
                        "scale": result["scale"],
                        "fallback": result["fallback"],
                        "training_cpu": training_cpu,
                        "cpu_seconds": time.process_time(),
                    }
                ),
                flush=True,
            )
    checkpoint = root / "selected.bin"
    persist(checkpoint, selected)
    final = {name: generate(model, value, guard) for name, value in batches.items()}
    body_state = optimizer.state_dict()
    body_optimizer = original.optimizer_for(model)
    body_optimizer.load_state_dict(
        detached_tree(
            {
                "param_groups": body_state["param_groups"][:1],
                "state": {
                    key: value
                    for key, value in body_state["state"].items()
                    if key < len(list(model.parameters()))
                },
            }
        )
    )
    inference = original.payload_for(
        model,
        body_optimizer,
        selected["update"],
        contract["data_digest"],
        batch["episode_ids"],
        selected["training_cpu"],
    )
    inference["training_provenance"] = {
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "accepted_parameter_updates": selected["accepted_parameter_updates"],
    }
    atomic_save(inference, root / "inference-only.bin")
    body_only, _ = original.restore(
        atomic_load(root / "inference-only.bin"), contract["data_digest"]
    )
    for name in batches:
        assert_values_equal(
            final[name],
            generate(body_only, batches[name], guard),
            "stable_credit.auxiliary_removed",
        )
    ablations = {}
    for kind in ("zero_rgb", "empty_question"):
        changed = dict(batches["dev"])
        if kind == "zero_rgb":
            changed["images"] = torch.zeros_like(changed["images"])
        else:
            changed["question_valid"] = torch.zeros_like(changed["question_valid"])
        ablations[kind] = generate(body_only, changed, guard)
        assert ablations[kind]["accuracy"] <= 0.5
    expected = {
        "measurement": content.measure(model, head, batch, classes, weights),
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
    assert_values_equal(expected, recovered["result"], "stable_credit.independent_recovery")
    cpu = time.process_time() + recovered["cpu_seconds"]
    if cpu > contract["cpu_cap"]:
        raise RuntimeError("stable_credit_budget_stop")
    passed = (
        final["train"]["accuracy"] >= contract["gates"]["train"]
        and final["dev"]["accuracy"] >= contract["gates"]["dev"]
        and min(item["accuracy"] for item in final["dev"]["by_question"].values())
        >= contract["gates"]["each_question"]
    )
    report = {
        "schema": "taiji-u2-stable-credit-v1",
        "verdict": (
            "first_round_calibrated_stable_credit" if passed else "stable_credit_not_calibrated"
        ),
        "final": final,
        "controls": ablations,
        "comparators": {
            key: {name: value["accuracy"] for name, value in item["final"].items()}
            for key, item in controls.items()
        },
        "optimizer_steps": STATS["optimizer_steps"],
        "attempts": len(decisions),
        "selected_attempts": selected["update"],
        "selected_accepted_parameter_updates": selected["accepted_parameter_updates"],
        "selected_examples_seen": selected["examples_seen"],
        "selected_training_cpu": selected["training_cpu"],
        "spent_training_cpu": training_cpu,
        "overshoot": overshoot,
        "decisions": decisions,
        "auxiliary_removed_exact": True,
        "independent_recovery_exact": True,
        "independent_recovery_gradient_tensors": len(expected["measurement"]["gradients"]),
        "runtime_parameter_count": sum(parameter.numel() for parameter in body_only.parameters()),
        "context_readings": content.context_summary(model, batch),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "run_root": str(root),
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "train": final["train"]["accuracy"],
                "dev": final["dev"]["accuracy"],
                "accepted_updates": selected["accepted_parameter_updates"],
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
            if not REPORT.exists():
                write_report(
                    REPORT,
                    {
                        "schema": "taiji-u2-stable-credit-v1",
                        "verdict": (
                            "budget_stop" if "budget_stop" in error else "engineering_failure"
                        ),
                        "error": error,
                        **STATS,
                        "cpu_seconds": time.process_time(),
                        "wall_seconds": time.monotonic() - STARTED,
                        "check_reads": 0,
                        "final_reads": 0,
                        "capability_qualifications": [],
                        "run_conditions": {
                            "device": "cpu",
                            "threads": 1,
                            "torch": torch.__version__,
                        },
                    },
                )
            raise
