"""Matched-initialization calibration with a train-only answer-context objective."""

# ruff: noqa: E402 -- startup belongs to the bounded calibration
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
from torch import nn
from torch.nn import functional

from scripts.training import calibrate_first_round_22_5 as original
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_round_baseline_22_5 import FirstRoundBaseline, generate, make_batch
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_content_credit_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_content_credit_20261113.json"
STATS: dict[str, Any] = {"optimizer_steps": 0, "selected_update": 0}


def sources():
    return {
        **original.sources(),
        "scripts/training/calibrate_content_credit_22_5.py": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
    }


def labels_for(batch):
    raw = [bytes(row[:-1].tolist()).hex() for row in batch["targets"]]
    labels = sorted(set(raw))
    classes = torch.tensor([labels.index(value) for value in raw], dtype=torch.long)
    counts = torch.bincount(classes, minlength=len(labels))
    weights = len(raw) / (len(labels) * counts.float())
    return labels, classes, weights


def head_for(seed, count):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Linear(64, count)


def terms(model, head, batch, classes, weights):
    context = model.context(
        batch["images"], batch["image_index"], batch["question_ids"], batch["question_valid"]
    )
    embedded = model.embedding(batch["previous"])
    inputs = torch.cat([embedded, context[:, None].expand(-1, embedded.shape[1], -1)], -1)
    output, _ = model.gru(inputs, context[None])
    logp = model.output(output).masked_fill(~batch["teacher_masks"], -torch.inf).log_softmax(-1)
    byte_loss = -logp.gather(-1, batch["targets"][..., None]).mean()
    class_logits = head(context)
    content_loss = (
        functional.cross_entropy(class_logits, classes, reduction="none") * weights[classes]
    ).mean()
    return {
        "byte": byte_loss,
        "content": content_loss,
        "total": byte_loss + content_loss,
        "logp": logp,
        "classification": class_logits,
        "context": context,
    }


def snapshot(model, head, optimizer, update, contract, batch, labels, weights, training_cpu):
    return {
        "schema": "taiji-u2-content-credit-learner-v1",
        "parameters": detached_tree(model.state_dict()),
        "head": detached_tree(head.state_dict()),
        "optimizer": detached_tree(optimizer.state_dict()),
        "body_seed": model.seed,
        "head_seed": contract["head_seed"],
        "labels": labels,
        "weights": weights.clone(),
        "data_digest": contract["data_digest"],
        "data_order": batch["episode_ids"],
        "update": update,
        "examples_seen": 180 * update,
        "training_cpu": training_cpu,
        "source_pins": sources(),
        "rng": torch.get_rng_state(),
        "threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
        "training": model.training,
        "head_training": head.training,
    }


def restore(payload, contract, batch):
    assert payload["schema"] == "taiji-u2-content-credit-learner-v1"
    assert payload["source_pins"] == sources()
    assert payload["threads"] == torch.get_num_threads() and payload["torch"] == str(
        torch.__version__
    )
    assert (
        payload["data_digest"] == contract["data_digest"]
        and payload["data_order"] == batch["episode_ids"]
    )
    assert payload["examples_seen"] == 180 * payload["update"]
    labels, _, weights = labels_for(batch)
    assert labels == payload["labels"]
    assert_values_equal(weights, payload["weights"])
    model = FirstRoundBaseline(payload["body_seed"])
    model.load_state_dict(payload["parameters"])
    model.train(payload["training"])
    head = head_for(payload["head_seed"], len(labels))
    head.load_state_dict(payload["head"])
    head.train(payload["head_training"])
    optimizer = original.optimizer_for(model)
    optimizer.add_param_group({"params": list(head.parameters())})
    optimizer.load_state_dict(detached_tree(payload["optimizer"]))
    torch.set_rng_state(payload["rng"])
    return model, head, optimizer


def measure(model, head, batch, classes, weights):
    result = terms(model, head, batch, classes, weights)
    gradients = torch.autograd.grad(
        result["total"], [*model.parameters(), *head.parameters()], allow_unused=True
    )
    return {
        "logp": result["logp"].detach(),
        "byte_loss": result["byte"].detach(),
        "content_loss": result["content"].detach(),
        "class_logits": result["classification"].detach(),
        "gradients": [None if value is None else value.detach() for value in gradients],
    }


def context_summary(model, batch):
    with torch.no_grad():
        context = model.context(
            batch["images"], batch["image_index"], batch["question_ids"], batch["question_valid"]
        )
        return {
            "rms_variation": float(context.std(0, unbiased=False).square().mean().sqrt()),
            "absolute_above_099_fraction": float((context.abs() > 0.99).float().mean()),
            "exact_unit_fraction": float((context.abs() == 1).float().mean()),
        }


def persist(path, payload, model, head, optimizer, contract, batch, classes, weights):
    atomic_save(payload, path)
    other, other_head, other_optimizer = restore(atomic_load(path), contract, batch)
    assert_values_equal(model.state_dict(), other.state_dict())
    assert_values_equal(head.state_dict(), other_head.state_dict())
    assert_values_equal(optimizer.state_dict(), other_optimizer.state_dict())
    assert_values_equal(
        measure(model, head, batch, classes, weights),
        measure(other, other_head, batch, classes, weights),
    )


def worker(checkpoint, destination):
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batch = make_batch(data["train"])
    _, classes, weights = labels_for(batch)
    with StepGuard() as blocked:
        model, head, optimizer = restore(atomic_load(checkpoint), contract, batch)
        result = {
            "measurement": measure(model, head, batch, classes, weights),
            "generated": generate(model, batch),
            "optimizer_digest": value_digest(optimizer.state_dict()),
        }
    atomic_save(
        {
            "result": result,
            "optimizer_steps": blocked.calls,
            "pid": os.getpid(),
            "cpu_seconds": time.process_time(),
        },
        destination,
    )


def run():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert sources() == contract["source_pins"]
    assert (
        hashlib.sha256(Path(contract["control_report"]).read_bytes()).hexdigest()
        == contract["control_report_sha256"]
    )
    control = json.loads(Path(contract["control_report"]).read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    batches = {name: make_batch(data[name]) for name in ("train", "dev")}
    batch = batches["train"]
    labels, classes, weights = labels_for(batch)
    assert len(labels) == contract["supervision_classes"]
    parent_path = Path(contract["initial_checkpoint"])
    assert (
        hashlib.sha256(parent_path.read_bytes()).hexdigest()
        == contract["initial_checkpoint_sha256"]
    )
    initial_payload = atomic_load(parent_path)
    assert initial_payload["update"] == 0
    model, optimizer = original.restore(initial_payload, contract["data_digest"])
    initial_context = context_summary(model, batch)
    head = head_for(contract["head_seed"], len(labels))
    optimizer.add_param_group({"params": list(head.parameters())})
    root = Path(get_external_path(f"data/experiments/taiji_u2/content_credit/{time.time_ns()}"))
    STATS["run_root"] = str(root)

    def guard():
        assert sources() == contract["source_pins"]
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("content_credit_budget_stop")

    selected = snapshot(model, head, optimizer, 0, contract, batch, labels, weights, 0.0)
    persist(
        root / "update-000000.bin",
        selected,
        model,
        head,
        optimizer,
        contract,
        batch,
        classes,
        weights,
    )
    training_cpu, overshoot = 0.0, None
    for update in range(1, contract["max_updates"] + 1):
        guard()
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        result = terms(model, head, batch, classes, weights)
        result["total"].backward()
        assert all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in [*model.parameters(), *head.parameters()]
        )
        optimizer.step()
        STATS["optimizer_steps"] += 1
        training_cpu += time.process_time() - started
        guard()
        current = snapshot(
            model, head, optimizer, update, contract, batch, labels, weights, training_cpu
        )
        if training_cpu > contract["training_cpu_cap"]:
            atomic_save(current, root / "overshoot.bin")
            overshoot = {"update": update, "training_cpu": training_cpu}
            model, head, optimizer = restore(selected, contract, batch)
            break
        selected = current
        STATS["selected_update"] = update
        if update == 1 or update % 50 == 0:
            persist(
                root / f"update-{update:06}.bin",
                selected,
                model,
                head,
                optimizer,
                contract,
                batch,
                classes,
                weights,
            )
        if update == 1 or update % 100 == 0:
            print(
                json.dumps(
                    {
                        "update": update,
                        "training_cpu": training_cpu,
                        "byte_loss": float(result["byte"].detach()),
                        "content_loss": float(result["content"].detach()),
                        "cpu_seconds": time.process_time(),
                    }
                ),
                flush=True,
            )
    checkpoint = root / "selected.bin"
    persist(checkpoint, selected, model, head, optimizer, contract, batch, classes, weights)
    final = {name: generate(model, value, guard) for name, value in batches.items()}
    body_state = optimizer.state_dict()
    body_state = {
        "param_groups": [body_state["param_groups"][0]],
        "state": {
            key: value
            for key, value in body_state["state"].items()
            if key < len(list(model.parameters()))
        },
    }
    body_optimizer = original.optimizer_for(model)
    body_optimizer.load_state_dict(detached_tree(body_state))
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
        "train_only_supervisor": "answer-context balanced CE",
    }
    atomic_save(inference, root / "inference-only.bin")
    body_only, _ = original.restore(
        atomic_load(root / "inference-only.bin"), contract["data_digest"]
    )
    assert_values_equal(
        final["train"], generate(body_only, batch, guard), "supervisor_removed_train"
    )
    assert_values_equal(
        final["dev"], generate(body_only, batches["dev"], guard), "supervisor_removed_dev"
    )
    controls = {}
    for kind in ("zero_rgb", "empty_question"):
        changed = dict(batches["dev"])
        if kind == "zero_rgb":
            changed["images"] = torch.zeros_like(changed["images"])
        else:
            changed["question_valid"] = torch.zeros_like(changed["question_valid"])
        controls[kind] = generate(body_only, changed, guard)
        assert controls[kind]["accuracy"] <= 0.5
    expected = {
        "measurement": measure(model, head, batch, classes, weights),
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
    assert_values_equal(expected, recovered["result"], "content_credit.independent_recovery")
    assert (
        hashlib.sha256(Path(control["checkpoint"]).read_bytes()).hexdigest()
        == control["checkpoint_sha256"]
    )
    control_model, _ = original.restore(atomic_load(control["checkpoint"]), contract["data_digest"])
    context_readings = {
        "initial": initial_context,
        "byte_only_control": context_summary(control_model, batch),
        "content_supervised": context_summary(model, batch),
    }
    classifier_accuracy = {}
    with torch.no_grad():
        for name, value in batches.items():
            target_classes = torch.tensor(
                [labels.index(bytes(row[:-1].tolist()).hex()) for row in value["targets"]]
            )
            context = model.context(
                value["images"],
                value["image_index"],
                value["question_ids"],
                value["question_valid"],
            )
            classifier_accuracy[name] = float(
                (head(context).argmax(-1) == target_classes).float().mean()
            )
    cpu = time.process_time() + recovered["cpu_seconds"]
    if cpu > contract["cpu_cap"]:
        raise RuntimeError("content_credit_budget_stop")
    passed = (
        final["train"]["accuracy"] >= contract["gates"]["train"]
        and final["dev"]["accuracy"] >= contract["gates"]["dev"]
        and min(item["accuracy"] for item in final["dev"]["by_question"].values())
        >= contract["gates"]["each_question"]
    )
    report = {
        "schema": "taiji-u2-content-credit-v1",
        "verdict": (
            "first_round_calibrated_with_content_credit"
            if passed
            else "content_credit_not_calibrated"
        ),
        "final": final,
        "controls": controls,
        "context_readings": context_readings,
        "diagnostic_head_accuracy": classifier_accuracy,
        "control_accuracy": {name: value["accuracy"] for name, value in control["final"].items()},
        "matched_updates": selected["update"] == control["selected_updates"],
        "optimizer_steps": STATS["optimizer_steps"],
        "selected_updates": selected["update"],
        "selected_examples_seen": selected["examples_seen"],
        "selected_training_cpu": selected["training_cpu"],
        "spent_training_cpu": training_cpu,
        "overshoot": overshoot,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "run_root": str(root),
        "train_only_labels_hex": labels,
        "train_only_weights": weights.tolist(),
        "runtime_parameter_count": sum(parameter.numel() for parameter in body_only.parameters()),
        "auxiliary_removed_exact": True,
        "independent_recovery_gradient_tensors": len(expected["measurement"]["gradients"]),
        "independent_recovery_exact": True,
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
                "schema": "taiji-u2-content-credit-v1",
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
