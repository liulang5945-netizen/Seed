"""Paired learning with actual-CPU-selected durable endpoints and inherited state."""

# ruff: noqa: E402 -- account for import time
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner, save_learner
from scripts.training.first_a_support_22_5 import evaluate as evaluate_single
from scripts.training.paired_episode_22_5 import episode_loss, rollout_episode
from scripts.training.prepare_paired_learning_22_5 import MANIFEST as ADMISSION
from scripts.training.prepare_paired_learning_22_5 import sources as learner_sources
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

MANIFEST = ROOT / "plans/manifests/taiji_u2_paired_learning_execution_20261113.json"
REPORT = ROOT / "reports/taiji_u2_paired_learning_20261113.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    result = learner_sources()
    result["scripts/training/run_paired_learning_22_5.py"] = hashlib.sha256(
        Path(__file__).read_text(encoding="utf-8-sig").encode()
    ).hexdigest()
    return result


def evaluate(model, partition, guard):
    rows: list[dict[str, Any]] = []
    for episode in partition["episodes"]:
        guard()
        result = rollout_episode(model, partition, episode, budget_guard=guard)
        rows.extend({**row, "episode_id": episode["id"]} for row in result["rows"])
    assert len(rows) == 360
    return {
        "correct": sum(row["correct"] for row in rows),
        "denominator": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "rows": rows,
    }


def ledger(root, row):
    with (root / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def control_partition(partition, kind):
    changed = copy.deepcopy(partition)
    if kind == "without_rgb":
        for world in changed["worlds"].values():
            world["public"]["observations"].pop("rgb", None)
            world["public"]["coordinates"].pop("rgb", None)
    elif kind == "without_questions":
        for episode in changed["episodes"]:
            episode["requests"] = ("请回答。".encode(),) * 2
    else:
        raise ValueError("unknown_paired_input_control")
    return changed


def prepare():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert admission["source_pins"] == learner_sources()
    contract = {
        "schema": "taiji-u2-paired-learning-execution-v1",
        "admission_sha256": digest(ADMISSION),
        "run_root": admission["run_root"],
        "source_pins": sources(),
        "train_cpu": admission["contract"]["train_cpu_per_learner"],
        "max_updates": admission["contract"]["maximum_updates"],
        "cpu_cap": 2400,
        "wall_cap": 4800,
        "max_single_update_cpu": max(
            2.0, 3 * max(row["median_training_cpu"] for row in admission["learners"].values())
        ),
        "compute_tolerance": 0.05,
        "endpoint": admission["contract"]["cpu_endpoint"],
        "gates": admission["design"]["main_gates"],
        "optimizer_policy": "inherit real moments; phase update/cursor start at0; no teacher feedback in evaluation",
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "budget_accounting": "runner2400 newly reserved, project unallocated10290 CPU",
        "runner_preparation_cpu_seconds": time.process_time(),
    }
    MANIFEST.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "admitted",
                "train_cpu_per_learner": contract["train_cpu"],
                "runner_cpu_cap": contract["cpu_cap"],
            }
        ),
        flush=True,
    )


def run():
    torch.set_num_threads(1)
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert (
        digest(ADMISSION) == contract["admission_sha256"] and sources() == contract["source_pins"]
    )
    root = Path(contract["run_root"])
    if REPORT.exists() or (root / "started.bin").exists():
        raise FileExistsError("paired_learning_already_started")
    data = atomic_load(admission["data"]["data_path"])
    assert value_digest(data) == admission["data"]["data_digest"]
    old_data_manifest = json.loads(
        (ROOT / "plans/manifests/taiji_u2_evidence_A_price_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    old_data = atomic_load(Path(old_data_manifest["data"]["run_root"]) / "data.bin")
    assert value_digest(old_data) == old_data_manifest["data"]["digest"]
    results: dict[str, Any] = {}
    calls = 0

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("paired_learning_budget_stop")

    with writer_lease(root / "train.lock"):
        atomic_save({"execution_sha256": digest(MANIFEST)}, root / "started.bin")
        try:
            frozen, _, _ = load_learner(admission["learners"]["native-query"]["initial"])
            frozen_digest = value_digest(frozen.state_dict())
            results["native-query-frozen"] = {
                "train": evaluate(frozen, data["train"], guard),
                "dev": evaluate(frozen, data["dev"], guard),
                "parameter_digest": frozen_digest,
                "optimizer_steps": 0,
                "old_dev32": evaluate_single(frozen, old_data["fit"], budget_guard=guard),
            }
            assert value_digest(frozen.state_dict()) == frozen_digest
            for name, recipe in admission["learners"].items():
                guard()
                assert (
                    sources() == contract["source_pins"]
                    and digest(recipe["initial"]) == recipe["initial_sha256"]
                )
                model, optimizer, payload = load_learner(recipe["initial"])
                if name == "native-query":
                    assert value_digest(model.state_dict()) == frozen_digest
                    initial = {
                        "train": results["native-query-frozen"]["train"],
                        "dev": results["native-query-frozen"]["dev"],
                        "old_dev32": results["native-query-frozen"]["old_dev32"],
                    }
                else:
                    initial = {
                        "train": evaluate(model, data["train"], guard),
                        "dev": evaluate(model, data["dev"], guard),
                        "old_dev32": evaluate_single(model, old_data["fit"], budget_guard=guard),
                    }
                directory = root / name
                directory.mkdir()
                spent = selected_cpu = 0.0
                selected_update = cursor = 0
                selected_path = Path(recipe["initial"])
                first_credit: dict[str, Any] = {}
                overshoot = None
                for update in range(1, contract["max_updates"] + 1):
                    guard()
                    assert sources() == contract["source_pins"]
                    episode = data["train"]["episodes"][admission["order"][cursor % 180]]
                    ledger(
                        root,
                        {"learner": name, "update": update, "cursor": cursor, "status": "opened"},
                    )
                    watched = {
                        key: p.detach().clone()
                        for key, p in model.named_parameters()
                        if update == 1
                        and (
                            key.startswith("evidence_reader.")
                            or key
                            in {"encoder.rgb.weight", "text_input.weight", "byte_head.weight"}
                        )
                    }
                    start = time.process_time()
                    optimizer.zero_grad(set_to_none=True)
                    loss = episode_loss(model, data["train"], episode)
                    loss.backward()
                    assert all(
                        p.grad is None or bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    )
                    optimizer.step()
                    calls += 1
                    model.parameter_revision += 1
                    step_cpu = time.process_time() - start
                    spent += step_cpu
                    cursor += 1
                    if step_cpu > contract["max_single_update_cpu"]:
                        raise RuntimeError("paired_learning_unbounded_single_update")
                    if update == 1:
                        named = dict(model.named_parameters())
                        first_credit = {
                            key: {
                                "gradient_norm": (
                                    None
                                    if named[key].grad is None
                                    else float(named[key].grad.norm())
                                ),
                                "max_parameter_change": float(
                                    (named[key].detach() - before).abs().max()
                                ),
                            }
                            for key, before in watched.items()
                        }
                        assert all(
                            row["gradient_norm"] is not None
                            and row["gradient_norm"] > 0
                            and row["max_parameter_change"] > 0
                            for key, row in first_credit.items()
                            if key.startswith("evidence_reader.")
                        )
                    beyond = spent > contract["train_cpu"]
                    path = directory / f"{'overshoot' if beyond else 'update'}-{update:06}.bin"
                    save_learner(
                        path,
                        model,
                        optimizer,
                        name,
                        payload["lineage"],
                        update=update,
                        cursor=cursor,
                        order=admission["order"],
                        data_digest=admission["data"]["data_digest"],
                    )
                    status = "quarantined" if beyond else "accepted"
                    ledger(
                        root,
                        {
                            "learner": name,
                            "update": update,
                            "cursor": cursor,
                            "status": status,
                            "checkpoint": str(path),
                            "checkpoint_sha256": digest(path),
                            "training_cpu": spent,
                            "step_cpu": step_cpu,
                        },
                    )
                    assert sources() == contract["source_pins"]
                    if beyond:
                        overshoot = {
                            "update": update,
                            "checkpoint": str(path),
                            "training_cpu": spent,
                            "step_cpu": step_cpu,
                        }
                        model, optimizer, restored = load_learner(selected_path)
                        assert restored["update"] == selected_update
                        cursor = restored["cursor"]
                        break
                    selected_update, selected_cpu, selected_path = update, spent, path
                    if update == 1 or update % 256 == 0:
                        print(
                            json.dumps(
                                {
                                    "learner": name,
                                    "update": update,
                                    "training_cpu": spent,
                                    "loss": float(loss.detach()),
                                    "cpu_seconds": time.process_time(),
                                }
                            ),
                            flush=True,
                        )
                if overshoot is None or selected_update == 0:
                    raise RuntimeError("paired_learning_compute_endpoint_unavailable")
                terminal = value_digest(model.state_dict())
                final = {
                    "train": evaluate(model, data["train"], guard),
                    "dev": evaluate(model, data["dev"], guard),
                    "old_dev32": evaluate_single(model, old_data["fit"], budget_guard=guard),
                }
                controls = {}
                for kind in ("without_rgb", "without_questions"):
                    changed = control_partition(data["dev"], kind)
                    controls[kind] = evaluate(model, changed, guard)
                assert value_digest(model.state_dict()) == terminal
                results[name] = {
                    "initial": initial,
                    "final": final,
                    "controls": controls,
                    "selected_updates": selected_update,
                    "selected_training_cpu": selected_cpu,
                    "spent_training_cpu": spent,
                    "overshoot": overshoot,
                    "checkpoint": str(selected_path),
                    "parameter_digest": terminal,
                    "first_update_credit": first_credit,
                }
                atomic_save(results, root / "partial-results.bin")
                print(
                    json.dumps(
                        {
                            "learner": name,
                            "complete": True,
                            "train": final["train"]["accuracy"],
                            "dev": final["dev"]["accuracy"],
                            "selected_cpu": selected_cpu,
                            "updates": selected_update,
                        }
                    ),
                    flush=True,
                )
            values = [results[name]["selected_training_cpu"] for name in admission["learners"]]
            parity = (max(values) - min(values)) / max(values) <= contract["compute_tolerance"]
            primary = results["native-query"]
            gates = contract["gates"]
            passed = (
                primary["final"]["train"]["accuracy"] >= gates["train_accuracy"]
                and primary["final"]["dev"]["accuracy"] >= gates["dev_accuracy"]
                and primary["final"]["dev"]["accuracy"]
                - results["native-query-frozen"]["dev"]["accuracy"]
                >= gates["dev_gain_over_frozen_parent"]
                and primary["final"]["dev"]["accuracy"]
                - primary["controls"]["without_rgb"]["accuracy"]
                >= gates["dev_rgb_cut_accuracy_drop"]
                and primary["final"]["dev"]["accuracy"]
                - primary["controls"]["without_questions"]["accuracy"]
                >= gates["dev_no_question_accuracy_drop"]
            )
            verdict = "paired_learning_supported" if passed else "paired_learning_not_supported"
            if not parity:
                verdict = "resource_parity_unavailable"
        except Exception as error:
            verdict = "budget_stop" if "budget_stop" in str(error) else "engineering_failure"
            results["error"] = f"{type(error).__name__}: {error}"
            parity = False
    report = {
        "schema": "taiji-u2-paired-learning-v1",
        "verdict": verdict,
        "results": results,
        "training_compute_matched": parity,
        "optimizer_steps": calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "threads": 1,
            "dtype": "float32",
            "torch": torch.__version__,
        },
        "execution_sha256": digest(MANIFEST),
    }
    if report["cpu_seconds"] > contract["cpu_cap"] or report["wall_seconds"] > contract["wall_cap"]:
        report["verdict"] = "budget_stop"
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "optimizer_steps": calls,
                "cpu_seconds": report["cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
