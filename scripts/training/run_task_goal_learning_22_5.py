"""Same-parent factorial with actual-CPU endpoints and conditional history training."""

# ruff: noqa: E402 -- include startup in the bounded runner cost
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import evaluate as evaluate_single
from scripts.training.prepare_task_goal_learning_22_5 import sources as learner_sources
from scripts.training.run_paired_learning_22_5 import control_partition, digest, evaluate, ledger
from scripts.training.task_goal_learner_22_5 import load_learner, save_learner
from scripts.training.task_history_loss_22_5 import history_loss
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

ADMISSION = ROOT / "plans/manifests/taiji_u2_task_goal_learning_admission_20261113_v2.json"
EXECUTION = ROOT / "plans/manifests/taiji_u2_task_goal_learning_execution_20261113.json"
REPORT = ROOT / "reports/taiji_u2_task_goal_learning_20261113.json"


def sources():
    pins = learner_sources()
    for name in (
        "scripts/training/run_task_goal_learning_22_5.py",
        "scripts/training/run_paired_learning_22_5.py",
    ):
        pins[name] = hashlib.sha256(
            (ROOT / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return pins


def prepare():
    if EXECUTION.exists():
        raise FileExistsError(EXECUTION)
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert admission["source_pins"] == learner_sources()
    estimated = 0
    for recipe in admission["groups"].values():
        assert digest(recipe["initial"]) == recipe["initial_sha256"]
        model, optimizer, payload = load_learner(recipe["initial"])
        assert payload["update"] == payload["cursor"] == 0 and optimizer.state
        assert payload["order"] == admission["order"]
        per_file = Path(recipe["initial"]).stat().st_size + 2048 * len(list(model.parameters()))
        estimated += per_file * (admission["contract"]["max_updates"] + 1)
    free = shutil.disk_usage(admission["run_root"]).free
    assert free > estimated * 1.25
    contract = {
        "schema": "taiji-u2-task-goal-execution-v1",
        "status": "runner_admitted",
        "admission_sha256": digest(ADMISSION),
        "run_root": admission["run_root"],
        "source_pins": sources(),
        "cpu_cap": 3600,
        "wall_cap": 7200,
        "cpu_per_group": 300,
        "max_updates": 8192,
        "max_single_update_cpu": 2.0,
        "endpoint_compute_tolerance": 0.05,
        "groups": list(admission["groups"]),
        "main_group": "D",
        "gates": admission["contract"]["gates"],
        "checkpoint_estimated_bytes": estimated,
        "disk_free": free,
        "initial_reuse": "Exact parameter digest plus previously verified neutral mode at zero projection; no behavior selection",
        "budget_accounting": "admission120 and runner3600 reserved, project unallocated6030 CPU",
        "learner_optimizer_steps_before_run": 0,
        "check_reads": 0,
        "final_reads": 0,
    }
    EXECUTION.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "runner_admitted",
                "groups": contract["groups"],
                "cpu_per_group": 300,
                "disk_estimate": estimated,
            }
        ),
        flush=True,
    )


def train_group(name, recipe, admission, contract, data, root, guard, counters):
    assert digest(recipe["initial"]) == recipe["initial_sha256"]
    model, optimizer, payload = load_learner(recipe["initial"])
    directory = root / name
    directory.mkdir()
    spent = selected_cpu = 0.0
    selected_update = cursor = 0
    selected_path = Path(recipe["initial"])
    first_credit: dict[str, Any] = {}
    overshoot = None
    blocked_count = 0
    for update in range(1, contract["max_updates"] + 1):
        guard()
        assert sources() == contract["source_pins"]
        episode = data["train"]["episodes"][admission["order"][cursor % 180]]
        ledger(root, {"learner": name, "update": update, "cursor": cursor, "status": "opened"})
        watched = {
            key: p.detach().clone()
            for key, p in model.named_parameters()
            if update == 1
            and (
                key == "native_task_projection.weight"
                or key in {"encoder.rgb.weight", "text_input.weight", "byte_head.weight"}
            )
        }
        start = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        result = history_loss(model, data["train"], episode, recipe["history"])
        loss = result["loss"]
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step()
        counters["optimizer_steps"] += 1
        model.parameter_revision += 1
        step_cpu = time.process_time() - start
        spent += step_cpu
        cursor = update
        blocked_count += not result["first_ended"]
        if step_cpu > contract["max_single_update_cpu"]:
            raise RuntimeError("task_goal_unbounded_single_update")
        if update == 1:
            named = dict(model.named_parameters())
            first_credit = {
                key: {
                    "gradient_norm": (
                        None if named[key].grad is None else float(named[key].grad.norm())
                    ),
                    "max_parameter_change": float((named[key].detach() - before).abs().max()),
                }
                for key, before in watched.items()
            }
            if "native_task_projection.weight" in first_credit:
                assert first_credit["native_task_projection.weight"]["max_parameter_change"] > 0
        beyond = spent > contract["cpu_per_group"]
        checkpoint = directory / f"{'overshoot' if beyond else 'update'}-{update:06}.bin"
        save_learner(
            checkpoint,
            model,
            optimizer,
            recipe["architecture"],
            payload["lineage"],
            update=update,
            cursor=cursor,
            order=admission["order"],
            data_digest=admission["data"]["data_digest"],
        )
        ledger(
            root,
            {
                "learner": name,
                "update": update,
                "cursor": cursor,
                "status": "quarantined" if beyond else "accepted",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "training_cpu": spent,
                "step_cpu": step_cpu,
                "first_ended": result["first_ended"],
            },
        )
        assert sources() == contract["source_pins"]
        if beyond:
            overshoot = {
                "update": update,
                "checkpoint": str(checkpoint),
                "training_cpu": spent,
                "step_cpu": step_cpu,
            }
            model, optimizer, restored = load_learner(selected_path)
            assert restored["update"] == selected_update
            break
        selected_update, selected_cpu, selected_path = update, spent, checkpoint
        if update == 1 or update % 256 == 0:
            print(
                json.dumps(
                    {
                        "group": name,
                        "update": update,
                        "training_cpu": spent,
                        "loss": float(loss.detach()),
                        "first_ended": result["first_ended"],
                        "cpu_seconds": time.process_time(),
                    }
                ),
                flush=True,
            )
    if overshoot is None or selected_update == 0:
        raise RuntimeError("task_goal_compute_endpoint_unavailable")
    summary = {
        "selected_updates": selected_update,
        "model_optimizer_steps": selected_update + 1,
        "selected_training_cpu": selected_cpu,
        "spent_training_cpu": spent,
        "overshoot": overshoot,
        "checkpoint": str(selected_path),
        "parameter_digest": value_digest(model.state_dict()),
        "first_update_credit": first_credit,
        "blocked_training_episodes": blocked_count,
    }
    return model, summary


def accuracy_by_round(item):
    return {
        str(turn): sum(row["correct"] for row in item["rows"] if row["turn"] == turn) / 180
        for turn in (0, 1)
    }


def run():
    torch.set_num_threads(1)
    contract = json.loads(EXECUTION.read_text(encoding="utf-8"))
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert (
        digest(ADMISSION) == contract["admission_sha256"] and sources() == contract["source_pins"]
    )
    root = Path(contract["run_root"])
    if REPORT.exists() or (root / "started.bin").exists():
        raise FileExistsError("task_goal_learning_already_started")
    data = atomic_load(admission["data"]["data_path"])
    assert value_digest(data) == admission["data"]["data_digest"]
    old_meta = json.loads(
        (ROOT / "plans/manifests/taiji_u2_evidence_A_price_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    old_data = atomic_load(Path(old_meta["data"]["run_root"]) / "data.bin")
    assert value_digest(old_data) == old_meta["data"]["digest"]
    results: dict[str, Any] = {}
    initial_cache = {}
    counters = {"optimizer_steps": 0}

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("task_goal_learning_budget_stop")

    with writer_lease(root / "train.lock"):
        atomic_save({"execution_sha256": digest(EXECUTION)}, root / "started.bin")
        try:
            for name in contract["groups"]:
                recipe = admission["groups"][name]
                guard()
                model, _, _ = load_learner(recipe["initial"])
                key = value_digest(model.state_dict())
                if key not in initial_cache:
                    initial_cache[key] = {
                        "train": evaluate(model, data["train"], guard),
                        "dev": evaluate(model, data["dev"], guard),
                        "old_dev32": evaluate_single(model, old_data["fit"], budget_guard=guard),
                    }
                initial = initial_cache[key]
                model, summary = train_group(
                    name, recipe, admission, contract, data, root, guard, counters
                )
                final = {
                    "train": evaluate(model, data["train"], guard),
                    "dev": evaluate(model, data["dev"], guard),
                    "old_dev32": evaluate_single(model, old_data["fit"], budget_guard=guard),
                }
                controls = {
                    kind: evaluate(model, control_partition(data["dev"], kind), guard)
                    for kind in ("without_rgb", "without_questions")
                }
                assert value_digest(model.state_dict()) == summary["parameter_digest"]
                results[name] = {
                    **summary,
                    "mode": recipe["mode"],
                    "history": recipe["history"],
                    "initial": initial,
                    "final": final,
                    "controls": controls,
                    "dev_by_round": accuracy_by_round(final["dev"]),
                }
                atomic_save(results, root / "partial-results.bin")
                print(
                    json.dumps(
                        {
                            "group": name,
                            "complete": True,
                            "train": final["train"]["accuracy"],
                            "dev": final["dev"]["accuracy"],
                            "by_round": results[name]["dev_by_round"],
                            "selected_cpu": summary["selected_training_cpu"],
                            "updates": summary["selected_updates"],
                        }
                    ),
                    flush=True,
                )
            scores = {
                name: results[name]["final"]["dev"]["accuracy"] for name in ("A", "B", "C", "D")
            }
            factorial = {
                "goal_effect": ((scores["C"] - scores["A"]) + (scores["D"] - scores["B"])) / 2,
                "history_effect": ((scores["B"] - scores["A"]) + (scores["D"] - scores["C"])) / 2,
                "interaction": scores["D"] - scores["C"] - scores["B"] + scores["A"],
                "point_estimates_only": True,
            }
            cpus = [value["selected_training_cpu"] for value in results.values()]
            parity = (max(cpus) - min(cpus)) / max(cpus) <= contract["endpoint_compute_tolerance"]
            primary, gates = results["D"], contract["gates"]
            passed = (
                primary["final"]["train"]["accuracy"] >= gates["train_accuracy"]
                and primary["final"]["dev"]["accuracy"] >= gates["dev_accuracy"]
                and min(primary["dev_by_round"].values()) >= gates["each_dev_round_accuracy"]
                and primary["final"]["dev"]["accuracy"] - primary["initial"]["dev"]["accuracy"]
                >= gates["gain_over_frozen"]
                and primary["final"]["dev"]["accuracy"]
                - primary["controls"]["without_rgb"]["accuracy"]
                >= gates["rgb_cut_drop"]
                and primary["final"]["dev"]["accuracy"]
                - primary["controls"]["without_questions"]["accuracy"]
                >= gates["question_cut_drop"]
            )
            verdict = (
                "state_goal_factorial_supported" if passed else "state_goal_factorial_not_supported"
            )
            if not parity:
                verdict = "resource_parity_unavailable"
        except Exception as error:
            verdict = "budget_stop" if "budget_stop" in str(error) else "engineering_failure"
            results["error"] = f"{type(error).__name__}: {error}"
            parity, factorial = False, {}
    actual_calls = sum(
        value.get("model_optimizer_steps", 0) for name, value in results.items() if name != "error"
    )
    report = {
        "schema": "taiji-u2-task-goal-learning-v1",
        "verdict": verdict,
        "results": results,
        "factorial": factorial,
        "training_compute_matched": parity,
        "completed_group_optimizer_steps": actual_calls,
        "learner_optimizer_steps": counters["optimizer_steps"],
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "execution_sha256": digest(EXECUTION),
    }
    if report["cpu_seconds"] > contract["cpu_cap"] or report["wall_seconds"] > contract["wall_cap"]:
        report["verdict"] = "budget_stop"
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "completed_group_steps": actual_calls,
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
