"""Bounded four-arm shared-fusion learning with durable actual-CPU endpoints."""

# ruff: noqa: E402 -- account for startup
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
from scripts.training.joint_token_learner_22_5 import load_learner, save_learner
from scripts.training.prepare_joint_learning_22_5 import ADMISSION
from scripts.training.prepare_joint_learning_22_5 import sources as learner_sources
from scripts.training.run_paired_learning_22_5 import control_partition, digest, evaluate
from scripts.training.run_task_goal_learning_22_5 import accuracy_by_round, train_group
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

EXECUTION = ROOT / "plans/manifests/taiji_u2_joint_learning_execution_20261113.json"
REPORT = ROOT / "reports/taiji_u2_joint_learning_20261113.json"


def sources():
    pins = learner_sources()
    for name in (
        "run_joint_learning_22_5.py",
        "run_task_goal_learning_22_5.py",
        "run_paired_learning_22_5.py",
    ):
        relative = f"scripts/training/{name}"
        pins[relative] = hashlib.sha256(
            (ROOT / relative).read_text(encoding="utf-8-sig").encode()
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
        estimated += (
            Path(recipe["initial"]).stat().st_size + 2048 * len(list(model.parameters()))
        ) * (admission["contract"]["max_updates"] + 1)
    assert shutil.disk_usage(admission["run_root"]).free > 1.25 * estimated
    contract = {
        "schema": "taiji-u2-joint-learning-execution-v1",
        "admission_sha256": digest(ADMISSION),
        "source_pins": sources(),
        "run_root": admission["run_root"],
        "groups": list(admission["groups"]),
        "main_group": "joint-query",
        "cpu_per_group": admission["contract"]["cpu_per_group"],
        "cpu_cap": 3600,
        "wall_cap": 7200,
        "max_updates": admission["contract"]["max_updates"],
        "max_single_update_cpu": max(
            2.0, 3 * max(max(group["price_samples"]) for group in admission["groups"].values())
        ),
        "endpoint_compute_tolerance": 0.05,
        "gates": admission["contract"]["gates"],
        "checkpoint_estimated_bytes": estimated,
        "initial_policy": "Evaluate all four initial functions separately; query and goal-control have identical weights but different nonzero functions",
        "optimizer_policy": "Real inherited moments, new fusion moments absent; one isolated overshoot and full rollback",
        "budget_accounting": "price120 and runner3600 reserved; unallocated1230 CPU before audit",
        "check_reads": 0,
        "final_reads": 0,
    }
    EXECUTION.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": "joint_runner_admitted",
                "cpu_per_group": contract["cpu_per_group"],
                "checkpoint_estimated_bytes": estimated,
            }
        ),
        flush=True,
    )


def run():
    torch.set_num_threads(1)
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert (
        digest(ADMISSION) == execution["admission_sha256"] and execution["source_pins"] == sources()
    )
    root = Path(execution["run_root"])
    if REPORT.exists() or (root / "started.bin").exists():
        raise FileExistsError("joint_learning_already_started")
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
    counters = {"optimizer_steps": 0}
    comparisons = {}

    def guard():
        if (
            time.process_time() > execution["cpu_cap"]
            or time.monotonic() - STARTED > execution["wall_cap"]
        ):
            raise RuntimeError("joint_learning_budget_stop")

    def full_evaluation(model):
        return {
            "train": evaluate(model, data["train"], guard),
            "dev": evaluate(model, data["dev"], guard),
            "old_dev32": evaluate_single(model, old_data["fit"], budget_guard=guard),
        }

    with writer_lease(root / "train.lock"):
        atomic_save({"execution_sha256": digest(EXECUTION)}, root / "started.bin")
        try:
            for name in execution["groups"]:
                recipe = admission["groups"][name]
                guard()
                initial_model, _, _ = load_learner(recipe["initial"])
                initial = full_evaluation(initial_model)
                model, summary = train_group(
                    name,
                    recipe,
                    admission,
                    execution,
                    data,
                    root,
                    guard,
                    counters,
                    loader=load_learner,
                    saver=save_learner,
                    source_reader=sources,
                    watched_prefixes=("joint_fusion.",),
                )
                final = full_evaluation(model)
                controls = {
                    kind: evaluate(model, control_partition(data["dev"], kind), guard)
                    for kind in ("without_rgb", "without_questions")
                }
                assert value_digest(model.state_dict()) == summary["parameter_digest"]
                results[name] = {
                    **summary,
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
                            "updates": summary["selected_updates"],
                            "selected_cpu": summary["selected_training_cpu"],
                        }
                    ),
                    flush=True,
                )
            cpus = [item["selected_training_cpu"] for item in results.values()]
            parity = (max(cpus) - min(cpus)) / max(cpus) <= execution["endpoint_compute_tolerance"]
            main, gates = results["joint-query"], execution["gates"]
            passed = (
                main["final"]["train"]["accuracy"] >= gates["train_accuracy"]
                and main["final"]["dev"]["accuracy"] >= gates["dev_accuracy"]
                and min(main["dev_by_round"].values()) >= gates["each_dev_round_accuracy"]
                and main["final"]["dev"]["accuracy"] - main["initial"]["dev"]["accuracy"]
                >= gates["gain_over_frozen"]
                and main["final"]["dev"]["accuracy"] - main["controls"]["without_rgb"]["accuracy"]
                >= gates["rgb_cut_drop"]
                and main["final"]["dev"]["accuracy"]
                - main["controls"]["without_questions"]["accuracy"]
                >= gates["question_cut_drop"]
            )
            comparisons = {
                "query_minus_goal_control": main["final"]["dev"]["accuracy"]
                - results["joint-goal"]["final"]["dev"]["accuracy"],
                "query_minus_native_base": main["final"]["dev"]["accuracy"]
                - results["native-base"]["final"]["dev"]["accuracy"],
                "query_minus_transformer": main["final"]["dev"]["accuracy"]
                - results["transformer"]["final"]["dev"]["accuracy"],
                "point_estimates_only": True,
            }
            verdict = "joint_task_supported" if passed else "joint_task_not_supported"
            if not parity:
                verdict = "resource_parity_unavailable"
        except Exception as error:
            verdict = "budget_stop" if "budget_stop" in str(error) else "engineering_failure"
            results["error"] = f"{type(error).__name__}: {error}"
            parity = False
    report = {
        "schema": "taiji-u2-joint-learning-v1",
        "verdict": verdict,
        "results": results,
        "comparisons": comparisons,
        "training_compute_matched": parity,
        "learner_optimizer_steps": counters["optimizer_steps"],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "execution_sha256": digest(EXECUTION),
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    if (
        report["cpu_seconds"] > execution["cpu_cap"]
        or report["wall_seconds"] > execution["wall_cap"]
    ):
        report["verdict"] = "budget_stop"
    write_report(REPORT, report)
    print(
        json.dumps(
            {key: report[key] for key in ("verdict", "learner_optimizer_steps", "cpu_seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
