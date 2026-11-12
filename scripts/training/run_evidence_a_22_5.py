"""Frozen four-arm evidence learning, per-update durable checkpoints and task exits."""

# ruff: noqa: E402 -- include startup in resource accounting
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
from scripts.training.conditional_a_support_22_5 import paired_example
from scripts.training.evidence_a_learner_22_5 import load_learner, save_learner, source_pins
from scripts.training.first_a_support_22_5 import evaluate, task_loss
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

MANIFEST = ROOT / "plans/manifests/taiji_u2_evidence_A_execution_20261112.json"
REPORT = ROOT / "reports/taiji_u2_evidence_A_learning_20261112.json"
ARMS = ("native-old", "native-query", "native-capacity", "transformer")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    pins = source_pins()
    for name in (
        "scripts/training/run_evidence_a_22_5.py",
        "scripts/training/conditional_a_support_22_5.py",
    ):
        pins[name] = hashlib.sha256(
            (ROOT / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return pins


def append_ledger(root, row):
    with (root / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("evidence_A_execution_already_frozen")
    torch.set_num_threads(1)
    price = json.loads(
        (ROOT / "plans/manifests/taiji_u2_evidence_A_price_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    for name, pin in price["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        )
    data = atomic_load(Path(price["data"]["run_root"]) / "data.bin")
    assert value_digest(data) == price["data"]["digest"]
    root = Path(get_external_path(f"data/experiments/taiji_u2/evidence_A_run/{time.time_ns()}"))
    estimate = 0
    with StepGuard() as blocked:
        for arm in ARMS:
            recipe = price["arms"][arm]
            assert digest(recipe["initial"]) == recipe["initial_sha256"]
            model, optimizer, payload = load_learner(recipe["initial"])
            assert (
                payload["data_digest"] == price["data"]["digest"]
                and payload["order"] == price["data"]["order"]
            )
            assert not optimizer.state and payload["update"] == payload["cursor"] == 0
            assert value_digest(model.state_dict()) == recipe["parameter_digest"]
            parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            per_checkpoint = (
                Path(recipe["initial"]).stat().st_size
                + 3 * parameter_bytes
                + 2048 * len(list(model.parameters()))
            )
            estimate += per_checkpoint * recipe["proposed_updates"]
    free = shutil.disk_usage(root).free
    assert free >= estimate * 1.25
    contract = {
        "schema": "taiji-u2-evidence-A-execution-v1",
        "status": "runner_admitted",
        "run_root": str(root),
        "source_pins": sources(),
        "data": price["data"],
        "arms": price["arms"],
        "arm_order": list(ARMS),
        "main_gate": price["main_gate"],
        "bootstrap": price["bootstrap"],
        "cpu_cap": 3240,
        "wall_cap": 6480,
        "preparation_cpu_seconds": time.process_time(),
        "preparation_wall_seconds": time.monotonic() - STARTED,
        "preparation_reserve_limit_cpu": 60,
        "optimizer_steps_before_run": blocked.calls,
        "training_compute_tolerance": 0.1,
        "checkpoint_estimated_bytes": estimate,
        "disk_free_at_admission": free,
        "checkpoint_policy": "Every update claims, performs one Adam, saves and reads full state, then appends accepted. Existing run marker or partial ledger stops implicit restart.",
        "diagnostics": [
            "rgb_zero",
            "spatial_coordinates_zero",
            "different_entity_request",
            "time_coordinates_zero",
            "episode_identity_changed",
            "new_residual_off",
        ],
        "diagnostic_scope": "Full fit32, frozen final parameters. Source identity change is an episode audit-id invariance check. Single-event time cut does not establish temporal task ability.",
        "budget_accounting": "Runner3240 plus parent180 plus price180 totals stage3600; runner preparation counted inside3240. After reserving3240, project unallocated13350 CPU.",
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    assert contract["preparation_cpu_seconds"] <= 60 and contract["preparation_wall_seconds"] <= 120
    MANIFEST.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "runner_admitted",
                "updates": sum(row["proposed_updates"] for row in price["arms"].values()),
                "estimated_checkpoint_bytes": estimate,
                "disk_free": free,
                "cpu_seconds": contract["preparation_cpu_seconds"],
            }
        ),
        flush=True,
    )


def diagnostics(model, architecture, examples, guard):
    rows = {}
    for name in (
        "rgb_zero",
        "spatial_coordinates_zero",
        "different_entity_request",
        "time_coordinates_zero",
        "episode_identity_changed",
    ):
        changed = copy.deepcopy(examples)
        for example in changed:
            public = example["public"]
            if name == "rgb_zero":
                public["observations"]["rgb"].zero_()
            elif name == "spatial_coordinates_zero":
                public["coordinates"]["rgb"][:, :2].zero_()
            elif name == "different_entity_request":
                example["public"] = paired_example(example)["public"]
            elif name == "time_coordinates_zero":
                for value in public["coordinates"].values():
                    value[:, 3:5].zero_()
            else:
                example["id"] = "audit-id-" + example["id"]
        rows[name] = evaluate(model, changed, budget_guard=guard)
    if architecture in {"native-query", "native-capacity"}:
        previous = model._verification_cuts
        try:
            model._verification_cuts = frozenset({"query_evidence"})
            rows["new_residual_off"] = evaluate(model, examples, budget_guard=guard)
        finally:
            model._verification_cuts = previous
    return rows


def bootstrap(query, other, settings):
    assert [row["id"] for row in query["rows"]] == [row["id"] for row in other["rows"]]
    values = torch.tensor(
        [
            int(a["correct"]) - int(b["correct"])
            for a, b in zip(query["rows"], other["rows"], strict=True)
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(settings["seed"])
    indices = torch.randint(len(values), (settings["replicates"], len(values)), generator=generator)
    samples = values[indices].mean(-1)
    tail = (1 - settings["confidence"]) / 2
    return {
        "gain": float(values.mean()),
        "lower": float(samples.quantile(tail)),
        "upper": float(samples.quantile(1 - tail)),
        "report_only": True,
    }


def run():
    torch.set_num_threads(1)
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert contract["status"] == "runner_admitted" and contract["source_pins"] == sources()
    root = Path(contract["run_root"])
    if REPORT.exists() or (root / "started.bin").exists():
        raise FileExistsError("evidence_A_already_started")
    data = atomic_load(Path(contract["data"]["run_root"]) / "data.bin")
    assert value_digest(data) == contract["data"]["digest"]
    results: dict[str, Any] = {}
    calls, accepted = 0, 0

    def guard():
        if (
            time.process_time() + contract["preparation_cpu_seconds"] > contract["cpu_cap"]
            or time.monotonic() - STARTED + contract["preparation_wall_seconds"]
            > contract["wall_cap"]
        ):
            raise RuntimeError("evidence_A_budget_stop")

    with writer_lease(root / "train.lock"):
        atomic_save({"manifest_sha256": digest(MANIFEST)}, root / "started.bin")
        try:
            for arm in contract["arm_order"]:
                recipe = contract["arms"][arm]
                assert (
                    digest(recipe["initial"]) == recipe["initial_sha256"]
                    and sources() == contract["source_pins"]
                )
                model, optimizer, payload = load_learner(recipe["initial"])
                initial = evaluate(model, data["fit"], budget_guard=guard)
                training_cpu, cursor = 0.0, 0
                directory = root / arm
                directory.mkdir()
                first_credit: dict[str, Any] = {}
                for update in range(1, recipe["proposed_updates"] + 1):
                    guard()
                    assert sources() == contract["source_pins"]
                    batch = [
                        data["train"][contract["data"]["order"][(cursor + i) % 64]]
                        for i in range(2)
                    ]
                    atomic_save(
                        {"update": update, "cursor": cursor, "arm": arm},
                        directory / f"claim-{update:06}.bin",
                    )
                    append_ledger(
                        root, {"arm": arm, "update": update, "cursor": cursor, "status": "opened"}
                    )
                    watched = {
                        name: p.detach().clone()
                        for name, p in model.named_parameters()
                        if update == 1
                        and (name.startswith("evidence_reader.") or name == "encoder.rgb.weight")
                    }
                    start = time.process_time()
                    optimizer.zero_grad(set_to_none=True)
                    loss = task_loss(model, batch)
                    loss.backward()
                    assert all(
                        p.grad is None or bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    )
                    optimizer.step()
                    calls += 1
                    model.parameter_revision += 1
                    training_cpu += time.process_time() - start
                    if update == 1:
                        named = dict(model.named_parameters())
                        first_credit = {
                            name: {
                                "gradient_norm": (
                                    None
                                    if named[name].grad is None
                                    else float(named[name].grad.norm())
                                ),
                                "max_parameter_change": float(
                                    (named[name].detach() - before).abs().max()
                                ),
                            }
                            for name, before in watched.items()
                        }
                        assert all(
                            row["max_parameter_change"] > 0
                            for name, row in first_credit.items()
                            if name.startswith("evidence_reader.")
                        )
                    cursor += 2
                    checkpoint = directory / f"update-{update:06}.bin"
                    save_learner(
                        checkpoint,
                        model,
                        optimizer,
                        arm,
                        payload["lineage"],
                        update=update,
                        cursor=cursor,
                        order=contract["data"]["order"],
                        data_digest=contract["data"]["digest"],
                    )
                    assert optimizer.state and sources() == contract["source_pins"]
                    append_ledger(
                        root,
                        {
                            "arm": arm,
                            "update": update,
                            "cursor": cursor,
                            "status": "accepted",
                            "checkpoint_sha256": digest(checkpoint),
                            "training_cpu": training_cpu,
                        },
                    )
                    accepted += 1
                    if update == 1 or update % 128 == 0 or update == recipe["proposed_updates"]:
                        print(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "update": update,
                                    "planned": recipe["proposed_updates"],
                                    "loss": float(loss.detach()),
                                    "training_cpu": training_cpu,
                                    "accounted_cpu": time.process_time()
                                    + contract["preparation_cpu_seconds"],
                                }
                            ),
                            flush=True,
                        )
                terminal = value_digest(model.state_dict())
                final = evaluate(model, data["fit"], budget_guard=guard)
                interventions = diagnostics(model, arm, data["fit"], guard)
                assert value_digest(model.state_dict()) == terminal
                assert [row["output_hex"] for row in final["rows"]] == [
                    row["output_hex"] for row in interventions["episode_identity_changed"]["rows"]
                ]
                results[arm] = {
                    "initial": initial,
                    "final": final,
                    "diagnostics": interventions,
                    "updates": recipe["proposed_updates"],
                    "training_cpu": training_cpu,
                    "first_update_credit": first_credit,
                    "final_parameter_digest": terminal,
                    "checkpoint": str(checkpoint),
                }
                atomic_save(results, root / "partial-results.bin")
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "completed": True,
                            "fit": final["accuracy"],
                            "diagnostics": {
                                name: row["accuracy"] for name, row in interventions.items()
                            },
                        }
                    ),
                    flush=True,
                )
            comparisons = {
                name: bootstrap(
                    results["native-query"]["final"], results[name]["final"], contract["bootstrap"]
                )
                for name in ("native-old", "native-capacity", "transformer")
            }
            training = [row["training_cpu"] for row in results.values()]
            parity = (max(training) - min(training)) / max(training) <= contract[
                "training_compute_tolerance"
            ]
            gate = contract["main_gate"]
            passed = (
                results["native-query"]["final"]["accuracy"] >= gate["query_fit_accuracy"]
                and comparisons["native-old"]["gain"] >= gate["query_gain_over_old"]
                and comparisons["native-capacity"]["gain"] >= gate["query_gain_over_capacity"]
            )
            verdict = "development_comparison_passed" if passed else "task_gain_not_supported"
            if not parity:
                verdict = "resource_parity_unavailable"
        except Exception as error:
            verdict = "budget_stop" if "budget_stop" in str(error) else "engineering_failure"
            comparisons, parity = {}, False
            results["error"] = f"{type(error).__name__}: {error}"
    report = {
        "schema": "taiji-u2-evidence-A-learning-v1",
        "verdict": verdict,
        "arms": results,
        "comparisons": comparisons,
        "training_compute_matched": parity,
        "optimizer_steps": calls,
        "accepted_checkpoints": accepted,
        "cpu_seconds": time.process_time(),
        "accounted_cpu_seconds": time.process_time() + contract["preparation_cpu_seconds"],
        "wall_seconds": time.monotonic() - STARTED + contract["preparation_wall_seconds"],
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
        "manifest_sha256": digest(MANIFEST),
    }
    if (
        report["accounted_cpu_seconds"] > contract["cpu_cap"]
        or report["wall_seconds"] > contract["wall_cap"]
    ):
        report["verdict"] = "budget_stop"
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "optimizer_steps": calls,
                "accepted_checkpoints": accepted,
                "accounted_cpu": report["accounted_cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
