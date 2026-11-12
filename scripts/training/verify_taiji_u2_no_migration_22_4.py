"""Consume the frozen u3/u4 slots as zero-step recovery when no growth occurred."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.u2_attempt_ledger_22_3_4_3 import AttemptLedger
from scripts.training.u2_update_executor_22_4_1 import load_json, write_once
from scripts.training.u2_update_support_22_3_4_3 import file_sha, fingerprint, readback
from scripts.training.u2_update_support_22_4_1 import normalized_sha256
from scripts.training.verify_taiji_u2_cross_process_22_3 import (
    legal_operation,
    packet_for,
    prepare_action,
)
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save, detached_tree
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import U2Runtime, run_conditions
from taiji.u2_subject import synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_1_no_migration_20261112_v2.json"
U3 = "u3-migration-original"
U4 = "u4-migration-recovery"


def validate_unchanged_width(runtime: U2Runtime) -> None:
    model = runtime.model
    if model.c.group_units is not None or model.c.base_units is not None:
        raise ValueError("The frozen no_migration_rule does not apply to a grown layout")
    if model.group_extensions or model.native_r.shape[1] != model.c.units:
        raise ValueError("Actual neural widths differ from the declared dense base")
    if not model.c.record_metadata or len(list(model.parameters())) != 194:
        raise ValueError("The source must be the accepted metadata-enabled u1 graph")


def measure_continuation(runtime: U2Runtime, directory: Path) -> dict[str, Any]:
    """Compare four loss credit maps after the same legal sensor continuation."""
    before = fingerprint(runtime)
    records = {record.record_id: record.raw for record in runtime.state["records"]}
    runtime.set_goal("逐字复述红盒记录。".encode())
    runtime.choose(legal_operation(0))
    prepare_action(runtime, directory)
    fixture = synthetic_fixture(runtime.model.c, runtime.model.native_w.dtype)
    packet = packet_for(runtime, fixture, "22.4.1-no-migration-continuation")
    receipt = runtime.receive(packet)
    model = runtime.model
    utterance = model.begin_utterance(runtime.state)
    goal = model.goal(utterance["goal_raw"])
    terms = []
    for symbol in [*records["red"], 256]:
        probability, _, output = model.output(utterance, goal, utterance["prediction_action"])
        terms.append(-probability[symbol])
        utterance = model.accept_byte(utterance, symbol, output)
    losses = {
        "task": torch.stack(terms).mean(),
        "predictive": -receipt["predictive_logp"].sum(),
        "actor": receipt["actor"],
        "critic": receipt["critic"],
    }
    parameters = dict(model.named_parameters())
    gradients = {}
    for kind, loss in losses.items():
        values = torch.autograd.grad(
            loss, list(parameters.values()), allow_unused=True, retain_graph=True
        )
        gradients[kind] = dict(zip(parameters, detached_tree(values), strict=True))
    after = fingerprint(runtime)
    for name in ("parameter_sha256", "moment_sha256", "revision"):
        assert before[name] == after[name], name
    assert runtime.state["settled"].count(packet.receipt_id) == 1
    return {
        "fingerprint": after,
        "structure": model.structure(),
        "log_output": runtime.output()[0].detach(),
        "policy": model.policy(runtime.state)["log_policy"].detach(),
        "gradients": gradients,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "effect_count": runtime.executor.effect_count,
        "settlements": runtime.state["settled"].count(packet.receipt_id),
        "run_conditions": run_conditions(model),
    }


def compare_measurements(original: dict, restored: dict) -> list[dict[str, Any]]:
    for name in original.keys() - {"gradients"}:
        assert_values_equal(original[name], restored[name], name)
    assert original["gradients"].keys() == restored["gradients"].keys()
    rows = []
    for kind, left in original["gradients"].items():
        right = restored["gradients"][kind]
        assert left.keys() == right.keys()
        for name, value in left.items():
            peer = right[name]
            if value is None:
                assert peer is None, (kind, name)
                rows.append({"loss": kind, "parameter": name, "status": "unused", "delta": 0.0})
                continue
            assert peer is not None and bool(torch.isfinite(value).all()), (kind, name)
            assert torch.equal(value, peer), (kind, name)
            rows.append(
                {
                    "loss": kind,
                    "parameter": name,
                    "status": "nonzero" if bool(value.any()) else "zero",
                    "gradient_norm": float(value.norm()),
                    "delta": float((value - peer).abs().max()),
                }
            )
    return rows


def check_sources(contract: dict) -> None:
    if contract.get("authorized") is not True or contract["optimizer_steps_authorized"] != 0:
        raise ValueError("The two no-migration slots require a zero-step contract")
    for path, expected in contract["source_pins"].items():
        if normalized_sha256(ROOT / path) != expected:
            raise ValueError(f"Source pin differs: {path}")
    for row in contract["references"].values():
        if normalized_sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Reference pin differs: {row['path']}")


def worker(contract_path: Path) -> int:
    torch.set_num_threads(1)
    contract = load_json(contract_path)
    check_sources(contract)
    root = ROOT / contract["run_root"]
    with StepGuard() as guard:
        runtime, proof = readback(root / "u3.bin", contract["parent"]["fingerprint"])
        validate_unchanged_width(runtime)
        rebuilt, _ = rebuild_state(runtime.model, runtime.history, runtime.state)
        assert_values_equal(runtime.state, rebuilt, "worker.rebuild")
        runtime.save(root / "u4.bin")
        _, post_proof = readback(root / "u4.bin", contract["parent"]["fingerprint"])
        result = measure_continuation(runtime, root / "recovered")
        atomic_save(result, root / "u4-measurement.bin")
    payload = {
        "schema": "taiji-u2-22.4.1-no-migration-worker-v1",
        "verdict": "pass",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "parent_readback": proof,
        "post_readback": post_proof,
        "state_rebuild_equal": True,
        "optimizer_step_calls": guard.calls,
        "run_conditions": result["run_conditions"],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
    }
    destination = ROOT / contract["worker_report"]
    if destination.exists():
        raise FileExistsError(destination)
    write_report(destination, payload)
    return 0


def verify(contract_path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    contract = load_json(contract_path)
    check_sources(contract)
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / "run.json", contract)
    parent = contract["parent"]
    shutil.copyfile(ROOT / parent["checkpoint"], root / "u1-post.bin")
    assert file_sha(root / "u1-post.bin") == parent["checkpoint_sha256"]
    ledger = AttemptLedger(
        ROOT / contract["ledger"],
        {
            "run_id": contract["run_id"],
            "authorized": True,
            "owner_approval_ref": contract["owner_approval_ref"],
            "manifest_sha256": normalized_sha256(contract_path),
        },
        contract["source_pins"],
        {
            "run": {
                "attempt_order": [U3, U4],
                "budget_max": 2,
                "attempt_parents": {U3: "u1-post.bin", U4: "u3.bin"},
                "attempt_posts": {U3: "u3.bin", U4: "u4.bin"},
            }
        },
    )
    ledger.start()
    identity = U3
    with StepGuard() as guard:
        try:
            ledger.open(U3, parent_checkpoint="u1-post.bin", zero_step_recovery=True)
            runtime, source_proof = readback(root / "u1-post.bin", parent["fingerprint"])
            validate_unchanged_width(runtime)
            rebuilt, _ = rebuild_state(runtime.model, runtime.history, runtime.state)
            assert_values_equal(runtime.state, rebuilt, "original.rebuild")
            runtime.save(root / "u3.bin")
            _, u3_proof = readback(root / "u3.bin", parent["fingerprint"])
            original = measure_continuation(runtime, root / "original")
            atomic_save(original, root / "u3-measurement.bin")
            u3_cpu, u3_wall = time.process_time(), time.monotonic() - STARTED
            assert u3_cpu <= 30 and u3_wall <= 60, "u3 resource cap"
            u3 = {
                "id": U3,
                "status": "accepted",
                "parent_readback": source_proof,
                "post_readback": u3_proof,
                "state_rebuild_equal": True,
                "optimizer_step_calls": guard.calls,
                "cpu_seconds": u3_cpu,
                "wall_seconds": u3_wall,
                "run_conditions": run_conditions(runtime.model),
            }
            u3_path = ROOT / contract["receipts"][U3]
            write_report(u3_path, u3)
            ledger.close(
                U3,
                outcome="accepted",
                post_checkpoint="u3.bin",
                optimizer_step_calls=0,
                receipt_sha256=normalized_sha256(u3_path),
            )

            identity = U4
            ledger.open(U4, parent_checkpoint="u3.bin", zero_step_recovery=True)
            cpu_start, wall_start = time.process_time(), time.monotonic()
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.training.verify_taiji_u2_no_migration_22_4",
                    "--worker",
                    "--contract",
                    str(contract_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
            if child.returncode:
                raise RuntimeError(f"Recovery worker failed: {child.stderr[-2000:]}")
            worker_report = load_json(ROOT / contract["worker_report"])
            assert worker_report["verdict"] == "pass" and worker_report["pid"] != os.getpid()
            assert worker_report["optimizer_step_calls"] == 0
            rows = compare_measurements(original, atomic_load(root / "u4-measurement.bin"))
            u4_cpu = time.process_time() - cpu_start + worker_report["cpu_seconds"]
            u4_wall = time.monotonic() - wall_start
            assert u4_cpu <= 30 and u4_wall <= 60, "u4 resource cap"
            u4 = {
                "id": U4,
                "status": "accepted",
                "worker": contract["worker_report"],
                "worker_sha256": normalized_sha256(ROOT / contract["worker_report"]),
                "gradient_rows": rows,
                "max_abs_gradient_delta": 0.0,
                "optimizer_step_calls": 0,
                "cpu_seconds": u4_cpu,
                "wall_seconds": u4_wall,
                "run_conditions": original["run_conditions"],
            }
            u4_path = ROOT / contract["receipts"][U4]
            write_report(u4_path, u4)
            ledger.close(
                U4,
                outcome="accepted",
                post_checkpoint="u4.bin",
                optimizer_step_calls=0,
                receipt_sha256=normalized_sha256(u4_path),
            )
        except Exception as error:
            state = ledger.read()
            if identity in state["opened"] and identity not in state["closed"]:
                ledger.close(
                    identity,
                    outcome="exception",
                    optimizer_step_calls=guard.calls,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
    state = ledger.read()
    return {
        "schema": "taiji-u2-22.4.1-no-migration-summary-v1",
        "verdict": "passed_no_migration_zero_step_recovery",
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": normalized_sha256(contract_path),
        "parent": parent,
        "actual_group_widths": [runtime.model.c.units] * runtime.model.c.groups,
        "slots_consumed": state["consumed"],
        "ledger_rows": len(state["rows"]),
        "ledger": contract["ledger"],
        "ledger_head": state["head"],
        "optimizer_step_calls": guard.calls,
        "compared_parameters": len(list(runtime.model.parameters())),
        "compared_losses": list(original["gradients"]),
        "max_abs_gradient_delta": 0.0,
        "cpu_seconds": time.process_time() + worker_report["cpu_seconds"],
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": original["run_conditions"],
        "capability_qualifications": [],
        "scope": "Parent side is a restored durable u1 checkpoint. No live old process, new growth, optimizer update or task learning is certified.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.contract.resolve())
    contract = load_json(args.contract)
    destination = ROOT / contract["summary"]
    if destination.exists():
        raise FileExistsError(destination)
    report = verify(args.contract.resolve())
    write_report(destination, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "verdict",
                    "slots_consumed",
                    "optimizer_step_calls",
                    "compared_parameters",
                    "max_abs_gradient_delta",
                    "cpu_seconds",
                    "wall_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
