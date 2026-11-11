"""Execute the frozen six-update/two-fault U2 engineering protocol.

--dry-run materializes and reads back both parents with zero optimizer calls.
--run requires the frozen execution manifest and the recorded owner reference.
Recovery step2 runs in a separate process. Every attempt is opened durably first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.u2_attempt_ledger_22_3_4_3 import (
    AttemptLedger,
    AuthorizationError,
    LedgerError,
)
from scripts.training.u2_update_executor_22_3_4_3 import (
    failed_attempt,
    load_manifest,
    perform,
    write_once,
)
from scripts.training.u2_update_support_22_3_4_3 import (
    ReadbackError,
    StepCounter,
    distributions,
    file_sha,
    fingerprint,
    fresh_load,
    prepare_parent,
    readback,
)
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save
from taiji.u2_recovery import run_conditions

DEFAULT_MANIFEST = ROOT / "plans/manifests/taiji_u2_22_3_4_3_execution_20261111.json"


@contextmanager
def step_guard(counter):
    original_adam, original_sgd = torch.optim.Adam.step, torch.optim.SGD.step
    torch.optim.Adam.step = counter.wrapper(original_adam)

    def forbidden(*args, **kwargs):
        raise LedgerError("SGD is outside the frozen execution recipe")

    torch.optim.SGD.step = forbidden
    try:
        yield
    finally:
        torch.optim.Adam.step, torch.optim.SGD.step = original_adam, original_sgd


def precontract(manifest):
    path = ROOT / manifest["precontract_path"]
    if file_sha(path) != manifest["precontract_sha256"]:
        raise ValueError("Frozen precontract changed")
    return json.loads(path.read_text(encoding="utf-8"))


def context(root, manifest, manifest_path):
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if (
        run["manifest_sha256"] != file_sha(manifest_path)
        or run["owner_approval_ref"] != manifest["owner_approval_ref"]
    ):
        raise AuthorizationError("Run authorization/manifest identity mismatch")
    ledger_path = ROOT / f"reports/taiji_u2_22_3_4_3_attempts_{run['run_id']}.ndjson"
    return run, AttemptLedger(ledger_path, run, manifest["source_pins"], precontract(manifest))


def prepare(root, manifest):
    registry = {}
    all_proofs = []
    for group in ("F1", "F2"):
        directory = root / group
        directory.mkdir(exist_ok=True)
        path = directory / "parent.bin"
        if path.exists():
            runtime = fresh_load(path)
        else:
            runtime = prepare_parent(group, manifest, directory)
            runtime.save(path)
            all_proofs.extend(runtime.readbacks)
        expected = fingerprint(runtime)
        _, proof = readback(path, expected)
        registry[f"{group}/parent.bin"] = proof
        all_proofs.append(proof)
    return registry, all_proofs


def attempt_worker(root, identity, manifest_path, output_path):
    torch.set_num_threads(1)
    manifest = load_manifest(manifest_path)
    run, ledger = context(root, manifest, manifest_path)
    spec = precontract(manifest)
    state = ledger.read()
    if (
        identity not in state["opened"]
        or identity in state["closed"]
        or identity in state["reconciled"]
    ):
        raise LedgerError("Recovery worker has no durable open attempt")
    parent = root / spec["run"]["attempt_parents"][identity]
    runtime = fresh_load(parent)
    opened = state["opened"][identity]
    assert file_sha(parent) == opened["parent_sha256"]
    assert fingerprint(runtime)["state_sha256"] == opened["pre_state_sha256"]
    counter = StepCounter(ledger, root / "internal/step_claims")
    with step_guard(counter):
        result = perform(
            runtime,
            identity[:2].upper(),
            identity,
            manifest,
            counter,
            root / spec["run"]["attempt_posts"][identity],
        )
    result.update(
        worker_pid=os.getpid(),
        worker_parent_pid=os.getppid(),
        worker_cpu_seconds=time.process_time(),
        worker_wall_seconds=time.monotonic() - STARTED,
    )
    atomic_save(result, output_path)
    return 0


def launch_attempt(root, identity, manifest_path):
    directory = root / "internal"
    directory.mkdir(exist_ok=True)
    output = directory / (identity + ".bin")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--attempt-worker",
            identity,
            "--run-root",
            str(root),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    payload = atomic_load(output)
    assert payload["worker_pid"] != os.getpid() and payload["worker_parent_pid"] == os.getpid()
    return payload


def s12_path(run_id, identity):
    return ROOT / f"reports/taiji_u2_22_3_4_3_s12_{run_id}_{identity}.json"


def execute(root, manifest, manifest_path, registry, preparation_readbacks):
    run, ledger = context(root, manifest, manifest_path)
    if ledger.path.exists():
        existing = ledger.read()
        if len(existing["closed"]) == 8:
            raise LedgerError("Run already complete; optimizer attempts cannot be repeated")
        ledger.reconcile()
        raise LedgerError("Existing run reconciled; consumed attempts are not repeated")
    ledger.start()
    spec = precontract(manifest)
    counter = StepCounter(ledger, root / "internal/step_claims")
    attempts = []
    child_cpu = 0.0
    all_readbacks = list(preparation_readbacks)
    with step_guard(counter):
        for identity in spec["run"]["attempt_order"]:
            cpu_before, wall_before = time.process_time(), time.monotonic()
            group = "F1" if identity.startswith("fault-") else identity[:2].upper()
            relative = spec["run"]["attempt_parents"][identity]
            parent = root / relative
            runtime, proof = readback(parent, registry[relative]["fingerprint"])
            if file_sha(parent) != registry[relative]["sha256"]:
                raise ReadbackError("Frozen parent file identity changed")
            all_readbacks.append(proof)
            before = fingerprint(runtime)
            opened = ledger.open(
                identity,
                parent_checkpoint=relative,
                parent_sha256=file_sha(parent),
                pre_state_sha256=before["state_sha256"],
                pre_parameter_sha256=before["parameter_sha256"],
                pre_moment_sha256=before["moment_sha256"],
            )
            post_relative = spec["run"]["attempt_posts"][identity]
            failure = None
            failed_path = None
            try:
                if identity.endswith("-recovery"):
                    result = launch_attempt(root, identity, manifest_path)
                    child_cpu += result["worker_cpu_seconds"]
                else:
                    result = perform(
                        runtime,
                        group,
                        identity,
                        manifest,
                        counter,
                        None if post_relative is None else root / post_relative,
                    )
                assert_values_equal(before, result["pre"], "attempt.parent")
                if post_relative is None:
                    _, post_proof = readback(parent, before)
                else:
                    _, post_proof = readback(root / post_relative, result["post"])
                    registry[post_relative] = post_proof
            except Exception as error:
                failure = error
                failed_path = post_relative
                runtime, post_proof = readback(parent, before)
                result = failed_attempt(
                    runtime,
                    error,
                    None if identity.endswith("-recovery") else int(identity in counter.called),
                )
                post_relative = None
            all_readbacks.extend(result["readbacks"])
            all_readbacks.append(post_proof)
            close_seq = len(ledger.read()["rows"])
            path = s12_path(run["run_id"], identity)
            payload = {
                "schema": "taiji-u2-s12-v1",
                "run_id": run["run_id"],
                "attempt_id": identity,
                "group": group,
                "dtype": manifest["groups"][group]["dtype"],
                "status": result["status"],
                "ledger_seq_open": opened["seq"],
                "ledger_seq_close": close_seq,
                "parent_checkpoint": str(parent.relative_to(ROOT).as_posix()),
                "parent_sha256": file_sha(parent),
                "post_checkpoint": (
                    None
                    if post_relative is None
                    else str((root / post_relative).relative_to(ROOT).as_posix())
                ),
                "post_sha256": None if post_relative is None else file_sha(root / post_relative),
                "pre_state_sha256": before["state_sha256"],
                "pre_parameter_sha256": before["parameter_sha256"],
                "pre_moment_sha256": before["moment_sha256"],
                "post_state_sha256": result["post"]["state_sha256"],
                "post_parameter_sha256": result["post"]["parameter_sha256"],
                "post_moment_sha256": result["post"]["moment_sha256"],
                "history_reference": result["post"]["history_reference"],
                "revision": result["post"]["revision"],
                "gradient_sha256": result["gradient_sha256"],
                "next_output": result["next_output"],
                "policy": result["policy"],
                "run_conditions": result["run_conditions"],
                "parameter_rows": result["parameter_rows"],
                "losses": result["losses"],
                "optimizer_step_calls": result["optimizer_step_calls"],
                "migration": result["migration"],
                "cpu_seconds": time.process_time()
                - cpu_before
                + result.get("worker_cpu_seconds", 0),
                "wall_seconds": time.monotonic() - wall_before,
                "execution_manifest": str(manifest_path.relative_to(ROOT).as_posix()),
                "manifest_sha256": file_sha(manifest_path),
                "run_root": str(root.relative_to(ROOT).as_posix()),
                "parent_readback": proof,
                "post_readback": post_proof,
                "failed_checkpoint": failed_path,
            }
            write_once(path, payload)
            assert (
                json.loads(path.read_text(encoding="utf-8"))["gradient_sha256"]
                == result["gradient_sha256"]
            )
            closed = ledger.close(
                identity,
                outcome=(
                    "readback_failed" if isinstance(failure, ReadbackError) else result["status"]
                ),
                readback_ok=not isinstance(failure, ReadbackError),
                post_checkpoint=post_relative,
                post_sha256=payload["post_sha256"],
                post_state_sha256=payload["post_state_sha256"],
                post_parameter_sha256=payload["post_parameter_sha256"],
                post_moment_sha256=payload["post_moment_sha256"],
                gradient_sha256=payload["gradient_sha256"],
                cpu_seconds=payload["cpu_seconds"],
                wall_seconds=payload["wall_seconds"],
                s12=str(path.relative_to(ROOT).as_posix()),
                s12_sha256=file_sha(path),
                optimizer_step_calls=result["optimizer_step_calls"],
            )
            assert closed["seq"] == close_seq
            attempts.append(payload)
            if failure is not None:
                raise failure
            if (
                time.process_time() + child_cpu > manifest["formal_budget"]["cpu_seconds"]
                or time.monotonic() - STARTED > manifest["formal_budget"]["wall_seconds"]
            ):
                raise RuntimeError("budget_stop")
    assert sum(item["optimizer_step_calls"] for item in attempts) == 6
    return attempts, registry, all_readbacks, child_cpu


def launch_s12(root, identity, manifest):
    path = s12_path(root.name.removeprefix("taiji_u2_22_3_4_3_"), identity)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/training/verify_taiji_u2_s12_22_3_4_3.py"),
            "--s12",
            str(path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    output = path.with_name(path.stem + "_worker.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--attempt-worker")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--owner-approval-ref")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.run_root.resolve()
    if root.parent != (ROOT / "output").resolve() or not re.fullmatch(
        r"taiji_u2_22_3_4_3_\d{8}T\d{6}Z", root.name
    ):
        parser.error("Run root must use the frozen output directory convention")
    if args.attempt_worker:
        return attempt_worker(root, args.attempt_worker, args.manifest.resolve(), args.output)
    manifest = load_manifest(args.manifest.resolve())
    if args.run and args.owner_approval_ref != manifest["owner_approval_ref"]:
        raise AuthorizationError("Explicit recorded owner reference is required for --run")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        registry, proofs = prepare(root, manifest)
        if args.dry_run:
            report = {
                "schema": "taiji-u2-22.3.4.3-dry-run-v1",
                "run_root": str(root.relative_to(ROOT).as_posix()),
                "manifest_sha256": file_sha(args.manifest.resolve()),
                "checkpoints": registry,
                "readback_calls": len(proofs),
                "optimizer_step_calls": 0,
                "cpu_seconds": time.process_time(),
                "wall_seconds": time.monotonic() - STARTED,
                "run_conditions": run_conditions(fresh_load(root / "F1/parent.bin").model),
            }
            write_once(root / "dry-run.json", report)
            print(json.dumps({"verdict": "dry_run_pass", "parents": 2, "optimizer_step_calls": 0}))
            return 0
        run_path = root / "run.json"
        if not run_path.exists():
            write_once(
                run_path,
                {
                    "run_id": root.name.removeprefix("taiji_u2_22_3_4_3_"),
                    "created_at": datetime.now(UTC).isoformat(),
                    "authorized": True,
                    "owner_approval_ref": args.owner_approval_ref,
                    "manifest_sha256": file_sha(args.manifest.resolve()),
                },
            )
        attempts, registry, proofs, child_cpu = execute(
            root, manifest, args.manifest.resolve(), registry, proofs
        )
        workers = [
            launch_s12(root, identity, manifest)
            for identity in precontract(manifest)["run"]["attempt_order"]
        ]
        for group in ("F1", "F2"):
            left = fresh_load(root / group / "step2-orig.bin")
            right = fresh_load(root / group / "step2-recovery.bin")
            assert_values_equal(fingerprint(left), fingerprint(right), group + ".step2")
            assert_values_equal(distributions(left), distributions(right), group + ".output")
        run, ledger = context(root, manifest, args.manifest.resolve())
        state = ledger.read()
        cpu = time.process_time() + child_cpu + sum(item["cpu_seconds"] for item in workers)
        wall = time.monotonic() - STARTED
        assert (
            cpu <= manifest["formal_budget"]["cpu_seconds"]
            and wall <= manifest["formal_budget"]["wall_seconds"]
        )
        report = {
            "schema": "taiji-u2-22.3.4.3-engineering-updates-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "passed_six_updates_two_faults_only",
            "run_id": run["run_id"],
            "execution_manifest": str(args.manifest.resolve().relative_to(ROOT).as_posix()),
            "manifest_sha256": run["manifest_sha256"],
            "ledger": str(ledger.path.relative_to(ROOT).as_posix()),
            "ledger_sha256": file_sha(ledger.path),
            "attempt_open": len(state["opened"]),
            "attempt_close": len(state["closed"]),
            "optimizer_step_calls": sum(item["optimizer_step_calls"] for item in attempts),
            "canonical_checkpoints": registry,
            "checkpoint_coverage": len(registry),
            "readback_calls": len(proofs),
            "attempts": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"next_output", "policy", "parameter_rows"}
                }
                for item in attempts
            ],
            "worker_reports": [worker["attempt_id"] for worker in workers],
            "cpu_seconds_including_workers": cpu,
            "wall_seconds": wall,
            "training_steps": 0,
            "project_dataset_reads": 0,
            "confirm_reads": 0,
            "whole_22_3_passed": False,
            "run_conditions": workers[0]["run_conditions"],
        }
        target = ROOT / f"reports/taiji_u2_22_3_4_3_summary_{run['run_id']}.json"
        write_once(target, report)
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "attempts": 8,
                    "optimizer_step_calls": 6,
                    "cpu_seconds": cpu,
                    "report": str(target),
                }
            )
        )
        return 0
    except Exception as error:
        if args.run:
            run_id = root.name.removeprefix("taiji_u2_22_3_4_3_")
            target = ROOT / f"reports/taiji_u2_22_3_4_3_summary_{run_id}.json"
            if not target.exists():
                failure = {
                    "schema": "taiji-u2-22.3.4.3-failure-v1",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                    "run_id": run_id,
                    "threads": torch.get_num_threads(),
                    "parent_cpu_seconds": time.process_time(),
                    "wall_seconds": time.monotonic() - STARTED,
                    "resource_accounting": "partial_on_failure; durable ledger retains consumed attempts",
                    "whole_22_3_passed": False,
                }
                try:
                    _, failure_ledger = context(root, manifest, args.manifest.resolve())
                    state = failure_ledger.read()
                    failure.update(
                        attempt_open=len(state["opened"]), attempt_close=len(state["closed"])
                    )
                except (OSError, ValueError, LedgerError) as ledger_error:
                    failure["ledger_error"] = str(ledger_error)
                write_once(target, failure)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LedgerError, ReadbackError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(error.exit_code) from error
