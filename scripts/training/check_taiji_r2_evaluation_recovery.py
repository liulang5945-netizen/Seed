"""Read-only evidence audit and model-free bootstrap probe for CC-1 recovery.

This entry never runs evaluation, selects prompts, restores checkpoints, or
modifies the failed attempt. Its disposable IPC inputs are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_failed_attempt(prefix: Path) -> dict[str, Any]:
    """Read small metadata only. Zero outputs alone never proves non-exposure."""
    paths = {
        key: prefix.with_name(f"{prefix.name}_{key}.json")
        for key in ("attempt", "failure", "result", "face", "generations")
    }
    ledger = prefix.with_name(f"{prefix.name}_ledger.jsonl")
    attempt = json.loads(paths["attempt"].read_text(encoding="utf-8"))
    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    identity = attempt["attempt_id"]
    if any(item.get("attempt_id") != identity for item in (failure, result)):
        raise ValueError("failed attempt identities differ")
    if any(item.get("status") != "failed" for item in (failure, result)):
        raise ValueError("only an already failed attempt may be audited")
    exposed_events = {"prompt_selection_started", "prompt_face_persisted", "query_started"}
    event_names = [row.get("event") for row in events]
    actual_events = exposed_events.intersection(event_names)
    counts = [failure.get("query_attempts"), failure.get("persisted_generation_outputs")]
    stopped_before_selection = (
        counts == [0, 0]
        and not actual_events
        and not paths["face"].exists()
        and not paths["generations"].exists()
        and failure.get("fresh_process_restores") == {}
    )
    tails = [row for row in events if row.get("event") == "worker_startup_tail_audited"]
    tail = tails[-1] if tails else None
    boot_delta = None
    if tail:
        before, after = tail.get("boot_identity_before"), tail.get("boot_identity_at_audit")
        if isinstance(before, (float, int)) and isinstance(after, (float, int)):
            boot_delta = abs(after - before)
    mismatches = []
    for name, expected in attempt.get("source_hashes", {}).items():
        path = Path(name)
        # Do not read checkpoint/data bytes. Only bound source files are audited.
        if path.suffix not in {".py", ".md"} or not path.is_relative_to(ROOT):
            continue
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(str(path.relative_to(ROOT)))
    return {
        "attempt_id": identity,
        "raw_sha256": {key: sha256(paths[key]) for key in ("attempt", "failure", "result")},
        "ledger_sha256": sha256(ledger),
        "events": event_names,
        "query_attempts": failure.get("query_attempts"),
        "persisted_generation_outputs": failure.get("persisted_generation_outputs"),
        "fresh_process_restores": failure.get("fresh_process_restores"),
        "stopped_before_selection_in_this_attempt": stopped_before_selection,
        "exposure_scope": "this instrumented attempt only; not a machine-wide access audit",
        "source_mismatches_after_fix": mismatches,
        "published_cpu_seconds": failure.get("cpu_seconds_total_at_failure"),
        "published_wall_seconds": failure.get("wall_seconds_at_failure"),
        "latest_startup_tail_unknown": bool(tail and tail.get("unknown_tail")),
        "latest_startup_tail_upper": tail.get("charged_child_cpu_upper") if tail else None,
        "boot_identity_difference_seconds": boot_delta,
        "no_retry_authorized": True,
    }


def probe_bootstrap(*, recovery_entry: bool = False) -> dict[str, Any]:
    """Exercise the real Windows Job/IPC entry without sending any model task."""
    if os.name != "nt":
        raise ValueError("real bootstrap probe requires Windows Job Objects")
    # Import exactly the real entry's dependencies; never patch frozen globals.
    from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery
    from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter

    scratch = ROOT / ".seed_test_tmp" / "r2-evaluation-bootstrap"
    scratch.mkdir(parents=True, exist_ok=True)
    cpu_start, wall_start = time.process_time(), time.monotonic()
    worker = None
    with tempfile.TemporaryDirectory(prefix="ipc-", dir=scratch) as directory:
        temporary = Path(directory)
        attempt, ledger = temporary / "synthetic-attempt.json", temporary / "synthetic-ledger.jsonl"
        attempt.write_text(
            json.dumps({"attempt_id": f"bootstrap-probe-{uuid.uuid4().hex}"}), encoding="utf-8"
        )
        ledger.write_text("", encoding="utf-8")
        try:
            command = adapter.extension_worker_command(attempt, ledger)
            if recovery_entry:
                command[command.index("-m") + 1] = (
                    "scripts.training.evaluate_taiji_r2_engineering_recovery_v1"
                )
            worker = recovery.SupervisedWorker(
                attempt,
                ledger,
                command=command,
                budget=recovery.EvaluationBudget(cpu_limit_seconds=240, wall_limit_seconds=300),
                startup_cpu_limit_seconds=120,
                startup_wall_limit_seconds=120,
            )
            # Construction authenticates ready against the actual Job and attempt.
            ready_cpu = worker.sample_tree_cpu_seconds()
            stopped = worker.terminate_tree()
            child_cpu = worker.sample_tree_cpu_seconds()
            if not stopped or not worker.ipc_write_thread_stopped:
                raise RuntimeError("bootstrap probe did not confirm complete worker-tree exit")
            if ledger.read_text(encoding="utf-8"):
                raise RuntimeError("bootstrap-only probe unexpectedly wrote a task ledger")
            return {
                "status": "passed",
                "worker_module": command[command.index("-m") + 1],
                "authenticated_ready": True,
                "worker_tree_stopped": stopped,
                "exit_code": worker.process.returncode,
                "ready_child_cpu_seconds": ready_cpu,
                "child_cpu_seconds": child_cpu,
                "parent_cpu_seconds_after_import": time.process_time() - cpu_start,
                "wall_seconds_after_import": time.monotonic() - wall_start,
                "model_tasks_sent": 0,
                "checkpoint_restores": 0,
                "prompt_queries": 0,
                "training_updates": 0,
                "termination": "intentional Job termination after ready, not four-restore shutdown",
            }
        finally:
            if worker is not None and not worker._closed:
                worker.terminate_tree()


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bootstrap", action="store_true")
    parser.add_argument("--recovery-entry", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != (ROOT / "reports").resolve() or output.exists():
        parser.error("output must be a new file directly under reports/")
    prefix = ROOT / "reports" / PREFIX
    report: dict[str, Any] = {
        "schema": "taiji-r2-evaluation-recovery-preparation-v1",
        "status": "prepared",
        "can_promote": False,
        "formal_evaluation_authorized": False,
        "failed_attempt": audit_failed_attempt(prefix),
        "source_sha256": sha256(Path(__file__)),
    }
    before = dict(report["failed_attempt"]["raw_sha256"])
    try:
        if args.probe_bootstrap:
            report["bootstrap"] = probe_bootstrap(recovery_entry=args.recovery_entry)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    after = audit_failed_attempt(prefix)
    if (
        after["raw_sha256"] != before
        or after["ledger_sha256"] != report["failed_attempt"]["ledger_sha256"]
    ):
        report.update(status="failed", error="original evidence changed during preparation")
    report["preparation_parent_cpu_seconds"] = time.process_time() - started_cpu
    report["preparation_wall_seconds"] = time.monotonic() - started_wall
    if not all(
        math.isfinite(report[key])
        for key in ("preparation_parent_cpu_seconds", "preparation_wall_seconds")
    ):
        raise ValueError("preparation cost is not finite")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "prepared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
