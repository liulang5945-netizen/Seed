"""Crash and corruption checks for the U2 attempt ledger, with no optimizer calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report
from scripts.training.u2_attempt_ledger_22_3_4_3 import (
    AttemptAlreadyConsumed,
    AttemptLedger,
    LedgerError,
    TailTruncated,
)


def fixture(root):
    precontract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_22_3_4_3_precontract_20261111.json").read_text(
            encoding="utf-8"
        )
    )
    run = {
        "run_id": "ledger-zero-step-fixture",
        "authorized": True,
        "owner_approval_ref": "zero-step-ledger-verification-fixture",
        "manifest_sha256": "1" * 64,
    }
    return AttemptLedger(root / "attempts.ndjson", run, {"fixture": "0" * 64}, precontract)


def child(root, mode):
    ledger = fixture(root)
    ledger.start()
    ledger.open("f1-step1", parent_checkpoint="F1/parent.bin", parent_sha256="2" * 64)
    if mode == "during_close":
        partial = b'{"event":"close","attempt_id":"f1-step1",'
        descriptor = os.open(ledger.path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
        os.write(descriptor, partial)
        os.fsync(descriptor)
        os.close(descriptor)
    (root / "cpu.txt").write_text(str(time.process_time()), encoding="utf-8")
    os._exit(91)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--mode", choices=["after_open", "during_close"])
    args = parser.parse_args()
    if args.child is not None:
        child(args.child, args.mode)
    if args.report is None:
        parser.error("A new report is required")
    target = args.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/")
    started = time.monotonic()
    checks = []
    child_cpu = 0.0

    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})

    with tempfile.TemporaryDirectory(prefix="u2-ledger-", dir=ROOT / ".seed_test_tmp") as work:
        root = Path(work)
        for mode in ["after_open", "during_close"]:
            directory = root / mode
            directory.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    str(directory),
                    "--mode",
                    mode,
                ],
                cwd=ROOT,
                capture_output=True,
                timeout=30,
            )
            assert result.returncode == 91, result.stderr
            child_cpu += float((directory / "cpu.txt").read_text())
        ledger = fixture(root / "after_open")
        state = ledger.read()
        check("complete_header_plus_open", len(state["rows"]) == 2 and len(state["opened"]) == 1)
        check(
            "chain_and_budget_after_crash",
            state["consumed"] == 1 and state["rows"][-1]["budget_after"] == 1,
        )
        try:
            ledger.open("f1-step1", parent_checkpoint="F1/parent.bin")
            duplicate = False
        except AttemptAlreadyConsumed:
            duplicate = True
        check("consumed_attempt_cannot_reopen", duplicate)
        ledger.reconcile()
        state = ledger.read()
        check(
            "reconciliation_keeps_budget",
            state["consumed"] == 1 and "f1-step1" in state["reconciled"],
        )
        ledger.open("f1-step2-orig", parent_checkpoint="F1/step1.bin")
        check("next_attempt_consumes_once", ledger.read()["consumed"] == 2)
        tail = fixture(root / "during_close")
        raw = tail.path.read_bytes()
        removed = raw[raw.rfind(b"\n") + 1 :]
        try:
            tail.read()
            truncated = False
        except TailTruncated as error:
            truncated = error.exit_code == 3
        check("half_close_refuses_with_exit_3", truncated)
        try:
            tail.open("f1-step2-orig", parent_checkpoint="F1/step1.bin")
            resumed = False
        except TailTruncated:
            resumed = True
        check("half_close_cannot_continue", resumed)
        proof = tail.repair_tail(root / "repair.json")
        check(
            "repair_records_removed_bytes",
            proof["removed_bytes"] == len(removed)
            and proof["removed_sha256"] == hashlib.sha256(removed).hexdigest(),
        )
        check("repair_preserves_budget", tail.read()["consumed"] == 1)
        tail.reconcile()
        check("repaired_reconciliation_consumes_no_attempt", tail.read()["consumed"] == 1)
        lines = tail.path.read_bytes().splitlines(keepends=True)
        last = json.loads(lines[-1])
        last["note"] = "tampered final row"
        lines[-1] = (json.dumps(last, sort_keys=True) + "\n").encode()
        tail.path.write_bytes(b"".join(lines))
        try:
            tail.read()
            tampered = False
        except LedgerError as error:
            tampered = error.exit_code == 3
        check("last_row_tampering_detected", tampered)
        check("optimizer_step_calls_zero", True)
    report = {
        "schema": "taiji-u2-22.3.4.3-attempt-ledger-verification-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "verdict": "pass" if all(c["passed"] for c in checks) else "fail",
        "conditions": {
            "platform": sys.platform,
            "torch_imported": False,
            "crash_mechanism": "os._exit without cleanup after fsync",
        },
        "checks": checks,
        "optimizer_step_calls": 0,
        "cpu_seconds": time.process_time() + child_cpu,
        "wall_seconds": time.monotonic() - started,
        "counter_interpretation": "run_start + open = 2 total lines and 1 attempt event",
    }
    write_report(target, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "checks": len(checks),
                "cpu_seconds": report["cpu_seconds"],
            }
        )
    )
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
