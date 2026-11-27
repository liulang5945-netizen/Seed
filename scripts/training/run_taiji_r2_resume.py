"""Single R2 continuation entry point; historical launchers are provenance only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training import taiji_r2_dialogue_continuation_extension as ledger
from scripts.training.run_taiji_r2_checkpoint_resume_adapter import validate_prepared
from scripts.training.taiji_r2_resume_ledger_anchor import (
    current_fast_charges,
    current_fast_reservations,
    verified_snapshot,
)

MEMORY_FLOOR_GIB = 7.0
SOURCES = (
    "scripts/training/run_taiji_r2_resume.py",
    "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py",
    "scripts/training/taiji_r2_dialogue_continuation_extension.py",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
)


REVISION_PATH = ledger.REPORTS_DIR / f"{ledger.EXTENSION_PREFIX}_budget_amendment_v1_4.json"
SOURCE_REVISION_PATH = ledger.REPORTS_DIR / f"{ledger.EXTENSION_PREFIX}_source_revision_v1_3.json"
SOURCE_REVISION_SHA256 = "2f6938afb910a7bca2c9d5bd5c3d4f79e5a67aa2cd19da08c0a9577589640bf7"


def read_validated_amendment():
    for path in (REVISION_PATH, SOURCE_REVISION_PATH, ledger.CONTRACT_PATH):
        if path.is_symlink() or not path.is_file():
            raise runner.Refuse("immutable budget provenance missing or linked")
    record = ledger._read_json(REVISION_PATH)
    expected = {
        "schema": "taiji-r2-continuation-budget-amendment-v1.4",
        "extension_id": ledger.EXTENSION_PREFIX,
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "contract_file_sha256": runner.sha256_file(ledger.CONTRACT_PATH),
        "contract_object_sha256": "7685c69a1f1ea2cd80990f04e06241757d9f1360d661c896545693805e01ad5a",
        "previous_budget": {"per_arm": 1500000.0, "aggregate": 4300000.0},
        "approved_budget": {"per_arm": 1550000.0, "aggregate": 4400000.0},
    }
    payload = {k: v for k, v in record.items() if k != "amendment_sha256"}
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        any(record.get(k) != v for k, v in expected.items())
        or runner.sha256_file(SOURCE_REVISION_PATH) != SOURCE_REVISION_SHA256
        or record.get("amendment_sha256") != digest
    ):
        raise runner.Refuse("budget amendment lineage, cap or digest differs")
    return record


def activate_budget():
    record = read_validated_amendment()
    previous = (
        ledger.PER_ARM_CPU_LIMIT,
        ledger.AGGREGATE_CPU_LIMIT,
        ledger.effective_budget_limits,
    )
    limits = (record["approved_budget"]["per_arm"], record["approved_budget"]["aggregate"])
    ledger.PER_ARM_CPU_LIMIT, ledger.AGGREGATE_CPU_LIMIT = limits
    ledger.effective_budget_limits = lambda: limits
    try:
        with ledger._aggregate_lock(ledger.LOCK_PATH):
            contract, _, _, _ = ledger._read_summary_fast()
            if contract.get("contract_sha256") != record["contract_object_sha256"]:
                raise runner.Refuse("budget contract differs")
    except Exception:
        ledger.PER_ARM_CPU_LIMIT, ledger.AGGREGATE_CPU_LIMIT, ledger.effective_budget_limits = (
            previous
        )
        raise
    return record


def reconcile_start_progress(arm, progress_path):
    """Resolve recorded settlements by sequence; scan legacy records at most once."""
    events = runner.read_jsonl(progress_path)
    active = runner.latest_reservations().get(arm)
    recorded = {
        row.get("reservation_id"): row
        for row in events
        if isinstance(row.get("ledger_sequence"), int) and row.get("event") != "work_unit_reserved"
    }
    legacy = None
    for intent in [row for row in events if row.get("event") == "continuation_start_intent"]:
        token = intent.get("reservation_id")
        if not isinstance(token, str) or not token:
            raise runner.Refuse("start intent lacks reservation token")
        if active is not None and active.get("reservation_id") == token:
            continue
        row = recorded.get(token)
        if row is not None:
            event = runner.read_json(
                ledger._event_path(ledger.AGGREGATE_LEDGER_PREFIX, row["ledger_sequence"])
            )
            if (
                event.get("event_sha256") != row.get("ledger_event_sha256")
                or event.get("event_sha256") != ledger._digest(event, omit="event_sha256")
                or event.get("reservation_id") != token
                or event.get("arm") != arm
                or event.get("event") == "work_unit_reserved"
            ):
                raise runner.Refuse("recorded start settlement differs from ledger")
            continue
        if legacy is None:
            legacy = runner._verified_ledger_events(arm)
        matches = [
            row
            for row in legacy
            if row.get("reservation_id") == token and row.get("event") != "work_unit_reserved"
        ]
        if len(matches) != 1:
            raise runner.Refuse("start intent requires exactly one settlement")
        runner._append_progress_from_ledger(progress_path, events, matches[0])
    return events


def resource_snapshot():
    available = runner.psutil.virtual_memory().available / 1024**3
    if available < MEMORY_FLOOR_GIB:
        raise runner.Refuse(f"available memory below {MEMORY_FLOOR_GIB:g} GiB: {available:.2f}")
    return {"available_memory_gib": round(available, 3), "memory_floor_gib": MEMORY_FLOOR_GIB}


class ResumeBudget(runner.ExtensionBudget):
    """One memory policy; retain atomic CPU reservation and stop reserves."""

    def guard(self, reserve=None, *, terminal=False):
        if self.active_reservation_id is not None:
            self.settle_reservation("work_unit_settled")
        reserve = self.episode_reserve if reserve is None else float(reserve)
        upper = reserve + runner.LEDGER_EVENT_RESERVE_SECONDS
        if not terminal:
            upper += runner.TERMINAL_STOP_RESERVE_SECONDS
        charge = self.used()
        try:
            if charge + upper >= self.limit:
                raise runner.Refuse("cumulative CPU ceiling exhausted")
            resource_snapshot()
            token = runner.ledger_guard(self.arm, charge, reserve_cpu_seconds=upper)
        except (runner.Refuse, ledger.ExtensionLedgerRefuse) as exc:
            self.pending_stop_details = {"stop_reason": "work_unit_refused", "refusal": str(exc)}
            if not terminal and charge + runner.LEDGER_EVENT_RESERVE_SECONDS < self.limit:
                try:
                    stop_token = runner.ledger_guard(
                        self.arm, charge, reserve_cpu_seconds=runner.LEDGER_EVENT_RESERVE_SECONDS
                    )
                except (runner.Refuse, ledger.ExtensionLedgerRefuse):
                    stop_token = None
                if isinstance(stop_token, str) and stop_token:
                    self.active_reservation_id = stop_token
                    self.active_reservation_upper = charge + runner.LEDGER_EVENT_RESERVE_SECONDS
            raise runner.Refuse(str(exc)) from exc
        if not isinstance(token, str) or not token:
            raise runner.Refuse("ledger did not return reservation token")
        self.active_reservation_id = token
        self.active_reservation_upper = charge + upper


def budget_status(arm, _contexts, **_cursor):
    """Periodic metadata uses current hard limits, without rereading three histories."""
    charges = current_fast_charges()
    return {
        "feasible": True,
        "projection_scope": "selected_arm",
        "policy": "actual_cpu_guard; no_remaining_run_forecast_veto",
        "arm": arm,
        "arm_headroom_cpu_seconds": runner.PER_ARM_CPU_LIMIT - charges[arm],
        "aggregate_headroom_cpu_seconds": runner.AGGREGATE_CPU_LIMIT - sum(charges.values()),
    }


def install_runtime(prepared):
    append_snapshot = runner._append_cpu_snapshot

    def report_snapshot(arm, event, *args, **kwargs):
        record = append_snapshot(arm, event, *args, **kwargs)
        if event in {"continuation_resumed", "durable_training_commit", "continuation_completed"}:
            print(f"[resume] {event} ledger_sequence={record.get('sequence')}", flush=True)
        return record

    runner._append_cpu_snapshot = report_snapshot
    runner._validate_prepared = lambda **_kwargs: prepared
    runner.latest_charges = current_fast_charges
    runner.latest_reservations = current_fast_reservations
    runner._reconcile_start_progress = reconcile_start_progress
    runner._throughput_projection = budget_status
    runner.resource_snapshot = resource_snapshot
    runner.ExtensionBudget = ResumeBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["warm_start"], default="warm_start")
    parser.add_argument(
        "--check-only", action="store_true", help="verify checkpoint and ledger without training"
    )
    args = parser.parse_args()
    amendment = activate_budget()
    runner.PER_ARM_CPU_LIMIT, runner.AGGREGATE_CPU_LIMIT = ledger.effective_budget_limits()
    runner.torch.set_num_threads(1)
    print("[resume] verifying ledger and checkpoint storage", flush=True)
    snapshot = verified_snapshot()
    prepared = validate_prepared(
        allow_terminal_artifacts=True, ledger_snapshot=snapshot["effective_charges"]
    )
    resource_snapshot()
    if args.check_only:
        print("[resume] checkpoint ancestry, storage, source and ledger checks passed", flush=True)
        return
    manifest = {
        "schema": "taiji-r2-resume-v1",
        "arm": args.arm,
        "approved_budget": amendment["approved_budget"],
        "budget_amendment_sha256": amendment["amendment_sha256"],
        "source_revision_sha256": runner.sha256_file(SOURCE_REVISION_PATH),
        "extension_attempt_sha256": runner.sha256_file(runner.paths(args.arm)["extension_attempt"]),
        "latest_commit": next(
            row["receipt"]
            for row in reversed(runner.read_jsonl(runner.paths(args.arm)["extension_progress"]))
            if row.get("event") == "durable_training_commit"
        ),
        "ledger_anchor": snapshot,
        "sources": {str(ROOT / name): runner.sha256_file(ROOT / name) for name in SOURCES},
        "resource_policy": {"memory_floor_gib": MEMORY_FLOOR_GIB, "scope": "warm_start_only"},
        "utc": runner.utc_now(),
    }
    runner._write_json_once(
        runner.REPORTS_DIR / f"taiji_r2_resume_{args.arm}_{uuid.uuid4().hex}.json", manifest
    )
    install_runtime(prepared)
    print("[resume] reconciling start and restoring latest durable checkpoint", flush=True)
    result = runner.execute_arm(args.arm, resume=True)
    print(json.dumps({"status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
