"""N34.1.02 (DRAFT, not executed until N34 admission is signed): 100-session / 10,000-event durability run.

Pre-registered: plans/reference/DESIGN-N34-1-02_20261126.md (D-20261126-612).
Zero training. Exercises taiji/long_run_harness.py: 100 sessions, >=10,000 events, ten
deterministic crash injections before commit, resume to the last consistent state every time,
audit-chain verification, and sealed check/final read blocking.

Usage (after admission): python scripts/training/n34_1_02_durability.py
Output: reports/n34_1_02_durability.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.long_run_harness import CrashInjected, LongRunHarness, SealViolation  # noqa: E402

CARD = "N34.1.02"
SESSIONS = 100
EVENTS_PER_SESSION = 100
CRASH_POINTS = (9, 19, 29, 39, 49, 59, 69, 79, 89, 99)
CAP_CPU = 20.0
OUT = ROOT / "reports" / "n34_1_02_durability.json"


def session_body(record, i: int) -> None:
    record.add_pending({"id": f"task-{i}", "kind": "unfinished"})
    for k in range(EVENTS_PER_SESSION):
        record.append_experience("step", {"session": i, "k": k})
    if i % 5 == 0 and record.pending:
        record.resolve_pending(record.pending[0]["id"], "done")
    if i % 10 == 0:
        record.append_experience("recall", {"session": i})


def main() -> int:
    started = time.process_time()
    with tempfile.TemporaryDirectory() as d:
        h = LongRunHarness(d, "n34-subject-0")
        crashes = 0
        digest_before = None
        i = 0
        while i < SESSIONS:
            crash = i in CRASH_POINTS
            if crash:
                digest_before = h.resume()[0].digest() if h.session_index else "0" * 64
            try:
                h.run_session(session_body, crash_before_commit=crash)
                i += 1
            except CrashInjected:
                crashes += 1
                record, idx = h.resume()
                after = record.digest() if idx else "0" * 64
                if after != digest_before:
                    raise AssertionError(
                        f"resume digest mismatch after crash at session {i}"
                    ) from None
                # retry the same session (i unchanged)
        record, nxt = h.resume()
        chain_ok = h.verify_chain()
        sealed_attempts = 0
        for split in ("check", "final"):
            try:
                h.read_split(split)
            except SealViolation:
                sealed_attempts += 1
        checks = {
            "sessions_completed": nxt == SESSIONS,
            "events_at_least_10k": len(record.experience) >= 10_000,
            "crashes_all_recovered": crashes == len(CRASH_POINTS),
            "audit_chain_ok": chain_ok,
            "sealed_attempts_blocked": sealed_attempts == 2 and h.guard.attempts_count() == 2,
            "pending_consistent": len(record.pending) == SESSIONS - SESSIONS // 5,
        }
    exit_ = "durability_ready" if all(checks.values()) else "durability_failure"
    payload = {
        "schema": "taiji-n34-1-02-v1",
        "card": CARD,
        "design": "DESIGN-N34-1-02_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "sessions": SESSIONS,
        "events": len(record.experience),
        "crash_points": list(CRASH_POINTS),
        "crashes_recovered": crashes,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False
        )
    )
    return 0 if exit_ == "durability_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
