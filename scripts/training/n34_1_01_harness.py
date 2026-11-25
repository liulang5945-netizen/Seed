"""N34.1.01 (zero training): long-run durability harness card.

Pre-registered draft: plans/reference/DESIGN-N34-1-01_20261126.md (D-20261126-610/612).
Executed under the owner's standing delegation ("reasonable steps approved") for zero-training
N34 preparation; N34 training/confirmation cards remain gated on the signed admission terms.

Checks: 50 sessions, crash injection at session 24 then resume, audit chain verification,
sealed check/final read attempts blocked, deterministic chain on a second identical run.
Output: reports/n34_1_01_harness.json
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

CARD = "N34.1.01"
SESSIONS = 50
CRASH_SESSION = 24
CAP_CPU = 10.0
OUT = ROOT / "reports" / "n34_1_01_harness.json"


def _body(record, i: int) -> None:
    record.add_pending({"id": f"task-{i}", "kind": "unfinished"})
    record.append_experience("session", {"i": i})


def _run(root: Path) -> dict:
    h = LongRunHarness(root, "n34-subject-0")
    i = 0
    crashes = 0
    crashed: set[int] = set()
    while i < SESSIONS:
        crash = i == CRASH_SESSION and i not in crashed
        try:
            h.run_session(_body, crash_before_commit=crash)
            i += 1
        except CrashInjected:
            crashed.add(i)
            crashes += 1
            record, idx = h.resume()
            if idx != CRASH_SESSION:
                raise AssertionError(f"resume index {idx} != {CRASH_SESSION}") from None
            if len(record.pending) != CRASH_SESSION:
                raise AssertionError("resume pending count mismatch") from None
    record, nxt = h.resume()
    sealed = 0
    for split in ("check", "final"):
        try:
            h.read_split(split)
        except SealViolation:
            sealed += 1
    return {
        "sessions": nxt,
        "pending": len(record.pending),
        "experience": len(record.experience),
        "crashes_recovered": crashes,
        "chain": list(h.chain),
        "chain_ok": h.verify_chain(),
        "sealed_blocked": sealed,
        "guard_attempts": h.guard.attempts_count(),
        "digest": record.digest(),
    }


def main() -> int:
    started = time.process_time()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _run(Path(d1))
        b = _run(Path(d2))
    checks = {
        "sessions_50": a["sessions"] == SESSIONS,
        "pending_50": a["pending"] == SESSIONS,
        "experience_50": a["experience"] == SESSIONS,
        "crash_recovered": a["crashes_recovered"] == 1,
        "audit_chain_ok": a["chain_ok"],
        "sealed_blocked": a["sealed_blocked"] == 2 and a["guard_attempts"] == 2,
        "deterministic_chain": a["chain"] == b["chain"],
        "deterministic_digest": a["digest"] == b["digest"],
    }
    exit_ = "harness_ready" if all(checks.values()) else "harness_failure"
    payload = {
        "schema": "taiji-n34-1-01-v1",
        "card": CARD,
        "design": "DESIGN-N34-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "sessions": SESSIONS,
        "crash_session": CRASH_SESSION,
        "checks": checks,
        "chain_prefix": [c[:12] for c in a["chain"][:3]],
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
        "authorization": "zero-training preparation under owner standing delegation; training cards remain gated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False
        )
    )
    return 0 if exit_ == "harness_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
