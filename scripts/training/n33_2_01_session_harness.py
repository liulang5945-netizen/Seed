"""N33.2.01 (zero training): cross-session persistence harness replay.

Pre-registered: plans/reference/DESIGN-N33-2-01_20261125.md (D-20261125-577).
Checks round-trip, cross-process load, tamper detection and subject isolation on
taiji/session_store.py. No policy is trained; no confirmation seed is read.

Usage: python scripts/training/n33_2_01_session_harness.py
Output: reports/n33_2_01_harness.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.session_store import SessionRecord, SessionStore  # noqa: E402

CARD = "N33.2.01"
OUT = ROOT / "reports" / "n33_2_01_harness.json"


def make_record() -> SessionRecord:
    r = SessionRecord(subject_id="taiji-subject-0")
    r.capsule = {"h": torch.tensor([1.5, -2.0, 0.25]), "w": torch.zeros(2, 3)}
    r.add_pending({"id": "task-1", "kind": "unfinished", "state": "waiting"})
    r.append_experience("task_started", {"task": "task-1"}, source="owner")
    return r


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        store = SessionStore(d)
        rec = make_record()
        digest = rec.digest()
        store.save(rec)
        back = store.load("taiji-subject-0")
        roundtrip = back.digest() == digest and torch.equal(back.capsule["h"], rec.capsule["h"])

        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{ROOT}")
            from taiji.session_store import SessionStore
            print(SessionStore(r"{d}").load("taiji-subject-0").digest())
            """)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        cross_process = proc.returncode == 0 and proc.stdout.strip() == digest

        path = Path(d) / "taiji-subject-0.session"
        tampered = torch.load(path, weights_only=False)
        tampered["metadata"]["clock"] = 999
        torch.save(tampered, path)
        try:
            store.load("taiji-subject-0")
            tamper_detected = False
        except ValueError:
            tamper_detected = True

        store2 = SessionStore(str(Path(d) / "iso"))
        store2.save(rec)
        other = SessionRecord(subject_id="other")
        other.capsule = {"h": torch.tensor([9.0])}
        store2.save(other)
        a = store2.load("taiji-subject-0")
        b = store2.load("other")
        isolated = torch.equal(a.capsule["h"], rec.capsule["h"]) and torch.equal(
            b.capsule["h"], torch.tensor([9.0])
        )

    checks = {
        "roundtrip_bit_exact": bool(roundtrip),
        "cross_process_digest": bool(cross_process),
        "tamper_detected": bool(tamper_detected),
        "subject_isolation": bool(isolated),
    }
    exit_ = "harness_ready" if all(checks.values()) else "harness_failure"
    payload = {
        "schema": "taiji-n33-2-01-v1",
        "card": CARD,
        "design": "DESIGN-N33-2-01_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "store_schema": "taiji-session-store-v1",
        "checks": checks,
        "digest_prefix": digest[:16],
        "tamper_probe": "clock field modified in stored session",
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": 10.0},
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
    return 0 if exit_ == "harness_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
