"""SUB.1.01 (zero training): integration smoke of the subject loop.

Design: plans/reference/DESIGN-SUB-1-01_20261126.md (D-20261126-619).
Runs five sessions through taiji/subject_loop.py with the scripted control policy: SessionStore
persistence, Chinese continuous task, credibility ledger rebuild, uncertainty verbalisation and
ECE, sha256 audit chain, crash-before-commit recovery, sealed check/final blocking, and
determinism (two identical independent runs).

Usage: python scripts/training/sub_1_01_loop_smoke.py
Output: reports/sub_1_01_loop.json
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

from taiji.subject_loop import SubjectLoop  # noqa: E402

CARD = "SUB.1.01"
SESSIONS = 5
CRASH_SESSION = 2
SEEDS = [5200 + i for i in range(SESSIONS)]
CAP_CPU = 10.0
OUT = ROOT / "reports" / "sub_1_01_loop.json"


def _run(root: Path) -> dict:
    loop = SubjectLoop(root, "subject-0")
    predicted: list[float] = []
    outcomes: list[int] = []
    confidences: list[str] = []
    i = 0
    crashes = 0
    crashed: set[int] = set()
    while i < SESSIONS:
        crash = i == CRASH_SESSION and i not in crashed
        try:
            out = loop.run_session(SEEDS[i], crash_before_commit=crash)
            predicted.extend(out["predicted"])
            outcomes.extend(out["outcomes"])
            confidences.append(out["confidence"])
            i += 1
        except RuntimeError:
            crashed.add(i)
            crashes += 1
            assert loop.session_index == CRASH_SESSION
    summary = loop.summary(predicted, outcomes, confidences)
    sealed = 0
    for split in ("check", "final"):
        try:
            loop.sealed_read(split)
        except PermissionError:
            sealed += 1
    return {
        "sessions": summary.sessions,
        "steps": summary.steps,
        "chain": summary.chain,
        "posterior": summary.ledger_posterior,
        "ece": summary.ece,
        "confidences": confidences,
        "crashes": crashes,
        "sealed_blocked": sealed,
        "digest": summary.record_digest,
    }


def main() -> int:
    torch_free_start = time.process_time()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _run(Path(d1))
        b = _run(Path(d2))
    checks = {
        "sessions_5": a["sessions"] == SESSIONS,
        "steps_positive": a["steps"] >= SESSIONS * 3,
        "audit_chain_len": len(a["chain"]) == SESSIONS,
        "crash_recovered": a["crashes"] == 1,
        "sealed_blocked": a["sealed_blocked"] == 2,
        "ledger_posterior_high": a["posterior"] >= 0.5,
        "ece_in_range": 0.0 <= a["ece"] <= 1.0,
        "confidence_levels": all(
            c in {"我很确定", "比较确定", "不太确定", "不确定"} for c in a["confidences"]
        ),
        "deterministic_chain": a["chain"] == b["chain"],
        "deterministic_digest": a["digest"] == b["digest"],
    }
    exit_ = "loop_ready" if all(checks.values()) else "loop_failure"
    payload = {
        "schema": "taiji-sub-1-01-v1",
        "card": CARD,
        "design": "DESIGN-SUB-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "sessions": SESSIONS,
        "seeds": SEEDS,
        "crash_session": CRASH_SESSION,
        "steps": a["steps"],
        "ledger_posterior": a["posterior"],
        "ece": a["ece"],
        "confidences": a["confidences"],
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - torch_free_start, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
            "policy": "scripted_oracle_control",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "steps": a["steps"],
                "ece": a["ece"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "loop_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
