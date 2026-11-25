"""SUB.1.02 (zero training): mixed-task subject-loop integration smoke (Chinese + physics).

Design: plans/reference/DESIGN-SUB-1-02_20261126.md (D-20261126-622).
Runs four sessions alternating Chinese continuous tasks and multimodal-physics tasks through the
same subject record: persistence, evidence/credibility accumulation, uncertainty verbalisation,
audit chain, crash-before-commit recovery, sealed blocking, determinism. Scripted control arms
only; no capability claim.

Usage: python scripts/training/sub_1_02_mixed_loop.py
Output: reports/sub_1_02_mixed_loop.json
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

CARD = "SUB.1.02"
PLAN = ("chinese", "physics", "chinese", "physics")
SEEDS = (5500, 5600, 5501, 5601)
CRASH_INDEX = 1
CAP_CPU = 10.0
OUT = ROOT / "reports" / "sub_1_02_mixed_loop.json"


def _run(root: Path) -> dict:
    loop = SubjectLoop(root, "subject-0")
    predicted: list[float] = []
    outcomes: list[int] = []
    confidences: list[str] = []
    i = 0
    crashes = 0
    crashed: set[int] = set()
    while i < len(PLAN):
        crash = i == CRASH_INDEX and i not in crashed
        try:
            if PLAN[i] == "physics":
                out = loop.run_physics_session(SEEDS[i], crash_before_commit=crash)
            else:
                out = loop.run_session(SEEDS[i], crash_before_commit=crash)
            predicted.extend(out["predicted"])
            outcomes.extend(out["outcomes"])
            confidences.append(out["confidence"])
            i += 1
        except RuntimeError:
            crashed.add(i)
            crashes += 1
            assert loop.session_index == CRASH_INDEX
    summary = loop.summary(predicted, outcomes, confidences)
    sealed = 0
    for split in ("check", "final"):
        try:
            loop.sealed_read(split)
        except PermissionError:
            sealed += 1
    record = loop.store.load("subject-0")
    tasks = set()
    for e in record.experience:
        if e.get("kind") == "session":
            tasks.add(e["payload"].get("task", "chinese"))
    return {
        "summary": summary,
        "crashes": crashes,
        "sealed": sealed,
        "tasks": sorted(tasks),
        "entries": len(loop.chain),
    }


def main() -> int:
    started = time.process_time()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _run(Path(d1))
        b = _run(Path(d2))
    s = a["summary"]
    checks = {
        "sessions_4": s.sessions == 4,
        "both_tasks": a["tasks"] == ["chinese", "physics"],
        "steps_positive": s.steps >= 16,
        "audit_chain_len": a["entries"] == 4,
        "crash_recovered": a["crashes"] == 1,
        "sealed_blocked": a["sealed"] == 2,
        "ledger_posterior_in_range": 0.0 <= s.ledger_posterior <= 1.0,
        "ece_in_range": 0.0 <= s.ece <= 1.0,
        "confidences_valid": all(
            c in {"我很确定", "比较确定", "不太确定", "不确定"} for c in s.confidences
        ),
        "deterministic_chain": s.chain == b["summary"].chain,
        "deterministic_digest": s.record_digest == b["summary"].record_digest,
    }
    exit_ = "mixed_loop_ready" if all(checks.values()) else "mixed_loop_failure"
    payload = {
        "schema": "taiji-sub-1-02-v1",
        "card": CARD,
        "design": "DESIGN-SUB-1-02_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": list(PLAN),
        "seeds": list(SEEDS),
        "steps": s.steps,
        "ledger_posterior": s.ledger_posterior,
        "ece": s.ece,
        "confidences": s.confidences,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
            "policy": "scripted_controls",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "steps": s.steps,
                "ece": s.ece,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "mixed_loop_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
