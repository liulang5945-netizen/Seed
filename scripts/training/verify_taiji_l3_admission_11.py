"""11 admission gate: read the existing reports and decide whether L3 scope may open.

This step measures nothing. Every precondition names the file and field it read, so
a reader can re-run the check instead of trusting the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
L2_REPORT = "reports/taiji_native_core_r5c_evaluation_20261031.json"
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
PRICING_REPORT = "reports/taiji_core_r028_life_fragment_pricing_20261101.json"
CANDIDATE_REPORTS = (
    "reports/taiji_core_goal_world_10_1_20261101.json",
    "reports/taiji_core_goal_world_10_1b_20261101.json",
    "reports/taiji_core_goal_binding_10_2_20261101.json",
    "reports/taiji_core_goal_planner_10_3_20261101.json",
    "reports/taiji_core_goal_planner_10_3b_20261101.json",
    "reports/taiji_core_subject_selection_10_4_20261101.json",
    "reports/taiji_core_goal_conditioned_10_5_20261101.json",
    "reports/taiji_core_action_verdict_10_6_20261101.json",
    "reports/taiji_core_goal_training_10_7_20261101.json",
)
INSTRUMENT_REPORT = "reports/taiji_core_instrument_11_7_v2_20261101.json"
INSTRUMENT_SCOPE = ("static_declared_path", "runtime_hooked_open_socket_subprocess")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11 admission report is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    l2 = load(L2_REPORT)
    card = load(CLOSEOUT_CARD)
    pricing = load(PRICING_REPORT)

    l2_status = l2.get("status")
    p1 = {
        "name": "P1_l2_holds",
        "passed": isinstance(l2_status, str) and l2_status.startswith("PASS"),
        "evidence": f"{L2_REPORT}:status={l2_status}",
    }

    model_candidates = []
    for relative in CANDIDATE_REPORTS:
        report = load(relative)
        model_candidates.append(
            {
                "report": relative,
                "status": report.get("status"),
                "model_involved": report.get("model_involved"),
            }
        )
    passing_models = [
        row
        for row in model_candidates
        if row["model_involved"] is True and str(row["status"]).startswith("passed")
    ]
    p2 = {
        "name": "P2_one_capability_by_the_model",
        "passed": bool(passing_models),
        "evidence": {"candidates": model_candidates, "passing": len(passing_models)},
    }

    recovery = bool(card.get("restore", {}).get("spawn_bitwise_equal")) and bool(
        card.get("inheritance", {}).get("equal") == card.get("inheritance", {}).get("total")
    )
    performance = str(pricing.get("status", "")).startswith("passed")
    safety_reports = [
        relative
        for relative in CANDIDATE_REPORTS
        if load(relative).get("safety_checked") is True or "safety" in load(relative)
    ]
    try:
        instrument = load(INSTRUMENT_REPORT)
    except OSError:
        instrument = {}
    scope = instrument.get("safety_scope")
    side_effects = instrument.get("side_effects") or {}
    instrument_ok = (
        instrument.get("status") == "instrument_measured_quiet"
        and instrument.get("safety_checked") is True
        and isinstance(scope, list)
        and set(INSTRUMENT_SCOPE).issubset(scope)
        and side_effects.get("writes") == []
        and side_effects.get("sockets") == 0
        and side_effects.get("subprocesses") == 0
        and side_effects.get("system_calls") == 0
        and instrument.get("deterministic_outputs") is True
    )
    if instrument_ok:
        safety_reports.append(INSTRUMENT_REPORT)
    p3 = {
        "name": "P3_safety_recovery_performance",
        "passed": recovery and performance and bool(safety_reports),
        "evidence": {
            "recovery": {"passed": recovery, "source": f"{CLOSEOUT_CARD}:restore/inheritance"},
            "performance": {"passed": performance, "source": f"{PRICING_REPORT}:status"},
            "safety": {
                "passed": bool(safety_reports),
                "sources": safety_reports or "no runtime safety reading exists",
                "scope": instrument.get("safety_scope"),
            },
        },
    }

    identity = card.get("identity", {}).get("parent_model_digest")
    p4 = {
        "name": "P4_same_candidate_identity",
        "passed": bool(passing_models) and bool(identity),
        "evidence": {
            "closeout_digest": identity,
            "capability_reports": [row["report"] for row in passing_models],
        },
    }

    preconditions = [p1, p2, p3, p4]
    blockers = [row["name"] for row in preconditions if not row["passed"]]
    evidence_ok = all("evidence" in row and row["evidence"] for row in preconditions)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall

    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not evidence_ok:
        status = "no_evidence_path"
    elif blockers:
        status = "step11_not_admitted"
    else:
        status = "admitted_to_l3_scope"

    report = {
        "schema": "taiji-l3-admission-11-v1",
        "step": "11",
        "date": "2026-11-01",
        "status": status,
        "preconditions": preconditions,
        "blockers": blockers,
        "candidate_reports": model_candidates,
        "training_admitted": False,
        "l3_scope_started": False,
        "limits": (
            "Admission gate only. A not-admitted verdict closes nothing except step 11's opening; it "
            "does not invalidate any 09/10 engineering result. Admission still would not be the L3 "
            "acceptance itself and would not authorise training."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                L2_REPORT,
                CLOSEOUT_CARD,
                PRICING_REPORT,
                INSTRUMENT_REPORT,
                "scripts/training/verify_taiji_l3_admission_11.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "blockers": blockers,
                "preconditions": {row["name"]: row["passed"] for row in preconditions},
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
