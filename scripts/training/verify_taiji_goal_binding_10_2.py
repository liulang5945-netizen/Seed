"""10.2 verification: the binding contract, five public counter-examples, one baseline.

No model is bound here. The rule baseline must bind the unambiguous Chinese goals
and refuse the missing/ambiguous ones with a declared reason; every fixture counts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import taiji.goal_binding as binder  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES, build_fixtures  # noqa: E402

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
FORBIDDEN_IDENTIFIERS = ("Goal", "gold", "terminal_state", "reward", "score")
FIXTURES = (
    ("b1-paraphrase-a", "把蓝色杯子放到窗台", {"conditions": [(OBJECTS[0], PLACES[1], 1)]}),
    ("b2-paraphrase-b", "蓝色杯子应该出现在窗台上", {"conditions": [(OBJECTS[0], PLACES[1], 1)]}),
    ("b3-negation", "红色印章不要留在左盒", {"conditions": [(OBJECTS[1], PLACES[2], -1)]}),
    (
        "b4-conjunction",
        "把蓝色杯子放到窗台，并且红色印章不要留在左盒",
        {
            "conditions": [
                (OBJECTS[0], PLACES[1], 1),
                (OBJECTS[1], PLACES[2], -1),
            ]
        },
    ),
    ("b5-missing", "把它放到窗台", {"abstain": True, "reason": "missing_referent"}),
    ("b6-ambiguous", "把蓝色杯子放到窗台或木桌", {"abstain": True, "reason": "ambiguous"}),
)


def scan_binder_source() -> dict:
    tree = ast.parse(inspect.getsource(binder))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    seen = names | attributes
    return {token: int(token in seen) for token in FORBIDDEN_IDENTIFIERS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.2 report is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    state = build_fixtures()[0].initial
    rows = []
    for fixture_id, text, expectation in FIXTURES:
        binding = binder.bind(text, state)
        conditions = [list(condition) for condition in binding.conditions]
        rows.append(
            {
                "fixture_id": fixture_id,
                "goal_text": text,
                "conditions": conditions,
                "abstain": binding.abstain,
                "reason": binding.reason,
                "unknowns": list(binding.unknowns),
                "expectation": {
                    "conditions": [
                        list(condition) for condition in expectation.get("conditions", [])
                    ],
                    "abstain": expectation.get("abstain", False),
                    "reason": expectation.get("reason"),
                },
            }
        )

    by_id = {row["fixture_id"]: row for row in rows}
    dropped = len(rows) != len(FIXTURES)
    paraphrase_ok = by_id["b1-paraphrase-a"]["conditions"] == by_id["b2-paraphrase-b"]["conditions"]
    negation_ok = by_id["b3-negation"]["conditions"] == [[OBJECTS[1], PLACES[2], -1]]
    conjunction_ok = {tuple(condition) for condition in by_id["b4-conjunction"]["conditions"]} == {
        (OBJECTS[0], PLACES[1], 1),
        (OBJECTS[1], PLACES[2], -1),
    }
    abstain_ok = all(
        by_id[fixture_id]["abstain"] is True
        and by_id[fixture_id]["reason"] == reason
        and by_id[fixture_id]["conditions"] == []
        for fixture_id, reason in (
            ("b5-missing", "missing_referent"),
            ("b6-ambiguous", "ambiguous"),
        )
    )
    leak_counts = scan_binder_source()
    leak_free = all(count == 0 for count in leak_counts.values())
    model_binding_delivered = False

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {"id": "fixture_dropped", "passed": not dropped, "observed": [len(rows), len(FIXTURES)]},
        {
            "id": "paraphrase_mismatch",
            "passed": paraphrase_ok,
            "observed": [
                by_id["b1-paraphrase-a"]["conditions"],
                by_id["b2-paraphrase-b"]["conditions"],
            ],
        },
        {
            "id": "negation_or_conjunction_wrong",
            "passed": negation_ok and conjunction_ok,
            "observed": [by_id["b3-negation"]["conditions"], by_id["b4-conjunction"]["conditions"]],
        },
        {
            "id": "abstain_not_enforced",
            "passed": abstain_ok,
            "observed": {
                fixture_id: [by_id[fixture_id]["abstain"], by_id[fixture_id]["reason"]]
                for fixture_id in ("b5-missing", "b6-ambiguous")
            },
        },
        {"id": "gold_leaked_into_binder", "passed": leak_free, "observed": leak_counts},
        {
            "id": "model_binding_claimed",
            "passed": not model_binding_delivered,
            "observed": model_binding_delivered,
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    failed = [row["id"] for row in checks if not row["passed"]]
    if "budget_not_exceeded" in failed:
        status = "budget_stop"
    elif "fixture_dropped" in failed:
        status = "fixture_dropped"
    elif "paraphrase_mismatch" in failed:
        status = "paraphrase_mismatch"
    elif "negation_or_conjunction_wrong" in failed:
        status = "negation_or_conjunction_wrong"
    elif "abstain_not_enforced" in failed:
        status = "abstain_not_enforced"
    elif "gold_leaked_into_binder" in failed:
        status = "gold_leaked_into_binder"
    elif "model_binding_claimed" in failed:
        status = "model_binding_claimed"
    elif failed:
        status = "engineering_failure"
    else:
        status = "passed_10_2_binding_contract_only"

    report = {
        "schema": "taiji-goal-binding-10-2-v1",
        "step": "10.2",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "binder": "rule_baseline",
            "reasons": list(binder.REASONS),
            "fixtures": len(FIXTURES),
            "denominator": len(FIXTURES),
            "model_binding_delivered": model_binding_delivered,
        },
        "fixtures": rows,
        "checks": checks,
        "model_involved": False,
        "training_admitted": False,
        "task_capability_claimed": False,
        "limits": (
            "Binding contract and counter-examples only. The binder is a rule baseline over the "
            "10.1 vocabulary; model-side binding of natural Chinese goals is not delivered and no "
            "task capability is claimed."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world.py",
                "taiji/goal_binding.py",
                "scripts/training/verify_taiji_goal_binding_10_2.py",
                "tests/taiji_native/test_goal_binding.py",
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
                "fixtures": len(rows),
                "abstained": [row["fixture_id"] for row in rows if row["abstain"]],
                "cpu_seconds": round(cpu_seconds, 3),
                "failed": failed,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
