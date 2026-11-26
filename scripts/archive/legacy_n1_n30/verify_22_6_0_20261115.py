"""Guard for the 22.6.0 fixture receipt.

The receipt is produced by ``record_22_6_0_fixture_20261115.py``. This guard does
not import the module's assessment: it recomputes every published number from the
schedule table the receipt was built from, using its own value function. If the
producer and this reader ever disagree, the guard is the one that is allowed to
speak.

It also refuses to pass on a claim the card cannot make: a parent that is not
present, a capability count above zero, or a seed that is not deterministic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/taiji_u2_22_6_0_fixture_20261115.json"
REPORT = ROOT / "reports/taiji_u2_22_6_0_fixture_20261115.json"
ENTRY = ROOT / "reports/taiji_u2_22_6_entry_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_22_6_0_guard_20261115.json"
HEAD_FILES = (
    ROOT / "plans/MASTER-PLAN.md",
    ROOT / "plans/MAINLINE.md",
    ROOT / "plans/reference/ROADMAP-NEXT_20261104.md",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def value_at(opportunity: dict, tick: int) -> int:
    """Independent value function: act only, integer, closed after the deadline."""
    if tick < 0 or tick > opportunity["deadline_tick"]:
        return 0
    if tick < opportunity["reveal_tick"]:
        return opportunity["value_guess"]
    start, end = opportunity["window"]
    return opportunity["value_full"] if start <= tick <= end else 0


def recompute(schedule: dict) -> dict:
    rows = schedule["sessions"]
    flat = [opportunity for row in rows for opportunity in row["opportunities"]]
    now = [value_at(o, 0) for o in flat]
    deferred = [max(value_at(o, tick) for tick in range(1, o["deadline_tick"] + 1)) for o in flat]
    failures = []
    if len(rows) != 3 or any(len(row["opportunities"]) != 16 for row in rows):
        failures.append("shape")
    if sum(max(a, b) for a, b in zip(now, deferred, strict=True)) == 0:
        failures.append("wait_idle_optimal")
    for opportunity in flat:
        if value_at(opportunity, opportunity["deadline_tick"] + 1) != 0:
            failures.append("deadline_not_binding")
            break
    if not any(o["window"][1] < o["deadline_tick"] for o in flat):
        failures.append("window_equals_deadline_everywhere")
    if not any(b > a for a, b in zip(now, deferred, strict=True)):
        failures.append("wait_never_optimal")
    if not any(a > b for a, b in zip(now, deferred, strict=True)):
        failures.append("wait_always_optimal")
    for row in rows:
        quiet = row["quiet_run"]
        if len(quiet) < 3 or quiet[-1] - quiet[0] + 1 != len(quiet):
            failures.append("no_contiguous_prompt_free_run")
        elif any(row["opportunities"][i]["prompt"] is not None for i in quiet):
            failures.append("prompt_inside_the_quiet_run")
        change = row["feedback_change"]
        if not change["affects"] or min(change["affects"]) <= change["at"]:
            failures.append("no_feedback_change")
        if not any(o["kind"] == "pending" for o in row["opportunities"]):
            failures.append("no_pending_result")
        if row["help"]["bytes"] <= 0 or not row["help"]["removed_arm"]:
            failures.append("help_not_metered_or_not_removed")
    return {
        "valid": not failures,
        "failures": sorted(set(failures)),
        "totals": {
            "act_now": sum(now),
            "deferred": sum(deferred),
            "oracle": sum(max(a, b) for a, b in zip(now, deferred, strict=True)),
        },
        "counts": {
            "requires_waiting": sum(1 for a, b in zip(now, deferred, strict=True) if b > a),
            "requires_immediate": sum(1 for a, b in zip(now, deferred, strict=True) if a > b),
            "pending_results": sum(1 for o in flat if o["kind"] == "pending"),
            "prompt_free": sum(1 for o in flat if o["prompt"] is None),
            "sessions": len(rows),
            "opportunities": len(flat),
        },
    }


def evaluate_entry(contract: dict, entry: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["entry_verdict_is_limited"] = (
        entry["verdict"] == "entry_limited_to_fixture_parent_absent"
    )
    checks["capability_entry_is_false"] = entry["capability_entry"] is False
    checks["fixture_entry_is_true"] = entry["fixture_entry"] is True
    checks["gate_rows_complete"] = all(
        row.get("id") and row.get("requirement") and row.get("evidence") for row in entry["gate"]
    )
    checks["blocking_rows_are_the_three_parent_rows"] = entry["blocking_rows"] == [
        "G1",
        "G2",
        "G3",
    ]
    checks["forbidden_lists_every_subcard"] = all(
        any(card in item for item in entry["forbidden_now"])
        for card in ("22.6.1", "22.6.2", "22.6.3", "22.6.4", "22.6.5")
    )
    checks["entry_claims_nothing"] = (
        entry["discipline"]["capability_claims"] == 0
        and entry["discipline"]["checks_consumed"] == 0
        and entry["discipline"]["budget_consumed_cpu"] == 0.0
        and entry["discipline"]["optimizer_calls"] == 0
    )
    checks["entry_pins_live"] = entry["source_pins"]["module"] == sha_of(
        ROOT / "taiji/u2_drive_task.py"
    ) and entry["source_pins"]["manifest"] == sha_of(MANIFEST)
    checks["entry_matches_contract_parent"] = (
        entry["fixture"]["criteria_sha256"] == contract["criteria_sha256"]
    )
    return checks


def heads_agree() -> bool:
    lines = []
    for path in HEAD_FILES:
        found = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("**队首（唯一）**：")
        ]
        if len(found) != 1:
            return False
        lines.append(found[0])
    return len(set(lines)) == 1


def evaluate(contract: dict, report: dict, schedules: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract["status"] == "frozen"
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    pins = contract["source_pins"]
    checks["module_pin_live"] = pins["module"] == sha_of(ROOT / "taiji/u2_drive_task.py")
    checks["test_pin_live"] = pins["test"] == sha_of(
        ROOT / "tests/taiji_native/test_u2_drive_task.py"
    )
    checks["report_verdict_is_valid"] = report["verdict"] == "fixture_valid"
    checks["report_has_no_failures"] = report["failures"] == []
    checks["seed_set_matches"] = [row["seed"] for row in report["seeds"]] == contract["seeds"]
    checks["every_seed_valid_and_deterministic"] = all(
        row["valid"] and row["deterministic"] for row in report["seeds"]
    )
    checks["parent_is_absent"] = (
        report["parent"]["present"] is False
        and report["parent"]["native_parent_qualified"] is False
    )
    checks["zero_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_optimizer_no_forward_no_world_step"] = (
        report["discipline"]["optimizer_calls"] == 0
        and report["discipline"]["model_forwards"] == 0
        and report["discipline"]["world_steps"] == 0
        and report["discipline"]["sealed_reads"] == 0
    )
    recomputed = {row["seed"]: recompute(schedules[row["seed"]]) for row in report["seeds"]}
    checks["recompute_matches_every_seed"] = all(
        recomputed[row["seed"]]["totals"] == row["totals"]
        and recomputed[row["seed"]]["counts"] == row["counts"]
        and recomputed[row["seed"]]["failures"] == row["failures"]
        for row in report["seeds"]
    )
    checks["recompute_says_valid"] = all(block["valid"] for block in recomputed.values())
    checks["composition_is_seed_invariant"] = (
        len({tuple(sorted(row["totals"].items())) for row in report["seeds"]}) == 1
    )
    checks["digests_differ_by_seed"] = len({row["digest"] for row in report["seeds"]}) == len(
        report["seeds"]
    )
    checks["cost_within_cap"] = (
        report["measured"]["cpu_seconds"] <= contract["caps"]["cpu"]
        and report["measured"]["wall_seconds"] <= contract["caps"]["wall"]
    )
    checks["cpu_below_timer_tick_is_declared"] = (
        report["measured"]["cpu_seconds"] >= 0.0156 or report["measured"]["wall_seconds"] > 0
    )
    checks["nav_heads_agree"] = heads_agree()
    return checks


def drills(contract: dict, report: dict, schedules: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = not evaluate(contract, mutated, schedules)[
        "zero_capability_claims"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["parent"]["present"] = True
    out["claim_a_parent"] = not evaluate(contract, mutated, schedules)["parent_is_absent"]
    mutated = json.loads(json.dumps(report))
    mutated["seeds"][1]["totals"]["oracle"] += 1
    out["inflate_oracle_total"] = not evaluate(contract, mutated, schedules)[
        "recompute_matches_every_seed"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["seeds"][0]["counts"]["requires_waiting"] = 0
    out["erase_waiting_count"] = not evaluate(contract, mutated, schedules)[
        "recompute_matches_every_seed"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["verdict"] = "fixture_invalid"
    out["flip_verdict"] = not evaluate(contract, mutated, schedules)["report_verdict_is_valid"]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["optimizer_calls"] = 3
    out["hide_an_optimizer_call"] = not evaluate(contract, mutated, schedules)[
        "no_optimizer_no_forward_no_world_step"
    ]
    entry = json.loads(ENTRY.read_text(encoding="utf-8"))
    entry_checks = evaluate_entry(contract, entry)
    mutated = json.loads(json.dumps(entry))
    mutated["capability_entry"] = True
    out["open_capability_entry"] = not evaluate_entry(contract, mutated)[
        "capability_entry_is_false"
    ]
    mutated = json.loads(json.dumps(entry))
    mutated["verdict"] = "entered"
    out["rename_the_verdict"] = not evaluate_entry(contract, mutated)["entry_verdict_is_limited"]
    out["entry_checks_all_hold"] = all(entry_checks.values())
    return out


def main() -> int:
    import time

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    sys.path.insert(0, str(ROOT))
    from taiji.u2_drive_task import build_schedule

    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    schedules = {seed: build_schedule(int(seed)) for seed in contract["seeds"]}
    checks = evaluate(contract, report, schedules)
    entry = json.loads(ENTRY.read_text(encoding="utf-8"))
    checks.update(evaluate_entry(contract, entry))
    fired = drills(contract, report, schedules)
    passed = all(checks.values())
    receipt = {
        "schema": "taiji-u2-22-6-0-guard-v1",
        "card": "22.6.0",
        "verdict": "guard_passed" if passed and all(fired.values()) else "guard_failed",
        "checks": checks,
        "drills": fired,
        "failed": [name for name, value in checks.items() if not value],
        "drills_not_firing": [name for name, value in fired.items() if not value],
    }
    receipt["measured"] = {
        "cpu_seconds": round(time.process_time() - cpu_start, 6),
        "wall_seconds": round(time.perf_counter() - wall_start, 6),
    }
    GUARD_REPORT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"verdict": receipt["verdict"], "failed": receipt["failed"]},
            ensure_ascii=False,
        )
    )
    return 0 if receipt["verdict"] == "guard_passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
