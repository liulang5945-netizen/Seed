"""Guard for the P3.8.4 global closeout ledger.

Every check re-reads the cited reports from disk. Nothing here loads a model,
runs a forward pass, writes a checkpoint, or touches a sealed face.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLOSEOUT = ROOT / "reports/taiji_u2_p3_8_4_global_closeout_20261114.json"
RECEIPT = ROOT / "reports/taiji_u2_p3_8_4_global_closeout_guard_20261114.json"
PARENT = (
    ROOT
    / "data/experiments/taiji_u2/joint_learning/1794554480667378100/joint-query/update-000758.bin"
)
HEADLINE_MARK = "**队首（唯一）**："
HEADLINE_FILES = (
    ROOT / "plans/MAINLINE.md",
    ROOT / "plans/MASTER-PLAN.md",
    ROOT / "plans/reference/ROADMAP-NEXT_20261104.md",
)


def sha_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def headline_of(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.find(HEADLINE_MARK)
        if index >= 0:
            return line[index + len(HEADLINE_MARK) :].strip()
    return ""


def cpu_of(data: dict):
    for key in ("cpu_seconds", "measured_cpu_seconds"):
        if isinstance(data.get(key), (int, float)):
            return float(data[key])
    measured = data.get("measured")
    if isinstance(measured, dict) and isinstance(measured.get("cpu_seconds"), (int, float)):
        return float(measured["cpu_seconds"])
    return None


def verdict_of(data: dict):
    return data.get("verdict") or data.get("audited_verdict")


def group_of(card: str) -> str:
    if card.startswith("P3.8.2b"):
        return "p3_8_2b_engineering_20261114"
    if card.startswith("P3.8.3"):
        return "p3_8_3_runner_20261114"
    return "p3_8_1_settled_20261113"


def evaluate(report: dict) -> tuple[dict[str, bool], dict]:
    checks: dict[str, bool] = {}
    detail: dict = {}

    stage_sha: list[bool] = []
    stage_verdict: list[bool] = []
    stage_cpu: list[bool] = []
    group_sum: dict[str, float] = {}
    for stage in report["stages"]:
        path = ROOT / stage["report"]
        exists = path.exists()
        data = load_json(path) if exists else {}
        stage_sha.append(exists and sha_of(path) == stage["report_sha256"])
        stage_verdict.append(exists and verdict_of(data) == stage["verdict"])
        stage_cpu.append(cpu_of(data) == stage["cpu_seconds"])
        if stage["cpu_seconds"] is not None and exists:
            key = group_of(stage["card"])
            group_sum[key] = round(group_sum.get(key, 0.0) + float(stage["cpu_seconds"]), 6)

    checks["all_stage_sha_match"] = all(stage_sha) and len(stage_sha) == 16
    checks["all_stage_verdict_match"] = all(stage_verdict)
    checks["all_stage_cpu_match"] = all(stage_cpu)
    declared = report["cost_ledger"]["instrumented_report_cpu_by_group"]
    checks["group_sums_match"] = all(
        abs(group_sum.get(key, -1.0) - float(value)) < 1e-6 for key, value in declared.items()
    ) and set(group_sum) == set(declared)
    total = round(sum(group_sum.values()), 6)
    checks["total_and_unused_match"] = (
        abs(total - report["cost_ledger"]["instrumented_report_cpu_total"]) < 1e-6
        and abs(
            report["cost_ledger"]["unused_unallocated_cpu"]
            - (790 - declared["p3_8_2b_engineering_20261114"] - declared["p3_8_3_runner_20261114"])
        )
        < 1e-6
    )

    unsigned = dict(report)
    unsigned.pop("closeout_sha256", None)
    checks["closeout_sha_match"] = (
        hashlib.sha256(canonical(unsigned)).hexdigest() == report["closeout_sha256"]
    )

    conditions = report["closeout_conditions"]
    checks["conditions_all_passed"] = len(conditions) == 8 and all(
        item["passed"] for item in conditions
    )
    checks["condition_evidence_exists"] = all(
        (ROOT / evidence).exists() for item in conditions for evidence in item["evidence"]
    )

    discipline = report["discipline"]
    zero_keys = [
        key
        for key in discipline
        if key.startswith("closeout_card_") and isinstance(discipline[key], int)
    ]
    checks["discipline_all_zero"] = bool(zero_keys) and all(
        discipline[key] == 0 for key in zero_keys
    )

    exits = report["exit_semantics"]
    boundary_keys = [key for key in exits if key.startswith("no_") or key.startswith("not_a_")]
    checks["exit_semantics_boundaries"] = bool(boundary_keys) and all(
        exits[key] is True for key in boundary_keys
    )
    checks["parent_sha_match"] = (
        PARENT.exists() and sha_of(PARENT) == report["parent_identity"]["sha256"]
    )

    headlines = [headline_of(path) for path in HEADLINE_FILES]
    checks["headline_three_identical"] = len(set(headlines)) == 1 and bool(headlines[0])
    checks["headline_marks_closeout_and_handoff"] = all(
        "P3.8" in text and "收束" in text and "P3.9.1" in text for text in headlines
    )
    detail["group_sums"] = group_sum
    detail["total_cpu"] = total
    detail["headline_sha256"] = hashlib.sha256(headlines[0].encode("utf-8")).hexdigest()
    detail["headline"] = headlines[0]
    return checks, detail


def red_drills(report: dict) -> dict[str, bool]:
    """Each mutation must make at least one guard check fail."""

    def mutate_verdict(candidate: dict) -> None:
        candidate["stages"][0]["verdict"] = "tampered"

    def mutate_sha(candidate: dict) -> None:
        candidate["stages"][3]["report_sha256"] = "0" * 64

    def mutate_cpu(candidate: dict) -> None:
        candidate["stages"][0]["cpu_seconds"] = 999.0

    def mutate_group(candidate: dict) -> None:
        candidate["cost_ledger"]["instrumented_report_cpu_by_group"]["p3_8_3_runner_20261114"] = 1.0

    def mutate_condition(candidate: dict) -> None:
        candidate["closeout_conditions"][0]["passed"] = False

    def mutate_evidence(candidate: dict) -> None:
        candidate["closeout_conditions"][0]["evidence"] = ["reports/does_not_exist.json"]

    def mutate_discipline(candidate: dict) -> None:
        candidate["discipline"]["closeout_card_optimizer_steps"] = 1

    def mutate_exit(candidate: dict) -> None:
        candidate["exit_semantics"]["no_dose_increase"] = False

    def mutate_parent(candidate: dict) -> None:
        candidate["parent_identity"]["sha256"] = "f" * 64

    def mutate_closeout_sha(candidate: dict) -> None:
        candidate["closeout_sha256"] = "e" * 64

    mutations = {
        "stage_verdict": mutate_verdict,
        "stage_report_sha": mutate_sha,
        "stage_cpu": mutate_cpu,
        "group_sum": mutate_group,
        "condition_passed": mutate_condition,
        "condition_evidence": mutate_evidence,
        "discipline_zero": mutate_discipline,
        "exit_boundary": mutate_exit,
        "parent_sha": mutate_parent,
        "closeout_sha": mutate_closeout_sha,
    }
    results: dict[str, bool] = {}
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(report)
        mutate(candidate)
        checks, _ = evaluate(candidate)
        results[name] = not all(checks.values())
    return results


def file_drill(report: dict) -> dict:
    """Real on-disk headline mutation: green -> red -> byte-identical restore -> green."""

    target = HEADLINE_FILES[0]
    original = target.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    target_index = next((i for i, line in enumerate(lines) if HEADLINE_MARK in line), None)
    if target_index is None or "P3.9.1" not in lines[target_index]:
        return {"passed": False, "reason": "headline_marker_not_found"}
    lines[target_index] = lines[target_index].replace("P3.9.1", "P3.9.9", 1)
    mutated = "".join(lines)
    if mutated == original.decode("utf-8"):
        return {"passed": False, "reason": "headline_mutation_failed"}
    try:
        target.write_bytes(mutated.encode("utf-8"))
        red_checks, _ = evaluate(report)
        red_fired = not all(red_checks.values())
    finally:
        target.write_bytes(original)
    restored = target.read_bytes()
    green_checks, green_detail = evaluate(report)
    return {
        "passed": red_fired and restored == original and all(green_checks.values()),
        "red_fired": red_fired,
        "restored_byte_identical": restored == original,
        "green_after_restore": all(green_checks.values()),
        "failing_checks_while_mutated": [key for key, value in red_checks.items() if not value],
        "headline_sha256": green_detail["headline_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--red-drill", action="store_true", help="also mutate a real headline file, then restore it"
    )
    args = parser.parse_args()

    started_cpu, started_wall = time.process_time(), time.monotonic()
    report = load_json(CLOSEOUT)
    checks, detail = evaluate(report)
    drills = red_drills(report)
    drill = file_drill(report) if args.red_drill else None

    passed = all(checks.values()) and all(drills.values()) and (drill is None or drill["passed"])
    receipt = {
        "schema": "taiji-u2-p3-8-4-global-closeout-guard-v1",
        "date": "2026-11-14",
        "closeout": CLOSEOUT.name,
        "closeout_sha256": report["closeout_sha256"],
        "checks": checks,
        "in_memory_red_drills": drills,
        "real_file_red_drill": drill,
        "detail": detail,
        "passed": passed,
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
        "note": "0 model loads / 0 forward passes / 0 optimizer steps / 0 sealed-face reads; the guard only re-reads reports, the frozen parent blob sha and three headline files",
    }
    RECEIPT.write_bytes(json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        json.dumps(
            {"passed": passed, "checks": checks, "red_drills": drills, "file_drill": drill},
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
