"""Guard for the frozen P3.9.1-4 pre-registration.

Re-reads the manifest, the pricing report, the frozen parent blob and the three
headline documents. No model, forward pass, optimizer step or sealed face.
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
MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_9_prereg_20261114.json"
SPEC = ROOT / "plans/reference/SPEC-TAIJI_U2_P3_9_PREREG_20261114.md"
PRICE = ROOT / "reports/taiji_u2_p3_9_1_price_20261114.json"
PARENT = (
    ROOT
    / "data/experiments/taiji_u2/joint_learning/1794554480667378100/joint-query/update-000758.bin"
)
RECEIPT = ROOT / "reports/taiji_u2_p3_9_prereg_guard_20261114.json"
HEADLINE_MARK = "**队首（唯一）**："
HEADLINE_FILES = (
    ROOT / "plans/MAINLINE.md",
    ROOT / "plans/MASTER-PLAN.md",
    ROOT / "plans/reference/ROADMAP-NEXT_20261104.md",
)
FORBIDDEN_LITERALS = (
    "paired_updates",
    "min_meaningful_effect",
    "cpu_ratio_band",
    "bootstrap_rounds",
    "0.0312",
    "2303",
)


def sha_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def headline_of(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.find(HEADLINE_MARK)
        if index >= 0:
            return line[index + len(HEADLINE_MARK) :].strip()
    return ""


def evaluate(manifest: dict, spec_text: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["criteria_sha_matches"] = (
        hashlib.sha256(canonical(manifest["criteria"])).hexdigest() == manifest["criteria_sha256"]
    )
    checks["price_sha_matches"] = PRICE.exists() and sha_of(PRICE) == manifest["price"]["sha256"]
    checks["price_verdict_admitted"] = (
        manifest["price"]["verdict"] == "price_measured_admission_possible"
    )
    checks["parent_sha_matches"] = (
        PARENT.exists() and sha_of(PARENT) == manifest["parent"]["sha256"]
    )
    checks["parent_qualification_registered"] = (
        manifest["parent"]["native_parent_qualified"] is False
    )
    checks["spec_has_no_second_gate_set"] = not any(
        literal in spec_text for literal in FORBIDDEN_LITERALS
    )
    checks["budget_not_transferred"] = (
        manifest["criteria"]["budget"]["p3_8_budget_transferred"] is False
    )
    checks["execution_hashes_not_prefilled"] = (
        manifest["execution_runner_sha256"] is None and manifest["execution_report_sha256"] is None
    )
    checks["decision_rule_complete"] = set(manifest["criteria"]["behavior"]["decision_rule"]) == {
        "comparison_not_admitted",
        "compute_parity_unavailable",
        "budget_cannot_express",
        "b_superior_in_window",
        "a_superior_in_window",
        "inconclusive_within_window",
    }
    headlines = [headline_of(path) for path in HEADLINE_FILES]
    checks["headline_three_identical"] = len(set(headlines)) == 1 and bool(headlines[0])
    checks["headline_names_p3_9"] = all("P3.9" in text for text in headlines)
    return checks


def red_drills(manifest: dict, spec_text: str) -> dict[str, bool]:
    def mutate_gate(candidate: dict) -> None:
        candidate["criteria"]["behavior"]["min_meaningful_effect"] = 0.9

    def mutate_price(candidate: dict) -> None:
        candidate["price"]["sha256"] = "0" * 64

    def mutate_parent(candidate: dict) -> None:
        candidate["parent"]["native_parent_qualified"] = True

    def mutate_budget(candidate: dict) -> None:
        candidate["criteria"]["budget"]["p3_8_budget_transferred"] = True

    def mutate_execution(candidate: dict) -> None:
        candidate["execution_runner_sha256"] = "a" * 64

    def mutate_sha(candidate: dict) -> None:
        candidate["criteria_sha256"] = "b" * 64

    results = {}
    for name, mutate in {
        "gate_value": mutate_gate,
        "price_sha": mutate_price,
        "parent_qualification": mutate_parent,
        "budget_transfer": mutate_budget,
        "execution_hash": mutate_execution,
        "criteria_sha": mutate_sha,
    }.items():
        candidate = copy.deepcopy(manifest)
        mutate(candidate)
        results[name] = not all(evaluate(candidate, spec_text).values())
    injected = spec_text + "\npaired_updates: 999\n"
    results["spec_second_gate_set"] = not all(evaluate(manifest, injected).values())
    return results


def file_drill(manifest: dict, spec_text: str) -> dict:
    target = HEADLINE_FILES[0]
    original = target.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    index = next((i for i, line in enumerate(lines) if HEADLINE_MARK in line), None)
    if index is None or "P3.9" not in lines[index]:
        return {"passed": False, "reason": "headline_marker_not_found"}
    lines[index] = lines[index].replace("P3.9", "P3.8", 1)
    try:
        target.write_bytes("".join(lines).encode("utf-8"))
        red_checks = evaluate(manifest, spec_text)
        red_fired = not all(red_checks.values())
    finally:
        target.write_bytes(original)
    restored = target.read_bytes()
    green = evaluate(manifest, spec_text)
    return {
        "passed": red_fired and restored == original and all(green.values()),
        "red_fired": red_fired,
        "restored_byte_identical": restored == original,
        "green_after_restore": all(green.values()),
        "failing_checks_while_mutated": [key for key, value in red_checks.items() if not value],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red-drill", action="store_true")
    args = parser.parse_args()
    started_cpu, started_wall = time.process_time(), time.monotonic()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec_text = SPEC.read_text(encoding="utf-8")
    checks = evaluate(manifest, spec_text)
    drills = red_drills(manifest, spec_text)
    drill = file_drill(manifest, spec_text) if args.red_drill else None
    passed = all(checks.values()) and all(drills.values()) and (drill is None or drill["passed"])
    receipt = {
        "schema": "taiji-u2-p3-9-prereg-guard-v1",
        "date": "2026-11-14",
        "manifest": MANIFEST.name,
        "criteria_sha256": manifest["criteria_sha256"],
        "checks": checks,
        "in_memory_red_drills": drills,
        "real_file_red_drill": drill,
        "passed": passed,
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
        "note": "0 model loads / 0 forward passes / 0 optimizer steps / 0 sealed-face reads",
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
