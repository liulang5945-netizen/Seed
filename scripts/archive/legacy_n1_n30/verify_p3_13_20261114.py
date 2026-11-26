"""Guard for the P3.13 accepted-update arm.

Recomputes the final dose from its bundle, re-derives the verdict from the frozen
rule, checks the acceptance counters and the discipline block, and proves the
guard rings with in-memory and real-file mutations. No training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import evaluate, load_learner
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_13_acceptance_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_13_acceptance_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_13_guard_20261114.json"
GATE = 0.80
LOW = 0.35


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def verdict_for(report: dict) -> str:
    final = report["doses"][-1]
    if final["accuracy"] >= GATE:
        return "marginal_solution_broken"
    if final["accuracy"] <= LOW and final["train_accuracy"] <= LOW:
        return "acceptance_alone_insufficient"
    return "partial_movement_without_gate"


def evaluate_checks(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract.get("status") == "frozen"
    checks["authority_present"] = bool(contract.get("training_authority"))
    checks["caps_set"] = bool(contract.get("cpu_cap")) and bool(contract.get("wall_cap"))
    checks["criteria_sha_matches"] = (
        hashlib.sha256(canonical(contract["criteria"])).hexdigest() == contract["criteria_sha256"]
    )
    checks["verdict_matches_rule"] = report["verdict"] == verdict_for(report)
    checks["final_dose_matches_recompute"] = (
        report["doses"][-1]["correct"] == recomputed["correct"]
        and report["doses"][-1]["accuracy"] == recomputed["accuracy"]
        and report["doses"][-1]["train_correct"] == recomputed["train_correct"]
    )
    checks["distinct_outputs_matches"] = (
        report["doses"][-1]["distinct_outputs"] == recomputed["distinct_outputs"]
    )
    checks["acceptance_counters_consistent"] = (
        report["acceptance"]["accepted"] + report["acceptance"]["rejected"] == 1844
        and report["acceptance"]["optimizer_steps"] == 1844
    )
    checks["cost_within_cap"] = (
        report["cost"]["total_cpu"] <= contract["cpu_cap"]
        and report["cost"]["total_wall"] <= contract["wall_cap"]
    )
    checks["no_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_sealed_reads"] = all(
        report["discipline"][key] == 0 for key in ("check_reads", "final_reads", "confirm_reads")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-drills", action="store_true")
    args = parser.parse_args()
    started, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    final = report["doses"][-1]["update"]
    model, _, _ = load_learner(Path(contract["run_root"]) / f"dose-{final:04d}.bin")
    scored = evaluate(model, data["fit"])
    train = evaluate(model, data["train"])
    recomputed = {
        "correct": scored["correct"],
        "accuracy": scored["accuracy"],
        "train_correct": train["correct"],
        "distinct_outputs": len({row["output_hex"] for row in scored["rows"]}),
    }
    checks = evaluate_checks(contract, report, recomputed)

    drills: dict[str, bool] = {}
    real: dict = {"performed": False}
    if not args.skip_drills:
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "marginal_solution_broken"
        drills["tamper_verdict"] = not evaluate_checks(contract, tampered, recomputed)[
            "verdict_matches_rule"
        ]
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["correct"] = 30
        drills["tamper_final_correct"] = not evaluate_checks(contract, tampered, recomputed)[
            "final_dose_matches_recompute"
        ]
        tampered = copy.deepcopy(report)
        tampered["acceptance"]["rejected"] += 1
        drills["tamper_acceptance"] = not evaluate_checks(contract, tampered, recomputed)[
            "acceptance_counters_consistent"
        ]
        tampered = copy.deepcopy(report)
        tampered["discipline"]["capability_claims"] = 1
        drills["claim_capability"] = not evaluate_checks(contract, tampered, recomputed)[
            "no_capability_claims"
        ]
        original = REPORT.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["accuracy"] = 0.9
        REPORT.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            red = evaluate_checks(
                contract, json.loads(REPORT.read_text(encoding="utf-8")), recomputed
            )
        finally:
            REPORT.write_bytes(original)
        after = hashlib.sha256(REPORT.read_bytes()).hexdigest()
        green = evaluate_checks(
            contract, json.loads(REPORT.read_text(encoding="utf-8")), recomputed
        )
        real = {
            "performed": True,
            "red_checks": sorted(name for name, ok in red.items() if not ok),
            "restored_byte_exact": before == after,
            "green_after_restore": all(green.values()),
        }
    drills_ok = all(drills.values()) and (
        not real.get("performed")
        or (
            bool(real["red_checks"]) and real["restored_byte_exact"] and real["green_after_restore"]
        )
    )
    payload = {
        "schema": "taiji-u2-p3-13-guard-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.13",
        "checks": checks,
        "red_drills": drills,
        "real_file_drill": real,
        "recomputed": recomputed,
        "verdict": "pass" if all(checks.values()) and drills_ok else "fail",
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "discipline": {"training_runs": 0, "optimizer_calls": 0},
        "measured": {
            "cpu_seconds": round(time.process_time() - started, 6),
            "wall_seconds": round(time.perf_counter() - wall, 6),
        },
    }
    write_report(OUT, payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "failed": payload["failed"],
                "drills": drills,
                "real": real,
                "recomputed": recomputed,
                "measured": payload["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
