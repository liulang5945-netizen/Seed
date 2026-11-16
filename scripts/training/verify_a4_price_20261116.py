"""Guard for the A.4 pricing receipt.

The price report is the only thing standing between a design and an optimizer
call, so the guard recomputes the cap from the measured unit costs instead of
trusting the number in the report, checks the pins and the frozen criteria, and
refuses a receipt that claims training happened.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "plans/manifests/taiji_u2_a4_column_read_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a4_column_read_price_20261116.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a4_column_read_guard_20261116.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def recompute_cap(report: dict) -> dict:
    """Rebuild the budget from the measured unit costs, independently."""
    unit = report["unit_cpu"]
    plan = report["plan"]
    seeds = int(plan["seeds"])
    updates = int(plan["updates_per_seed"])
    dose_points = int(plan["dose_points"])
    per_seed_training = updates * unit["train_step_proxy_per_example"]
    per_seed_evaluation = dose_points * ((32 + 64) * unit["byte_round_per_example_cached_read"])
    per_seed_probe = unit["prefix0_capture_fit32"]
    per_seed_recovery = unit["recovery_replay_fit8"] * (32 / 8)
    per_seed_checkpoints = dose_points * 0.05
    per_seed = (
        per_seed_training
        + per_seed_evaluation
        + per_seed_probe
        + per_seed_recovery
        + per_seed_checkpoints
    )
    estimate = per_seed * seeds
    contingency = estimate * 0.15
    return {
        "per_seed_training": round(per_seed_training, 3),
        "per_seed_evaluation": round(per_seed_evaluation, 3),
        "per_seed_probe": round(per_seed_probe, 3),
        "per_seed_recovery": round(per_seed_recovery, 3),
        "per_seed_checkpoints": round(per_seed_checkpoints, 3),
        "per_seed_total": round(per_seed, 3),
        "estimate_cpu": round(estimate, 3),
        "contingency_cpu": round(contingency, 3),
        "cap_cpu": round(estimate + contingency, 3),
    }


def evaluate(contract: dict, report: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_is_design_priced"] = contract["status"] == "design_priced"
    checks["training_not_authorised"] = contract["training_authority"] is None
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    pins = contract["source_pins"]
    checks["script_pin_live"] = pins["script_sha256"] == sha_of(ROOT / pins["script"])
    checks["read_module_pin_live"] = pins["read_module_sha256"] == sha_of(
        ROOT / pins["read_module"]
    )
    checks["read_identity_declared"] = contract["candidate"]["read_identity"] == "column-read-v1"
    checks["read_scalar_count_matches_contract"] = (
        report["read_parameters"]["scalars"] == contract["candidate"]["read_scalars_expected"]
    )
    checks["read_under_design_cap"] = (
        report["read_parameters"]["scalars"] <= contract["structure"]["design_scalar_cap"]
    )
    checks["body_identity_unchanged"] = (
        report["body"]["parameters"] == 22742 and report["body"]["tensors"] == 194
    )
    recomputed = recompute_cap(report)
    checks["cap_recomputes_from_unit_costs"] = all(
        abs(recomputed[key] - report["plan"][key]) < 0.01
        for key in ("per_seed_total", "estimate_cpu", "contingency_cpu", "cap_cpu")
    )
    checks["cap_within_design_cap"] = report["plan"]["cap_cpu"] <= float(
        contract["plan"]["design_cap_cpu"]
    )
    checks["replay_is_byte_stable"] = report["unit_cpu"]["replay_stable"] is True
    checks["no_training_happened"] = (
        report["discipline"]["optimizer_calls"] == 0
        and report["discipline"]["optimizer_steps"] == 0
        and report["discipline"]["training_updates"] == 0
        and report["discipline"]["checkpoints_written"] == 0
        and report["discipline"]["bundles_written"] == 0
    )
    checks["no_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_sealed_reads"] = (
        report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0
    )
    checks["gate_r_carries_the_three_conditions"] = (
        contract["criteria"]["gate_r_read_prefix0"]["margin_over_shuffle_null"] >= 0.20
        and contract["criteria"]["gate_r_read_prefix0"]["painted_column_colour_probe"] >= 0.50
        and contract["criteria"]["gate_r_read_prefix0"]["prefix"] == 0
    )
    checks["correction_identifiability_declared"] = bool(
        contract["criteria"]["gate_b_byte"]["correction_identifiability"]
    )
    checks["scope_limit_declared"] = bool(contract["criteria"]["scope_limit"])
    return checks


def drills(contract: dict, report: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mutated = json.loads(json.dumps(report))
    mutated["plan"]["cap_cpu"] = 10.0
    out["shrink_the_cap"] = not evaluate(contract, mutated)["cap_recomputes_from_unit_costs"]
    mutated = json.loads(json.dumps(report))
    mutated["unit_cpu"]["train_step_proxy_per_example"] *= 2
    out["inflate_a_unit_cost"] = not evaluate(contract, mutated)["cap_recomputes_from_unit_costs"]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["optimizer_calls"] = 600
    out["hide_an_optimizer_call"] = not evaluate(contract, mutated)["no_training_happened"]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = not evaluate(contract, mutated)["no_capability_claims"]
    mutated = json.loads(json.dumps(report))
    mutated["body"]["parameters"] = 23000
    out["claim_a_changed_body"] = not evaluate(contract, mutated)["body_identity_unchanged"]
    mutated = json.loads(json.dumps(report))
    mutated["read_parameters"]["scalars"] = 9000
    out["hide_an_oversized_read"] = not evaluate(contract, mutated)["read_under_design_cap"]
    return out


def main() -> int:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checks = evaluate(contract, report)
    fired = drills(contract, report)
    passed = all(checks.values()) and all(fired.values())
    receipt = {
        "schema": "taiji-u2-a4-column-read-guard-v1",
        "card": "22.5.A.4",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "failed": [name for name, value in checks.items() if not value],
        "drills_not_firing": [name for name, value in fired.items() if not value],
        "recomputed_cap": recompute_cap(report),
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
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
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
