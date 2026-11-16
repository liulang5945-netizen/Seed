"""Guard for the A.5 protocol price receipt.

The price receipt is the only thing standing between a protocol card and an
optimizer call, so the guard recomputes the cap from the measured unit costs
instead of trusting the number in the report, checks the pins and the frozen
criteria, and refuses a receipt that claims training happened.

It also drills the red directions: shrinking the cap, inflating a unit cost,
hiding an optimizer call, injecting a capability claim, widening the seed
count and claiming a parent.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
PROBE_REPORT = ROOT / "reports/taiji_u2_a5_feasibility_probe_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a5_protocol_price_20261116.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a5_protocol_guard_20261116.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def recompute_cap(report: dict) -> dict:
    """Rebuild the budget from the measured unit costs, independently."""
    unit = report["unit_cpu"]
    plan = report["plan"]
    seeds = int(plan["seeds"])
    updates = int(plan["updates_per_seed"])
    dose_points = int(plan["dose_points"])
    error_cap = int(plan["feedback_error_allowance"])
    a4_updates = int(plan["a4_protocol_updates_per_error"])
    rounds = 32 + 64

    per_seed_training = updates * unit["train_step_proxy_per_batch"]
    per_seed_evaluation = dose_points * rounds * unit["byte_round_per_example"]
    per_seed_feedback = error_cap * (
        unit["single_error_correction_per_error"] + unit["byte_round_per_example_untrained"]
    )
    per_seed_control = (
        error_cap
        * (
            a4_updates * unit["train_step_proxy_per_batch"]
            + unit["single_error_correction_per_error"]
        )
        + 32 * unit["byte_round_per_example_untrained"]
    )
    per_seed_recovery = unit["recovery_replay_fit8"] * (32 / 8)
    per_seed_checkpoints = dose_points * 0.05
    per_seed = (
        per_seed_training
        + per_seed_evaluation
        + per_seed_feedback
        + per_seed_control
        + per_seed_recovery
        + per_seed_checkpoints
    )
    estimate = per_seed * seeds
    contingency = round(estimate * 0.15, 3)
    return {
        "per_seed_training": round(per_seed_training, 3),
        "per_seed_evaluation": round(per_seed_evaluation, 3),
        "per_seed_feedback": round(per_seed_feedback, 3),
        "per_seed_control_arm": round(per_seed_control, 3),
        "per_seed_recovery": round(per_seed_recovery, 3),
        "per_seed_checkpoints": round(per_seed_checkpoints, 3),
        "per_seed_total": round(per_seed, 3),
        "estimate_cpu": round(estimate, 3),
        "contingency_cpu": contingency,
        "cap_cpu": round(estimate + contingency, 3),
    }


def evaluate(contract: dict, report: dict, probe: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_state_is_recorded"] = contract["status"] in {
        "design_priced",
        "price_frozen_authorization_pending",
        "frozen",
    }
    checks["authority_state_is_consistent"] = (
        contract["training_authority"] is None
        if contract["status"] in {"design_priced", "price_frozen_authorization_pending"}
        else bool(contract["training_authority"])
    )
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    pins = contract["source_pins"]
    for name, path in (
        ("probe", "scripts/training/probe_a5_feasibility_20261116.py"),
        ("price", "scripts/training/price_a5_protocol_20261116.py"),
        ("subject", "taiji/u2_subject.py"),
        ("events", "taiji/u2_events.py"),
        ("contract_module", "taiji/u2_task_contract.py"),
        ("a2_runner", "scripts/training/run_a2_byte_20261115.py"),
        ("support", "scripts/training/first_a_support_22_5.py"),
        ("columns", "taiji/u2_observation_columns.py"),
    ):
        checks[f"pin_{name}_live"] = pins.get(name) == sha_of(ROOT / path)

    recomputed = recompute_cap(report)
    checks["cap_recomputes_from_unit_costs"] = all(
        abs(recomputed[key] - report["plan"][key]) < 0.01
        for key in (
            "per_seed_training",
            "per_seed_evaluation",
            "per_seed_feedback",
            "per_seed_control_arm",
            "per_seed_recovery",
            "per_seed_total",
            "estimate_cpu",
            "contingency_cpu",
            "cap_cpu",
        )
    )
    checks["cap_within_design_cap"] = report["plan"]["cap_cpu"] <= float(
        contract["plan"]["design_cap_cpu"]
    )
    checks["budget_uses_the_expensive_round"] = (
        report["plan"]["per_seed_feedback"]
        >= int(contract["plan"]["feedback_error_allowance"])
        * report["unit_cpu"]["byte_round_per_example_untrained"]
    )
    checks["budget_uses_the_expensive_round_strictly"] = (
        abs(
            report["plan"]["per_seed_feedback"]
            - int(contract["plan"]["feedback_error_allowance"])
            * (
                report["unit_cpu"]["single_error_correction_per_error"]
                + report["unit_cpu"]["byte_round_per_example_untrained"]
            )
        )
        < 0.01
    )
    checks["trained_and_untrained_rounds_are_distinguished"] = (
        report["unit_cpu"]["byte_round_per_example_untrained"]
        > report["unit_cpu"]["byte_round_per_example"]
    )
    checks["replay_is_byte_stable"] = report["raw"]["replay_stable"] is True
    checks["no_training_happened"] = (
        report["discipline"]["optimizer_calls"] == 0
        and report["discipline"]["optimizer_steps"] == 0
        and report["discipline"]["training_updates"] == 0
        and report["discipline"]["checkpoints_written"] == 0
        and report["discipline"]["bundles_written"] == 0
    )
    checks["no_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_parent_claimed"] = report["discipline"]["accepted_parent"] is False
    checks["no_sealed_reads"] = (
        report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0
    )

    # The protocol itself, not just the price.
    checks["three_seeds_declared"] = len(contract["plan"]["seeds"]) == 3
    checks["world_pairing_unit_is_the_example"] = "example" in contract["criteria"]["world_pairing"]
    checks["bootstrap_resamples_declared"] = int(contract["plan"]["bootstrap_resamples"]) == 2000
    checks["correction_protocol_is_one_error_at_a_time"] = (
        "one first-pass error" in contract["criteria"]["correction_protocol"]
    )
    checks["feedback_is_an_environment_result"] = (
        contract["criteria"]["feedback_carrier"] == "released record (environment result)"
        and contract["criteria"]["teacher_forcing_forbidden"] is True
    )
    checks["a_retention_declared_not_applicable"] = (
        contract["criteria"]["a_retention"] == "not_applicable_no_A_parent"
    )
    checks["accepted_parent_is_false"] = contract["criteria"]["accepted_parent"] is False
    checks["gate_thresholds_are_the_2251_ones"] = (
        contract["criteria"]["main_gate"] == 0.8
        and contract["criteria"]["correction_gate"] == 0.8
        and contract["criteria"]["feedback_gain_gate"] == 0.05
    )

    # The feasibility probe is the card's evidence base.
    findings = probe["findings"]
    checks["probe_ran_without_training"] = (
        probe["discipline"]["optimizer_steps"] == 0
        and probe["discipline"]["checkpoints_written"] == 0
    )
    checks["records_channel_is_live"] = (
        findings["F2_fact_to_byte_routes"]["deltas"]["record_added"] > 1.0
    )
    checks["records_channel_survives_the_read_code_hook"] = (
        findings["F2_fact_to_byte_routes"]["deltas"]["record_added_under_read_code_hook"] > 1.0
    )
    checks["fast_path_is_recorded_as_cut"] = (
        findings["F2_fact_to_byte_routes"]["deltas"]["fast_memory_perturbed_under_read_code_hook"]
        == 0.0
    )
    checks["illegal_release_is_refused"] = bool(
        findings["F1_environment_result_channel"].get("set_records_after_utterance_refused")
    )
    return checks


def drills(contract: dict, report: dict, probe: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}

    def fires(mutated_report: dict, key: str) -> bool:
        return not evaluate(contract, mutated_report, probe)[key]

    mutated = json.loads(json.dumps(report))
    mutated["plan"]["cap_cpu"] = 10.0
    out["shrink_the_cap"] = fires(mutated, "cap_recomputes_from_unit_costs")

    mutated = json.loads(json.dumps(report))
    mutated["unit_cpu"]["train_step_proxy_per_batch"] *= 2
    out["inflate_a_unit_cost"] = fires(mutated, "cap_recomputes_from_unit_costs")

    mutated = json.loads(json.dumps(report))
    # The red direction is not "change a unit cost" - the cap would simply be
    # recomputed from it and stay self-consistent.  It is "keep the measured
    # units and shrink the cap anyway": the budget then claims less than the
    # measurements support.
    mutated["plan"]["cap_cpu"] = round(
        mutated["plan"]["per_seed_total"] * int(mutated["plan"]["seeds"]) * 0.5, 3
    )
    out["understate_the_expensive_round"] = fires(mutated, "cap_recomputes_from_unit_costs")

    mutated = json.loads(json.dumps(report))
    # A second, sharper red direction: price the feedback round with the cheap
    # round instead of the expensive one.
    mutated["plan"]["per_seed_feedback"] = round(
        int(contract["plan"]["feedback_error_allowance"])
        * report["unit_cpu"]["single_error_correction_per_error"],
        3,
    )
    out["price_the_feedback_round_with_the_cheap_round"] = fires(
        mutated, "budget_uses_the_expensive_round_strictly"
    )

    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["optimizer_steps"] = 600
    out["hide_an_optimizer_step"] = fires(mutated, "no_training_happened")

    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = fires(mutated, "no_capability_claims")

    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["accepted_parent"] = True
    out["claim_a_parent"] = fires(mutated, "no_parent_claimed")

    mutated = json.loads(json.dumps(report))
    mutated["plan"]["seeds"] = 1
    out["drop_to_one_seed"] = fires(mutated, "cap_recomputes_from_unit_costs")

    mutated = json.loads(json.dumps(report))
    mutated["raw"]["replay_stable"] = False
    out["hide_an_unstable_replay"] = fires(mutated, "replay_is_byte_stable")

    mutated_probe = json.loads(json.dumps(probe))
    mutated_probe["findings"]["F2_fact_to_byte_routes"]["deltas"][
        "record_added_under_read_code_hook"
    ] = 0.0
    out["claim_the_records_channel_is_cut"] = not evaluate(contract, report, mutated_probe)[
        "records_channel_survives_the_read_code_hook"
    ]

    mutated_contract = json.loads(json.dumps(contract))
    mutated_contract["criteria"]["a_retention"] = "passed"
    out["claim_a_retention_that_has_no_parent"] = not evaluate(mutated_contract, report, probe)[
        "a_retention_declared_not_applicable"
    ]

    mutated_contract = json.loads(json.dumps(contract))
    mutated_contract["criteria"]["teacher_forcing_forbidden"] = False
    out["restore_the_byte_supervision_protocol"] = not evaluate(mutated_contract, report, probe)[
        "feedback_is_an_environment_result"
    ]

    return out


def main() -> int:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
    checks = evaluate(contract, report, probe)
    fired = drills(contract, report, probe)
    passed = all(checks.values()) and all(fired.values())
    receipt = {
        "schema": "taiji-u2-a5-protocol-guard-v1",
        "card": "22.5.A.5",
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
            {
                "verdict": receipt["verdict"],
                "failed": receipt["failed"],
                "drills_not_firing": receipt["drills_not_firing"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
