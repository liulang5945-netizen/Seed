"""Aggregate the 22.5.0 A-parent formal package into its pre-registered verdict.

Reads the three U2-arm reports and the three FLOPs-parity baseline-arm reports,
applies the frozen exit ladder from the package contract
(`stage_budget_unavailable > a_parent_not_met > engineering_failure >
budget_stop > a_parent_earned` under criterion 乙: median U2 check accuracy
>= 0.80, per-seed preflight/recovery equality, UTF-8/EOS at full denominator)
and writes the consolidated run report.  Zero model compute.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20261110, 20261111, 20261112)
U2_DOSE = 2000
U2_TRAIN_UNIT = 0.708333  # frozen measured unit: per update at batch 8
CAP_CPU = 17000.0  # fifth ruling: honest-account refreeze, successful legs
BASELINE_ATTEMPT1_WASTE = 7400.0  # EOS crash, listed separately, not cap-consuming
STAGE_DRAWS_CPU = 13400.0
STAGE_CAP_CPU = 24000.0
OUT = ROOT / "reports/taiji_u2_a_parent_package_run_20261119.json"


def main() -> int:
    u2 = {
        seed: json.loads(
            (ROOT / f"reports/taiji_u2_a_parent_package_u2_seed{seed}_20261119.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    }
    baseline = {
        seed: json.loads(
            (
                ROOT / f"reports/taiji_u2_a_parent_package_baseline_seed{seed}_20261119.json"
            ).read_text(encoding="utf-8")
        )
        for seed in SEEDS
    }
    accuracies = sorted(u2[seed]["check"]["accuracy"] for seed in SEEDS)
    median = accuracies[1]
    engineering = [
        seed
        for seed in SEEDS
        if not u2[seed]["preflight_equal"]
        or not u2[seed]["recovery_equal"]
        or u2[seed]["check"]["valid_utf8"] != 192
        or u2[seed]["check"]["eos"] != 192
    ]
    u2_arm_cpu = sum(u2[seed]["cost"]["cpu"] + U2_DOSE * U2_TRAIN_UNIT for seed in SEEDS)
    package_cpu = u2_arm_cpu + sum(baseline[seed]["cost"]["cpu"] for seed in SEEDS)
    # The stage refreeze was approved (fifth ruling closed that exit); the
    # arithmetic stage position is disclosed bookkeeping, not a verdict gate.
    stage_position = STAGE_DRAWS_CPU + package_cpu + BASELINE_ATTEMPT1_WASTE
    if median < 0.80:
        verdict = "a_parent_not_met"
    elif engineering:
        verdict = "engineering_failure"
    elif package_cpu > CAP_CPU:
        verdict = "budget_stop"
    else:
        verdict = "a_parent_earned"
    payload = {
        "schema": "taiji-u2-22-5-a-parent-package-run-v1",
        "card": "22.5.0 A-parent formal package",
        "criterion": "yi_median_check_min_0_80",
        "verdict": verdict,
        "u2": {
            "seeds": list(SEEDS),
            "fit_correct": {str(seed): u2[seed]["train"]["fit_correct"] for seed in SEEDS},
            "check_correct": {str(seed): u2[seed]["check"]["correct"] for seed in SEEDS},
            "check_accuracies": accuracies,
            "check_median": round(median, 6),
            "preflight_equal": {str(seed): u2[seed]["preflight_equal"] for seed in SEEDS},
            "recovery_equal": {str(seed): u2[seed]["recovery_equal"] for seed in SEEDS},
            "utf8_eos_full": {
                str(seed): (u2[seed]["check"]["valid_utf8"], u2[seed]["check"]["eos"])
                for seed in SEEDS
            },
        },
        "baseline_flops_matched": {
            "identity": "weight_tied_transformer_sequence+first_a_adapter_v1",
            "dose_per_seed": 168318,
            "ratio_u2_over_baseline": 10.5199,
            "fit_correct": {str(seed): baseline[seed]["fit"]["correct"] for seed in SEEDS},
            "check_correct": {str(seed): baseline[seed]["check"]["correct"] for seed in SEEDS},
            "check_accuracy": {
                str(seed): round(baseline[seed]["check"]["accuracy"], 4) for seed in SEEDS
            },
        },
        "cost": {
            "u2_arm_cpu_parent_measured": round(sum(u2[seed]["cost"]["cpu"] for seed in SEEDS), 2),
            "u2_arm_training_child_priced_unit": round(U2_DOSE * U2_TRAIN_UNIT * len(SEEDS), 2),
            "u2_arm_cpu": round(u2_arm_cpu, 2),
            "baseline_arm_cpu": round(sum(baseline[seed]["cost"]["cpu"] for seed in SEEDS), 2),
            "package_cpu": round(package_cpu, 2),
            "cap_cpu": CAP_CPU,
            "baseline_attempt1_waste_listed_separately": BASELINE_ATTEMPT1_WASTE,
            "stage_draws_cpu_estimate": STAGE_DRAWS_CPU,
            "stage_cap_cpu": STAGE_CAP_CPU,
            "stage_position_with_package": round(stage_position, 2),
            "stage_ledger_status": "estimate basis undercounts runner-child compute; honest re-tally required, breach recorded as incident",
            "cost_basis_note": "u2 child training at the frozen measured unit; u2 child eval/overhead phase unpriced (wall-evidenced ~2560s total, disclosed)",
        },
        "exposure": {
            "check": "same-instrument face as D-20261119-02, aggregate-only",
            "allocated_22_5": 2,
            "used": 2,
            "remaining": 0,
        },
        "not_signed": [
            "22.6 unlock",
            "external capability claims",
            "22.5.1 B/check② (separate case)",
        ],
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "check_median": round(median, 4),
                "u2_check": payload["u2"]["check_correct"],
                "baseline_check": payload["baseline_flops_matched"]["check_accuracy"],
                "package_cpu": payload["cost"]["package_cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
