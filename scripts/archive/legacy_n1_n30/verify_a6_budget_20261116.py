"""Independent guard for the A.6 budget curve.

Three things are checked, and the third is the one that makes the others mean
something:

1. the baseline is re-derived from the start bundle - the first pass must still be
   0 of 24 and the A task 32 of 32 - so the arms start where the card says;
2. every arm's arithmetic is checked against the frozen rules: ``updates`` must
   equal ``budget`` times the error count, the damage must equal
   ``baseline_a - a_fit``, and the verdict must be the one the frozen priority
   order produces;
3. one arm (k=1, the cheapest) is re-run end to end and must reproduce its stored
   row exactly.  The other two arms are not re-derived because re-running them
   costs more than the card has left; that limitation is written into the guard
   report rather than hidden.

Drills corrupt each published field in memory; the matching check must go red.
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

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_a6_budget_20261116 as a6
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_a6_budget_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a6_budget_run_20261116.json"
RUNNER = ROOT / "scripts/training/run_a6_budget_20261116.py"
GUARD_REPORT = ROOT / "reports/taiji_u2_a6_budget_run_guard_20261116.json"
SPOT_BUDGET = 1


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def verdict_of(payload: dict, contract: dict) -> str:
    arms = payload["arms"]
    rates = [entry["second_pass_accuracy"] for entry in arms]
    retention_gate = float(contract["criteria"]["retention_gate"])
    gate = float(contract["criteria"]["correction_gate"])
    zero_band = float(contract["criteria"]["zero_band"])
    broken = [entry["budget"] for entry in arms if entry["damage"] > retention_gate]
    if broken:
        return "retention_broken_by_budget"
    monotone = all(
        later >= earlier - 1e-12 for earlier, later in zip(rates, rates[1:], strict=False)
    )
    endpoint = rates[-1] - rates[0]
    if (
        max(rates) >= gate
        and monotone
        and endpoint >= 2 / payload["denominator"]["task_b_examples"]
    ):
        return "correction_scales_with_budget"
    if max(rates) <= zero_band:
        return "correction_zero_at_every_budget"
    return "correction_plateaus_below_gate"


def evaluate(payload: dict, contract: dict, baseline: dict, spot: dict | None) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    criteria = contract["criteria"]
    arms = payload["arms"]
    errors = payload["denominator"]["first_pass_errors"]

    checks["criteria_unchanged"] = (
        hashlib.sha256(
            json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        == contract["criteria_sha256"]
    )
    checks["runner_pin_matches"] = sha_of(RUNNER) == contract["source_pins"]["runner"]
    checks["baseline_first_pass_is_zero"] = abs(payload["first_pass_accuracy"]) < 1e-12
    checks["baseline_a_fit_is_one"] = abs(payload["baseline_a_fit"] - 1.0) < 1e-12
    checks["budgets_measured_are_declared"] = all(
        entry["budget"] in payload["budgets_declared"] for entry in arms
    )
    checks["stopped_budgets_are_reported"] = set(payload["budgets_declared"]) - set(
        payload["budgets_measured"]
    ) == set(payload["budgets_not_measured"])
    for entry in arms:
        checks[f"updates_match_budget_k{entry['budget']}"] = (
            entry["updates"] == entry["budget"] * errors
        )
        checks[f"damage_is_a_difference_k{entry['budget']}"] = (
            abs(entry["damage"] - (payload["baseline_a_fit"] - entry["a_fit_accuracy"])) < 1e-12
        )
        checks[f"correct_matches_rate_k{entry['budget']}"] = (
            abs(entry["second_pass_accuracy"] - entry["correct"] / errors) < 1e-12
        )
    checks["verdict_matches_frozen_order"] = verdict_of(payload, contract) == payload["verdict"]
    checks["retention_list_matches_gate"] = [
        entry["budget"] for entry in arms if entry["damage"] > float(criteria["retention_gate"])
    ] == payload["retention_broken_at"]
    checks["discipline_zero"] = (
        payload["discipline"]["capability_claims"] == 0
        and payload["discipline"]["accepted_parent"] is False
    )
    if baseline is not None:
        checks["start_bundle_reproduces_baseline"] = (
            baseline["first_pass_accuracy"] == payload["first_pass_accuracy"]
            and baseline["a_fit_accuracy"] == payload["baseline_a_fit"]
        )
    if spot is not None:
        stored = next(entry for entry in arms if entry["budget"] == SPOT_BUDGET)
        checks["spot_arm_reproduces"] = (
            spot["correct"] == stored["correct"]
            and spot["second_pass_accuracy"] == stored["second_pass_accuracy"]
            and spot["a_fit_accuracy"] == stored["a_fit_accuracy"]
            and spot["a_fit_digest"] == stored["a_fit_digest"]
            and abs(spot["parameter_abs_sum"] - stored["parameter_abs_sum"]) < 1e-6
        )
    return checks


def drills(payload: dict, contract: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}

    def fires(mutated: dict, key: str) -> bool:
        return not evaluate(mutated, contract, None, None)[key]

    mutated = json.loads(json.dumps(payload))
    mutated["arms"][1]["damage"] = 0.0
    out["hide_the_retention_damage"] = fires(mutated, "damage_is_a_difference_k2")

    mutated = json.loads(json.dumps(payload))
    mutated["arms"][1]["damage"] = 0.0
    mutated["verdict"] = "correction_zero_at_every_budget"
    out["drop_the_retention_exit"] = fires(mutated, "verdict_matches_frozen_order")

    mutated = json.loads(json.dumps(payload))
    mutated["arms"][0]["updates"] = 1
    out["undercharge_the_budget"] = fires(mutated, "updates_match_budget_k1")

    mutated = json.loads(json.dumps(payload))
    mutated["arms"][2]["correct"] = 8
    out["claim_the_arm_learned"] = fires(mutated, "correct_matches_rate_k4")

    mutated = json.loads(json.dumps(payload))
    mutated["budgets_not_measured"] = []
    out["hide_the_missing_budget"] = fires(mutated, "stopped_budgets_are_reported")

    mutated = json.loads(json.dumps(payload))
    mutated["first_pass_accuracy"] = 1.0
    out["claim_the_first_pass_was_right"] = fires(mutated, "baseline_first_pass_is_zero")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spot", action="store_true")
    args = parser.parse_args()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    seed = int(contract["plan"]["seeds"][0])
    start = a6.load_start(contract, seed)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(start, head, prior, data["train"] + data["fit"], contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    first = a5.b_pass(start, items, codes, template_of)
    baseline = {
        "first_pass_accuracy": a5.accuracy(first),
        "a_fit_accuracy": a5.accuracy(
            [a5.a_round(start, example, codes[example["id"]]) for example in data["fit"]]
        ),
    }
    errors = [items[index] for index, row in enumerate(first) if not row["correct"]]
    error_templates = [template_of(index) for index, row in enumerate(first) if not row["correct"]]
    spot = (
        None
        if args.skip_spot
        else a6.arm(
            contract,
            seed,
            SPOT_BUDGET,
            codes,
            errors,
            error_templates,
            data,
            lambda: None,
        )
    )
    checks = evaluate(payload, contract, baseline, spot)
    fired = drills(payload, contract)
    passed = all(checks.values()) and all(fired.values())
    report = {
        "schema": "taiji-u2-a6-budget-run-guard-v1",
        "card": "22.5.A.6",
        "date": "2026-11-16",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "baseline_rederived": baseline,
        "spot_arm": (
            None
            if spot is None
            else {
                "budget": SPOT_BUDGET,
                "correct": spot["correct"],
                "second_pass_accuracy": spot["second_pass_accuracy"],
                "a_fit_accuracy": spot["a_fit_accuracy"],
                "a_fit_digest_matches": spot["a_fit_digest"],
            }
        ),
        "limitation": (
            "only the k=1 arm is re-derived from the start bundle. k=2 and k=4 are checked "
            "arithmetically but not re-run, because re-running them costs more than the card has "
            "left. Their rows are therefore trusted as written and flagged as such"
        ),
        "published_verdict": payload["verdict"],
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
    }
    GUARD_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "checks": f"{sum(checks.values())}/{len(checks)}",
                "drills": f"{sum(fired.values())}/{len(fired)}",
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
