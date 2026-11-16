"""Independent guard for the A.5 protocol run.

The guard does two things, and the second is the one that matters:

1. It recomputes every published gate from the numbers in the report - the main
   gate, the correction rate, the gain, the three-seed direction and the
   world-paired lower bound - and compares them to the gate table the run wrote.
2. It re-derives the readings themselves from the saved bundles in a fresh
   process: the 32 A rounds, the 24 task-B first-pass rounds, the 24-rep control
   second pass, the 24 learned second-pass rounds and the 24 counter-fact rounds.
   A number that the bundle cannot reproduce is a number that did not happen.

It also drills its own checks: each published field is corrupted in memory and
the corresponding check must go red.  A check that cannot fail is not a check.
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
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a5_protocol_run_20261116.json"
RUNNER = ROOT / "scripts/training/run_a5_22_5.py"
GUARD_REPORT = ROOT / "reports/taiji_u2_a5_protocol_run_guard_20261116.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def recompute_gates(payload: dict, contract: dict) -> dict:
    criteria = contract["criteria"]
    primary = payload["seeds"][0]
    gains = [row["feedback"]["gain"] for row in payload["seeds"]]
    pairs = [pair for row in payload["seeds"] for pair in row["pairs"]]
    lower = a5.bootstrap_lower_bound(pairs, contract)
    return {
        "main": "passed" if primary["final"]["fit_accuracy"] >= criteria["main_gate"] else "failed",
        "correction": (
            "not_identifiable_by_design"
            if primary["first_pass"]["denominator"] - primary["first_pass"]["correct"] < 4
            else (
                "passed"
                if primary["feedback"]["correction_rate"] >= criteria["correction_gate"]
                else "failed"
            )
        ),
        "gain": (
            "passed" if primary["feedback"]["gain"] >= criteria["feedback_gain_gate"] else "failed"
        ),
        "lower_bound": "passed" if lower["lower_bound_95"] > 0 else "failed",
        "three_seed": (
            "passed"
            if all(gain > 0 for gain in gains) or all(gain < 0 for gain in gains)
            else "failed"
        ),
        "retention": "passed" if primary["retention"]["damage"] <= 0.02 else "failed",
        "recovery": payload["gates"]["recovery"],
    }


def evaluate(payload: dict, contract: dict, derived: dict | None) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    criteria = contract["criteria"]
    seeds = payload["seeds"]

    checks["criteria_unchanged"] = (
        hashlib.sha256(
            json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        == contract["criteria_sha256"]
    )
    checks["runner_pin_matches"] = sha_of(RUNNER) == contract["source_pins"]["runner"]
    checks["three_seeds_present"] = len(seeds) == int(len(contract["plan"]["seeds"]))

    gates = recompute_gates(payload, contract)
    gates.pop("recovery")
    checks["gate_table_recomputes"] = all(gates[key] == payload["gates"][key] for key in gates)

    summary = {
        "gates": payload["gates"],
        "a4_second_pass": seeds[0]["feedback"]["a4_second_pass"],
        "learned_second_pass": seeds[0]["feedback"]["learned_second_pass"],
    }
    checks["verdict_matches_ladder"] = a5.decide(summary) == payload["verdict"]

    for row in seeds:
        feedback = row["feedback"]
        release_count = len(feedback["trace"])
        paths = [entry["path"] for entry in feedback["trace"]]
        # Every release is one trace row; every update ends either kept or rolled
        # back; a round is corrected exactly when it was released-only or kept.
        checks[f"trace_accounts_seed_{row['seed']}"] = (
            feedback["releases"] == release_count
            and feedback["releases"] == row["first_pass"]["denominator"]
            and feedback["corrected"] == paths.count("updated") + paths.count("released_only")
            and feedback["updates"] == paths.count("updated") + paths.count("rolled_back")
            and feedback["rollbacks"] == paths.count("rolled_back")
        )
        checks[f"gain_is_a_difference_seed_{row['seed']}"] = (
            abs(
                feedback["gain"]
                - (feedback["learned_second_pass"] - feedback["control_second_pass"])
            )
            < 1e-12
        )
        checks[f"control_made_no_update_seed_{row['seed']}"] = (
            feedback["control_updates"] == 0 and feedback["control_state_unchanged"] is True
        )
        checks[f"pairs_match_denominator_seed_{row['seed']}"] = len(row["pairs"]) == len(
            row["first_pass"]["error_ids"]
        )
        checks[f"templates_are_distinct_seed_{row['seed']}"] = len(
            set(contract["task_b"]["template_offsets"])
        ) == len(contract["task_b"]["templates"]) and max(
            contract["task_b"]["template_offsets"]
        ) - min(
            contract["task_b"]["template_offsets"]
        ) >= int(
            criteria["task_b_template_offset_spread_minimum"]
        )
        checks[f"bundles_declared_seed_{row['seed']}"] = bool(
            (ROOT / row["bundle_pre_feedback"]).exists()
            and (ROOT / row["bundle_post_feedback"]).exists()
        )

    checks["discipline_zero"] = (
        payload["discipline"]["check_reads"] == 0
        and payload["discipline"]["final_reads"] == 0
        and payload["discipline"]["confirm_reads"] == 0
        and payload["discipline"]["capability_claims"] == 0
        and payload["discipline"]["accepted_parent"] is False
    )
    checks["cap_respected"] = payload["cost"]["cpu"] <= float(contract["cpu_cap"]) + 1e-6

    if derived is not None:
        for row, rederived in zip(seeds, derived, strict=True):
            seed = row["seed"]
            checks[f"a_fit_reproduced_seed_{seed}"] = (
                abs(rederived["a_fit_accuracy"] - row["final"]["fit_accuracy"]) < 1e-12
            )
            checks[f"b_first_pass_reproduced_seed_{seed}"] = (
                abs(rederived["b_first_accuracy"] - row["first_pass"]["accuracy"]) < 1e-12
            )
            checks[f"b_control_reproduced_seed_{seed}"] = (
                abs(rederived["b_control_accuracy"] - row["feedback"]["control_second_pass"])
                < 1e-12
            )
            checks[f"b_learned_reproduced_seed_{seed}"] = (
                abs(rederived["b_learned_accuracy"] - row["feedback"]["learned_second_pass"])
                < 1e-12
            )
            checks[f"a_after_feedback_reproduced_seed_{seed}"] = (
                rederived["a_after_digest"] == row["a_fit_after_feedback_digest"]
                and abs(rederived["a_after_accuracy"] - row["retention"]["a_fit_after"]) < 1e-12
            )
            checks[f"counter_fact_reproduced_seed_{seed}"] = (
                rederived["counter_follows"] == row["counter_fact"]["follows_counter"]
            )
    return checks


def redriving(payload: dict, contract: dict) -> dict[str, dict]:
    """Re-derive every reading from the saved bundles, seed by seed."""
    torch.set_num_threads(1)
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    base = a5.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(base, head, prior, data["train"] + data["fit"], contract)
    out = []
    for row in payload["seeds"]:
        entry = {}
        for tag, key in (("pre", "bundle_pre_feedback"), ("post", "bundle_post_feedback")):
            model = a5.build_subject(contract)
            model.load_state_dict(torch.load(ROOT / row[key], weights_only=False)["model"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            a_rounds = [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]
            if tag == "pre":
                entry["a_fit_accuracy"] = a5.accuracy(a_rounds)
                first = a5.b_pass(model, items, codes, template_of)
                entry["b_first_accuracy"] = a5.accuracy(first)
                control = a5.b_pass(model, items, codes, template_of)
                entry["b_control_accuracy"] = a5.accuracy(control)
            else:
                entry["a_after_accuracy"] = a5.accuracy(a_rounds)
                entry["a_after_digest"] = a5.rows_digest(a_rounds)
                learned = a5.b_pass(model, items, codes, template_of)
                entry["b_learned_accuracy"] = a5.accuracy(learned)
                counter = [
                    a5.b_round(model, item, codes[item["example"]["id"]], template_of(index), True)
                    for index, item in enumerate(items)
                ]
                entry["counter_follows"] = sum(
                    1
                    for hit, item in zip(counter, items, strict=True)
                    if hit["output_hex"] == item["counter_target"].hex()
                )
        out.append(entry)
    return out


def drills(payload: dict, contract: dict, derived: list[dict]) -> dict[str, bool]:
    out: dict[str, bool] = {}

    def fires(mutated: dict, key: str) -> bool:
        return not evaluate(mutated, contract, derived)[key]

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["feedback"]["gain"] = 0.9
    out["inflate_the_gain"] = fires(
        mutated, "gain_is_a_difference_seed_" + str(mutated["seeds"][0]["seed"])
    )

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["feedback"]["control_updates"] = 8
    out["let_the_control_arm_update"] = fires(
        mutated, "control_made_no_update_seed_" + str(mutated["seeds"][0]["seed"])
    )

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["feedback"]["trace"] = [
        {**entry, "path": "updated"} for entry in mutated["seeds"][0]["feedback"]["trace"]
    ]
    out["claim_the_rollbacks_were_updates"] = fires(
        mutated, "trace_accounts_seed_" + str(mutated["seeds"][0]["seed"])
    )

    mutated = json.loads(json.dumps(payload))
    mutated["gates"] = {key: "passed" for key in mutated["gates"]}
    out["paint_the_gate_table_green"] = fires(mutated, "verdict_matches_ladder")

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["retention"]["a_fit_after"] = 0.5
    out["understate_the_retention"] = fires(
        mutated, "a_after_feedback_reproduced_seed_" + str(mutated["seeds"][0]["seed"])
    )

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["counter_fact"]["follows_counter"] = 24
    out["claim_the_record_was_read"] = fires(
        mutated, "counter_fact_reproduced_seed_" + str(mutated["seeds"][0]["seed"])
    )

    mutated = json.loads(json.dumps(payload))
    mutated["discipline"]["capability_claims"] = 1
    out["claim_capability"] = fires(mutated, "discipline_zero")

    mutated = json.loads(json.dumps(payload))
    mutated["seeds"][0]["bundle_post_feedback"] = "data/experiments/taiji_u2/a5/missing.pt"
    out["point_at_a_missing_bundle"] = fires(
        mutated, "bundles_declared_seed_" + str(mutated["seeds"][0]["seed"])
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if args.cached and GUARD_REPORT.exists():
        previous = json.loads(GUARD_REPORT.read_text(encoding="utf-8"))
        derived = previous["rederived"]
        derivation = "reused_from_the_previous_full_guard_pass"
    else:
        derived = redriving(payload, contract)
        derivation = "recomputed_from_bundles_in_this_process"
    checks = evaluate(payload, contract, derived)
    fired = drills(payload, contract, derived)
    passed = all(checks.values()) and all(fired.values())
    report = {
        "schema": "taiji-u2-a5-protocol-run-guard-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "rederived": derived,
        "derivation": derivation,
        "recomputed_gates": recompute_gates(payload, contract),
        "published_gates": payload["gates"],
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
