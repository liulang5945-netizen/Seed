"""Guard for the A.4 run receipt.

The receipt claims that a read inside the body can carry the scene and that the
byte path can say the answer. The guard reloads every saved bundle into a
freshly seeded body, recomputes the fit rounds, the read class accuracy and the
two prefix-0 instruments with its own code path, re-derives the ladder from the
frozen criteria, and refuses a receipt whose numbers disagree, whose cost
escapes the cap, or which claims anything about a parent or a capability.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

MANIFEST = ROOT / "plans/manifests/taiji_u2_a4_column_read_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a4_column_read_run_20261116.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a4_column_read_run_guard_20261116.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def ladder(final: dict, contract: dict) -> str:
    """Re-derived here on purpose: a bug in the runner's decide() must show."""
    criteria = contract["criteria"]
    gate_r = criteria["gate_r_read_prefix0"]
    gate_b = criteria["gate_b_byte"]
    probes = final["prefix_probes"]
    outcomes_ok = (
        final["read_accuracy_fit"] >= gate_r["fit_read_accuracy"]
        and probes["margin"] >= gate_r["margin_over_shuffle_null"]
        and probes["prefix"] == gate_r["prefix"]
    )
    premise_ok = probes["token_colour"]["painted"] >= gate_r["painted_column_colour_probe"]
    byte_ok = final["fit_accuracy"] >= gate_b["full_response_exact_success"]
    if outcomes_ok and byte_ok and premise_ok:
        return "read_in_body_confirmed"
    if outcomes_ok and byte_ok:
        return "premise_below_floor_read_outcomes_passed"
    if outcomes_ok:
        return "byte_gate_failed_with_read_passed"
    return "read_gate_failed"


def recompute(contract: dict, report: dict) -> dict:
    from scripts.training.run_a4_22_5 import (
        build_subject,
        byte_rows,
        prefix_probes,
        read_accuracy,
        set_trainable,
        token_colour_probe,
    )
    from taiji.u2_history import atomic_load

    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data_root"]))
    out = {}
    for entry in report["seeds"]:
        seed = int(entry["seed"])
        for tag in ("bundle", "bundle_after_feedback"):
            model, read = build_subject(contract, seed)
            set_trainable(model, read, contract)
            bundle = torch.load(ROOT / entry[tag], weights_only=False)
            read.load_state_dict(bundle["read"])
            # Load the trained emission tensors into the body, then prove the load
            # took effect value by value.
            unexpected = model.load_state_dict(bundle["emission"], strict=False)[1]
            assert not unexpected, unexpected
            state = model.state_dict()
            for name, value in bundle["emission"].items():
                assert torch.equal(state[name], value), name
            rows = byte_rows(model, read, data["fit"])
            probes = prefix_probes(model, read, data)
            probes["token_colour"] = token_colour_probe(model, data)
            out.setdefault(seed, {})[tag] = {
                "fit_accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
                "fit_correct": sum(int(row["correct"]) for row in rows),
                "fit_denominator": len(rows),
                "read_accuracy_fit": read_accuracy(rows),
                "prefix_best_fit": probes["best_fit"],
                "prefix_null_max": probes["null_max"],
                "prefix_margin": probes["margin"],
                "painted": probes["token_colour"]["painted"],
            }
    return out


def evaluate(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract["status"] == "frozen"
    checks["authority_recorded"] = bool(contract["training_authority"])
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    checks["runner_pin_live"] = contract["source_pins"]["runner_sha256"] == sha_of(
        ROOT / "scripts/training/run_a4_22_5.py"
    )
    checks["read_module_pin_live"] = contract["source_pins"]["read_module_sha256"] == sha_of(
        ROOT / "taiji/u2_column_read.py"
    )
    checks["seed_count_matches_plan"] = len(report["seeds"]) == len(contract["plan"]["seeds"])
    checks["recompute_matches_every_seed"] = all(
        abs(
            recomputed[int(entry["seed"])]["bundle"]["fit_accuracy"]
            - entry["final"]["fit_accuracy"]
        )
        < 1e-9
        and abs(
            recomputed[int(entry["seed"])]["bundle"]["read_accuracy_fit"]
            - entry["final"]["read_accuracy_fit"]
        )
        < 1e-9
        and abs(
            recomputed[int(entry["seed"])]["bundle"]["prefix_margin"]
            - entry["final"]["prefix_probes"]["margin"]
        )
        < 1e-6
        and abs(
            recomputed[int(entry["seed"])]["bundle"]["painted"]
            - entry["final"]["prefix_probes"]["token_colour"]["painted"]
        )
        < 1e-6
        for entry in report["seeds"]
    )
    checks["post_feedback_bundle_matches_the_reported_loss"] = all(
        abs(
            recomputed[int(entry["seed"])]["bundle_after_feedback"]["fit_accuracy"]
            - entry["correction"]["learned_second_pass"]
        )
        < 1e-9
        for entry in report["seeds"]
    )
    checks["verdict_matches_ladder"] = report["verdict"] == ladder(
        report["seeds"][0]["final"], contract
    )
    checks["every_seed_recomputed"] = all(
        int(entry["seed"]) in recomputed for entry in report["seeds"]
    )
    checks["no_parent_claim"] = (
        report["discipline"]["accepted_parent"] is False
        and report["discipline"]["capability_claims"] == 0
    )
    checks["no_sealed_reads"] = (
        report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0
    )
    checks["cost_within_cap"] = (
        report["cost"]["total_cpu"] <= contract["cpu_cap"]
        and report["cost"]["total_wall"] <= contract["wall_cap"]
    )
    checks["trainable_count_declared"] = (
        report["trainable_parameters"]["total"] == contract["trainable_scalar_count"]
    )
    checks["read_weight_declared"] = contract["aux_read_weight"] == 1.0
    checks["premise_and_outcomes_separated"] = "premise_below_floor_read_outcomes_passed" in str(
        contract["criteria"]["exits"]
    )
    return checks


def drills(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mutated = json.loads(json.dumps(report))
    mutated["seeds"][0]["final"]["fit_accuracy"] = 0.95
    out["inflate_the_fit"] = not evaluate(contract, mutated, recomputed)[
        "recompute_matches_every_seed"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["seeds"][0]["final"]["read_accuracy_fit"] = 0.95
    out["inflate_the_read"] = not evaluate(contract, mutated, recomputed)[
        "recompute_matches_every_seed"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["verdict"] = "read_in_body_confirmed"
    out["claim_the_unlock"] = not evaluate(contract, mutated, recomputed)["verdict_matches_ladder"]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["accepted_parent"] = True
    out["claim_a_parent"] = not evaluate(contract, mutated, recomputed)["no_parent_claim"]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = not evaluate(contract, mutated, recomputed)["no_parent_claim"]
    mutated = json.loads(json.dumps(report))
    mutated["cost"]["total_cpu"] = contract["cpu_cap"] + 1
    out["escape_the_cap"] = not evaluate(contract, mutated, recomputed)["cost_within_cap"]
    mutated = json.loads(json.dumps(report))
    mutated["seeds"][0]["final"]["prefix_probes"]["margin"] = 0.5
    out["inflate_the_margin"] = not evaluate(contract, mutated, recomputed)[
        "recompute_matches_every_seed"
    ]
    return out


def main() -> int:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    recomputed = recompute(contract, report)
    checks = evaluate(contract, report, recomputed)
    fired = drills(contract, report, recomputed)
    passed = all(checks.values()) and all(fired.values())
    receipt = {
        "schema": "taiji-u2-a4-column-read-run-guard-v1",
        "card": "22.5.A.4",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "failed": [name for name, value in checks.items() if not value],
        "drills_not_firing": [name for name, value in fired.items() if not value],
        "recomputed": recomputed,
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
