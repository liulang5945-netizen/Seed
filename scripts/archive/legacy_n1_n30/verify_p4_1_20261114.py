"""Guard for P4.1: recompute both arms from their checkpoints.

The card's formal exit is ``encoder_cannot_carry_colour``, fired by a branch that
assumes a linear probe at 0.95 is necessary for the task. Arm B reaches fit 0.906
with the probe at 0.608, so the branch contradicts its own reading. This guard
therefore checks the reading and the rule separately: it recomputes the numbers
from the saved checkpoints, re-applies the frozen rule, and asserts the
contradiction explicitly rather than letting either side hide it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.first_a_support_22_5 import make_model
from scripts.training.run_p4_1_22_5 import (
    CONTRACT,
    TokenDetector,
    accuracy,
    probe,
)
from taiji.u2_history import atomic_load

RUN_REPORT = ROOT / "reports/taiji_u2_p4_1_column_supervision_20261114.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_p4_1_guard_20261114.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def load_arm(contract: dict, arm: str, update: int):
    checkpoint = torch.load(
        Path(contract["run_root"]) / f"arm{arm}-{update:06d}.pt", weights_only=True
    )
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.encoder["rgb"].load_state_dict(checkpoint["encoder"])
    module = TokenDetector(model, contract["candidate"]["tau"])
    module.load_state_dict(checkpoint["module"])
    model.eval()
    module.eval()
    return model, module


def rule(probe_painted: float, fit: float, criteria: dict) -> str:
    """The frozen ladder, copied so the guard does not depend on the runner."""
    if probe_painted >= criteria["probe_gate"] and fit >= criteria["fit_gate"]:
        return "column_supervision_makes_parent_reachable"
    if probe_painted >= criteria["probe_gate"] and fit <= criteria["fit_low"]:
        return "colour_in_tokens_readout_still_blocks"
    if probe_painted < criteria["probe_gate"]:
        return "encoder_cannot_carry_colour"
    return "inconclusive_column_supervision"


def evaluate(contract: dict, report: dict, data, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen_and_authorized"] = contract["status"] == "frozen" and bool(
        contract["training_authority"]
    )
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    checks["runner_pin_live"] = contract["source_pins"]["runner"] == sha_of(
        ROOT / "scripts/training/run_p4_1_22_5.py"
    )
    checks["report_carries_the_frozen_criteria"] = (
        report["criteria_sha256"] == contract["criteria_sha256"]
        and report["criteria"] == contract["criteria"]
    )
    checks["two_arms_three_doses"] = set(report["arms"]) == {"A", "B"} and all(
        len(arm["doses"]) == 3 for arm in report["arms"].values()
    )
    checks["arm_a_has_no_scaffold"] = report["arms"]["A"]["scaffold_parameters"] == 0
    checks["arm_b_scaffold_is_63"] = report["arms"]["B"]["scaffold_parameters"] == 63
    checks["arms_share_trainable_count"] = (
        report["arms"]["A"]["trainable_parameters"]
        == report["arms"]["B"]["trainable_parameters"]
        == 451
    )
    checks["zero_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_sealed_reads"] = (
        report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0
    )
    checks["cost_within_cap"] = (
        report["cost"]["total_cpu"] <= contract["cpu_cap"]
        and report["cost"]["total_wall"] <= contract["wall_cap"]
    )

    matches = True
    for arm in ("A", "B"):
        for row in report["arms"][arm]["doses"]:
            mine = recomputed[(arm, row["update"])]
            matches = matches and mine["fit_correct"] == row["fit_correct"]
            matches = matches and mine["train_correct"] == row["train_correct"]
            matches = matches and (abs(mine["probe_painted"] - row["probe_painted_fit"]) < 1e-5)
            matches = matches and (abs(mine["probe_overall"] - row["probe_overall_fit"]) < 1e-5)
            matches = matches and (
                abs(mine["colour_accuracy_fit"] - row["colour_accuracy_fit"]) < 1e-5
            )
    checks["recompute_matches_every_row"] = matches

    final = report["arms"]["B"]["doses"][-1]
    checks["verdict_matches_frozen_rule"] = report["verdict"] == rule(
        final["probe_painted_fit"], final["fit_accuracy"], contract["criteria"]
    )
    checks["arm_b_fit_passes_the_gate"] = final["fit_accuracy"] >= contract["criteria"]["fit_gate"]
    checks["verdict_is_contradicted_by_its_own_reading"] = (
        report["verdict"] == "encoder_cannot_carry_colour"
        and final["fit_accuracy"] >= contract["criteria"]["fit_gate"]
        and not contract["criteria"]["probe_gate"] <= final["probe_painted_fit"]
    )
    checks["arm_b_beats_arm_a_on_fit"] = (
        final["fit_accuracy"] > report["arms"]["A"]["doses"][-1]["fit_accuracy"] + 0.2
    )
    checks["probe_is_not_monotone"] = (
        report["arms"]["B"]["doses"][0]["probe_painted_fit"] > final["probe_painted_fit"]
    )
    checks["baseline_probe_published_with_floors"] = (
        abs(report["baseline_probe"]["fit"]["painted"] - 0.3692) < 5e-3
        and abs(report["baseline_probe"]["fit"]["background"] - 0.7801) < 5e-3
        and abs(report["baseline_probe"]["fit"]["painted_majority"] - 0.2077) < 5e-3
    )
    return checks


def drills(contract: dict, report: dict, data, recomputed: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = not evaluate(contract, mutated, data, recomputed)[
        "zero_capability_claims"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["arms"]["B"]["doses"][-1]["fit_correct"] = 32
    out["inflate_arm_b_fit"] = not evaluate(contract, mutated, data, recomputed)[
        "recompute_matches_every_row"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["verdict"] = "column_supervision_makes_parent_reachable"
    out["relabel_the_verdict"] = not evaluate(contract, mutated, data, recomputed)[
        "verdict_matches_frozen_rule"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["arms"]["B"]["doses"][0]["probe_painted_fit"] = 0.60
    out["flatten_the_probe_dip"] = not evaluate(contract, mutated, data, recomputed)[
        "probe_is_not_monotone"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["arms"]["B"]["scaffold_parameters"] = 0
    out["hide_the_scaffold"] = not evaluate(contract, mutated, data, recomputed)[
        "arm_b_scaffold_is_63"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["baseline_probe"]["fit"]["background"] = 0.676
    out["quote_overall_as_background"] = not evaluate(contract, mutated, data, recomputed)[
        "baseline_probe_published_with_floors"
    ]
    return out


def main() -> int:
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    recomputed: dict = {}
    for arm in ("A", "B"):
        for row in report["arms"][arm]["doses"]:
            model, module = load_arm(contract, arm, row["update"])
            scored = accuracy(model, module, data["fit"])
            trained = accuracy(model, module, data["train"])
            measured = probe(model, data, 1.0)
            recomputed[(arm, row["update"])] = {
                "fit_correct": scored["correct"],
                "train_correct": trained["correct"],
                "colour_accuracy_fit": scored["colour_accuracy"],
                "probe_painted": measured["fit"]["painted"],
                "probe_overall": measured["fit"]["overall"],
            }
    checks = evaluate(contract, report, data, recomputed)
    fired = drills(contract, report, data, recomputed)
    passed = all(checks.values()) and all(fired.values())
    receipt = {
        "schema": "taiji-u2-p4-1-guard-v1",
        "card": "22.5.P4.1",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "failed": [name for name, value in checks.items() if not value],
        "drills_not_firing": [name for name, value in fired.items() if not value],
        "recomputed": {f"{arm}-{update}": values for (arm, update), values in recomputed.items()},
        "note": (
            "the contradiction check is deliberate: the frozen rule labels this card "
            "encoder_cannot_carry_colour while arm B passes the fit gate"
        ),
    }
    GUARD_REPORT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps({"verdict": receipt["verdict"], "failed": receipt["failed"]}, ensure_ascii=False)
    )
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
