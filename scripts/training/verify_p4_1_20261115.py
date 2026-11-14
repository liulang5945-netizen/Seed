"""Guard for the P4.1 column-supervision receipt.

The producer trained ``encoder.rgb`` with a per-column prototype-distance CE and
published dose readings. This guard reloads every saved bundle into a freshly
seeded body, recomputes the fit, colour, probe and geometry numbers with the same
math the producer used, re-derives the verdict ladder from the frozen criteria on
its own, and refuses to pass a receipt whose numbers disagree, whose cost escapes
its cap, whose baseline check failed, or which claims anything about capability.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "plans/manifests/taiji_u2_p4_1_column_supervision_20261115.json"
REPORT = ROOT / "reports/taiji_u2_p4_1_column_supervision_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_p4_1_guard_20261115.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def ladder(final: dict, criteria: dict) -> str:
    """Re-derived here on purpose: a bug in the producer's decide() must show."""
    if final["geometry_collapse"]:
        return "column_supervision_collapsed"
    if final["fit_accuracy"] >= criteria["fit_gate"]:
        if final["probe_fit"] >= criteria["probe_gate"]:
            return "pathway_unlocked"
        return "task_carries_probe_lags"
    if final["fit_accuracy"] <= criteria["fit_low"]:
        if final["probe_fit"] >= criteria["probe_gate"]:
            return "colour_learned_task_blocked"
        return "encoder_dead_on_this_task"
    return "inconclusive_column_supervision"


def recompute(contract: dict, report: dict) -> dict[int, dict]:
    import torch

    from scripts.training.first_a_support_22_5 import make_model
    from scripts.training.run_p4_1_22_5_arm_c_20261115 import (
        TokenDetector,
        accuracy,
        geometry,
        ridge_probe,
    )
    from taiji.u2_history import atomic_load

    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data_root"]))
    out = {}
    for row in report["doses"]:
        bundle = torch.load(
            Path(contract["run_root"]) / f"bundle-{row['update']:06d}.pt", weights_only=False
        )
        model = make_model("native", int(contract["candidate"]["seed"]))
        unexpected = model.load_state_dict(bundle["encoder_rgb"], strict=False)[1]
        assert not unexpected, unexpected
        state = model.state_dict()
        for name, value in bundle["encoder_rgb"].items():
            assert torch.equal(state[name], value), name
        module = TokenDetector(float(contract["candidate"]["tau"]))
        module.load_state_dict(bundle["module"])
        module.eval()
        train = accuracy(model, module, data["train"])
        fit = accuracy(model, module, data["fit"])
        probe = ridge_probe(model, data)
        out[row["update"]] = {
            "train_accuracy": train["accuracy"],
            "fit_accuracy": fit["accuracy"],
            "colour_accuracy_fit": fit["colour_accuracy"],
            "centroid_column_hits_fit": fit["centroid_column_hits"],
            "centroid_column_total_fit": fit["centroid_column_total"],
            "abstain_on_other_fit": fit["abstain_on_other"],
            "probe_train": round(probe["train"], 6),
            "probe_fit": round(probe["fit"], 6),
            "prototype_pairwise_min": geometry(model)["prototype_pairwise_min"],
        }
    return out


def evaluate(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract["status"] == "frozen"
    criteria = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    checks["criteria_sha256_matches"] = (
        hashlib.sha256(criteria.encode()).hexdigest() == contract["criteria_sha256"]
    )
    pins = contract["source_pins"]
    checks["runner_pin_live"] = pins["runner"] == sha_of(
        ROOT / "scripts/training/run_p4_1_22_5_arm_c_20261115.py"
    )
    checks["support_pin_live"] = pins["support"] == sha_of(
        ROOT / "scripts/training/first_a_support_22_5.py"
    )
    checks["column_pathway_pin_live"] = pins["column_pathway"] == sha_of(
        ROOT / "taiji/u2_observation_columns.py"
    )
    checks["authority_recorded"] = bool(contract["training_authority"])
    checks["dose_set_matches"] = [row["update"] for row in report["doses"]] == contract[
        "dose_points"
    ]
    checks["baseline_check_passed"] = (
        report["baseline_probe"]["fit"] is not None
        and abs(
            report["baseline_probe"]["fit"]
            - contract["criteria"]["baseline_check"]["expected_probe_fit_at_zero"]
        )
        <= contract["criteria"]["baseline_check"]["tolerance"]
    )
    checks["recompute_matches_every_dose"] = all(
        recomputed[row["update"]]["fit_accuracy"] == row["fit_accuracy"]
        and recomputed[row["update"]]["train_accuracy"] == row["train_accuracy"]
        and recomputed[row["update"]]["probe_fit"] == row["probe_fit"]
        and recomputed[row["update"]]["colour_accuracy_fit"] == row["colour_accuracy_fit"]
        and recomputed[row["update"]]["centroid_column_hits_fit"] == row["centroid_column_hits_fit"]
        and recomputed[row["update"]]["prototype_pairwise_min"] == row["prototype_pairwise_min"]
        for row in report["doses"]
    )
    final = report["doses"][-1]
    checks["verdict_matches_ladder"] = report["verdict"] == ladder(final, contract["criteria"])
    checks["no_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_sealed_reads"] = (
        report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0
    )
    checks["cost_within_cap"] = (
        report["cost"]["total_cpu"] <= contract["cpu_cap"]
        and report["cost"]["total_wall"] <= contract["wall_cap"]
    )
    checks["single_variable_honoured"] = (
        report["trainable_parameters"]["encoder_rgb"] == 392
        and report["trainable_parameters"]["module"] == 59
    )
    checks["collapse_override_is_explicit"] = isinstance(final["geometry_collapse"], bool)
    return checks


def drills(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    mutated = json.loads(json.dumps(report))
    mutated["verdict"] = "pathway_unlocked"
    out["claim_the_unlock"] = not evaluate(contract, mutated, recomputed)["verdict_matches_ladder"]
    mutated = json.loads(json.dumps(report))
    mutated["doses"][-1]["probe_fit"] = 0.99
    out["inflate_the_probe"] = not evaluate(contract, mutated, recomputed)[
        "recompute_matches_every_dose"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["doses"][-1]["fit_accuracy"] = 0.9
    out["inflate_the_fit"] = not evaluate(contract, mutated, recomputed)[
        "recompute_matches_every_dose"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["discipline"]["capability_claims"] = 1
    out["inject_capability_claim"] = not evaluate(contract, mutated, recomputed)[
        "no_capability_claims"
    ]
    mutated = json.loads(json.dumps(report))
    mutated["cost"]["total_cpu"] = contract["cpu_cap"] + 1
    out["escape_the_cap"] = not evaluate(contract, mutated, recomputed)["cost_within_cap"]
    mutated = json.loads(json.dumps(report))
    mutated["trainable_parameters"]["encoder_rgb"] = 8192
    out["hide_a_widened_encoder"] = not evaluate(contract, mutated, recomputed)[
        "single_variable_honoured"
    ]
    return out


def main() -> int:
    import time

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    recomputed = recompute(contract, report)
    checks = evaluate(contract, report, recomputed)
    fired = drills(contract, report, recomputed)
    passed = all(checks.values()) and all(fired.values())
    receipt = {
        "schema": "taiji-u2-p4-1-guard-v1",
        "card": "22.5.P4.1",
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
    return 0 if receipt["verdict"] == "guard_passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
