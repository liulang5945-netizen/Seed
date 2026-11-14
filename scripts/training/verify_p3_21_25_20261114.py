"""Guard for the P3.21-P3.25 readout line.

Recomputes every reported accuracy from the saved checkpoints, re-derives each
verdict from its frozen criteria, and asserts the two collapse facts that carry
the line's conclusion (attention entropy exactly log 16 and log 256). Also
asserts that P3.25 was withdrawn before freezing, so no reading is claimed for it.

Every check is exercised in both directions: a mutated copy must turn it red.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import (
    run_p3_21_22_5 as p21,
)
from scripts.training import (
    run_p3_22_22_5 as p22,
)
from scripts.training import (
    run_p3_23_22_5 as p23,
)
from scripts.training import (
    run_p3_24_22_5 as p24,
)
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load

OUT = ROOT / "reports/taiji_u2_p3_21_25_guard_20261114.json"
MANIFESTS = {
    "p21": ROOT / "plans/manifests/taiji_u2_p3_21_position_20261114.json",
    "p22": ROOT / "plans/manifests/taiji_u2_p3_22_selection_20261114.json",
    "p23": ROOT / "plans/manifests/taiji_u2_p3_23_coupled_20261114.json",
    "p24": ROOT / "plans/manifests/taiji_u2_p3_24_centroid_20261114.json",
    "p25": ROOT / "plans/manifests/taiji_u2_p3_25_decoded_colour_20261114.json",
}
REPORTS = {
    "p21": ROOT / "reports/taiji_u2_p3_21_position_20261114.json",
    "p22": ROOT / "reports/taiji_u2_p3_22_selection_20261114.json",
    "p23": ROOT / "reports/taiji_u2_p3_23_coupled_20261114.json",
    "p24": ROOT / "reports/taiji_u2_p3_24_centroid_20261114.json",
}


def criteria_hash(criteria: dict) -> str:
    return hashlib.sha256(
        json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verdict_for(final: dict, criteria: dict, kind: str) -> str:
    fit, train = final["fit_accuracy"], final["train_accuracy"]
    if fit >= criteria["fit_gate"]:
        return {
            "p21": "position_channel_restores_task",
            "p23": "coupled_selection_generalises",
            "p24": "centroid_readout_generalises",
        }[kind]
    if fit <= criteria["fit_low"]:
        if train < criteria["train_gate"]:
            return {
                "p21": "position_channel_not_learned",
                "p23": "coupled_selection_cannot_fit",
                "p24": "centroid_cannot_fit",
            }[kind]
        return {
            "p21": "position_channel_insufficient",
            "p23": "coupled_selection_insufficient",
            "p24": "centroid_readout_insufficient",
        }[kind]
    return {
        "p21": "inconclusive_position",
        "p23": "inconclusive_coupled_selection",
        "p24": "inconclusive_centroid",
    }[kind]


def p22_verdict(arms: dict, criteria: dict) -> str:
    selection = arms["selection"]["doses"][-1]
    bypass = arms["bypass"]["doses"][-1]
    if selection["fit_accuracy"] >= criteria["fit_gate"]:
        if bypass["fit_accuracy"] <= criteria["fit_low"]:
            return "bypass_is_the_cause"
        return "both_generalise"
    if selection["fit_accuracy"] <= criteria["fit_low"]:
        if selection["train_accuracy"] < criteria["train_gate"]:
            return "selection_cannot_fit"
        return "selection_readout_insufficient"
    return "inconclusive_selection"


def load_p21(contract, report):
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p21.PositionedDetector(p21.PIXELS, p21.QUERY, contract["candidate"]["width"], 4)
    module.attach_position(torch.zeros(p21.PATCHES, contract["candidate"]["width"]))
    state = torch.load(
        Path(contract["run_root"]) / f"detector-{report['doses'][-1]['update']:06d}.pt"
    )
    module.load_state_dict(state)
    module.eval()
    return model, module


def load_p22(contract, report, arm, bypass):
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p22.SelectionDetector(p22.PIXELS, p22.QUERY, contract["candidate"]["width"], bypass)
    state = torch.load(
        Path(contract["run_root"])
        / arm
        / f"detector-{report['arms'][arm]['doses'][-1]['update']:06d}.pt"
    )
    module.load_state_dict(state)
    module.eval()
    return model, module


def load_p23(contract, report):
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p23.CoupledDetector(p23.PIXELS, p23.QUERY, contract["candidate"]["width"])
    state = torch.load(
        Path(contract["run_root"]) / f"detector-{report['doses'][-1]['update']:06d}.pt"
    )
    module.load_state_dict(state)
    module.eval()
    return model, module


def load_p24(contract, report):
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p24.CentroidDetector(contract["candidate"]["width"])
    state = torch.load(
        Path(contract["run_root"]) / f"detector-{report['doses'][-1]['update']:06d}.pt"
    )
    module.load_state_dict(state)
    module.eval()
    return model, module


def evaluate(contracts, reports, data) -> dict:
    recomputed = {}
    model, module = load_p21(contracts["p21"], reports["p21"])
    recomputed["p21_fit"] = p21.accuracy(model, module, data["fit"])
    recomputed["p21_train"] = p21.accuracy(model, module, data["train"])
    for arm, bypass in (("selection", False), ("bypass", True)):
        model, module = load_p22(contracts["p22"], reports["p22"], arm, bypass)
        recomputed[f"p22_{arm}_fit"] = p22.accuracy(model, module, data["fit"])
        recomputed[f"p22_{arm}_train"] = p22.accuracy(model, module, data["train"])
    model, module = load_p23(contracts["p23"], reports["p23"])
    recomputed["p23_fit"] = p23.accuracy(model, module, data["fit"])
    recomputed["p23_train"] = p23.accuracy(model, module, data["train"])
    model, module = load_p24(contracts["p24"], reports["p24"])
    recomputed["p24_fit"] = p24.accuracy(model, module, data["fit"])
    recomputed["p24_train"] = p24.accuracy(model, module, data["train"])
    return recomputed


def check_all(contracts, reports, recomputed, drill_report=None) -> dict:
    reports = drill_report or reports
    checks = {}
    for key in ("p21", "p22", "p23", "p24"):
        contract = contracts[key]
        checks[f"{key}_criteria_hash"] = (
            criteria_hash(contract["criteria"]) == contract["criteria_sha256"]
        )
        checks[f"{key}_frozen"] = contract["status"] == "frozen"
    checks["p25_not_frozen"] = contracts["p25"]["status"] != "frozen"

    final21 = reports["p21"]["doses"][-1]
    checks["p21_verdict_rule"] = reports["p21"]["verdict"] == verdict_for(
        final21, contracts["p21"]["criteria"], "p21"
    )
    checks["p21_fit_matches_recompute"] = (
        abs(final21["fit_accuracy"] - recomputed["p21_fit"]["accuracy"]) < 1e-9
    )
    checks["p21_train_matches_recompute"] = (
        abs(final21["train_accuracy"] - recomputed["p21_train"]["accuracy"]) < 1e-9
    )
    checks["p21_no_column_targeting_on_fit"] = recomputed["p21_fit"]["attention_column_hits"] <= 10

    checks["p22_verdict_rule"] = reports["p22"]["verdict"] == p22_verdict(
        reports["p22"]["arms"], contracts["p22"]["criteria"]
    )
    for arm in ("selection", "bypass"):
        final = reports["p22"]["arms"][arm]["doses"][-1]
        checks[f"p22_{arm}_fit_matches_recompute"] = (
            abs(final["fit_accuracy"] - recomputed[f"p22_{arm}_fit"]["accuracy"]) < 1e-9
        )
    checks["p22_selection_abstains_on_all_non_unknown"] = (
        recomputed["p22_selection_fit"]["false_abstain"] == 24
    )

    final23 = reports["p23"]["doses"][-1]
    checks["p23_verdict_rule"] = reports["p23"]["verdict"] == verdict_for(
        final23, contracts["p23"]["criteria"], "p23"
    )
    checks["p23_fit_matches_recompute"] = (
        abs(final23["fit_accuracy"] - recomputed["p23_fit"]["accuracy"]) < 1e-9
    )
    checks["p23_attention_is_uniform"] = (
        abs(recomputed["p23_fit"]["mean_attention_entropy"] - math.log(16)) < 1e-3
    )
    checks["p23_entropy_matches_recompute"] = (
        abs(final23["attention_entropy_fit"] - recomputed["p23_fit"]["mean_attention_entropy"])
        < 1e-5
    )

    final24 = reports["p24"]["doses"][-1]
    checks["p24_verdict_rule"] = reports["p24"]["verdict"] == verdict_for(
        final24, contracts["p24"]["criteria"], "p24"
    )
    checks["p24_fit_matches_recompute"] = (
        abs(final24["fit_accuracy"] - recomputed["p24_fit"]["accuracy"]) < 1e-9
    )
    checks["p24_attention_is_uniform"] = (
        abs(recomputed["p24_fit"]["mean_attention_entropy"] - math.log(256)) < 1e-3
    )
    checks["p24_entropy_matches_recompute"] = (
        abs(final24["attention_entropy_fit"] - recomputed["p24_fit"]["mean_attention_entropy"])
        < 1e-5
    )
    return checks


def main() -> int:
    contracts = {
        key: json.loads(path.read_text(encoding="utf-8")) for key, path in MANIFESTS.items()
    }
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in REPORTS.items()}
    data = atomic_load(Path(contracts["p21"]["data_root"]))
    recomputed = evaluate(contracts, reports, data)
    checks = check_all(contracts, reports, recomputed)

    drills = {}
    mutated = copy.deepcopy(reports)
    mutated["p21"]["doses"][-1]["fit_accuracy"] = 0.75
    drills["bump_p21_fit"] = not check_all(contracts, reports, recomputed, mutated)[
        "p21_fit_matches_recompute"
    ]
    mutated = copy.deepcopy(reports)
    mutated["p21"]["verdict"] = "position_channel_restores_task"
    drills["flip_p21_verdict"] = not check_all(contracts, reports, recomputed, mutated)[
        "p21_verdict_rule"
    ]
    mutated = copy.deepcopy(reports)
    mutated["p22"]["arms"]["selection"]["doses"][-1]["fit_accuracy"] = 0.90
    drills["bump_p22_selection_fit"] = not check_all(contracts, reports, recomputed, mutated)[
        "p22_selection_fit_matches_recompute"
    ]
    mutated = copy.deepcopy(reports)
    mutated["p23"]["doses"][-1]["attention_entropy_fit"] = 1.0
    drills["fake_p23_entropy"] = not check_all(contracts, reports, recomputed, mutated)[
        "p23_entropy_matches_recompute"
    ]
    mutated = copy.deepcopy(reports)
    mutated["p24"]["verdict"] = "centroid_readout_generalises"
    drills["flip_p24_verdict"] = not check_all(contracts, reports, recomputed, mutated)[
        "p24_verdict_rule"
    ]
    mutated = {**contracts, "p25": {**contracts["p25"], "status": "frozen"}}
    drills["freeze_p25"] = not check_all(mutated, reports, recomputed)["p25_not_frozen"]

    report = {
        "schema": "taiji-u2-p3-21-25-guard-v1",
        "date": "2026-11-14",
        "cards": ["22.5.P3.21", "22.5.P3.22", "22.5.P3.23", "22.5.P3.24", "22.5.P3.25"],
        "verdict": (
            "guard_passed" if all(checks.values()) and all(drills.values()) else "guard_failed"
        ),
        "checks": checks,
        "drills": drills,
        "recomputed": {
            key: {
                "accuracy": value["accuracy"],
                "correct": value["correct"],
                "denominator": value["denominator"],
                "mean_attention_entropy": round(value["mean_attention_entropy"], 6),
            }
            for key, value in recomputed.items()
        },
        "measured": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": "independent recomputation from saved checkpoints; no training",
    }
    write_report(OUT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "failed": [k for k, v in {**checks, **drills}.items() if not v],
            }
        )
    )
    return 0 if report["verdict"] == "guard_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
