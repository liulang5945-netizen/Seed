"""Guard for P3.28: recompute both arms from the saved checkpoints.

Verifies the frozen criteria hash, re-derives the verdict ladder, recomputes the
four-way accuracy and the colour accuracy of both arms, and asserts the three
facts the card's conclusion rests on: the colour arm's colour accuracy is 1.000,
its column hits are 15 of 24, and the outcome arm stays far below it. Also
asserts that the position buffer is the column coordinate, which is the defect
that invalidated the earlier centroid cards.

Red directions: bumping a reported accuracy, flipping the verdict, faking the
colour accuracy, and rebuilding the module with the interleaved position buffer
must each turn a check red.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p3_28_22_5 as p28
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_28_palette_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_28_palette_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_28_guard_20261114.json"


def criteria_hash(criteria: dict) -> str:
    return hashlib.sha256(
        json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verdict_for(arms: dict, criteria: dict) -> str:
    outcome = arms["palette_outcome"]["doses"][-1]
    colour = arms["palette_colour"]["doses"][-1]
    gate, low = criteria["fit_gate"], criteria["fit_low"]
    if outcome["fit_accuracy"] >= gate:
        return "routing_learned_from_outcomes"
    if colour["fit_accuracy"] >= gate:
        return "palette_given_readout_works"
    if outcome["fit_accuracy"] <= low and colour["fit_accuracy"] <= low:
        return "centroid_path_broken_even_with_palette"
    return "inconclusive_palette"


def load(contract: dict, arm: str, update: int):
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p28.PrototypeDetector(
        contract["candidate"]["tau"], int(contract["candidate"]["seed"]) + 7
    )
    state = torch.load(Path(contract["run_root"]) / arm / f"detector-{update:06d}.pt")
    module.load_state_dict(state)
    module.eval()
    return model, module


def interleaved_positions() -> torch.Tensor:
    """The defective buffer: every pixel gets the coordinate of its row."""
    x = 2 * (torch.arange(p28.SIDE) + 0.5) / p28.SIDE - 1
    return x.repeat_interleave(p28.SIDE)


def check_all(contract, report, recomputed, drill_report=None, drill_positions=None) -> dict:
    report = drill_report or report
    checks = {}
    checks["criteria_hash"] = criteria_hash(contract["criteria"]) == contract["criteria_sha256"]
    checks["status_frozen"] = contract["status"] == "frozen"
    checks["verdict_rule"] = report["verdict"] == verdict_for(report["arms"], contract["criteria"])
    for arm in ("palette_outcome", "palette_colour"):
        final = report["arms"][arm]["doses"][-1]
        value = recomputed[arm]
        checks[f"{arm}_fit_matches_recompute"] = (
            abs(final["fit_accuracy"] - value["accuracy"]) < 1e-9
        )
        checks[f"{arm}_colour_matches_recompute"] = (
            abs(final["colour_accuracy_fit"] - value["colour_accuracy"]) < 1e-9
        )
    checks["colour_arm_reads_every_colour"] = recomputed["palette_colour"]["colour_accuracy"] == 1.0
    checks["colour_arm_column_hits_15"] = recomputed["palette_colour"]["centroid_column_hits"] == 15
    checks["outcome_arm_below_colour"] = (
        recomputed["palette_outcome"]["accuracy"] + 0.20 < recomputed["palette_colour"]["accuracy"]
    )
    positions = drill_positions if drill_positions is not None else p28.pixel_positions()
    checks["position_is_column_coordinate"] = bool(
        torch.equal(positions[:16], 2 * (torch.arange(16) + 0.5) / 16 - 1)
    )
    return checks


def main() -> int:
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    update = report["arms"]["palette_colour"]["doses"][-1]["update"]
    recomputed = {}
    for arm in ("palette_outcome", "palette_colour"):
        model, module = load(contract, arm, update)
        recomputed[arm] = p28.accuracy(model, module, data["fit"])
    checks = check_all(contract, report, recomputed)

    drills = {}
    mutated = copy.deepcopy(report)
    mutated["arms"]["palette_colour"]["doses"][-1]["fit_accuracy"] = 0.75
    drills["bump_colour_fit"] = not check_all(contract, report, recomputed, mutated)[
        "palette_colour_fit_matches_recompute"
    ]
    mutated = copy.deepcopy(report)
    mutated["verdict"] = "centroid_path_broken_even_with_palette"
    drills["flip_verdict"] = not check_all(contract, report, recomputed, mutated)["verdict_rule"]
    mutated = copy.deepcopy(report)
    mutated["arms"]["palette_colour"]["doses"][-1]["colour_accuracy_fit"] = 0.5
    drills["fake_colour_accuracy"] = not check_all(contract, report, recomputed, mutated)[
        "palette_colour_colour_matches_recompute"
    ]
    drills["interleaved_positions"] = not check_all(
        contract, report, recomputed, None, interleaved_positions()
    )["position_is_column_coordinate"]

    verdict = "guard_passed" if all(checks.values()) and all(drills.values()) else "guard_failed"
    write_report(
        OUT,
        {
            "schema": "taiji-u2-p3-28-guard-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.28",
            "verdict": verdict,
            "checks": checks,
            "drills": drills,
            "recomputed": {
                arm: {
                    "accuracy": value["accuracy"],
                    "correct": value["correct"],
                    "denominator": value["denominator"],
                    "colour_accuracy": value["colour_accuracy"],
                    "centroid_column_hits": value["centroid_column_hits"],
                    "centroid_column_total": value["centroid_column_total"],
                    "abstain_on_unknown": value["abstain_on_unknown"],
                    "abstain_on_other": value["abstain_on_other"],
                }
                for arm, value in recomputed.items()
            },
            "measured": {"threads": torch.get_num_threads(), "torch": torch.__version__},
            "scope": "independent recomputation from saved checkpoints; no training",
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "failed": [k for k, v in {**checks, **drills}.items() if not v],
                "recomputed": {
                    arm: {"fit": value["accuracy"], "colour": value["colour_accuracy"]}
                    for arm, value in recomputed.items()
                },
            }
        )
    )
    return 0 if verdict == "guard_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
