"""Guard for P3.29: recompute both reproduction cells from their checkpoints.

Checks the frozen criteria hash, re-derives the reproduction ladder, recomputes
the four-way and colour accuracy of both cells, and asserts the two facts the
card rests on: the paraphrase cell reads every colour correctly on a template it
never saw, and the seed_b cell reaches the gate. Also asserts that the
paraphrased request really differs from the original bytes, so a "pass" cannot
come from silently falling back to the original template.

Red directions: bumping a reported accuracy, flipping the verdict, zeroing the
paraphrase colour accuracy, and making the paraphrase identical to the original
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

from scripts.training import run_p3_29_22_5 as p29
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_29_reproduction_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_29_reproduction_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_29_guard_20261114.json"


def criteria_hash(criteria: dict) -> str:
    return hashlib.sha256(
        json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verdict_for(cells: dict, criteria: dict) -> str:
    gate = criteria["fit_gate"]
    seed_ok = cells["seed_b"]["doses"][-1]["fit_accuracy"] >= gate
    para_ok = cells["paraphrase"]["doses"][-1]["fit_accuracy"] >= gate
    if seed_ok and para_ok:
        return "reproduced_on_both_axes"
    if seed_ok and not para_ok:
        return "template_bound_routing"
    if para_ok and not seed_ok:
        return "seed_unstable"
    return "not_reproduced"


def load(contract: dict, name: str, update: int):
    spec = contract["cells"][name]
    model = make_model("native", int(spec["seed"]))
    model.eval()
    module = p29.PaletteDetector(contract["candidate"]["tau"], int(spec["seed"]) + 7)
    state = torch.load(Path(contract["run_root"]) / name / f"detector-{update:06d}.pt")
    module.load_state_dict(state)
    module.eval()
    return model, module, spec["style"]


def check_all(contract, report, recomputed, drill_report=None, drill_bytes=None) -> dict:
    report = drill_report or report
    checks = {}
    checks["criteria_hash"] = criteria_hash(contract["criteria"]) == contract["criteria_sha256"]
    checks["status_frozen"] = contract["status"] == "frozen"
    checks["verdict_rule"] = report["verdict"] == verdict_for(report["cells"], contract["criteria"])
    for name in ("seed_b", "paraphrase"):
        final = report["cells"][name]["doses"][-1]
        value = recomputed[name]
        checks[f"{name}_fit_matches_recompute"] = (
            abs(final["fit_accuracy"] - value["accuracy"]) < 1e-9
        )
        checks[f"{name}_colour_matches_recompute"] = (
            abs(final["colour_accuracy_fit"] - value["colour_accuracy"]) < 1e-9
        )
    checks["paraphrase_reads_every_colour"] = recomputed["paraphrase"]["colour_accuracy"] == 1.0
    checks["seed_b_reaches_gate"] = (
        recomputed["seed_b"]["accuracy"] >= contract["criteria"]["fit_gate"]
    )
    if drill_bytes is None:
        sample = atomic_load(Path(contract["data_root"]))["fit"][0]
        original = p29.request_bytes(sample, "original")
        paraphrase = p29.request_bytes(sample, "paraphrase")
    else:
        original, paraphrase = drill_bytes
    checks["paraphrase_differs_from_original"] = original != paraphrase
    return checks


def main() -> int:
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    update = report["cells"]["seed_b"]["doses"][-1]["update"]
    recomputed = {}
    for name in ("seed_b", "paraphrase"):
        model, module, style = load(contract, name, update)
        recomputed[name] = p29.accuracy(model, module, data["fit"], style)
    checks = check_all(contract, report, recomputed)

    drills = {}
    mutated = copy.deepcopy(report)
    mutated["cells"]["paraphrase"]["doses"][-1]["fit_accuracy"] = 0.5
    drills["bump_paraphrase_fit"] = not check_all(contract, report, recomputed, mutated)[
        "paraphrase_fit_matches_recompute"
    ]
    mutated = copy.deepcopy(report)
    mutated["verdict"] = "template_bound_routing"
    drills["flip_verdict"] = not check_all(contract, report, recomputed, mutated)["verdict_rule"]
    mutated = copy.deepcopy(report)
    mutated["cells"]["paraphrase"]["doses"][-1]["colour_accuracy_fit"] = 0.4
    drills["fake_colour_accuracy"] = not check_all(contract, report, recomputed, mutated)[
        "paraphrase_colour_matches_recompute"
    ]
    sample = data["fit"][0]
    drills["identical_template"] = not check_all(
        contract,
        report,
        recomputed,
        None,
        (p29.request_bytes(sample, "original"), p29.request_bytes(sample, "original")),
    )["paraphrase_differs_from_original"]

    verdict = "guard_passed" if all(checks.values()) and all(drills.values()) else "guard_failed"
    write_report(
        OUT,
        {
            "schema": "taiji-u2-p3-29-guard-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.29",
            "verdict": verdict,
            "checks": checks,
            "drills": drills,
            "recomputed": {
                name: {
                    "accuracy": value["accuracy"],
                    "correct": value["correct"],
                    "denominator": value["denominator"],
                    "colour_accuracy": value["colour_accuracy"],
                    "centroid_column_hits": value["centroid_column_hits"],
                    "abstain_on_unknown": value["abstain_on_unknown"],
                    "abstain_on_other": value["abstain_on_other"],
                }
                for name, value in recomputed.items()
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
                    name: {"fit": value["accuracy"], "colour": value["colour_accuracy"]}
                    for name, value in recomputed.items()
                },
            }
        )
    )
    return 0 if verdict == "guard_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
