"""Guard for P3.31: recompute the learned-scorer card from its checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p3_31_22_5 as p31
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_31_learned_scorer_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_31_learned_scorer_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_31_guard_20261114.json"


def criteria_hash(criteria: dict) -> str:
    return hashlib.sha256(
        json.dumps(criteria, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def check_all(contract, report, recomputed, drill_report=None) -> dict:
    report = drill_report or report
    final = report["doses"][-1]
    checks = {}
    checks["criteria_hash"] = criteria_hash(contract["criteria"]) == contract["criteria_sha256"]
    checks["status_frozen"] = contract["status"] == "frozen"
    checks["verdict_rule"] = report["verdict"] == p31.decide(final, contract["criteria"])
    checks["fit_matches_recompute"] = abs(final["fit_accuracy"] - recomputed["accuracy"]) < 1e-9
    checks["colour_matches_recompute"] = (
        abs(final["colour_accuracy_fit"] - recomputed["colour_accuracy"]) < 1e-9
    )
    checks["train_exceeds_fit_by_0_2"] = final["train_accuracy"] - final["fit_accuracy"] > 0.2
    checks["stop_rule_fires"] = (
        report["verdict"] == "column_tokens_insufficient_under_learned_scorer"
    )
    return checks


def main() -> int:
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    update = report["doses"][-1]["update"]
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p31.ScorerDetector(contract["candidate"]["tau"])
    module.load_state_dict(torch.load(Path(contract["run_root"]) / f"detector-{update:06d}.pt"))
    module.eval()
    recomputed = p31.accuracy(model, module, data["fit"])
    checks = check_all(contract, report, recomputed)

    drills = {}
    mutated = copy.deepcopy(report)
    mutated["doses"][-1]["fit_accuracy"] = 0.75
    drills["bump_fit"] = not check_all(contract, report, recomputed, mutated)[
        "fit_matches_recompute"
    ]
    mutated = copy.deepcopy(report)
    mutated["verdict"] = "column_tokens_suffice_with_learned_scorer"
    drills["flip_verdict"] = not check_all(contract, report, recomputed, mutated)["verdict_rule"]
    mutated = copy.deepcopy(report)
    mutated["doses"][-1]["colour_accuracy_fit"] = 0.1
    drills["fake_colour_accuracy"] = not check_all(contract, report, recomputed, mutated)[
        "colour_matches_recompute"
    ]

    verdict = "guard_passed" if all(checks.values()) and all(drills.values()) else "guard_failed"
    write_report(
        OUT,
        {
            "schema": "taiji-u2-p3-31-guard-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.31",
            "verdict": verdict,
            "checks": checks,
            "drills": drills,
            "recomputed": {
                "accuracy": recomputed["accuracy"],
                "correct": recomputed["correct"],
                "denominator": recomputed["denominator"],
                "colour_accuracy": recomputed["colour_accuracy"],
                "centroid_column_hits": recomputed["centroid_column_hits"],
                "centroid_column_total": recomputed["centroid_column_total"],
            },
            "measured": {"threads": torch.get_num_threads(), "torch": torch.__version__},
            "scope": "independent recomputation from the saved checkpoint; no training",
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "failed": [k for k, v in {**checks, **drills}.items() if not v],
                "recomputed": {
                    "fit": recomputed["accuracy"],
                    "colour": recomputed["colour_accuracy"],
                    "columns": f"{recomputed['centroid_column_hits']}/{recomputed['centroid_column_total']}",
                },
            }
        )
    )
    return 0 if verdict == "guard_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
