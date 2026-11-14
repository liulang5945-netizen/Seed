"""Guard for P3.30: recompute the token-space readout from its checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p3_30_22_5 as p30
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load
from taiji.u2_observation_columns import column_tokens

MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_30_column_readout_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_30_column_readout_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_30_guard_20261114.json"


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
    checks["verdict_rule"] = report["verdict"] == p30.decide(final, contract["criteria"])
    checks["fit_matches_recompute"] = abs(final["fit_accuracy"] - recomputed["accuracy"]) < 1e-9
    checks["colour_matches_recompute"] = (
        abs(final["colour_accuracy_fit"] - recomputed["colour_accuracy"]) < 1e-9
    )
    model = make_model("native", int(contract["candidate"]["seed"]))
    before = sum(parameter.numel() for parameter in model.parameters())
    before_tensors = len(list(model.named_parameters()))
    sample = atomic_load(Path(contract["data_root"]))["fit"][0]
    tokens = column_tokens(model, sample["public"]["observations"], sample["public"]["coordinates"])
    after = sum(parameter.numel() for parameter in model.parameters())
    checks["pathway_adds_no_parameters"] = (
        before == after == 22742 and len(list(model.named_parameters())) == before_tensors == 194
    )
    checks["token_shape_is_16_by_8"] = tuple(tokens.shape) == (16, 8)
    return checks


def main() -> int:
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    update = report["doses"][-1]["update"]
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    module = p30.TokenDetector(contract["candidate"]["tau"], p30.prototype_tokens(model))
    module.load_state_dict(torch.load(Path(contract["run_root"]) / f"detector-{update:06d}.pt"))
    module.eval()
    recomputed = p30.accuracy(model, module, data["fit"])
    checks = check_all(contract, report, recomputed)

    drills = {}
    mutated = copy.deepcopy(report)
    mutated["doses"][-1]["fit_accuracy"] = 0.75
    drills["bump_fit"] = not check_all(contract, report, recomputed, mutated)[
        "fit_matches_recompute"
    ]
    mutated = copy.deepcopy(report)
    mutated["verdict"] = "column_tokens_carry_the_task"
    drills["flip_verdict"] = not check_all(contract, report, recomputed, mutated)["verdict_rule"]
    mutated = copy.deepcopy(report)
    mutated["doses"][-1]["colour_accuracy_fit"] = 0.2
    drills["fake_colour_accuracy"] = not check_all(contract, report, recomputed, mutated)[
        "colour_matches_recompute"
    ]

    verdict = "guard_passed" if all(checks.values()) and all(drills.values()) else "guard_failed"
    write_report(
        OUT,
        {
            "schema": "taiji-u2-p3-30-guard-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.30",
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
                },
            }
        )
    )
    return 0 if verdict == "guard_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
