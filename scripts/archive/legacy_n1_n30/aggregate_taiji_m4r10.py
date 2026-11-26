"""M4.R10 formal aggregate: three-seed single-rule retention Gate.

Aggregates the per-seed M4.R10 rule-audit reports into the pre-registered
three-seed verdict.  The Gate mirrors M4.R7 formal semantics: the candidate
arm must show a positive C'' holdout gain on all seeds AND zero degradation
seeds on both C cycle-2 and cycle-3 retention deltas.  ``can_promote`` stays
false in the artifact; promotion is a plan-level decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORMAT = "taiji-m4r10-rule-audit-aggregate-v1"
VERSION = 1
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_FORMAT = "taiji-m4r10-rule-audit-v1"

GAIN_METRIC = "c3_holdout_gain_bpb"
DELTA_METRICS = ("c_cycle2_delta_bpb", "c_cycle3_delta_bpb")
REPORT_METRICS = (
    GAIN_METRIC,
    "c2_cycle3_delta_bpb",
    "c_cycle2_delta_bpb",
    "c_cycle3_delta_bpb",
    "active_a_retention_bpb",
)


def _load_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"unexpected report format in {path}: {report.get('format')}")
    if int(report.get("version", -1)) != VERSION:
        raise ValueError(f"unexpected report version in {path}")
    if not report.get("technical_gate_all_passed"):
        raise ValueError(f"seed report technical gate failed: {path}")
    if not all(bool(v) for v in report.get("data_checks", {}).values()):
        raise ValueError(f"seed report data chain mismatch: {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--train-bytes", type=int, required=True)
    parser.add_argument("--eval-bytes", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    reports = [_load_report(path) for path in args.reports]
    seeds = sorted(int(report["seed"]) for report in reports)
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {list(EXPECTED_SEEDS)}, got {seeds}")

    for report in reports:
        config = report["configuration"]
        if int(config["train_bytes"]) != int(args.train_bytes) or int(config["eval_bytes"]) != int(
            args.eval_bytes
        ):
            raise ValueError(f"budget mismatch in seed {report['seed']} report")
        if float(config["predictive_update_scale"]) != 0.5:
            raise ValueError(f"update-scale base must be 0.5 in seed {report['seed']}")
        arms = {arm["arm"]: arm for arm in report["arms"]}
        if (
            float(arms["baseline"]["consolidation_strength"]) != 0.0
            or float(arms["candidate"]["consolidation_strength"]) != 0.5
        ):
            raise ValueError(f"rule arms mismatch in seed {report['seed']}")

    variants: dict[str, Any] = {}
    for arm_name in ("baseline", "candidate"):
        metrics: dict[str, Any] = {}
        for metric in REPORT_METRICS:
            values = [
                float(next(a for a in r["arms"] if a["arm"] == arm_name)["capability"][metric])
                for r in reports
            ]
            entry = {
                "values": values,
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            if metric == GAIN_METRIC:
                entry["positive_seed_count"] = sum(1 for v in values if v > 0)
            else:
                entry["degradation_seed_count"] = sum(1 for v in values if v > 0)
            metrics[metric] = entry
        variants[arm_name] = {"metrics": metrics}

    candidate = variants["candidate"]["metrics"]
    gain_all_positive = candidate[GAIN_METRIC]["positive_seed_count"] == len(EXPECTED_SEEDS)
    no_cycle2_degradation = candidate["c_cycle2_delta_bpb"]["degradation_seed_count"] == 0
    no_cycle3_degradation = candidate["c_cycle3_delta_bpb"]["degradation_seed_count"] == 0
    formal_gate_passed = bool(gain_all_positive and no_cycle2_degradation and no_cycle3_degradation)

    aggregate = {
        "format": FORMAT,
        "version": VERSION,
        "status": "passed" if formal_gate_passed else "retention-gate-failed",
        "can_promote": False,
        "formal_gate_passed": formal_gate_passed,
        "retention_gate": {
            "candidate_c3_gain_all_positive": gain_all_positive,
            "candidate_cycle2_degradation_seeds": candidate["c_cycle2_delta_bpb"][
                "degradation_seed_count"
            ],
            "candidate_cycle3_degradation_seeds": candidate["c_cycle3_delta_bpb"][
                "degradation_seed_count"
            ],
        },
        "seeds": [
            {
                "seed": int(report["seed"]),
                "source_report": str(path),
                "technical_gate_all_passed": bool(report["technical_gate_all_passed"]),
            }
            for report, path in zip(reports, args.reports, strict=True)
        ],
        "variants": variants,
        "configuration": {
            "profile": "foundation",
            "train_bytes": int(args.train_bytes),
            "eval_bytes": int(args.eval_bytes),
            "predictive_update_scale": 0.5,
            "baseline_consolidation_strength": 0.0,
            "candidate_consolidation_strength": 0.5,
            "fixed_capacity": True,
        },
        "source_reports": [str(path) for path in args.reports],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "formal_gate_passed": formal_gate_passed,
                "candidate_gain_mean": candidate[GAIN_METRIC]["mean"],
                "candidate_gain_positive": candidate[GAIN_METRIC]["positive_seed_count"],
                "cycle2_degradation_seeds": candidate["c_cycle2_delta_bpb"][
                    "degradation_seed_count"
                ],
                "cycle3_degradation_seeds": candidate["c_cycle3_delta_bpb"][
                    "degradation_seed_count"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
