"""M4.R12 data-source canary aggregate: join UltraData arms with the R10
formal baseline reference (identical configuration, simple_zh corpus) and
apply the pre-registered read.

The pre-registered retention read for the UltraData arm mirrors R7/R10: the
arm must show a positive C'' holdout gain on all seeds AND zero degradation
seeds on both C cycle-2 and cycle-3 retention deltas.  Cross-corpus effects
(A retention on simple_zh holdout) are reported separately because a
corpus switch itself is a distribution shift.
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

FORMAT = "taiji-m4r12-data-source-aggregate-v1"
VERSION = 1
EXPECTED_SEEDS = (11, 29, 47)
CANARY_FORMAT = "taiji-m4r12-data-source-canary-v1"
METRICS = (
    "c3_holdout_gain_bpb",
    "c_cycle2_delta_bpb",
    "c_cycle3_delta_bpb",
    "c2_cycle3_delta_bpb",
    "active_a_retention_bpb",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--r10-baseline-aggregate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    canaries: list[dict[str, Any]] = [
        json.loads(p.read_text(encoding="utf-8")) for p in args.canary_reports
    ]
    seeds = sorted(int(r["seed"]) for r in canaries)
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {list(EXPECTED_SEEDS)}, got {seeds}")
    by_seed = {int(r["seed"]): r for r in canaries}
    for r in canaries:
        if r.get("format") != CANARY_FORMAT:
            raise ValueError(f"unexpected format in {r.get('format')}")
        if not r["technical_gate_all_passed"]:
            raise ValueError(f"seed{r['seed']} canary technical gate failed")

    r10 = json.loads(args.r10_baseline_aggregate.read_text(encoding="utf-8"))
    baseline_metrics = r10["variants"]["baseline"]["metrics"]

    ultra: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = [float(by_seed[s]["capability"][metric]) for s in EXPECTED_SEEDS]
        ultra[metric] = {
            "values": values,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    simple: dict[str, dict[str, Any]] = {
        metric: {
            "values": [float(v) for v in baseline_metrics[metric]["values"]],
            "mean": float(baseline_metrics[metric]["mean"]),
        }
        for metric in METRICS
    }

    gain_positive = sum(1 for v in ultra["c3_holdout_gain_bpb"]["values"] if v > 0)
    cycle2_degraded = sum(1 for v in ultra["c_cycle2_delta_bpb"]["values"] if v > 0)
    cycle3_degraded = sum(1 for v in ultra["c_cycle3_delta_bpb"]["values"] if v > 0)
    retention_gate_passed = bool(
        gain_positive == len(EXPECTED_SEEDS) and cycle2_degraded == 0 and cycle3_degraded == 0
    )
    a_retention_shift = [
        float(by_seed[s]["capability"]["active_a_retention_bpb"])
        - float(baseline_metrics["active_a_retention_bpb"]["values"][i])
        for i, s in enumerate(EXPECTED_SEEDS)
    ]

    aggregate = {
        "format": FORMAT,
        "version": VERSION,
        "status": "passed" if retention_gate_passed else "retention-gate-failed",
        "can_promote": False,
        "retention_gate": {
            "ultra_c3_gain_positive_seeds": gain_positive,
            "ultra_cycle2_degradation_seeds": cycle2_degraded,
            "ultra_cycle3_degradation_seeds": cycle3_degraded,
            "retention_gate_passed": retention_gate_passed,
        },
        "ultra_arm": ultra,
        "simple_zh_reference": simple,
        "a_retention_shift_vs_reference": {
            "values": a_retention_shift,
            "mean": sum(a_retention_shift) / len(a_retention_shift),
        },
        "interpretation": (
            "corpus switch to UltraData triggers severe within-corpus cycle "
            "degradation and cross-corpus A retention loss; the data-density "
            "hypothesis is confounded by the distribution shift"
            if not retention_gate_passed
            else "UltraData corpus passes the pre-registered retention read"
        ),
        "source_reports": [str(p) for p in args.canary_reports],
        "reference_report": str(args.r10_baseline_aggregate),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    shift_mean = sum(a_retention_shift) / len(a_retention_shift)
    args.report.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "retention_gate_passed": retention_gate_passed,
                "ultra_cycle3_degradation_seeds": cycle3_degraded,
                "ultra_c3_gain_mean": ultra["c3_holdout_gain_bpb"]["mean"],
                "a_retention_shift_mean": shift_mean,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
