"""M2.R1 verdict aggregation: versioned absolute/retention/incremental per seed.

Consumes the 1 MiB phase-C arm reports and judges the active_only result with
the R0.5 three-question contract:

- absolute: active readout must beat a fixed BPB threshold (6.5) on phase C;
- retention: the trained child must not be worse than the frozen parent on the
  phase-A retention slice;
- incremental: the pre-registered gain target (mean C-holdout gain >= 0.01 BPB
  across the confirmed seeds, with a per-seed gain > 0) must be met.  The
  target was registered in plans/01_SCOPE_AND_PHASES.md before the seeds ran.

Old reports are never rewritten; this is an additive read-only aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.measurement_verdict import (  # noqa: E402
    CapabilityVerdict,
    judge_absolute,
    judge_incremental,
    judge_retention,
)

AGG_FORMAT = "taiji-r1-verdict-aggregation-v2"
AGG_VERSION = 2
ABSOLUTE_BPB_THRESHOLD = 6.5
RETENTION_TOLERANCE_BPB = 0.05
GAIN_TARGET_BPB = 0.01


def _validate_data_contract(report: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Require the v2 evaluator's record-disjoint and lineage gates."""

    if report.get("format") != "taiji-r1-phase-c-arms-v2":
        raise ValueError(
            f"seed {seed} report is not the record-disjoint evaluator format; "
            "old R1 reports cannot be aggregated"
        )
    contract = report.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"seed {seed} report is missing data_contract")
    if contract.get("current_seed") != int(seed):
        raise ValueError(f"seed {seed} data_contract current_seed mismatch")
    source_lineage = contract.get("source_lineage")
    if not isinstance(source_lineage, dict) or not source_lineage.get("matches_source_checkpoint"):
        raise ValueError(f"seed {seed} source lineage Gate is not passed")
    if source_lineage.get("phase_a_expected_digest") != source_lineage.get(
        "phase_a_actual_digest"
    ) or source_lineage.get("phase_b_expected_digest") != source_lineage.get(
        "phase_b_actual_digest"
    ):
        raise ValueError(f"seed {seed} source lineage digest mismatch")
    gate = contract.get("gate")
    if not isinstance(gate, dict) or not all(
        bool(gate.get(name)) for name in ("record_disjoint", "source_lineage_matches")
    ):
        raise ValueError(f"seed {seed} record-disjoint data Gate is not passed")
    phase_chain = contract.get("phase_chain")
    if not isinstance(phase_chain, dict) or not phase_chain.get("record_disjoint"):
        raise ValueError(f"seed {seed} phase chain is not record-disjoint")
    overlap_counts = phase_chain.get("overlap_counts")
    if not isinstance(overlap_counts, dict) or any(
        bool(value) for value in overlap_counts.values()
    ):
        raise ValueError(f"seed {seed} phase chain contains record overlap")
    datasets = report.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("phase_c"), dict):
        raise ValueError(f"seed {seed} report is missing phase-C dataset metadata")
    phase_c = phase_chain.get("phase_c")
    if not isinstance(phase_c, dict) or datasets["phase_c"].get("digest") != phase_c.get("digest"):
        raise ValueError(f"seed {seed} report phase-C digest is not content-addressed")
    return {
        "phase_c_dataset_digest": str(phase_c["digest"]),
        "phase_c2_dataset_digest": str(phase_chain["phase_c2"]["digest"]),
        "cohort_seeds": tuple(int(value) for value in phase_chain["cohort_seeds"]),
    }


def judge_seed(
    report: dict[str, Any],
    *,
    seed: int,
    baseline_report: dict[str, Any] | None = None,
    arm: str = "active_only",
) -> dict[str, Any]:
    active = next(arm_run for arm_run in report["arms"] if arm_run["arm"] == arm)
    baseline = report if baseline_report is None else baseline_report
    no_update = next(arm_run for arm_run in baseline["arms"] if arm_run["arm"] == "no_update")
    a_c_bpb = float(no_update["capability"]["protected_c_holdout_bpb"])
    a_ret = float(no_update["capability"]["protected_a_retention_bpb"])
    f_c_bpb = float(active["capability"]["active_c_holdout_bpb"])
    f_ret = float(active["capability"]["active_a_retention_bpb"])

    verdicts: dict[str, CapabilityVerdict] = {
        "b1_sequence:absolute": judge_absolute(
            ability_id="b1_sequence",
            metric="bpb",
            value=f_c_bpb,
            threshold=ABSOLUTE_BPB_THRESHOLD,
            direction="lower_is_better",
            detail="active readout on phase-C holdout vs fixed threshold",
        ),
        "b1_sequence:retention": judge_retention(
            ability_id="b1_sequence",
            metric="bpb",
            child_value=f_ret,
            parent_value=a_ret,
            tolerance=RETENTION_TOLERANCE_BPB,
            direction="lower_is_better",
            detail="trained child vs frozen parent on phase-A retention",
        ),
        "b1_sequence:incremental": judge_incremental(
            ability_id="b1_sequence",
            metric="bpb",
            child_value=f_c_bpb,
            baseline_value=a_c_bpb,
            target_delta=GAIN_TARGET_BPB,
            direction="lower_is_better",
            require_delta=True,
            detail=f"pre-registered gain target {GAIN_TARGET_BPB} BPB",
        ),
    }
    return {
        "seed": int(seed),
        "parent_c_holdout_bpb": a_c_bpb,
        "active_c_holdout_bpb": f_c_bpb,
        "c_holdout_gain_bpb": a_c_bpb - f_c_bpb,
        "parent_a_retention_bpb": a_ret,
        "active_a_retention_bpb": f_ret,
        "a_retention_delta_bpb": f_ret - a_ret,
        "checks_passed": (
            f"{sum(int(v) for v in active['checks'].values())}/" f"{len(active['checks'])}"
        ),
        "verdicts": [verdict.to_payload() for verdict in verdicts.values()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--baseline-report",
        nargs="+",
        type=Path,
        help=(
            "Optional per-seed baseline reports containing the no_update arm; "
            "when omitted a seed report is expected to contain no_update itself."
        ),
    )
    parser.add_argument(
        "--arm",
        default="active_only",
        help="Arm name to judge inside each seed report (default: active_only).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / f"taiji_m2r_verdict_aggregation_{datetime.now(timezone.utc):%Y%m%d}.json",
    )
    args = parser.parse_args(argv)

    baselines: list[Path | None] = [None] * len(args.seed_report)
    if args.baseline_report is not None:
        if len(args.baseline_report) != len(args.seed_report):
            parser.error("--baseline-report count must match --seed-report count")
        baselines = list(args.baseline_report)

    entries: list[dict[str, Any]] = []
    contract_entries: list[dict[str, Any]] = []
    for path, baseline_path in zip(args.seed_report, baselines, strict=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = int(report["source_checkpoint"].rsplit("seed", 1)[1].split("-", 1)[0])
        contract_entries.append(_validate_data_contract(report, seed=seed))
        baseline_report = (
            None if baseline_path is None else json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        entries.append(judge_seed(report, seed=seed, baseline_report=baseline_report, arm=args.arm))

    gains = [float(entry["c_holdout_gain_bpb"]) for entry in entries]
    mean_gain = sum(gains) / len(gains) if gains else 0.0
    per_seed_positive = all(gain > 0.0 for gain in gains)
    overall_incremental = per_seed_positive and mean_gain >= GAIN_TARGET_BPB
    overall_retention = all(
        entry["a_retention_delta_bpb"] <= RETENTION_TOLERANCE_BPB for entry in entries
    )
    overall_absolute = all(
        any(v["kind"] == "absolute" and v["judgement"] == "passed" for v in entry["verdicts"])
        for entry in entries
    )
    phase_c_digests = {entry["phase_c_dataset_digest"] for entry in contract_entries}
    phase_c2_digests = {entry["phase_c2_dataset_digest"] for entry in contract_entries}
    cohort_sets = {entry["cohort_seeds"] for entry in contract_entries}
    if len(phase_c_digests) != 1 or len(phase_c2_digests) != 1 or len(cohort_sets) != 1:
        raise ValueError("seed reports must share one record-disjoint phase-C/C' chain and cohort")
    cohort_seeds = next(iter(cohort_sets))

    payload = {
        "format": AGG_FORMAT,
        "version": AGG_VERSION,
        "phase_c_dataset_digest": next(iter(phase_c_digests)),
        "phase_c2_dataset_digest": next(iter(phase_c2_digests)),
        "data_contract": {
            "format": "taiji-m2r1-record-disjoint-course-v1",
            "record_disjoint": True,
            "cohort_seeds": list(cohort_seeds),
        },
        "train_bytes": 1_048_576,
        "eval_bytes": 131_072,
        "seeds": entries,
        "aggregate": {
            "mean_c_holdout_gain_bpb": mean_gain,
            "per_seed_positive": per_seed_positive,
            "overall_absolute": overall_absolute,
            "overall_retention": overall_retention,
            "overall_incremental": overall_incremental,
            "overall": all((overall_absolute, overall_retention, overall_incremental)),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for entry in entries:
        print(
            f"seed {entry['seed']}: gain +{entry['c_holdout_gain_bpb']:.4f}  "
            f"ret_delta {entry['a_retention_delta_bpb']:+.4f}  "
            f"verdicts={[v['kind'] + ':' + v['judgement'] for v in entry['verdicts']]}"
        )
    print("aggregate:", json.dumps(payload["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
