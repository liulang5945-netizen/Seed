"""Aggregate the M4.R0 record-disjoint three-cycle continuation reports.

This is a read-only, fail-closed evidence aggregator.  It validates the
shared C/C'/C'' data lineage, source-checkpoint preflight, and the full
technical Gate before computing descriptive seed-panel statistics.  A
successful aggregation is not a promotion: the current fixed-capacity
cascade remains ``can_promote=false`` until the research exit criteria are
met by a later, explicitly reviewed experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGG_FORMAT = "taiji-m4r0-cascade3-aggregate-v1"
AGG_VERSION = 1
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_CONFIG = {
    "epochs": 1,
    "c_train_bytes": 1_048_576,
    "eval_bytes": 131_072,
    "chunk_bytes": 65_536,
    "checkpoint_interval": 1,
}
REQUIRED_METRICS = (
    "c3_holdout_gain_bpb",
    "c2_cycle3_delta_bpb",
    "c_cycle2_delta_bpb",
    "c_cycle3_delta_bpb",
    "active_a_retention_bpb",
)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_report(report: dict[str, Any], *, expected_seed: int) -> dict[str, Any]:
    if report.get("format") != "taiji-r1-phase-c-arms-v2":
        raise ValueError(f"seed {expected_seed} is not an M2.R1 v2 evaluator report")
    if report.get("status") != "passed":
        raise ValueError(f"seed {expected_seed} report status is not passed")
    if report.get("can_promote") is not False:
        raise ValueError(f"seed {expected_seed} report must remain non-promotable")

    for key, expected in EXPECTED_CONFIG.items():
        if report.get(key) != expected:
            raise ValueError(
                f"seed {expected_seed} {key} mismatch: "
                f"expected {expected}, got {report.get(key)!r}"
            )

    preflight = _require_mapping(report.get("preflight"), "preflight")
    if preflight.get("source_digest") != preflight.get("source_fresh_digest"):
        raise ValueError(f"seed {expected_seed} source checkpoint digest drifted")

    contract = _require_mapping(report.get("data_contract"), "data_contract")
    if contract.get("current_seed") != expected_seed:
        raise ValueError(f"seed {expected_seed} data_contract seed mismatch")
    source_lineage = _require_mapping(contract.get("source_lineage"), "source_lineage")
    if not source_lineage.get("matches_source_checkpoint"):
        raise ValueError(f"seed {expected_seed} source lineage Gate failed")
    if source_lineage.get("phase_a_expected_digest") != source_lineage.get(
        "phase_a_actual_digest"
    ) or source_lineage.get("phase_b_expected_digest") != source_lineage.get(
        "phase_b_actual_digest"
    ):
        raise ValueError(f"seed {expected_seed} source lineage digest mismatch")

    gate = _require_mapping(contract.get("gate"), "data_contract.gate")
    if not all(bool(gate.get(name)) for name in ("record_disjoint", "source_lineage_matches")):
        raise ValueError(f"seed {expected_seed} data contract Gate failed")
    phase_chain = _require_mapping(contract.get("phase_chain"), "data_contract.phase_chain")
    if not phase_chain.get("record_disjoint"):
        raise ValueError(f"seed {expected_seed} phase chain is not record-disjoint")
    overlaps = _require_mapping(phase_chain.get("overlap_counts"), "overlap_counts")
    if any(int(value) != 0 for value in overlaps.values()):
        raise ValueError(f"seed {expected_seed} phase chain contains record overlap")

    datasets = _require_mapping(report.get("datasets"), "datasets")
    chain_datasets: dict[str, dict[str, Any]] = {}
    for name in ("phase_c", "phase_c2", "phase_c3"):
        chain_entry = _require_mapping(phase_chain.get(name), f"phase_chain.{name}")
        dataset_entry = _require_mapping(datasets.get(name), f"datasets.{name}")
        if chain_entry.get("digest") != dataset_entry.get("digest"):
            raise ValueError(f"seed {expected_seed} {name} digest is not content-addressed")
        chain_datasets[name] = chain_entry

    arms = report.get("arms")
    if not isinstance(arms, list):
        raise ValueError(f"seed {expected_seed} arms must be a list")
    cascade_arms = [arm for arm in arms if isinstance(arm, dict) and arm.get("arm") == "cascade"]
    if len(cascade_arms) != 1:
        raise ValueError(f"seed {expected_seed} must contain exactly one cascade arm")
    arm = _require_mapping(cascade_arms[0], "cascade arm")
    checks = _require_mapping(arm.get("checks"), "cascade checks")
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError(f"seed {expected_seed} cascade technical Gate failed")
    capability = _require_mapping(arm.get("capability"), "cascade capability")
    for metric in REQUIRED_METRICS:
        if metric not in capability:
            raise ValueError(f"seed {expected_seed} is missing metric {metric}")
        float(capability[metric])

    return {
        "seed": expected_seed,
        "source_checkpoint": str(report.get("source_checkpoint")),
        "source_digest": str(preflight["source_digest"]),
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "data_chain": {
            "phase_c": str(chain_datasets["phase_c"]["digest"]),
            "phase_c2": str(chain_datasets["phase_c2"]["digest"]),
            "phase_c3": str(chain_datasets["phase_c3"]["digest"]),
            "cohort_seeds": [int(value) for value in phase_chain["cohort_seeds"]],
        },
        "metrics": {metric: float(capability[metric]) for metric in REQUIRED_METRICS},
        "resources": _require_mapping(arm.get("training"), "cascade training").get(
            "resource", {}
        ),
    }


def aggregate_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected exactly {len(EXPECTED_SEEDS)} seed reports")

    entries = [
        _validate_report(report, expected_seed=seed)
        for report, seed in zip(sorted(reports, key=lambda item: int(item["data_contract"]["current_seed"])), EXPECTED_SEEDS, strict=True)
    ]
    if [entry["seed"] for entry in entries] != list(EXPECTED_SEEDS):
        raise ValueError("seed panel must be exactly 11, 29, 47")

    chains = {json.dumps(entry["data_chain"], sort_keys=True) for entry in entries}
    if len(chains) != 1:
        raise ValueError("seed reports do not share one C/C'/C'' data chain")

    def values(metric: str) -> list[float]:
        return [entry["metrics"][metric] for entry in entries]

    c3_gains = values("c3_holdout_gain_bpb")
    c2_cycle3_deltas = values("c2_cycle3_delta_bpb")
    c_cycle2_deltas = values("c_cycle2_delta_bpb")
    c_cycle3_deltas = values("c_cycle3_delta_bpb")

    def mean(items: list[float]) -> float:
        return sum(items) / len(items)

    aggregate = {
        "technical_gate_all_passed": all(
            entry["checks_passed"] == entry["checks_total"] for entry in entries
        ),
        "c3_holdout_gain_bpb": {
            "mean": mean(c3_gains),
            "min": min(c3_gains),
            "max": max(c3_gains),
            "positive_seed_count": sum(value > 0.0 for value in c3_gains),
        },
        "c2_cycle3_delta_bpb": {
            "mean": mean(c2_cycle3_deltas),
            "degradation_seed_count": sum(value > 0.0 for value in c2_cycle3_deltas),
        },
        "c_cycle2_delta_bpb": {
            "mean": mean(c_cycle2_deltas),
            "degradation_seed_count": sum(value > 0.0 for value in c_cycle2_deltas),
        },
        "c_cycle3_delta_bpb": {
            "mean": mean(c_cycle3_deltas),
            "degradation_seed_count": sum(value > 0.0 for value in c_cycle3_deltas),
        },
        "interpretation": (
            "technical continuation and checkpoint gates passed, but fixed-capacity "
            "three-cycle cascade is not promoted: only one of three C'' holdouts "
            "improved and all three seeds degraded on the prior C retention after "
            "cycle 3"
        ),
    }
    return {
        "format": AGG_FORMAT,
        "version": AGG_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "can_promote": False,
        "seeds": entries,
        "data_chain": entries[0]["data_chain"],
        "configuration": dict(EXPECTED_CONFIG),
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r0_cascade3_aggregate_20260908.json",
    )
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.seed_report]
    payload = aggregate_reports(reports)
    payload["source_reports"] = [str(path) for path in args.seed_report]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "status": payload["status"], "can_promote": False, "aggregate": payload["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
