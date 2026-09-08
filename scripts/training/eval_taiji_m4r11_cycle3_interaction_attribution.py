"""M4.R11 cycle3 interaction attribution (read-only).

M4.R10 established a directional interaction: the fixed consolidation rule
improves long-run A retention and C'' gain yet worsens C cycle-3 boundary
forgetting, and the active-only update path showed milder cycle-3 degradation
than the R7 joint path.  This evaluator is read-only: it loads the R10 and
R7 formal artifacts, measures per-cycle active-readout parameter deltas, and
joins them with the recorded retention metrics.  It never trains or mutates
model state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
FORMAT = "taiji-m4r11-cycle3-interaction-attribution-v1"
R10_FORMAT = "taiji-m4r10-rule-audit-v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4r3_update_interference_canary import (  # noqa: E402
    _load_checkpoint,
)
from scripts.training.eval_taiji_m4r8_cycle3_failure_attribution import (  # noqa: E402
    _owner_delta,
)
from taiji import Taiji  # noqa: E402


def _readout_tensors(readout: Any) -> tuple[torch.Tensor, ...]:
    return (readout.synapses.edge_weight, readout.bias)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def _active_readout(model: Taiji) -> Any:
    readout = getattr(model, "_active_predictive_readout", None)
    if readout is None:
        raise ValueError("active predictive readout missing from artifact")
    return readout


def _distance_to_protected(active_model: Taiji) -> dict[str, float]:
    """L2 distance between the active readout and the frozen protected readout."""
    active = _active_readout(active_model)
    protected = active_model.predictive_readout
    active_tensors = _readout_tensors(active)
    protected_tensors = _readout_tensors(protected)
    squared = 0.0
    protected_squared = 0.0
    for left, right in zip(active_tensors, protected_tensors, strict=True):
        difference = (left.detach().cpu().float() - right.detach().cpu().float()).float()
        squared += float(torch.sum(difference * difference))
        protected_squared += float(torch.sum(right.detach().cpu().float() ** 2))
    return {
        "l2_distance": squared**0.5,
        "relative_to_protected_norm": squared**0.5 / max(protected_squared**0.5, 1e-12),
    }


def _cycle3_window(
    final_checkpoint_path: Path,
    cycle2_progress_path: Path,
) -> dict[str, Any]:
    """Cycle-3-only parameter displacement of the active readout.

    The cycle-2 progress checkpoint stores the model state at cycle-2
    completion; the final artifact stores the cycle-3 completed state.  Their
    difference is the single-cycle update the retention metric reacted to.
    """
    final_payload = torch.load(final_checkpoint_path, map_location="cpu", weights_only=False)
    progress_payload = torch.load(cycle2_progress_path, map_location="cpu", weights_only=False)
    if not isinstance(final_payload, dict) or not isinstance(progress_payload, dict):
        raise ValueError("R10 artifacts must be mappings")
    final_model = Taiji.from_checkpoint(final_payload)
    cycle2_model = Taiji.from_checkpoint(progress_payload["model"])
    cycle3_delta = _owner_delta(
        _readout_tensors(_active_readout(cycle2_model)),
        _readout_tensors(_active_readout(final_model)),
    )
    return {
        "final_checkpoint": str(final_checkpoint_path),
        "cycle2_progress_checkpoint": str(cycle2_progress_path),
        "cycle3_active_readout_delta": cycle3_delta,
        "final_active_distance_to_protected": _distance_to_protected(final_model),
        "cycle2_active_distance_to_protected": _distance_to_protected(cycle2_model),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r10-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--r7-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    r10_reports = [_load_json(path) for path in args.r10_reports]
    r7_reports = [_load_json(path) for path in args.r7_reports]
    seeds = tuple(sorted(int(report["seed"]) for report in r10_reports))
    if seeds != EXPECTED_SEEDS:
        raise ValueError("M4.R11 requires R10 formal reports for seed11/29/47")
    if tuple(sorted(int(report["seed"]) for report in r7_reports)) != EXPECTED_SEEDS:
        raise ValueError("M4.R11 requires R7 formal reports for seed11/29/47")
    r10_by_seed = {int(report["seed"]): report for report in r10_reports}
    r7_by_seed = {int(report["seed"]): report for report in r7_reports}

    checks = {
        "r10_reports_technical_all_passed": all(
            bool(report["technical_gate_all_passed"]) for report in r10_reports
        ),
        "r10_not_promoted": all(report["can_promote"] is False for report in r10_reports),
        "r10_data_chain_intact": all(
            all(bool(v) for v in report["data_checks"].values()) for report in r10_reports
        ),
        "r7_reports_technical_all_passed": all(
            bool(report["technical_gate_all_passed"]) for report in r7_reports
        ),
        "r10_same_update_scale_base": all(
            float(report["configuration"]["predictive_update_scale"]) == 0.5
            for report in r10_reports
        ),
        # Same source child file per seed across rounds.  Digest values are
        # NOT comparable across harnesses (R7 records a model-level digest,
        # R10 records the joint-payload digest); each report's own preflight
        # already gates its digest against the source file.
        "source_checkpoints_identical_across_rounds": all(
            _resolve_path(str(r10["source_checkpoint"])).resolve()
            == _resolve_path(str(r7["source_checkpoint"])).resolve()
            for r10, r7 in zip(
                sorted(r10_by_seed.values(), key=lambda r: r["seed"]),
                sorted(r7_by_seed.values(), key=lambda r: r["seed"]),
                strict=True,
            )
        ),
    }

    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        r10 = r10_by_seed[seed]
        r7 = r7_by_seed[seed]
        source_payload, source_model = _load_checkpoint(
            _resolve_path(str(r10["source_checkpoint"])), expected_seed=seed
        )
        del source_payload
        source_readout = _readout_tensors(source_model.predictive_readout)
        arm_rows: dict[str, Any] = {}
        for arm_name in ("baseline", "candidate"):
            arm = next(a for a in r10["arms"] if a["arm"] == arm_name)
            final_path = _resolve_path(str(arm["round_trip"]["final_checkpoint_path"]))
            train_bytes = int(r10["configuration"]["train_bytes"])
            cycle2_path = _resolve_path(
                str(
                    Path(arm["round_trip"]["final_checkpoint_path"]).parent
                    / f"seed{seed}_cascade_cycle2_c{train_bytes}.pt"
                )
            )
            if not final_path.is_file() or not cycle2_path.is_file():
                raise FileNotFoundError(f"missing R10 artifact for seed{seed} {arm_name}")
            window = _cycle3_window(final_path, cycle2_path)
            capability = arm["capability"]
            arm_rows[arm_name] = {
                "consolidation_strength": float(arm["consolidation_strength"]),
                "c_cycle3_delta_bpb": float(capability["c_cycle3_delta_bpb"]),
                "c_cycle2_delta_bpb": float(capability["c_cycle2_delta_bpb"]),
                "c2_cycle3_delta_bpb": float(capability["c2_cycle3_delta_bpb"]),
                "c3_holdout_gain_bpb": float(capability["c3_holdout_gain_bpb"]),
                "active_a_retention_bpb": float(capability["active_a_retention_bpb"]),
                **window,
            }
        # (b) joint reference from R7 scale-0.5 (protected joint update path).
        r7_variant = r7["variants"]["scale_0p5"]
        r7_artifact_path = _resolve_path(str(r7["artifacts"]["scale_0p5"]))
        r7_artifact = torch.load(r7_artifact_path, map_location="cpu", weights_only=False)
        r7_model = Taiji.from_checkpoint(r7_artifact["model"])
        joint_cycle3_delta = _owner_delta(
            source_readout,
            _readout_tensors(r7_model.predictive_readout),
        )
        joint_context_delta = _owner_delta(
            (source_model.predictive_context.recurrent.edge_weight,),
            (r7_model.predictive_context.recurrent.edge_weight,),
        )
        rows.append(
            {
                "seed": int(seed),
                "arms": arm_rows,
                "cycle3_delta_candidate_minus_baseline": float(
                    arm_rows["candidate"]["c_cycle3_delta_bpb"]
                    - arm_rows["baseline"]["c_cycle3_delta_bpb"]
                ),
                "joint_reference": {
                    "harness": "m4r7 protected joint update (context+readout), NOT the same owner path as R10",
                    "artifact": str(r7_artifact_path),
                    "c_cycle3_delta_bpb": float(r7_variant["metrics"]["c_cycle3_delta_bpb"]),
                    "c_cycle2_delta_bpb": float(r7_variant["metrics"]["c_cycle2_delta_bpb"]),
                    "c3_holdout_gain_bpb": float(r7_variant["metrics"]["c3_holdout_gain_bpb"]),
                    "predictive_readout_cycle3_relative_l2": joint_cycle3_delta[
                        "relative_l2_delta"
                    ],
                    "predictive_context_cycle3_relative_l2": joint_context_delta[
                        "relative_l2_delta"
                    ],
                },
            }
        )

    checks["all_rows_loaded"] = len(rows) == len(EXPECTED_SEEDS)
    finite_values: list[float] = []

    def _collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)
        elif isinstance(value, list):
            for item in value:
                _collect(item)
        elif isinstance(value, float):
            finite_values.append(value)

    _collect(rows)
    checks["all_metrics_finite"] = all(math.isfinite(v) for v in finite_values)
    checks["read_only_no_training"] = True
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    # Aggregate interaction evidence.
    per_seed = []
    for row in rows:
        base = row["arms"]["baseline"]
        cand = row["arms"]["candidate"]
        per_seed.append(
            {
                "seed": row["seed"],
                "cycle3_relative_l2_baseline": base["cycle3_active_readout_delta"][
                    "relative_l2_delta"
                ],
                "cycle3_relative_l2_candidate": cand["cycle3_active_readout_delta"][
                    "relative_l2_delta"
                ],
                "final_distance_to_protected_baseline": base["final_active_distance_to_protected"][
                    "relative_to_protected_norm"
                ],
                "final_distance_to_protected_candidate": cand["final_active_distance_to_protected"][
                    "relative_to_protected_norm"
                ],
                "cycle3_delta_candidate_minus_baseline": row[
                    "cycle3_delta_candidate_minus_baseline"
                ],
                "joint_cycle3_delta": row["joint_reference"]["c_cycle3_delta_bpb"],
                "active_only_minus_joint_cycle3": float(
                    base["c_cycle3_delta_bpb"] - row["joint_reference"]["c_cycle3_delta_bpb"]
                ),
            }
        )

    candidate_worse_cycle3_seeds = [
        entry["seed"] for entry in per_seed if entry["cycle3_delta_candidate_minus_baseline"] > 0.0
    ]
    candidate_closer_to_protected_seeds = [
        entry["seed"]
        for entry in per_seed
        if entry["final_distance_to_protected_candidate"]
        < entry["final_distance_to_protected_baseline"]
    ]
    active_only_better_than_joint_seeds = [
        entry["seed"] for entry in per_seed if entry["active_only_minus_joint_cycle3"] < 0.0
    ]
    interaction_consistent = bool(
        len(candidate_worse_cycle3_seeds) >= 2 and len(candidate_closer_to_protected_seeds) >= 2
    )

    diagnosis = {
        "candidate_worse_cycle3_seeds": candidate_worse_cycle3_seeds,
        "candidate_closer_to_protected_seeds": candidate_closer_to_protected_seeds,
        "active_only_better_than_joint_seeds": active_only_better_than_joint_seeds,
        "interaction_direction_consistent": interaction_consistent,
        "interpretation": (
            "CONSISTENT NEGATIVE INTERACTION: the consolidation rule pulls the "
            "active readout toward the frozen protected readout (long-run "
            "retention improves) while worsening cycle-3 boundary adaptation "
            "across the cohort."
            if interaction_consistent
            else "INCONCLUSIVE: the consolidation-cycle3 interaction is not "
            "consistent across the cohort; do not design a gated variant from "
            "this evidence alone."
        ),
        "gate_exit_recommendation": (
            "evidence supports a pre-registered conditional-rule candidate "
            "(cycle/uncertainty gated preservation) only if "
            "interaction_direction_consistent and the joint-vs-active-only "
            "contrast is confirmed by a dedicated single-variable smoke"
            if interaction_consistent
            else "freeze the fixed-capacity consolidation direction; treat "
            "cycle-3 degradation as a course-structure property pending "
            "further attribution"
        ),
    }

    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "technical_gate_all_passed": technical_gate_all_passed,
        "checks": checks,
        "inputs": {
            "r10_reports": [str(path) for path in args.r10_reports],
            "r7_reports": [str(path) for path in args.r7_reports],
        },
        "rows": rows,
        "per_seed_summary": per_seed,
        "diagnosis": diagnosis,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": technical_gate_all_passed,
                "diagnosis": {
                    key: diagnosis[key]
                    for key in (
                        "candidate_worse_cycle3_seeds",
                        "candidate_closer_to_protected_seeds",
                        "active_only_better_than_joint_seeds",
                        "interaction_direction_consistent",
                        "gate_exit_recommendation",
                    )
                },
            },
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
