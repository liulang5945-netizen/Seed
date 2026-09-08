"""M4.R10 single update/consolidation rule audit.

R8 attributed the cycle-3 retention failure neither to owner update magnitude
nor to an anomalous C3 byte distribution; R9 excluded an obvious C3 phase
boundary break.  The remaining hypothesis is the interaction between the
continuous update rule and the already-existing consolidation rule.  This
audit therefore reuses the M2.R1 cascade harness (isolated active execute
owner, record-disjoint C->C2->C3) on the stable foundation course with a
fixed ``predictive_update_scale`` base and exactly ONE rule variable: the
existing ``consolidation_strength`` (baseline ``0.0`` vs candidate ``0.5``).

Forbidden by the M4.R10 stop line: capacity changes, topology changes, data
source swaps (including UltraData), multi-scale sweeps, and peripheral
system wiring.  ``can_promote=false`` is fixed in this script; promotion
requires a pre-registered three-seed formal aggregate under the same
retention Gate as M4.R7.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    CHECKPOINT_OUTPUT_DIR,
    COHORT_SEEDS,
    _run_cascade_arm,
    build_disjoint_phase_chain,
    load_joint_child,
)

FORMAT = "taiji-m4r10-rule-audit-v1"
VERSION = 1

# Pre-registered foundation course digests (M2.R1 v2 / M4.R0 / M4.R9).
EXPECTED_PHASE_DIGESTS = {
    "phase_c": "ac97455f1583b90f7e2d62ca3c682c8e27f0c56bb7275ebee6abdb17fc6476bd",
    "phase_c2": "13dee7b2f7d38cd3688a5af76264c24cb6156ea371e9db88d812f5ea283598c1",
    "phase_c3": "33d3a0d9a755d4d315e875a42266a4f4ec74a6a7b54f9831bfa676bee4d2d1cd",
}

BASELINE_STRENGTH = 0.0
CANDIDATE_STRENGTH = 0.5

COMPARE_METRICS = (
    "c3_holdout_gain_bpb",
    "c2_cycle3_delta_bpb",
    "c_cycle2_delta_bpb",
    "c_cycle3_delta_bpb",
    "active_a_retention_bpb",
)


def _run_rule_arm(
    *,
    arm_name: str,
    consolidation_strength: float,
    source_model: Any,
    chain: Any,
    seed: int,
    train_bytes: int,
    eval_bytes: int,
    chunk_bytes: int,
    update_scale: float,
    progress_root: Path,
    resume: bool,
) -> dict[str, Any]:
    progress_dir = progress_root / arm_name
    progress_dir.mkdir(parents=True, exist_ok=True)
    result = _run_cascade_arm(
        source_model=source_model,
        c_train=chain.phase_c.train[:train_bytes],
        c2_train=chain.phase_c2.train[:train_bytes],
        c3_train=chain.phase_c3.train[:train_bytes],
        c_holdout=chain.phase_c.holdout[:eval_bytes],
        c2_holdout=chain.phase_c2.holdout[:eval_bytes],
        c3_holdout=chain.phase_c3.holdout[:eval_bytes],
        a_retention=chain.phase_a_by_seed[int(seed)].retention[:eval_bytes],
        epochs=1,
        seed=seed,
        chunk_bytes=chunk_bytes,
        checkpoint_interval=1,
        progress_dir=progress_dir,
        resume=resume,
        consolidation_strength=consolidation_strength,
        predictive_update_scale=update_scale,
    )
    # The cascade arm writes a fixed-name final checkpoint; archive it per arm
    # so the second arm cannot overwrite the first arm's evidence.
    final_name = f"seed{seed}_cascade_c{train_bytes}.pt"
    final_path = CHECKPOINT_OUTPUT_DIR / final_name
    if final_path.is_file():
        archived = progress_root / arm_name / final_name
        shutil.move(str(final_path), str(archived))
        result["round_trip"]["final_checkpoint_path"] = str(archived)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-bytes", type=int, default=16 * 1024)
    parser.add_argument("--eval-bytes", type=int, default=4 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024)
    parser.add_argument(
        "--update-scale",
        type=float,
        default=0.5,
        help="Fixed predictive_update_scale base (M4.R7 formal candidate).",
    )
    parser.add_argument(
        "--progress-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji_m4r10_rule_audit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not math.isfinite(args.update_scale) or args.update_scale <= 0.0:
        raise ValueError("--update-scale must be finite and positive")
    if args.train_bytes <= 0 or args.eval_bytes <= 0:
        raise ValueError("--train-bytes and --eval-bytes must be positive")

    started = time.perf_counter()
    joint_payload, source_model = load_joint_child(args.checkpoint, expected_seed=args.seed)
    lineage_seeds = tuple(dict.fromkeys((*COHORT_SEEDS, int(args.seed))))
    chain = build_disjoint_phase_chain(
        [args.corpus],
        cohort_seeds=lineage_seeds,
    )
    data_checks = {
        "phase_c_digest_matches_preregistration": (
            chain.phase_c.digest == EXPECTED_PHASE_DIGESTS["phase_c"]
        ),
        "phase_c2_digest_matches_preregistration": (
            chain.phase_c2.digest == EXPECTED_PHASE_DIGESTS["phase_c2"]
        ),
        "phase_c3_digest_matches_preregistration": (
            chain.phase_c3.digest == EXPECTED_PHASE_DIGESTS["phase_c3"]
        ),
        "source_ab_lineage_matches_checkpoint": (
            str(joint_payload.get("protected_dataset_digest"))
            == chain.phase_a_by_seed[int(args.seed)].digest
            and str(joint_payload.get("dataset_digest"))
            == chain.phase_b_by_seed[int(args.seed)].digest
        ),
    }
    if not all(data_checks.values()):
        raise RuntimeError(
            "M4.R10 data chain does not match the pre-registered foundation "
            f"course; refusing to train. checks={data_checks}"
        )
    partitions = (
        chain.phase_c.train,
        chain.phase_c2.train,
        chain.phase_c3.train,
        chain.phase_c3.holdout,
        chain.phase_a_by_seed[int(args.seed)].retention,
    )
    if args.train_bytes > min(len(chain.phase_c2.train), len(chain.phase_c3.train)):
        raise ValueError("train_bytes exceeds the cascade partitions")
    if args.eval_bytes > min(
        len(chain.phase_c.holdout),
        len(chain.phase_c2.holdout),
        len(chain.phase_c3.holdout),
        len(chain.phase_a_by_seed[int(args.seed)].retention),
    ):
        raise ValueError("eval_bytes exceeds the holdout partitions")
    del partitions

    arms: list[dict[str, Any]] = []
    for arm_name, strength in (
        ("baseline", BASELINE_STRENGTH),
        ("candidate", CANDIDATE_STRENGTH),
    ):
        arm_started = time.perf_counter()
        result = _run_rule_arm(
            arm_name=arm_name,
            consolidation_strength=strength,
            source_model=source_model,
            chain=chain,
            seed=args.seed,
            train_bytes=args.train_bytes,
            eval_bytes=args.eval_bytes,
            chunk_bytes=args.chunk_bytes,
            update_scale=args.update_scale,
            progress_root=args.progress_root,
            resume=args.resume,
        )
        arm_seconds = time.perf_counter() - arm_started
        arm_checks = dict(result["checks"])
        arm_checks["rule_parameter_recorded"] = float(result["consolidation_strength"]) == float(
            strength
        )
        arms.append(
            {
                "arm": arm_name,
                "consolidation_strength": float(strength),
                "predictive_update_scale": float(args.update_scale),
                "checks": arm_checks,
                "technical_gate_passed": all(bool(v) for v in arm_checks.values()),
                "capability": result["capability"],
                "round_trip": result["round_trip"],
                "training": result["training"],
                "resources": {**result["resources"], "arm_elapsed_seconds": arm_seconds},
            }
        )

    technical_gate_all_passed = all(arm["technical_gate_passed"] for arm in arms)
    by_arm = {arm["arm"]: arm for arm in arms}
    comparison: dict[str, Any] = {}
    for metric in COMPARE_METRICS:
        base = float(by_arm["baseline"]["capability"][metric])
        cand = float(by_arm["candidate"]["capability"][metric])
        comparison[metric] = {
            "baseline": base,
            "candidate": cand,
            "candidate_minus_baseline": cand - base,
        }
    diagnosis = (
        "technical consolidation and checkpoint gates "
        + ("passed" if technical_gate_all_passed else "FAILED")
        + "; single-rule candidate vs baseline on the scale-0.5 cascade base: "
        + json.dumps(
            {m: comparison[m]["candidate_minus_baseline"] for m in COMPARE_METRICS},
            sort_keys=True,
        )
    )

    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "seed": int(args.seed),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_digest": str(joint_payload.get("checkpoint_digest", "")),
        "corpus": str(args.corpus.as_posix()),
        "configuration": {
            "profile": "foundation",
            "epochs": 1,
            "train_bytes": int(args.train_bytes),
            "eval_bytes": int(args.eval_bytes),
            "chunk_bytes": int(args.chunk_bytes),
            "predictive_update_scale": float(args.update_scale),
            "baseline_consolidation_strength": BASELINE_STRENGTH,
            "candidate_consolidation_strength": CANDIDATE_STRENGTH,
            "fixed_capacity": True,
            "data_chain": dict(EXPECTED_PHASE_DIGESTS),
        },
        "data_checks": data_checks,
        "arms": arms,
        "comparison": comparison,
        "technical_gate_all_passed": technical_gate_all_passed,
        "diagnosis": diagnosis,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "technical_gate_all_passed": technical_gate_all_passed,
                "comparison": {
                    m: comparison[m]["candidate_minus_baseline"] for m in COMPARE_METRICS
                },
            },
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
