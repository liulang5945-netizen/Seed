"""M4.R5 single predictive-update-scale canary.

The canary is deliberately fixed-capacity and protected-owner only.  It
compares the legacy-equivalent scale ``1.0`` with a frozen ``0.0`` control and
one half-rate ``0.5`` candidate on the same record-disjoint continuation
course.  It does not change topology, data source, provider, or client code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    build_disjoint_phase_chain,
)
from scripts.training.eval_taiji_m4r3_update_interference_canary import (  # noqa: E402
    _cycle_metrics,
    _load_checkpoint,
    _score,
)
from seed.persistence import atomic_save  # noqa: E402
from taiji import Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4r5-update-scale-canary-v1"
VERSION = 1
DEFAULT_SEED = 11
COHORT_SEEDS = (11, 29, 47)
SCALES = (0.0, 0.5, 1.0)


def _owner_digests(model: Taiji) -> dict[str, str]:
    return {
        "fabric": content_digest(model.fabric.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "predictive_readout": content_digest(model.predictive_readout.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
    }


def _run_scale(
    source_payload: Mapping[str, Any],
    chain: Any,
    *,
    scale: float,
    train_bytes: int,
    eval_bytes: int,
    protected_baseline: Mapping[str, float],
) -> tuple[Taiji, dict[str, Any]]:
    model = Taiji.from_checkpoint(source_payload)
    before = _owner_digests(model)
    scores: list[dict[str, Any]] = []
    training: list[dict[str, float]] = []
    for phase in (chain.phase_c, chain.phase_c2, chain.phase_c3):
        started = time.perf_counter()
        metrics = model.learn_bytes(
            phase.train[:train_bytes],
            epochs=1,
            include_boundary=True,
            reset=True,
            use_memory=False,
            learn_fabric=False,
            learn_predictive_context=True,
            learn_predictive_readout=True,
            predictive_update_scale=scale,
        )
        training.append(
            {
                "observations": float(metrics["observations"]),
                "online_accuracy": float(metrics["online_accuracy"]),
                "mean_surprise": float(metrics["mean_surprise"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        c_score = _score(model, chain.phase_c.holdout[:eval_bytes])
        c2_score = _score(model, chain.phase_c2.holdout[:eval_bytes])
        c3_score = _score(model, chain.phase_c3.holdout[:eval_bytes])
        scores.append(
            {
                "c_bpb": c_score["bpb"],
                "c2_bpb": c2_score["bpb"],
                "c3_bpb": c3_score["bpb"],
                "read_only": bool(
                    c_score["read_only"] and c2_score["read_only"] and c3_score["read_only"]
                ),
                "owner": c_score["owner"],
                "scope": c_score["scope"],
            }
        )
    after = _owner_digests(model)
    return model, {
        "scale": scale,
        "training": training,
        "scores": scores,
        "metrics": _cycle_metrics(
            scores,
            protected_c3_bpb=float(protected_baseline["c3_bpb"]),
            protected_c_bpb=float(protected_baseline["c_bpb"]),
        ),
        "owner_before": before,
        "owner_after": after,
        "read_only_scoring": all(bool(item["read_only"]) for item in scores),
    }


def _save_artifact(
    artifact_dir: Path,
    scale: float,
    model: Taiji,
    *,
    seed: int,
    source_digest: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scale_name = str(scale).replace(".", "p")
    path = artifact_dir / f"scale_{scale_name}_seed{seed}.pt"
    atomic_save(
        {
            "format": FORMAT,
            "version": VERSION,
            "scale": scale,
            "source_checkpoint_digest": source_digest,
            "model": model.checkpoint(),
        },
        path,
    )
    return path


def run_canary(
    *,
    checkpoint: Path,
    corpus: Path,
    artifact_dir: Path,
    train_bytes: int,
    eval_bytes: int,
    report_path: Path,
    seed: int = DEFAULT_SEED,
    profile: str = "smoke",
) -> dict[str, Any]:
    source_payload, source_model = _load_checkpoint(checkpoint, expected_seed=seed)
    source_digest = content_digest(source_payload)
    chain = build_disjoint_phase_chain(
        [corpus],
        cohort_seeds=COHORT_SEEDS,
        profile=profile,
    )
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    if any(len(phase.train) < train_bytes or len(phase.holdout) < eval_bytes for phase in phases):
        raise ValueError("canary data does not satisfy requested budgets")
    protected_c = _score(source_model, chain.phase_c.holdout[:eval_bytes])
    protected_c3 = _score(source_model, chain.phase_c3.holdout[:eval_bytes])
    protected_baseline = {"c_bpb": protected_c["bpb"], "c3_bpb": protected_c3["bpb"]}
    started = time.perf_counter()
    models: dict[str, Taiji] = {}
    variants: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Path] = {}
    for scale in SCALES:
        name = f"scale_{str(scale).replace('.', 'p')}"
        model, variant = _run_scale(
            source_payload,
            chain,
            scale=scale,
            train_bytes=train_bytes,
            eval_bytes=eval_bytes,
            protected_baseline=protected_baseline,
        )
        models[name] = model
        variants[name] = variant
        artifacts[name] = _save_artifact(
            artifact_dir,
            scale,
            model,
            seed=seed,
            source_digest=source_digest,
        )
    probe = chain.phase_c3.holdout[:eval_bytes]
    fresh_scores: dict[str, dict[str, Any]] = {}
    for name, artifact_path in artifacts.items():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        fresh = Taiji.from_checkpoint(artifact["model"])
        fresh_scores[name] = _score(fresh, probe)
    checks: dict[str, bool] = {
        "record_disjoint_chain": all(value == 0 for value in chain.overlap_counts.values()),
        "source_checkpoint_unchanged": content_digest(source_model.checkpoint()) == source_digest,
        "scale_zero_context_unchanged": (
            variants["scale_0p0"]["owner_before"]["predictive_context"]
            == variants["scale_0p0"]["owner_after"]["predictive_context"]
        ),
        "scale_zero_readout_unchanged": (
            variants["scale_0p0"]["owner_before"]["predictive_readout"]
            == variants["scale_0p0"]["owner_after"]["predictive_readout"]
        ),
        "scale_zero_fabric_unchanged": (
            variants["scale_0p0"]["owner_before"]["fabric"]
            == variants["scale_0p0"]["owner_after"]["fabric"]
        ),
        "scale_zero_memory_unchanged": (
            variants["scale_0p0"]["owner_before"]["memory"]
            == variants["scale_0p0"]["owner_after"]["memory"]
        ),
        "scale_half_context_changes": (
            variants["scale_0p5"]["owner_before"]["predictive_context"]
            != variants["scale_0p5"]["owner_after"]["predictive_context"]
        ),
        "scale_half_readout_changes": (
            variants["scale_0p5"]["owner_before"]["predictive_readout"]
            != variants["scale_0p5"]["owner_after"]["predictive_readout"]
        ),
        "scale_one_context_changes": (
            variants["scale_1p0"]["owner_before"]["predictive_context"]
            != variants["scale_1p0"]["owner_after"]["predictive_context"]
        ),
        "scale_one_readout_changes": (
            variants["scale_1p0"]["owner_before"]["predictive_readout"]
            != variants["scale_1p0"]["owner_after"]["predictive_readout"]
        ),
    }
    for name in variants:
        checks[f"{name}_read_only_scoring"] = variants[name]["read_only_scoring"]
        checks[f"{name}_checkpoint_round_trip"] = (
            abs(
                float(variants[name]["metrics"]["c3_holdout_bpb"])
                - float(fresh_scores[name]["bpb"])
            )
            < 1e-9
        )
        checks[f"{name}_artifact_source_digest"] = (
            torch.load(artifacts[name], map_location="cpu", weights_only=False).get(
                "source_checkpoint_digest"
            )
            == source_digest
        )
    technical_gate = all(checks.values())
    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": time.time(),
        "status": "passed" if technical_gate else "failed",
        "can_promote": False,
        "seed": int(seed),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_digest": source_digest,
        "corpus": str(corpus),
        "configuration": {
            "profile": profile,
            "train_bytes": train_bytes,
            "eval_bytes": eval_bytes,
            "cohort_seeds": list(COHORT_SEEDS),
            "scales": list(SCALES),
            "learn_fabric": False,
            "fixed_capacity": True,
        },
        "protected_baseline": protected_baseline,
        "variants": variants,
        "fresh_restore": {
            name: {
                "bpb": fresh_scores[name]["bpb"],
                "read_only": fresh_scores[name]["read_only"],
            }
            for name in variants
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "checks": checks,
        "technical_gate_all_passed": technical_gate,
        "diagnosis": {
            "half_scale_candidate": (
                variants["scale_0p5"]["metrics"]["c3_holdout_gain_bpb"] > 0.0
                and variants["scale_0p5"]["metrics"]["c_cycle2_delta_bpb"] <= 0.0
                and variants["scale_0p5"]["metrics"]["c_cycle3_delta_bpb"] <= 0.0
            ),
            "interpretation": (
                "scale contract passed; half-rate candidate satisfies the seed11 "
                "smoke diagnostic, but cohort validation is still required"
                if technical_gate
                else "scale contract failed; do not interpret candidate metrics"
            ),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_bytes": {name: path.stat().st_size for name, path in artifacts.items()},
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--train-bytes", type=int, default=4_096)
    parser.add_argument("--eval-bytes", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "foundation"),
        default="smoke",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_canary(
        checkpoint=args.checkpoint,
        corpus=args.corpus,
        artifact_dir=args.artifact_dir,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        report_path=args.report,
        seed=args.seed,
        profile=args.profile,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "half_scale_candidate": report["diagnosis"]["half_scale_candidate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
