"""M4.R4 read-only course-shift and update-scale audit.

The audit does not train a model or change the Taiji runtime.  It rebuilds the
same record-disjoint pilot course used by M4.R3, measures phase difficulty and
byte-distribution differences, then inspects the already committed pilot
artifacts to estimate owner update magnitude.  Its purpose is to decide
whether the next intervention belongs in the data contract or in one update
rule, before any foundation-budget training is authorized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    build_disjoint_phase_chain,
)
from taiji import BytePredictiveReadout, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4r4-course-shift-audit-v1"
VERSION = 1
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_ARMS = ("readout_only", "context_only", "joint")
SCORE_NORMALIZER = math.log(2.0)


def _load_model(path: Path) -> tuple[dict[str, Any], Taiji]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    if "config" not in payload:
        payload = payload.get("model")
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint {path} envelope is missing model payload")
    model = Taiji.from_checkpoint(payload)
    return dict(payload), model


def _byte_probabilities(data: bytes) -> list[float]:
    counts = Counter(data)
    total = max(1, len(data))
    return [counts.get(index, 0) / total for index in range(256)]


def _entropy_bits(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0.0)


def _js_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    midpoint = [(left + right) * 0.5 for left, right in zip(first, second, strict=True)]

    def _kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0.0 and middle > 0.0
        )

    return 0.5 * (_kl(first) + _kl(second))


def _byte_stats(data: bytes) -> dict[str, float | int]:
    probabilities = _byte_probabilities(data)
    return {
        "bytes": len(data),
        "unique_bytes": sum(value > 0.0 for value in probabilities),
        "unique_ratio": sum(value > 0.0 for value in probabilities) / 256.0,
        "entropy_bits": _entropy_bits(probabilities),
        "top_byte_probability": max(probabilities),
    }


def _score(model: Taiji, data: bytes) -> dict[str, Any]:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(data)
    after = content_digest(model.checkpoint())
    return {
        "bpb": float(score["mean_surprise"]) / SCORE_NORMALIZER,
        "accuracy": float(score["accuracy"]),
        "observations": int(score["observations"]),
        "read_only": before == after,
        "owner": str(score["owner"]),
        "scope": str(score["scope"]),
    }


def _owner_tensors(model: Taiji, owner: str) -> tuple[torch.Tensor, ...]:
    if owner == "fabric":
        return tuple(model.fabric.parameter_tensors())
    if owner == "predictive_context":
        return (model.predictive_context.recurrent.edge_weight,)
    if owner == "predictive_readout":
        return (
            model.predictive_readout.synapses.edge_weight,
            model.predictive_readout.bias,
        )
    if owner == "active_predictive_readout":
        active = model._active_predictive_readout  # experiment-only introspection
        if not isinstance(active, BytePredictiveReadout):
            raise ValueError("active artifact is missing its active predictive readout")
        return active.synapses.edge_weight, active.bias
    if owner == "memory":
        return tuple(model.memory.parameter_tensors())
    raise ValueError(f"unsupported owner: {owner}")


def _update_summary(
    source: Taiji,
    candidate: Taiji,
    *,
    candidate_owner: str,
    reference_owner: str,
) -> dict[str, float | int]:
    reference = _owner_tensors(source, reference_owner)
    updated = _owner_tensors(candidate, candidate_owner)
    if len(reference) != len(updated):
        raise ValueError(f"owner tensor count changed for {candidate_owner}")
    squared = 0.0
    reference_squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for left, right in zip(reference, updated, strict=True):
        if left.shape != right.shape:
            raise ValueError(f"owner tensor shape changed for {candidate_owner}")
        delta = (right.detach().cpu().float() - left.detach().cpu().float()).abs()
        squared += float(delta.square().sum().item())
        reference_squared += float(left.detach().cpu().float().square().sum().item())
        max_abs = max(max_abs, float(delta.max().item()))
        changed += int((delta > 0.0).sum().item())
        total += int(delta.numel())
    return {
        "l2_delta": math.sqrt(squared),
        "relative_l2_delta": math.sqrt(squared) / max(math.sqrt(reference_squared), 1e-12),
        "max_abs_delta": max_abs,
        "changed_scalars": changed,
        "total_scalars": total,
    }


def _artifact_path(artifact_dir: Path, arm: str, seed: int) -> Path:
    return artifact_dir / f"{arm}_seed{seed}.pt"


def _phase_report(
    model: Taiji,
    phase_name: str,
    phase: Any,
    *,
    eval_bytes: int,
    previous_train: bytes | None,
    previous_holdout: bytes | None,
) -> dict[str, Any]:
    train = phase.train
    holdout = phase.holdout[:eval_bytes]
    train_probabilities = _byte_probabilities(train)
    holdout_probabilities = _byte_probabilities(holdout)
    phase_report: dict[str, Any] = {
        "phase": phase_name,
        "dataset_digest": phase.digest,
        "partition_seed": int(phase.partition_seed),
        "selected_record_count": len(phase.selected_record_digests),
        "train": _byte_stats(train),
        "holdout": _byte_stats(holdout),
        "train_holdout_js_nats": _js_divergence(train_probabilities, holdout_probabilities),
        "frozen_holdout": _score(model, holdout),
    }
    if previous_train is not None:
        phase_report["train_vs_previous_train_js_nats"] = _js_divergence(
            train_probabilities,
            _byte_probabilities(previous_train),
        )
    if previous_holdout is not None:
        phase_report["holdout_vs_previous_holdout_js_nats"] = _js_divergence(
            holdout_probabilities,
            _byte_probabilities(previous_holdout),
        )
    return phase_report


def _load_pilot_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("technical_gate_all_passed") is not True:
        raise ValueError(f"pilot report is not technically valid: {path}")
    if set(report.get("variants", {})) != set(EXPECTED_ARMS):
        raise ValueError(f"pilot report has an unexpected arm set: {path}")
    return report


def audit(
    *,
    checkpoints: Sequence[Path],
    corpus: Path,
    pilot_reports: Sequence[Path],
    artifact_dirs: Sequence[Path],
    train_bytes: int,
    eval_bytes: int,
    output: Path,
) -> dict[str, Any]:
    if len(checkpoints) != len(EXPECTED_SEEDS):
        raise ValueError("expected one checkpoint for each cohort seed")
    if len(pilot_reports) != len(EXPECTED_SEEDS):
        raise ValueError("expected one pilot report for each cohort seed")
    if len(artifact_dirs) != len(EXPECTED_SEEDS):
        raise ValueError("expected one artifact directory for each cohort seed")
    if train_bytes <= 0 or eval_bytes <= 0:
        raise ValueError("train_bytes and eval_bytes must be positive")
    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    pilot_payloads = [_load_pilot_report(path) for path in pilot_reports]
    checks: dict[str, bool] = {}
    for expected_seed, checkpoint, pilot, artifact_dir in zip(
        EXPECTED_SEEDS,
        checkpoints,
        pilot_payloads,
        artifact_dirs,
        strict=True,
    ):
        source_payload, source_model = _load_model(checkpoint)
        actual_seed = int(source_model.config.seed)
        checks[f"seed{expected_seed}_checkpoint_seed"] = actual_seed == expected_seed
        checks[f"seed{expected_seed}_pilot_seed"] = int(pilot["seed"]) == expected_seed
        source_digest = content_digest(source_payload)
        chain = build_disjoint_phase_chain(
            [corpus],
            cohort_seeds=EXPECTED_SEEDS,
            profile="pilot",
        )
        phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
        for phase in phases:
            if len(phase.train) < train_bytes or len(phase.holdout) < eval_bytes:
                raise ValueError("pilot data does not satisfy the requested audit budget")
        phase_names = ("phase_c", "phase_c2", "phase_c3")
        phase_reports: list[dict[str, Any]] = []
        previous_train: bytes | None = None
        previous_holdout: bytes | None = None
        for phase_name, phase in zip(phase_names, phases, strict=True):
            phase_report = _phase_report(
                source_model,
                phase_name,
                phase,
                eval_bytes=eval_bytes,
                previous_train=previous_train,
                previous_holdout=previous_holdout,
            )
            phase_reports.append(phase_report)
            previous_train = phase.train
            previous_holdout = phase.holdout[:eval_bytes]
        owner_updates: dict[str, Any] = {}
        artifact_checks: dict[str, bool] = {}
        for arm in EXPECTED_ARMS:
            artifact_path = _artifact_path(artifact_dir, arm, expected_seed)
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            artifact_model = Taiji.from_checkpoint(artifact["model"])
            artifact_checks[f"{arm}_source_digest"] = (
                artifact.get("source_checkpoint_digest") == source_digest
            )
            artifact_checks[f"{arm}_score_read_only"] = _score(
                artifact_model,
                phases[-1].holdout[:eval_bytes],
            )["read_only"]
            if arm == "readout_only":
                owner_updates[arm] = {
                    "active_predictive_readout": _update_summary(
                        source_model,
                        artifact_model,
                        candidate_owner="active_predictive_readout",
                        reference_owner="predictive_readout",
                    ),
                    "predictive_context": _update_summary(
                        source_model,
                        artifact_model,
                        candidate_owner="predictive_context",
                        reference_owner="predictive_context",
                    ),
                }
            else:
                owner_updates[arm] = {
                    "predictive_readout": _update_summary(
                        source_model,
                        artifact_model,
                        candidate_owner="predictive_readout",
                        reference_owner="predictive_readout",
                    ),
                    "predictive_context": _update_summary(
                        source_model,
                        artifact_model,
                        candidate_owner="predictive_context",
                        reference_owner="predictive_context",
                    ),
                }
        checks.update({f"seed{expected_seed}_{key}": value for key, value in artifact_checks.items()})
        checks[f"seed{expected_seed}_record_disjoint"] = all(
            value == 0 for value in chain.overlap_counts.values()
        )
        checks[f"seed{expected_seed}_frozen_scores_read_only"] = all(
            bool(item["frozen_holdout"]["read_only"]) for item in phase_reports
        )
        reports.append(
            {
                "seed": expected_seed,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_digest": source_digest,
                "pilot_report": str(pilot_reports[len(reports)]),
                "data_chain": {
                    "phase_c": {
                        "digest": chain.phase_c.digest,
                        "selected_record_count": len(chain.phase_c.selected_record_digests),
                    },
                    "phase_c2": {
                        "digest": chain.phase_c2.digest,
                        "selected_record_count": len(chain.phase_c2.selected_record_digests),
                    },
                    "phase_c3": {
                        "digest": chain.phase_c3.digest,
                        "selected_record_count": len(chain.phase_c3.selected_record_digests),
                    },
                    "overlap_counts": dict(chain.overlap_counts),
                },
                "phases": phase_reports,
                "owner_updates": owner_updates,
            }
        )
    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "configuration": {
            "profile": "pilot",
            "train_bytes": train_bytes,
            "eval_bytes": eval_bytes,
            "seeds": list(EXPECTED_SEEDS),
            "arms": list(EXPECTED_ARMS),
            "training_performed": False,
        },
        "corpus": str(corpus),
        "reports": reports,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": {
            "interpretation": (
                "course difficulty and owner update scale are now measured; no "
                "architecture or formal training decision is made by this audit"
                if all(checks.values())
                else "audit contract failed; do not interpret distribution or update metrics"
            ),
        },
        "resources": {"elapsed_seconds": time.perf_counter() - started},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "output/taiji-m2s-seed11-identity-generation-20260905/last.pt",
            PROJECT_ROOT / "output/taiji-m2u-seed29-identity-generation-20260905/last.pt",
            PROJECT_ROOT / "output/taiji-m2u-seed47-identity-generation-20260905/last.pt",
        ],
    )
    parser.add_argument(
        "--pilot-reports",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "reports/taiji_m4r3_update_interference_pilot_seed11_20260908.json",
            PROJECT_ROOT / "reports/taiji_m4r3_update_interference_pilot_seed29_20260908.json",
            PROJECT_ROOT / "reports/taiji_m4r3_update_interference_pilot_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--artifact-dirs",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "output/taiji-m4r3-update-interference-pilot-seed11",
            PROJECT_ROOT / "output/taiji-m4r3-update-interference-pilot-seed29",
            PROJECT_ROOT / "output/taiji-m4r3-update-interference-pilot-seed47",
        ],
    )
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/simple_zh/dialogue_extended_clean.jsonl")
    parser.add_argument("--train-bytes", type=int, default=16_384)
    parser.add_argument("--eval-bytes", type=int, default=4_096)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/taiji_m4r4_course_shift_audit_20260908.json",
    )
    args = parser.parse_args(argv)
    report = audit(
        checkpoints=args.checkpoints,
        corpus=args.corpus,
        pilot_reports=args.pilot_reports,
        artifact_dirs=args.artifact_dirs,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
