"""Diagnose native B1/B5 training signals without changing architecture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.train_taiji_joint import _config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryTask,
    FoundationTrainingDataset,
    Taiji,
    TaijiConfig,
)

FORMAT = "taiji-native-m1-signal-diagnostics-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_signal_diagnostics_20260902.json"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"


def _bpb(model: Taiji, data: bytes) -> float:
    return float(model.score_bytes(data)["mean_surprise"]) / math.log(2.0)


def _train_bytes(
    model: Taiji,
    data: bytes,
    *,
    chunk_bytes: int,
    include_boundary: bool,
) -> None:
    if int(chunk_bytes) <= 0:
        raise ValueError("signal diagnostic chunk_bytes must be positive")
    for start in range(0, len(data), int(chunk_bytes)):
        model.learn_bytes(
            data[start : start + int(chunk_bytes)],
            epochs=1,
            include_boundary=include_boundary,
        )


def _b1_variant(
    dataset: FoundationTrainingDataset,
    *,
    seed: int,
    chunk_bytes: int,
    include_boundary: bool,
) -> dict[str, object]:
    model = Taiji(_config(seed), episode_id=f"m1-19-b1-{seed}")
    parent = _bpb(model, dataset.holdout)
    _train_bytes(
        model,
        dataset.train,
        chunk_bytes=chunk_bytes,
        include_boundary=include_boundary,
    )
    return {
        "seed": seed,
        "chunk_bytes": int(chunk_bytes),
        "include_boundary": bool(include_boundary),
        "parent_holdout_bpb": parent,
        "holdout_bpb": _bpb(model, dataset.holdout),
        "retention_bpb": _bpb(model, dataset.retention),
        "parameter_count": model.parameter_count(),
    }


def run_b1_diagnostics(
    dataset: FoundationTrainingDataset,
    *,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    variants = (
        ("chunked_with_boundary", 1_024, True),
        ("chunked_without_boundary", 1_024, False),
        ("stream_with_boundary", len(dataset.train), True),
        ("stream_without_boundary", len(dataset.train), False),
    )
    records = {
        name: [
            _b1_variant(
                dataset,
                seed=seed,
                chunk_bytes=chunk_bytes,
                include_boundary=include_boundary,
            )
            for seed in seeds
        ]
        for name, chunk_bytes, include_boundary in variants
    }
    summary = {
        name: {
            "holdout_bpb_mean": sum(float(record["holdout_bpb"]) for record in values)
            / len(values),
            "holdout_bpb_min": min(float(record["holdout_bpb"]) for record in values),
            "retention_bpb_mean": sum(float(record["retention_bpb"]) for record in values)
            / len(values),
            "non_degrading_seed_count": sum(
                float(record["holdout_bpb"]) <= float(record["parent_holdout_bpb"])
                for record in values
            ),
        }
        for name, values in records.items()
    }
    best = min(summary, key=lambda name: float(summary[name]["holdout_bpb_mean"]))
    return {
        "dataset_digest": dataset.digest,
        "sample_counts": dataset.sample_counts,
        "best_variant_by_holdout_mean": best,
        "summary": summary,
        "records": records,
    }


def run_b5_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    replay_scales: tuple[float, ...],
    replay_targets: tuple[str, ...],
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    records: list[dict[str, object]] = []
    for scale in replay_scales:
        for targets in replay_targets:
            config = TaijiConfig(
                memory_action_decoder="shared",
                memory_confidence_decay=0.0,
                replay_memory_learning_scale=float(scale),
            )
            measurement = ContinualMemoryTask(
                config,
                seeds=seeds,
                replay_learning_targets=targets,
            ).evaluate(corpus)
            records.append(
                {
                    "replay_scale": float(scale),
                    "replay_learning_targets": targets,
                    "measurement": measurement.to_payload(),
                }
            )
    passing = [
        {
            "replay_scale": record["replay_scale"],
            "replay_learning_targets": record["replay_learning_targets"],
        }
        for record in records
        if record["measurement"]["status"] == "passed"  # type: ignore[index]
    ]
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "records": records,
        "candidates_passing_b5": passing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", type=Path, default=[DEFAULT_CORPUS])
    parser.add_argument("--partition-seed", type=int, default=11)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--replay-scales", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50, 1.0])
    parser.add_argument(
        "--replay-targets",
        nargs="+",
        choices=("all", "association", "readout"),
        default=["all", "association", "readout"],
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    dataset = FoundationTrainingDataset.from_jsonl(
        args.corpus,
        profile="smoke",
        partition_seed=args.partition_seed,
    )
    seeds = tuple(int(seed) for seed in args.seeds)
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared",
        "b1": run_b1_diagnostics(dataset, seeds=seeds),
        "b5": run_b5_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=seeds,
            replay_scales=tuple(float(value) for value in args.replay_scales),
            replay_targets=tuple(str(value) for value in args.replay_targets),
        ),
    }
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
