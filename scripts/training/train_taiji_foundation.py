"""Run a resumable native Taiji foundation training pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    FoundationTrainingDataset,
    FoundationTrainingRun,
    Taiji,
    TaijiConfig,
)


def _model_config(tier: str, seed: int) -> TaijiConfig:
    if tier == "micro":
        values = TaijiConfig(
            region_sizes=(8,),
            synapse_fan_in=2,
            motor_fan_in=4,
            memory_units=16,
            memory_fan_in=2,
            memory_readout_fan_in=2,
            memory_meta_dim=4,
            memory_time_dim=2,
            memory_episode_dim=2,
            lateral_fan_in=2,
            concept_capacity=8,
        ).to_dict()
    elif tier == "default":
        values = TaijiConfig().to_dict()
    else:
        raise ValueError(f"unsupported model tier: {tier}")
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "foundation"), default="pilot")
    parser.add_argument("--model-tier", choices=("micro", "default"), default="micro")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--partition-seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=1_024)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "taiji-foundation-pilot")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    dataset = FoundationTrainingDataset.from_jsonl(
        args.corpus,
        profile=args.profile,
        partition_seed=args.partition_seed,
    )
    if args.resume is not None:
        run = FoundationTrainingRun.from_checkpoint(
            args.resume,
            dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
    else:
        run = FoundationTrainingRun(
            Taiji(_model_config(args.model_tier, args.seed), episode_id="foundation-train"),
            dataset,
            output_dir=args.output_dir,
            profile=args.profile,
            model_tier=args.model_tier,
            epochs=args.epochs,
            chunk_bytes=args.chunk_bytes,
            checkpoint_interval=args.checkpoint_interval,
        )
    result: dict[str, Any]
    if args.eval_only:
        result = run.evaluate_only()
    else:
        result = run.run()
    report_path = args.report or args.output_dir / ("eval_report.json" if args.eval_only else "training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report_path"] = str(report_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
