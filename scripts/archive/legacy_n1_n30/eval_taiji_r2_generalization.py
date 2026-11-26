"""Evaluate the read-only H3.3 generalization surface of an R2 checkpoint.

The command loads a versioned R2 trainer checkpoint and a matching structured
corpus, then profiles train/dev/final transfer without training or changing
the default Seed entrypoint.  It is intentionally separate from the training
runner so a generalization report cannot accidentally become another training
run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji import (
    LANGUAGE_ALIGNMENT_GENERALIZATION_EVALUATION,
    LanguageAlignmentTrainer,
    LanguageEpisodeCorpus,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "dev", "final"),
        choices=("train", "dev", "final", "retention"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    before_digest = str(trainer.checkpoint()["checkpoint_digest"])
    diagnostic = trainer.generalization_diagnostic(tuple(args.splits))
    after_digest = str(trainer.checkpoint()["checkpoint_digest"])
    if before_digest != after_digest:
        raise RuntimeError("generalization evaluator changed the loaded checkpoint")
    report = {
        "format": LANGUAGE_ALIGNMENT_GENERALIZATION_EVALUATION,
        "status": "completed",
        "dataset": corpus.manifest(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_digest": before_digest,
        "diagnostic": diagnostic,
        "training_performed": False,
        "checkpoint_read_only": True,
        "native_mode_only": True,
        "external_provider": False,
    }
    report_path = args.report or args.checkpoint.with_name(
        args.checkpoint.stem + ".generalization.json"
    )
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
