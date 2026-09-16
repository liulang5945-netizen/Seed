"""Run the read-only R2-H3.4 conditional-credit comparison.

The evaluator loads one or more already-trained R2 checkpoints and records
position-wise native target credit plus the ordinary free-generation branch.
It never trains, applies result credit, or changes the default Seed runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji import LanguageAlignmentTrainer, LanguageEpisodeCorpus


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        type=Path,
        help="trained R2 checkpoint; repeat for the B/C comparison",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-generation-bytes", type=int, default=64)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    if args.max_generation_bytes <= 0:
        raise SystemExit("--max-generation-bytes must be positive")
    if not args.checkpoint:
        raise SystemExit("at least one checkpoint is required")

    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    runs: list[dict] = []
    for checkpoint_path in args.checkpoint:
        if not checkpoint_path.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus, device="cpu")
        if not (
            trainer.config.response_start_readout
            or trainer.config.response_phase_readout
        ):
            raise SystemExit(
                f"checkpoint has no isolated response candidate: {checkpoint_path}"
            )
        diagnostic = trainer.conditional_credit_diagnostic(
            max_generation_bytes=args.max_generation_bytes
        )
        runs.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_digest": payload.get("checkpoint_digest"),
                "config": trainer.config.to_payload(),
                "effective_active_parameters": trainer.model.parameter_count(),
                "readout_registry": trainer.model.readout_registry_status(),
                "diagnostic": diagnostic,
            }
        )

    comparison = {
        "checkpoint_count": len(runs),
        "same_corpus_digest": len({run["diagnostic"]["corpus_digest"] for run in runs}) == 1,
        "same_effective_active_parameters": len(
            {run["effective_active_parameters"] for run in runs}
        )
        == 1,
        "readout_owners": [
            {
                "checkpoint": run["checkpoint"],
                "response_start_readout": run["config"]["response_start_readout"],
                "response_phase_readout": run["config"]["response_phase_readout"],
            }
            for run in runs
        ],
    }
    report = {
        "format": "taiji-r2-h3-4-conditional-credit-comparison-v1",
        "status": "completed",
        "dataset": corpus.manifest(),
        "max_generation_bytes": int(args.max_generation_bytes),
        "comparison": comparison,
        "checkpoints": runs,
        "checkpoint_read_only": all(
            run["diagnostic"]["checkpoint_read_only"] for run in runs
        ),
        "recovery_repeatable": all(
            run["diagnostic"]["recovery_repeatable"] for run in runs
        ),
        "native_mode_only": all(run["diagnostic"]["native_mode_only"] for run in runs),
        "external_provider": any(
            run["diagnostic"]["external_provider"] for run in runs
        ),
        "result_credit_applied": any(
            run["diagnostic"]["result_credit_applied"] for run in runs
        ),
        "task_family_and_unknown_policy_forwarded": any(
            run["diagnostic"]["task_family_and_unknown_policy_forwarded"]
            for run in runs
        ),
    }
    _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
