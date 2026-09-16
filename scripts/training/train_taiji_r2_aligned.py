"""Run the isolated R2 structured language-alignment pilot.

The command always runs checkpoint_roundtrip_preflight before learning.  It
does not modify the default Seed entrypoint or any P3b/P5.2d artifact.
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
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisodeCorpus,
    Taiji,
    TaijiConfig,
    checkpoint_roundtrip_preflight,
    paired_checkpoint_diagnostic,
)

PROTECTED_NAMES = {
    "seed_beta.pt",
    "seed_corpus.pt",
    "resumed_seed_corpus.pt",
    "seed_corpus_prev_20260823.pt",
}
PRE_REGISTRATION = (
    "plans/reference/M5_R2_G1_CONDITIONAL_RESPONSE_PREREGISTRATION_20260916.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--parameter-budget", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--developmental-mode",
        choices=("static", "slow", "fast", "fast_slow"),
        default=None,
    )
    parser.add_argument(
        "--sequence-diagnostic",
        action="store_true",
        help="also compare greedy and bounded native sequence readout paths",
    )
    parser.add_argument(
        "--response-start-readout",
        action="store_true",
        help="enable the isolated native response-start readout candidate",
    )
    parser.add_argument(
        "--response-phase-readout",
        action="store_true",
        help="enable the isolated native full-response readout candidate",
    )
    parser.add_argument("--sequence-beam-width", type=int, default=4)
    parser.add_argument("--sequence-top-k", type=int, default=8)
    parser.add_argument("--sequence-max-bytes", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return torch.device(value)


def _refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name in PROTECTED_NAMES and resolved.parent.name == "checkpoints":
        raise SystemExit(f"refusing to overwrite protected checkpoint: {resolved}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise SystemExit("--max-episodes must be positive")
    if args.sequence_beam_width <= 0 or args.sequence_top_k <= 0 or args.sequence_max_bytes <= 0:
        raise SystemExit("sequence diagnostic parameters must be positive")
    if args.response_start_readout and args.response_phase_readout:
        raise SystemExit("response-start and response-phase readouts are mutually exclusive")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    _refuse_protected(args.checkpoint)
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    device = _resolve_device(args.device)

    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus, device=device)
    else:
        config = TaijiConfig.capacity_profile(
            args.parameter_budget,
            seed=args.seed,
            additional_predictive_readouts=int(
                args.response_start_readout or args.response_phase_readout
            ),
        )

        trainer = LanguageAlignmentTrainer(
            Taiji(config, device=device, episode_id="r2-aligned-pilot"),
            corpus,
            config=LanguageAlignmentConfig(
                developmental_mode=args.developmental_mode or "static",
                response_start_readout=args.response_start_readout,
                response_phase_readout=args.response_phase_readout,
            ),
        )

    additional_readouts = int(
        trainer.model.response_start_readout_enabled
        or trainer.model.response_phase_readout_enabled
    )

    # Keep the exact zero-step payload for the paired post-training diagnostic.
    baseline_payload = trainer.checkpoint()
    # Capture the zero-step generalization surface before any update.  The
    # preflight still performs its own fixed-train-episode round-trip, while
    # this paired baseline makes it impossible to read a post-training report
    # without a same-lineage dev/final comparator.
    baseline = {
        "train": trainer.evaluate("train"),
        "dev": trainer.evaluate("dev"),
        "final": trainer.evaluate("final"),
    }
    baseline_condition_route = {
        split: trainer.condition_route_diagnostic(split)
        for split in ("train", "dev", "final")
    }
    baseline_sequence_decode = None
    if args.sequence_diagnostic:
        baseline_sequence_decode = {
            split: trainer.sequence_decode_diagnostic(
                split,
                beam_width=args.sequence_beam_width,
                top_k=args.sequence_top_k,
                max_generation_bytes=args.sequence_max_bytes,
            )
            for split in ("train", "dev", "final")
        }
    baseline_response_start_margin = None
    if args.response_start_readout:
        baseline_response_start_margin = {
            split: trainer.response_start_margin_diagnostic(split)
            for split in ("train", "dev", "final")
        }
    baseline_response_phase_margin = None
    if args.response_phase_readout:
        baseline_response_phase_margin = {
            split: trainer.response_phase_margin_diagnostic(split)
            for split in ("train", "dev", "final")
        }
    preflight_dir = args.preflight_dir or args.checkpoint.parent / "preflight"
    preflight = checkpoint_roundtrip_preflight(trainer, directory=preflight_dir)
    training = trainer.train(epochs=args.epochs, max_episodes=args.max_episodes)
    checkpoint_path = trainer.save(args.checkpoint)
    paired = paired_checkpoint_diagnostic(trainer, baseline_payload)
    report = {
        "format": "taiji-r2-aligned-language-run-v2",
        "status": "completed",
        "pre_registration": PRE_REGISTRATION,
        "dataset": corpus.manifest(),
        "config": trainer.config.to_payload(),
        "capacity": {
            "target_active_parameters": int(args.parameter_budget),
            "planned_core_active_parameters": trainer.model.config.planned_active_parameter_count,
            "additional_predictive_readouts": additional_readouts,
            "additional_predictive_readout_parameters": int(
                additional_readouts
                * (
                    trainer.model.config.alphabet_size * trainer.model.config.motor_context_dim
                    + trainer.model.config.alphabet_size
                )
            ),
            "effective_active_parameters": trainer.model.parameter_count(),
            "within_target": trainer.model.parameter_count() <= int(args.parameter_budget),
        },
        "readout_registry": trainer.model.readout_registry_status(),
        "baseline": baseline,
        "preflight": preflight,
        "training": training,
        "paired_diagnostic": paired,
        "baseline_condition_route": baseline_condition_route,
        "baseline_sequence_decode": baseline_sequence_decode,
        "baseline_response_start_margin": baseline_response_start_margin,
        "baseline_response_phase_margin": baseline_response_phase_margin,
        "checkpoint": str(checkpoint_path),
        "train": trainer.evaluate("train"),
        "dev": trainer.evaluate("dev"),
        "final": trainer.evaluate("final"),
        "condition_route": {
            split: trainer.condition_route_diagnostic(split)
            for split in ("train", "dev", "final")
        },
        "sequence_decode": (
            {
                split: trainer.sequence_decode_diagnostic(
                    split,
                    beam_width=args.sequence_beam_width,
                    top_k=args.sequence_top_k,
                    max_generation_bytes=args.sequence_max_bytes,
                )
                for split in ("train", "dev", "final")
            }
            if args.sequence_diagnostic
            else None
        ),
        "response_start_margin": (
            {
                split: trainer.response_start_margin_diagnostic(split)
                for split in ("train", "dev", "final")
            }
            if args.response_start_readout
            else None
        ),
        "response_phase_margin": (
            {
                split: trainer.response_phase_margin_diagnostic(split)
                for split in ("train", "dev", "final")
            }
            if args.response_phase_readout
            else None
        ),
    }
    report_path = args.report or args.checkpoint.with_name(args.checkpoint.stem + ".json")
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
