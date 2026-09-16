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
    ResponsePlanTargetEncoder,
    Taiji,
    TaijiConfig,
    checkpoint_roundtrip_preflight,
    content_digest,
    paired_checkpoint_diagnostic,
)

PROTECTED_NAMES = {
    "seed_beta.pt",
    "seed_corpus.pt",
    "resumed_seed_corpus.pt",
    "seed_corpus_prev_20260823.pt",
}
PRE_REGISTRATION = "plans/reference/M5_R2_G1_CONDITIONAL_RESPONSE_PREREGISTRATION_20260916.md"
H36_PRE_REGISTRATION = "plans/reference/M5_R2_H3_6B_MATCHED_RUN_PREREGISTRATION_20260916.md"
H36_TARGET_GEOMETRY = "h3_6_whitened_native_compositional"


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
    parser.add_argument(
        "--response-plan-readout",
        action="store_true",
        help="enable the isolated persistent response-plan candidate",
    )
    parser.add_argument("--response-plan-width", type=int, default=32)
    parser.add_argument(
        "--response-plan-target-geometry",
        choices=("signed_hash_span", H36_TARGET_GEOMETRY),
        default="signed_hash_span",
        help="select the response-plan target geometry for a fresh candidate run",
    )
    parser.add_argument("--sequence-beam-width", type=int, default=4)
    parser.add_argument("--sequence-top-k", type=int, default=8)
    parser.add_argument("--sequence-max-bytes", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--defer-final",
        action="store_true",
        help="do not read final capability outputs during development runs",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run zero-step checkpoint/capacity preflight without capability training",
    )
    parser.add_argument(
        "--preregistration",
        choices=("legacy", "h3_6b"),
        default="legacy",
        help="select the frozen report contract for a fresh matched run",
    )
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
    if args.response_plan_width <= 0:
        raise SystemExit("--response-plan-width must be positive")
    if args.response_plan_target_geometry != "signed_hash_span" and not args.response_plan_readout:
        raise SystemExit("a non-legacy target geometry requires --response-plan-readout")
    if args.preregistration == "h3_6b" and not args.response_plan_readout:
        raise SystemExit("the H3.6-B contract requires --response-plan-readout")
    if (
        sum(
            bool(item)
            for item in (
                args.response_start_readout,
                args.response_phase_readout,
                args.response_plan_readout,
            )
        )
        > 1
    ):
        raise SystemExit("response candidate readouts are mutually exclusive")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    _refuse_protected(args.checkpoint)
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    device = _resolve_device(args.device)

    if args.resume is not None:
        if args.preregistration != "legacy":
            raise SystemExit(
                "--preregistration is only selectable for a fresh run; "
                "a resumed checkpoint owns its report contract"
            )
        if args.response_plan_target_geometry != "signed_hash_span":
            raise SystemExit(
                "--response-plan-target-geometry is only selectable for a fresh run; "
                "a resumed checkpoint owns its target geometry"
            )
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus, device=device)
    else:
        config = TaijiConfig.capacity_profile(
            args.parameter_budget,
            seed=args.seed,
            additional_predictive_readouts=int(
                args.response_start_readout
                or args.response_phase_readout
                or args.response_plan_readout
            ),
            response_plan_width=(args.response_plan_width if args.response_plan_readout else 0),
        )

        model = Taiji(config, device=device, episode_id="r2-aligned-pilot")
        target_encoder = None
        if args.response_plan_target_geometry == H36_TARGET_GEOMETRY:
            model.enable_response_plan_readout(plan_width=args.response_plan_width)
            target_encoder = ResponsePlanTargetEncoder.fit(model, corpus)

        trainer = LanguageAlignmentTrainer(
            model,
            corpus,
            config=LanguageAlignmentConfig(
                developmental_mode=args.developmental_mode or "static",
                response_start_readout=args.response_start_readout,
                response_phase_readout=args.response_phase_readout,
                response_plan_readout=args.response_plan_readout,
                response_plan_width=args.response_plan_width,
                response_plan_target_geometry=args.response_plan_target_geometry,
            ),
            response_plan_target_encoder=target_encoder,
        )

    additional_readouts = int(
        trainer.model.response_start_readout_enabled
        or trainer.model.response_phase_readout_enabled
        or trainer.model.response_plan_readout_enabled
    )

    # Keep the exact zero-step payload for the paired post-training diagnostic.
    baseline_payload = trainer.checkpoint()
    # Capture the zero-step generalization surface before any update.  The
    # preflight still performs its own fixed-train-episode round-trip, while
    # this paired baseline makes it impossible to read a post-training report
    # without a same-lineage dev/final comparator.
    evaluation_splits = ("train", "dev") if args.defer_final else ("train", "dev", "final")
    transfer_splits = ("dev",) if args.defer_final else ("dev", "final")
    baseline = {split: trainer.evaluate(split) for split in evaluation_splits}
    baseline_condition_route = {
        split: trainer.condition_route_diagnostic(split) for split in evaluation_splits
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
            for split in evaluation_splits
        }
    baseline_response_start_margin = None
    if args.response_start_readout:
        baseline_response_start_margin = {
            split: trainer.response_start_margin_diagnostic(split) for split in evaluation_splits
        }
    baseline_response_phase_margin = None
    if args.response_phase_readout:
        baseline_response_phase_margin = {
            split: trainer.response_phase_margin_diagnostic(split) for split in evaluation_splits
        }
    preflight_dir = args.preflight_dir or args.checkpoint.parent / "preflight"
    preflight = checkpoint_roundtrip_preflight(trainer, directory=preflight_dir)
    report_preregistration = (
        H36_PRE_REGISTRATION
        if args.preregistration == "h3_6b"
        or trainer.config.response_plan_target_geometry == H36_TARGET_GEOMETRY
        else PRE_REGISTRATION
    )
    if args.preflight_only:
        report = {
            "format": "taiji-r2-aligned-language-run-v2",
            "status": "preflight_passed",
            "training_performed": False,
            "pre_registration": report_preregistration,
            "model_seed": None if args.resume is not None else int(args.seed),
            "code_revision": trainer.code_revision,
            "dataset": corpus.manifest(),
            "config": trainer.config.to_payload(),
            "response_plan_target": {
                "geometry": trainer.config.response_plan_target_geometry,
                "encoder_digest": trainer.response_plan_target_digest,
                "encoder_parent_checkpoint_digest": (
                    None
                    if trainer.response_plan_target_encoder is None
                    else trainer.response_plan_target_encoder.parent_checkpoint_digest
                ),
                "encoder_corpus_digest": (
                    None
                    if trainer.response_plan_target_encoder is None
                    else trainer.response_plan_target_encoder.corpus_digest
                ),
                "train_target_map_digest": (
                    None
                    if not trainer.response_plan_targets
                    else content_digest(
                        {
                            key: value.detach().cpu().clone()
                            for key, value in sorted(trainer.response_plan_targets.items())
                        }
                    )
                ),
                "fit_episode_ids": (
                    []
                    if trainer.response_plan_target_encoder is None
                    else list(trainer.response_plan_target_encoder.fit_episode_ids)
                ),
            },
            "capacity": {
                "target_active_parameters": int(args.parameter_budget),
                "effective_active_parameters": trainer.model.parameter_count(),
                "within_target": trainer.model.parameter_count() <= int(args.parameter_budget),
            },
            "zero_step_trainer_checkpoint_digest": trainer.checkpoint()["checkpoint_digest"],
            "preflight": preflight,
            "checkpoint": None,
        }
        report_path = args.report or args.checkpoint.with_name(args.checkpoint.stem + ".json")
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    training = trainer.train(epochs=args.epochs, max_episodes=args.max_episodes)
    checkpoint_path = trainer.save(args.checkpoint)
    paired = paired_checkpoint_diagnostic(trainer, baseline_payload, splits=transfer_splits)
    report = {
        "format": "taiji-r2-aligned-language-run-v2",
        "status": "completed",
        "pre_registration": report_preregistration,
        "model_seed": None if args.resume is not None else int(args.seed),
        "code_revision": trainer.code_revision,
        "dataset": corpus.manifest(),
        "config": trainer.config.to_payload(),
        "response_plan_target": {
            "geometry": trainer.config.response_plan_target_geometry,
            "encoder_digest": trainer.response_plan_target_digest,
            "encoder_parent_checkpoint_digest": (
                None
                if trainer.response_plan_target_encoder is None
                else trainer.response_plan_target_encoder.parent_checkpoint_digest
            ),
            "encoder_corpus_digest": (
                None
                if trainer.response_plan_target_encoder is None
                else trainer.response_plan_target_encoder.corpus_digest
            ),
            "train_target_map_digest": (
                None
                if not trainer.response_plan_targets
                else content_digest(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in sorted(trainer.response_plan_targets.items())
                    }
                )
            ),
            "fit_episode_ids": (
                []
                if trainer.response_plan_target_encoder is None
                else list(trainer.response_plan_target_encoder.fit_episode_ids)
            ),
        },
        "final_deferred": bool(args.defer_final),
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
            "response_plan_parameters": (
                trainer.model.response_plan_readout.active_parameter_count
                - (
                    trainer.model.config.alphabet_size * trainer.model.config.motor_context_dim
                    + trainer.model.config.alphabet_size
                )
                if trainer.model.response_plan_readout_enabled
                else 0
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
        "final": None if args.defer_final else trainer.evaluate("final"),
        "condition_route": {
            split: trainer.condition_route_diagnostic(split) for split in evaluation_splits
        },
        "sequence_decode": (
            {
                split: trainer.sequence_decode_diagnostic(
                    split,
                    beam_width=args.sequence_beam_width,
                    top_k=args.sequence_top_k,
                    max_generation_bytes=args.sequence_max_bytes,
                )
                for split in evaluation_splits
            }
            if args.sequence_diagnostic
            else None
        ),
        "response_start_margin": (
            {split: trainer.response_start_margin_diagnostic(split) for split in evaluation_splits}
            if args.response_start_readout
            else None
        ),
        "response_phase_margin": (
            {split: trainer.response_phase_margin_diagnostic(split) for split in evaluation_splits}
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
