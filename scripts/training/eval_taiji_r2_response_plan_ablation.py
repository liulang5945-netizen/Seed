"""Read-only H3.5-A plan-state and parameter-matched bridge ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji import LanguageAlignmentTrainer, LanguageEpisode, LanguageEpisodeCorpus
from taiji.internalization import content_digest

H36_TARGET_GEOMETRY = "h3_6_whitened_native_compositional"
H36_PRE_REGISTRATION = "plans/reference/M5_R2_H3_6B_MATCHED_RUN_PREREGISTRATION_20260916.md"


def _target_for_diagnostic(
    trainer: LanguageAlignmentTrainer, episode: LanguageEpisode
) -> torch.Tensor | None:
    """Return a target only when the frozen contract makes one available.

    H3.6 deliberately fits targets on the train split only.  A dev ablation
    must therefore compare the normal and bridge-ablated renderer without
    asking for a new dev label or recomputing a target from the child model.
    """

    if trainer.config.response_plan_target_geometry == "signed_hash_span":
        return trainer._response_plan_target(episode)
    return trainer.response_plan_targets.get(episode.episode_id)


def _target_lineage(trainer: LanguageAlignmentTrainer) -> dict[str, object]:
    encoder = trainer.response_plan_target_encoder
    target_map_digest = None
    if trainer.response_plan_targets:
        target_map_digest = content_digest(
            {
                key: value.detach().cpu().clone()
                for key, value in sorted(trainer.response_plan_targets.items())
            }
        )
    return {
        "geometry": trainer.config.response_plan_target_geometry,
        "encoder_digest": None if encoder is None else encoder.target_digest,
        "encoder_parent_checkpoint_digest": (
            None if encoder is None else encoder.parent_checkpoint_digest
        ),
        "encoder_corpus_digest": None if encoder is None else encoder.corpus_digest,
        "train_target_map_digest": target_map_digest,
        "fit_episode_ids": [] if encoder is None else list(encoder.fit_episode_ids),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    if not trainer.model.response_plan_readout_enabled:
        raise SystemExit("checkpoint does not contain a response-plan candidate")
    original = trainer.checkpoint()
    original_digest = str(original["checkpoint_digest"])

    records = []
    for episode in corpus.for_split(args.split):
        trainer._prime(episode)
        predicted = trainer.model.begin_response_plan().detach().cpu()
        target = _target_for_diagnostic(trainer, episode)
        record = {
            "episode_id": episode.episode_id,
            "target_available": target is not None,
            "plan_norm": float(torch.linalg.vector_norm(predicted)),
            "plan_digest": content_digest(predicted.tolist()),
            "plan_cosine": None,
            "plan_l2": None,
            "target_digest": None,
        }
        if target is not None:
            target = target.detach().cpu()
            record.update(
                {
                    "plan_cosine": float(
                        torch.dot(predicted, target)
                        / max(1e-12, float(torch.linalg.vector_norm(predicted)))
                    ),
                    "plan_l2": float(torch.linalg.vector_norm(predicted - target)),
                    "target_digest": content_digest(target.tolist()),
                }
            )
        records.append(record)

    # Plan snapshots advance only transient dynamics.  Restore the exact
    # source before comparing normal and ablated generation so the final
    # digest guard covers both parameters and runtime state.
    trainer = LanguageAlignmentTrainer.from_checkpoint(original, corpus)

    normal = trainer.evaluate(args.split)
    bridge = trainer.model.response_plan_readout.plan_bridge.detach().clone()
    trainer.model.response_plan_readout.plan_bridge.zero_()
    ablated = trainer.evaluate(args.split)
    trainer.model.response_plan_readout.plan_bridge.copy_(bridge)
    if str(trainer.checkpoint()["checkpoint_digest"]) != original_digest:
        raise RuntimeError("plan ablation did not restore the source checkpoint")

    is_h36 = trainer.config.response_plan_target_geometry == H36_TARGET_GEOMETRY
    report = {
        "format": (
            "taiji-r2-h3-6b-response-plan-ablation-v1"
            if is_h36
            else "taiji-r2-h3-5a-response-plan-ablation-v1"
        ),
        "status": "completed",
        "pre_registration": H36_PRE_REGISTRATION if is_h36 else None,
        "split": args.split,
        "dataset_digest": corpus.digest,
        "checkpoint": str(args.checkpoint),
        "checkpoint_digest": original_digest,
        "checkpoint_read_only": True,
        "episodes": len(records),
        "target_lineage": _target_lineage(trainer),
        "target_available_count": sum(bool(item["target_available"]) for item in records),
        "mean_plan_cosine": (
            sum(item["plan_cosine"] for item in records if item["plan_cosine"] is not None)
            / sum(item["plan_cosine"] is not None for item in records)
            if any(item["plan_cosine"] is not None for item in records)
            else None
        ),
        "mean_plan_l2": (
            sum(item["plan_l2"] for item in records if item["plan_l2"] is not None)
            / sum(item["plan_l2"] is not None for item in records)
            if any(item["plan_l2"] is not None for item in records)
            else None
        ),
        "normal": normal,
        "plan_bridge_ablated": ablated,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "mean_plan_cosine": report["mean_plan_cosine"],
                "normal_sequence": normal["sequence_criterion_pass_rate"],
                "ablated_sequence": ablated["sequence_criterion_pass_rate"],
                "checkpoint_read_only": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
