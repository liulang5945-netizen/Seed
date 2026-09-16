"""Read-only H3.5-A plan-state and parameter-matched bridge ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji import LanguageAlignmentTrainer, LanguageEpisodeCorpus
from taiji.internalization import content_digest


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
        target = trainer._response_plan_target(episode).detach().cpu()
        cosine = float(torch.dot(predicted, target) / max(1e-12, float(torch.linalg.vector_norm(predicted))))
        records.append(
            {
                "episode_id": episode.episode_id,
                "plan_cosine": cosine,
                "plan_l2": float(torch.linalg.vector_norm(predicted - target)),
                "plan_digest": content_digest(predicted.tolist()),
                "target_digest": content_digest(target.tolist()),
            }
        )

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

    report = {
        "format": "taiji-r2-h3-5a-response-plan-ablation-v1",
        "status": "completed",
        "split": args.split,
        "dataset_digest": corpus.digest,
        "checkpoint": str(args.checkpoint),
        "checkpoint_digest": original_digest,
        "checkpoint_read_only": True,
        "episodes": len(records),
        "mean_plan_cosine": sum(item["plan_cosine"] for item in records) / len(records),
        "mean_plan_l2": sum(item["plan_l2"] for item in records) / len(records),
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
