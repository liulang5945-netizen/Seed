"""Run the H3.6 response-plan target encoder save/restore smoke."""

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
    content_digest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    model = Taiji(
        TaijiConfig.capacity_profile(
            300_000,
            additional_predictive_readouts=1,
            response_plan_width=32,
            seed=args.seed,
        ),
        episode_id="h3.6-target-encoder-smoke",
    )
    model.enable_response_plan_readout(plan_width=32)
    parent_digest = content_digest(model.checkpoint())
    encoder = ResponsePlanTargetEncoder.fit(
        model,
        corpus,
        parent_checkpoint_digest=parent_digest,
    )
    after_fit_digest = content_digest(model.checkpoint())
    targets = encoder.encode_corpus(model, corpus)
    after_encode_digest = content_digest(model.checkpoint())

    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=32,
            response_plan_target_geometry="h3_6_whitened_native_compositional",
            learn_fabric=False,
            learn_predictive_context=False,
        ),
        response_plan_target_encoder=encoder,
    )
    trainer_payload = trainer.checkpoint()
    trainer_checkpoint_digest = str(trainer_payload["checkpoint_digest"])

    target_payload_path = PROJECT_ROOT / "checkpoints" / ".r2-h3-6-target-encoder-smoke.pt"
    trainer_payload_path = PROJECT_ROOT / "checkpoints" / ".r2-h3-6-target-trainer-smoke.pt"
    target_payload_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(encoder.to_payload(), target_payload_path)
        restored_payload = torch.load(target_payload_path, map_location="cpu", weights_only=False)
        restored = ResponsePlanTargetEncoder.from_payload(
            restored_payload,
            expected_corpus_digest=corpus.digest,
            expected_parent_checkpoint_digest=parent_digest,
        )
        torch.save(trainer_payload, trainer_payload_path)
        restored_trainer_payload = torch.load(
            trainer_payload_path, map_location="cpu", weights_only=False
        )
        restored_trainer = LanguageAlignmentTrainer.from_checkpoint(
            restored_trainer_payload, corpus
        )
    finally:
        target_payload_path.unlink(missing_ok=True)
        trainer_payload_path.unlink(missing_ok=True)

    report = {
        "format": "taiji-r2-h3-6-target-encoder-smoke-v1",
        "status": "passed",
        "training_performed": False,
        "runtime_oracle_used": False,
        "dataset": corpus.manifest(),
        "seed": args.seed,
        "width": encoder.width,
        "model_context_dim": encoder.model_context_dim,
        "effective_rank": encoder.effective_rank,
        "fit_episode_ids": list(encoder.fit_episode_ids),
        "corpus_digest": encoder.corpus_digest,
        "parent_checkpoint_digest": encoder.parent_checkpoint_digest,
        "target_encoder_digest": encoder.target_digest,
        "trainer_target_encoder_digest": trainer.response_plan_target_digest,
        "trainer_checkpoint_digest": trainer_checkpoint_digest,
        "target_count": len(targets),
        "target_norms": {
            episode_id: float(torch.linalg.vector_norm(target))
            for episode_id, target in targets.items()
        },
        "payload_roundtrip_digest_match": restored.target_digest == encoder.target_digest,
        "trainer_checkpoint_roundtrip": (
            restored_trainer.checkpoint()["checkpoint_digest"] == trainer_checkpoint_digest
        ),
        "checkpoint_digest_after_fit": after_fit_digest,
        "checkpoint_digest_after_encode": after_encode_digest,
        "teacher_checkpoint_restored": (parent_digest == after_fit_digest == after_encode_digest),
    }
    if not report["payload_roundtrip_digest_match"]:
        raise RuntimeError("target encoder payload round-trip digest mismatch")
    if not report["trainer_checkpoint_roundtrip"]:
        raise RuntimeError("H3.6 trainer checkpoint round-trip digest mismatch")
    if not report["teacher_checkpoint_restored"]:
        raise RuntimeError("target encoder changed its teacher checkpoint")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "effective_rank": encoder.effective_rank,
                "target_encoder_digest": encoder.target_digest,
                "teacher_checkpoint_restored": report["teacher_checkpoint_restored"],
                "payload_roundtrip_digest_match": report["payload_roundtrip_digest_match"],
                "trainer_checkpoint_roundtrip": report["trainer_checkpoint_roundtrip"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
