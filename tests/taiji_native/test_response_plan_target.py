from __future__ import annotations

from pathlib import Path

import pytest
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

FIXTURE = Path(__file__).parents[1] / "fixtures" / "r2_h3_5a_response_plan_v3.jsonl"


def _corpus() -> LanguageEpisodeCorpus:
    return LanguageEpisodeCorpus.from_jsonl([FIXTURE])


def _model() -> Taiji:
    return Taiji(
        TaijiConfig.capacity_profile(300_000, seed=20260916),
        episode_id="h3.6-target-test",
    )


def test_response_plan_target_fit_is_train_only_and_restores_teacher() -> None:
    corpus = _corpus()
    model = _model()
    before = content_digest(model.checkpoint())

    encoder = ResponsePlanTargetEncoder.fit(model, corpus)

    assert encoder.width == 32
    assert encoder.fit_episode_ids == tuple(item.episode_id for item in corpus.for_split("train"))
    assert encoder.corpus_digest == corpus.digest
    assert content_digest(model.checkpoint()) == before

    targets = encoder.encode_corpus(model, corpus)
    assert set(targets) == {item.episode_id for item in corpus.episodes}
    assert all(target.shape == (32,) for target in targets.values())
    assert all(bool(torch.isfinite(target).all()) for target in targets.values())
    assert all(
        torch.isclose(torch.linalg.vector_norm(target), torch.tensor(1.0))
        for target in targets.values()
    )
    assert content_digest(model.checkpoint()) == before


def test_response_plan_target_payload_round_trip_and_binding() -> None:
    corpus = _corpus()
    model = _model()
    parent_digest = content_digest(model.checkpoint())
    encoder = ResponsePlanTargetEncoder.fit(
        model,
        corpus,
        parent_checkpoint_digest=parent_digest,
    )

    restored = ResponsePlanTargetEncoder.from_payload(
        encoder.to_payload(),
        expected_corpus_digest=corpus.digest,
        expected_parent_checkpoint_digest=parent_digest,
    )
    assert restored.target_digest == encoder.target_digest
    assert content_digest(restored.to_payload()) == content_digest(encoder.to_payload())

    with pytest.raises(ValueError, match="corpus digest"):
        ResponsePlanTargetEncoder.from_payload(
            encoder.to_payload(),
            expected_corpus_digest="0" * 64,
        )
    with pytest.raises(ValueError, match="parent checkpoint"):
        ResponsePlanTargetEncoder.from_payload(
            encoder.to_payload(),
            expected_parent_checkpoint_digest="1" * 64,
        )


def test_h36_target_is_checkpointed_and_reused_after_child_restore() -> None:
    corpus = _corpus()
    model = _model()
    model.enable_response_plan_readout(plan_width=32)
    encoder = ResponsePlanTargetEncoder.fit(model, corpus)
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

    assert trainer.response_plan_target_digest == encoder.target_digest
    assert set(trainer.response_plan_targets) == {
        item.episode_id for item in corpus.for_split("train")
    }
    trainer.train(epochs=1, max_episodes=1)
    checkpoint = trainer.checkpoint()
    restored = LanguageAlignmentTrainer.from_checkpoint(checkpoint, corpus)

    assert restored.config.response_plan_target_geometry == ("h3_6_whitened_native_compositional")
    assert restored.response_plan_target_digest == encoder.target_digest
    assert restored.checkpoint()["checkpoint_digest"] == checkpoint["checkpoint_digest"]
