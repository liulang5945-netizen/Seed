from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
import torch

from taiji import (
    FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE,
    FACTOR_RESPONSE_PLAN_TARGET_SLOT_WIDTH,
    FACTOR_RESPONSE_PLAN_TARGET_SLOTS,
    FactorizedResponsePlanTargetEncoder,
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisodeCorpus,
    ResponsePlanTargetEncoder,
    Taiji,
    TaijiConfig,
    checkpoint_roundtrip_preflight,
    content_digest,
    factorized_response_plan_preflight,
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


def test_factorized_response_target_is_train_bound_and_slot_normalized() -> None:
    corpus = _corpus()
    model = _model()
    before = content_digest(model.checkpoint())

    encoder = FactorizedResponsePlanTargetEncoder.fit(model, corpus)

    assert encoder.slots == FACTOR_RESPONSE_PLAN_TARGET_SLOTS
    assert encoder.slot_width == FACTOR_RESPONSE_PLAN_TARGET_SLOT_WIDTH
    assert encoder.phase_stride == FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE
    assert encoder.width == 48
    assert encoder.fit_episode_ids == tuple(
        item.episode_id for item in corpus.for_split("train")
    )
    targets = encoder.encode_corpus(model, corpus)
    assert set(targets) == {item.episode_id for item in corpus.episodes}
    for target in targets.values():
        assert target.shape == (48,)
        assert bool(torch.isfinite(target).all())
        slot_norms = torch.linalg.vector_norm(target.reshape(4, 12), dim=1)
        assert torch.allclose(slot_norms, torch.ones(4), atol=1e-6)
    assert content_digest(model.checkpoint()) == before

    restored = FactorizedResponsePlanTargetEncoder.from_payload(
        encoder.to_payload(),
        expected_corpus_digest=corpus.digest,
        expected_parent_checkpoint_digest=before,
    )
    assert restored.target_digest == encoder.target_digest
    assert content_digest(restored.to_payload()) == content_digest(encoder.to_payload())

    with pytest.raises(ValueError, match="parent checkpoint"):
        FactorizedResponsePlanTargetEncoder.from_payload(
            encoder.to_payload(),
            expected_parent_checkpoint_digest="1" * 64,
        )


def test_h37_factorized_plan_preflight_and_checkpoint_round_trip() -> None:
    corpus = _corpus()
    model = Taiji(
        TaijiConfig.capacity_profile(
            300_000,
            seed=20260917,
            additional_predictive_readouts=1,
            response_plan_width=48,
        ),
        episode_id="h3.7-factorized-test",
    )
    model.enable_response_plan_readout(
        plan_width=48,
        variant="factorized_v1",
        plan_slots=4,
        phase_stride=16,
    )
    encoder = FactorizedResponsePlanTargetEncoder.fit(model, corpus)
    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=48,
            response_plan_variant="factorized_v1",
            response_plan_slots=4,
            response_plan_phase_stride=16,
            response_plan_target_geometry="h3_7_factorized_response_chunks",
            learn_fabric=False,
            learn_predictive_context=False,
        ),
        response_plan_target_encoder=encoder,
    )

    root = Path(__file__).parents[2] / f"test-artifacts-h37-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        checkpoint_preflight = checkpoint_roundtrip_preflight(
            trainer,
            episode=corpus.for_split("train")[0],
            directory=root,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert checkpoint_preflight["status"] == "passed"
    factorized_preflight = factorized_response_plan_preflight(trainer)
    assert factorized_preflight["status"] == "passed"
    assert factorized_preflight["bridge_changed_after_byte_update"] or factorized_preflight[
        "slot_planner_changed_after_byte_update"
    ]
    assert trainer.model.parameter_count() <= 300_000

    trainer._prime(corpus.for_split("train")[0])
    trainer.model.begin_response_plan()
    for _ in range(16):
        trainer.model.advance_response_plan_phase()
    normal_phase_one = trainer.model.response_plan_probabilities()
    trainer.model.response_plan_readout.set_ablation_mode("slot_credit")
    ablated_phase_one = trainer.model.response_plan_probabilities()
    trainer.model.response_plan_readout.set_ablation_mode(None)
    assert not torch.equal(normal_phase_one, ablated_phase_one)

    trainer.train(epochs=1, max_episodes=1)
    checkpoint = trainer.checkpoint()
    restored = LanguageAlignmentTrainer.from_checkpoint(checkpoint, corpus)
    assert restored.config.response_plan_variant == "factorized_v1"
    assert restored.response_plan_target_digest == encoder.target_digest
    assert restored.checkpoint()["checkpoint_digest"] == checkpoint["checkpoint_digest"]
