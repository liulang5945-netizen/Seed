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
from taiji.response_plan_target import ByteAlignedResponsePlanTargetEncoder

FIXTURE = Path(__file__).parents[1] / "fixtures" / "r2_h3_5a_response_plan_v3.jsonl"


def _corpus() -> LanguageEpisodeCorpus:
    return LanguageEpisodeCorpus.from_jsonl([FIXTURE])


def _model() -> Taiji:
    return Taiji(
        TaijiConfig.capacity_profile(300_000, seed=20260916),
        episode_id="h3.6-target-test",
    )


def test_target_update_refreshes_prior_without_advancing_phase() -> None:
    model = _model()
    model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
    model.observe(65, learn=False, readout="predictive")
    model.begin_response_plan()
    protected = content_digest(model.predictive_readout.to_payload())
    before = model.snapshot().motor_probabilities.clone()
    model.learn_response_plan_target(torch.ones(48))
    readout = model.response_plan_readout
    assert readout.plan_step == 0
    assert not torch.equal(before, model.snapshot().motor_probabilities)
    torch.testing.assert_close(
        model.snapshot().motor_probabilities,
        model.response_plan_probabilities(),
        rtol=0,
        atol=0,
    )
    assert content_digest(model.predictive_readout.to_payload()) == protected


def test_training_first_byte_uses_post_target_update_prior(monkeypatch) -> None:
    corpus = _corpus()
    model = _model()
    model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=48,
            response_plan_variant="factorized_v1",
            response_plan_target_geometry="h3_7_factorized_response_chunks",
            learn_fabric=False,
            learn_predictive_context=False,
        ),
        response_plan_target_encoder=FactorizedResponsePlanTargetEncoder.fit(model, corpus),
    )
    readout = model.response_plan_readout
    original = readout.learn
    calls = []

    def checked(context, predicted, symbol, **kwargs):
        torch.testing.assert_close(predicted, readout.probabilities(context), rtol=0, atol=0)
        calls.append(symbol)
        return original(context, predicted, symbol, **kwargs)

    monkeypatch.setattr(readout, "learn", checked)
    trainer._target_pass(corpus.for_split("train")[0], learn=True)
    assert calls


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_active_plan_credit_matches_finite_difference(phase) -> None:
    model = _model()
    model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
    model.begin_response_plan()
    readout = model.response_plan_readout
    readout._plan_state = torch.linspace(-0.4, 0.4, 48, dtype=torch.float64)
    readout._plan_step = phase * 16
    upstream = torch.linspace(-0.7, 0.8, 48, dtype=torch.float64)
    analytical = readout._plan_state_feedback(upstream)
    numerical = torch.zeros_like(upstream)
    epsilon = 1e-6
    for index in range(48):
        original = readout._plan_state[index].item()
        readout._plan_state[index] = original + epsilon
        plus = float(readout._active_plan_vector() @ upstream)
        readout._plan_state[index] = original - epsilon
        minus = float(readout._active_plan_vector() @ upstream)
        readout._plan_state[index] = original
        numerical[index] = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(analytical, numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("response", ["中" * 24, "ab中😀" * 12, "a"])
def test_repaired_target_windows_follow_renderer_byte_phase(response) -> None:
    chunks = ByteAlignedResponsePlanTargetEncoder.chunks(response, slots=4, phase_stride=16)
    raw = response.encode("utf-8")
    assert b"".join(chunks) == raw
    for index, chunk in enumerate(chunks):
        for offset, symbol in enumerate(chunk):
            absolute = sum(map(len, chunks[:index])) + offset
            assert min(absolute // 16, 3) == index
            assert raw[absolute] == symbol


def test_repaired_target_rejects_legacy_geometry_and_roundtrips() -> None:
    model, corpus = _model(), _corpus()
    encoder = ByteAlignedResponsePlanTargetEncoder.fit(model, corpus)
    assert torch.equal(encoder.slot_scale, torch.ones(4))
    restored = ByteAlignedResponsePlanTargetEncoder.from_payload(encoder.to_payload())
    assert restored.target_digest == encoder.target_digest
    with pytest.raises(ValueError, match="format"):
        FactorizedResponsePlanTargetEncoder.from_payload(encoder.to_payload())
    with pytest.raises(ValueError, match="format"):
        ByteAlignedResponsePlanTargetEncoder.from_payload(
            FactorizedResponsePlanTargetEncoder.fit(model, corpus).to_payload()
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
    assert encoder.fit_episode_ids == tuple(item.episode_id for item in corpus.for_split("train"))
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
    assert (
        factorized_preflight["bridge_changed_after_byte_update"]
        or factorized_preflight["slot_planner_changed_after_byte_update"]
    )
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

    # The ablation is transient and must survive score_episode's read-only
    # model replacement without entering the checkpoint payload.
    before_evaluation = content_digest(restored.checkpoint())
    restored.model.response_plan_readout.set_ablation_mode("slot_credit")
    restored.evaluate("dev")
    assert restored.model.response_plan_readout.ablation_mode == "slot_credit"
    assert content_digest(restored.checkpoint()) == before_evaluation
    restored.model.response_plan_readout.set_ablation_mode(None)
