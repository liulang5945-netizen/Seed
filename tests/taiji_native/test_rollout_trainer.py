"""Scheduled-sampling trainer: zero-probability parity, RNG, formats and recovery."""

import pytest
import torch

from taiji.conditional_language_rollout import (
    ConditionalLanguageRolloutTrainer,
    rollout_logits,
)
from taiji.conditional_language_v2 import (
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
)
from taiji.internalization import content_digest
from taiji.task_core_v2 import TaskCoreConfigV2


def model() -> ConditionalLanguageCandidateV2:
    torch.set_num_threads(1)
    return ConditionalLanguageCandidateV2(
        TaskCoreConfigV2(
            width=16,
            embedding_width=8,
            key_width=8,
            slots=4,
            chunk_bytes=2,
            max_observation_bytes=128,
            max_evidence_bytes=64,
        )
    )


def observed(candidate, prompt=b"alpha evidence", *, scope="user-a", session="s1"):
    state = candidate.new_state(
        scope_id=scope, session_id=session, native_state_ref="cognitive:test:0"
    )
    return candidate.step_task(
        prompt,
        state,
        scope_id=scope,
        session_id=session,
        source_id="observed",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 128},
    )


def rollout(candidate, prompt, target, probability):
    return rollout_logits(
        candidate,
        prompt,
        target,
        probability=probability,
        scope_id="research",
        session_id="train",
        native_state_ref="research:train",
    )


def test_zero_probability_matches_teacher_forcing():
    candidate = model()
    expected = candidate.teacher_forced(b"alpha evidence", b"ABC")
    actual = rollout(candidate, b"alpha evidence", b"ABC", 0.0)
    assert torch.equal(actual, expected)


def test_greedy_sampling_is_deterministic_under_a_fixed_rng():
    candidate = model()
    rng = candidate.generator.get_state().clone()
    first = rollout(candidate, b"alpha evidence", b"ABC", 1.0)
    candidate.generator.set_state(rng)
    second = rollout(candidate, b"alpha evidence", b"ABC", 1.0)
    assert torch.equal(first, second)


def test_progress_and_probability_are_bounded():
    trainer = ConditionalLanguageRolloutTrainer(model())
    assert trainer.probability() == pytest.approx(1 / 256)
    trainer.steps = 255
    assert trainer.probability() == pytest.approx(0.5)
    trainer.steps = 10_000
    assert trainer.probability() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="max_probability"):
        ConditionalLanguageRolloutTrainer(model(), max_probability=1.5)
    with pytest.raises(ValueError, match="schedule_steps"):
        ConditionalLanguageRolloutTrainer(model(), schedule_steps=0)


def test_update_accounts_bytes_and_reaches_all_parameters():
    candidate = model()
    trainer = ConditionalLanguageRolloutTrainer(candidate)
    examples = [(b"alpha evidence", b"ABC"), (b"other evidence", b"DEF")]
    record = trainer.update(examples)
    assert trainer.steps == 1 and trainer.target_byte_exposures == 6
    assert candidate.parameter_revision == 1
    assert record["probability"] == pytest.approx(1 / 256)
    assert record["loss_nats"] > 0 and record["gradient_group_norms"]
    for parameter in candidate.parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)


def test_checkpoint_roundtrip_preserves_adam_rng_and_next_update(tmp_path):
    candidate = model()
    trainer = ConditionalLanguageRolloutTrainer(candidate, max_probability=0.5)
    examples = [(b"alpha evidence", b"ABC"), (b"other evidence", b"DEF")]
    trainer.update(examples)
    state = candidate.generate(observed(candidate), max_new_bytes=5)
    path = tmp_path / "rollout.pt"
    trainer.save(path, state, cursor={"update": 1})
    restored, resumed, cursor = ConditionalLanguageRolloutTrainer.from_checkpoint(
        torch.load(path, weights_only=True)
    )
    assert cursor == {"update": 1}
    assert restored.max_probability == 0.5 and restored.schedule_steps == 256
    assert (
        restored.checkpoint(resumed, cursor=cursor)["checkpoint_digest"]
        == trainer.checkpoint(state, cursor=cursor)["checkpoint_digest"]
    )
    assert (
        restored.model.generate(resumed, max_new_bytes=7).generated
        == candidate.generate(state, max_new_bytes=7).generated
    )
    trainer.update(examples)
    restored.update(examples)
    assert content_digest(trainer.model.state_dict()) == content_digest(restored.model.state_dict())
    assert content_digest(trainer.optimizer.state_dict()) == content_digest(
        restored.optimizer.state_dict()
    )


def test_rollout_and_teacher_forced_formats_are_mutually_rejected(tmp_path):
    candidate = model()
    state = observed(candidate)
    rollout_trainer = ConditionalLanguageRolloutTrainer(candidate)
    rollout_payload = rollout_trainer.checkpoint(state, cursor={"update": 0})
    with pytest.raises(ValueError, match="unsupported conditional language checkpoint"):
        ConditionalLanguageTrainerV2.from_checkpoint(rollout_payload)
    v2_trainer = ConditionalLanguageTrainerV2(model())
    v2_state = observed(v2_trainer.model)
    v2_payload = v2_trainer.checkpoint(v2_state, cursor={"update": 0})
    with pytest.raises(ValueError, match="unsupported rollout checkpoint"):
        ConditionalLanguageRolloutTrainer.from_checkpoint(v2_payload)
