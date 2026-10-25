"""Causal consumers, real sequence credit and recoverable opt-in task state."""

from dataclasses import replace

import pytest
import torch

from taiji.conditional_language import ConditionalLanguageCandidate, ConditionalLanguageTrainer
from taiji.internalization import content_digest
from taiji.task_core import TaskCoreConfig
from taiji.task_state import TaskStateV1


def model():
    torch.set_num_threads(1)
    return ConditionalLanguageCandidate(
        TaskCoreConfig(
            width=16,
            embedding_width=8,
            key_width=8,
            slots=4,
            chunk_bytes=2,
            max_observation_bytes=128,
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


def test_task_state_domains_provenance_and_strict_schema():
    candidate = model()
    state = observed(candidate)
    turn = state.reset("turn")
    assert torch.equal(turn.hidden, state.hidden) and turn.observations == state.observations
    assert len(state.workspace) == 4 and state.evidence_refs[-1][-1] == len(b"alpha evidence")
    cleared = state.reset("session")
    assert not cleared.observations and len(cleared.workspace) == 0
    restored = TaskStateV1.from_payload(state.to_payload())
    assert content_digest(restored.to_payload()) == content_digest(state.to_payload())
    malformed = state.to_payload()
    malformed.pop("scope_id")
    with pytest.raises(ValueError, match="schema"):
        TaskStateV1.from_payload(malformed)
    with pytest.raises(ValueError, match="scope/session"):
        candidate.step_task(
            b"b",
            state,
            scope_id="other",
            session_id="s1",
            source_id="next",
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 128},
        )
    with pytest.raises(ValueError, match="resource"):
        candidate.step_task(
            b"b",
            state,
            scope_id="user-a",
            session_id="s1",
            source_id="next",
            resource_snapshot={"remaining_steps": 0, "max_observation_bytes": 128},
        )


def test_teacher_forcing_is_causal_and_prefix_identity_is_not_a_model_input():
    candidate = model()
    first = candidate.teacher_forced(b"same prefix", b"ABX")
    second = candidate.teacher_forced(b"same prefix", b"ABY")
    assert torch.equal(first, second)  # target[t] is not an input to logits[t].
    assert not torch.equal(first[0], candidate.teacher_forced(b"different facts", b"ABX")[0])
    state = observed(candidate)
    renamed = replace(state, scope_id="different-id", session_id="different-id")
    assert torch.equal(candidate.next_logits(state)[0], candidate.next_logits(renamed)[0])


def test_actual_task_and_evidence_consumers_and_detach_credit_gate():
    candidate = model()
    state = observed(candidate)
    logits = candidate.next_logits(state)[0]
    changed = replace(state, hidden=-state.hidden, workspace=-state.workspace)
    assert not torch.equal(logits, candidate.next_logits(changed)[0])
    assert torch.equal(logits, candidate.next_logits(state)[0])
    candidate.teacher_forced(b"prefix with multiple distinct chunks", b"ABC").sum().backward()
    for parameter in (
        candidate.core.embedding.weight,
        candidate.core.encoder.weight_ih_l0,
        candidate.core.keys.weight,
        candidate.core.values.weight,
        candidate.query.weight,
        candidate.output.weight,
    ):
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
    assert bool(candidate.core.embedding.weight.grad[ord("x")].abs().sum() > 0)
    candidate.zero_grad(set_to_none=True)
    detached = TaskStateV1.from_payload(
        observed(candidate, b"prefix with multiple distinct chunks").to_payload()
    )
    candidate.next_logits(detached)[0].sum().backward()
    assert candidate.core.encoder.weight_ih_l0.grad is None
    assert not bool(candidate.core.embedding.weight.grad[ord("x")].abs().sum() > 0)
    assert candidate.output.weight.grad is not None


def test_state_and_evidence_cache_cannot_cross_scope_or_parameter_revision():
    candidate = model()
    state = observed(candidate)
    cache = candidate.evidence_cache(state)
    with pytest.raises(ValueError, match="cache"):
        candidate.next_logits(observed(candidate, b"other observations"), cache=cache)
    trainer = ConditionalLanguageTrainer(candidate)
    trainer.update([(b"alpha evidence", b"AB")])
    with pytest.raises(ValueError, match="parameter revision"):
        candidate.next_logits(state)


def test_full_checkpoint_preserves_generation_adam_rng_and_next_update(tmp_path):
    candidate = model()
    trainer = ConditionalLanguageTrainer(candidate)
    examples = [(b"alpha evidence", b"ABC"), (b"other evidence", b"DEF")]
    trainer.update(examples)
    state = candidate.generate(observed(candidate), max_new_bytes=5)
    path = tmp_path / "candidate.pt"
    trainer.save(path, state, cursor={"update": 1})
    restored, resumed, cursor = ConditionalLanguageTrainer.from_checkpoint(
        torch.load(path, weights_only=True)
    )
    assert cursor == {"update": 1}
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
    with pytest.raises(ValueError, match="write-once"):
        trainer.save(path, observed(candidate), cursor={"update": 2})


def test_native_opt_in_factory_does_not_modify_the_existing_model():
    from taiji.adapter import TSKV8Adapter

    native = TSKV8Adapter(episode_id="candidate-factory-test")
    before = content_digest(native.native_checkpoint())
    candidate, state = native.create_conditional_language_candidate(
        scope_id="user",
        session_id="s",
        config=TaskCoreConfig(width=16, embedding_width=8, key_width=8, slots=4),
    )
    assert candidate.parameter_count() > 0 and state.native_state_ref.startswith("cognitive:")
    assert content_digest(native.native_checkpoint()) == before
    assert "conditional_language" not in native.checkpoint()


def test_default_capacity_is_within_the_shared_one_million_parameter_ceiling():
    before = torch.get_rng_state().clone()
    candidate = ConditionalLanguageCandidate()
    assert candidate.parameter_count() == 938177
    assert torch.equal(torch.get_rng_state(), before)


def test_calibration_control_cannot_silently_update_the_legacy_model():
    from scripts.training.calibrate_taiji_n1_plan169_task_core import LocalLanguageControl

    with pytest.raises(ValueError, match="read-only"):
        LocalLanguageControl(object()).update_episode(None, 0)
