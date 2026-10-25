"""Per-byte evidence, positional addressing and v2 recovery for the candidate."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from taiji.conditional_language_v2 import (
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
)
from taiji.internalization import content_digest
from taiji.task_core_v2 import TaskCoreConfigV2
from taiji.task_state_v2 import TaskStateV2

V1_CHECKPOINT = Path("data/checkpoints/n1_plan169_task_core_20261025/final.pt")


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


def test_per_byte_memory_is_lossless_and_bounded():
    candidate = model()
    prompt = b"alpha evidence"
    state = observed(candidate, prompt)
    assert len(state.byte_memory) == len(prompt)
    assert len(state.byte_refs) == len(prompt)
    assert state.byte_refs[0] == (0, 0, 1) and state.byte_refs[-1] == (
        0,
        len(prompt) - 1,
        len(prompt),
    )
    assert len(state.workspace) == len(state.evidence_refs) <= 4
    assert state.observations[0] == ("observed", prompt)
    turn = state.reset("turn")
    assert torch.equal(turn.byte_memory, state.byte_memory)
    cleared = state.reset("session")
    assert len(cleared.byte_memory) == 0 and not cleared.byte_refs


def test_schema_is_strict_and_v1_payloads_are_rejected():
    state = observed(model())
    restored = TaskStateV2.from_payload(state.to_payload())
    assert content_digest(restored.to_payload()) == content_digest(state.to_payload())
    malformed = state.to_payload()
    malformed.pop("scope_id")
    with pytest.raises(ValueError, match="schema"):
        TaskStateV2.from_payload(malformed)
    v1 = state.to_payload()
    v1["format"] = "taiji-task-state-v1"
    with pytest.raises(ValueError, match="schema"):
        TaskStateV2.from_payload(v1)


def test_positional_keys_make_order_and_step_matter():
    candidate = model()
    first = observed(candidate, b"abcdef")
    swapped = replace(first, byte_memory=torch.flip(first.byte_memory, dims=(0,)))
    assert not torch.equal(candidate.next_logits(first)[0], candidate.next_logits(swapped)[0])
    assert torch.equal(candidate.next_logits(first)[0], candidate.next_logits(first)[0])
    renamed = replace(first, scope_id="other-id", session_id="other-id")
    assert torch.equal(candidate.next_logits(first)[0], candidate.next_logits(renamed)[0])
    weights = candidate.attention_weights(first)
    assert weights.shape == (len(first.byte_memory),)
    assert float(weights.detach().sum()) == pytest.approx(1.0, abs=1e-5)


def test_step_position_changes_the_same_state():
    candidate = model()
    state = observed(candidate, b"abcdef")
    advanced = replace(
        state,
        generated=b"x",
        previous_byte=ord("x"),
        decoder_hidden=torch.zeros(16),
    )
    assert not torch.equal(candidate.next_logits(state)[0], candidate.next_logits(advanced)[0])


def test_byte_evidence_is_consumed_and_detach_stops_encoder_credit():
    candidate = model()
    state = observed(candidate, b"prefix with multiple distinct chunks")
    logits = candidate.next_logits(state)[0]
    changed = replace(state, byte_memory=-state.byte_memory, workspace=-state.workspace)
    assert not torch.equal(logits, candidate.next_logits(changed)[0])
    candidate.teacher_forced(b"prefix with multiple distinct chunks", b"ABC").sum().backward()
    for parameter in (
        candidate.core.embedding.weight,
        candidate.core.encoder.weight_ih_l0,
        candidate.core.keys.weight,
        candidate.core.values.weight,
        candidate.query.weight,
        candidate.position.weight,
        candidate.output.weight,
    ):
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
    candidate.zero_grad(set_to_none=True)
    detached = replace(state, byte_memory=state.byte_memory.detach(), hidden=state.hidden.detach())
    candidate.next_logits(detached)[0].sum().backward()
    assert candidate.core.encoder.weight_ih_l0.grad is None
    assert candidate.output.weight.grad is not None


def test_teacher_forcing_is_causal():
    candidate = model()
    first = candidate.teacher_forced(b"same prefix", b"ABX")
    second = candidate.teacher_forced(b"same prefix", b"ABY")
    assert torch.equal(first, second)
    assert not torch.equal(first[0], candidate.teacher_forced(b"different facts", b"ABX")[0])


def test_checkpoint_v2_preserves_generation_adam_rng_and_next_update(tmp_path):
    candidate = model()
    trainer = ConditionalLanguageTrainerV2(candidate)
    examples = [(b"alpha evidence", b"ABC"), (b"other evidence", b"DEF")]
    trainer.update(examples)
    state = candidate.generate(observed(candidate), max_new_bytes=5)
    path = tmp_path / "candidate_v2.pt"
    trainer.save(path, state, cursor={"update": 1})
    restored, resumed, cursor = ConditionalLanguageTrainerV2.from_checkpoint(
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


@pytest.mark.skipif(not V1_CHECKPOINT.is_file(), reason="N1-169 evidence checkpoint is absent")
def test_v1_checkpoint_is_explicitly_rejected():
    payload = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=True)
    assert payload["version"] == 1
    with pytest.raises(ValueError, match="unsupported"):
        ConditionalLanguageTrainerV2.from_checkpoint(payload)


def test_default_capacity_stays_under_the_one_million_parameter_ceiling():
    before = torch.get_rng_state().clone()
    candidate = ConditionalLanguageCandidateV2()
    assert candidate.parameter_count() == 970945
    assert candidate.parameter_count() <= 1000000
    assert torch.equal(torch.get_rng_state(), before)
