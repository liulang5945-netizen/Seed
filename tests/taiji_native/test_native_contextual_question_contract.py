"""Ordered question branch: exact initial parent, full moments and credit barriers."""

from dataclasses import replace

import pytest
import torch

from scripts.training import audit_taiji_n1_contextual_question as audit
from tests.taiji_native._native_relation_fixtures import local_parent


def migrated():
    payload = local_parent()
    model, optimizer = audit.transfer.migrate(payload)
    audit.prepare(model)
    return model, optimizer, payload


def test_zero_gate_parent_identity_and_complete_restore(tmp_path):
    torch.set_num_threads(1)
    model, optimizer, payload = migrated()
    old = audit.parent.LocalQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    audit.prepare(old)
    assert sum(p.numel() for p in model.parameters()) == 55480
    assert len(optimizer.state) == len(list(old.parameters()))
    for name, parameter in model.named_parameters():
        if name in audit.transfer.NEW_PARAMETERS:
            assert parameter not in optimizer.state
        else:
            assert int(optimizer.state[parameter]["step"]) == (
                6144 if name in audit.parent.transfer.NEW_PARAMETERS else 12288
            )
    with torch.no_grad():
        a, aw = old.teacher_logits(
            old.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
        )
        b, bw = model.teacher_logits(
            model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
        )
        assert torch.equal(a, b) and torch.equal(aw, bw)
        assert old.generate_trace(
            old.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, 32
        ) == model.generate_trace(model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, 32)
    saved = audit.save(model, optimizer, tmp_path / "zero.pt")
    assert audit.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="explicit"):
        audit.transfer.ContextualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    broken = model.bundle(optimizer)
    del broken["state"]["context_question.weight_ih_l0"]
    with pytest.raises(RuntimeError, match="Missing key"):
        audit.transfer.ContextualQuestionEvidenceCandidate.restore_bundle(broken)


def test_gate_credit_context_credit_and_cut_observed_paths():
    model, _, _ = migrated()
    logits, _ = model.teacher_logits(
        model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
    )
    logits.square().mean().backward()
    assert model.context_gate.grad.abs().sum() > 0
    assert model.context_question.weight_ih_l0.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.context_gate.fill_(0.15)
    logits, _ = model.teacher_logits(
        model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
    )
    logits.square().mean().backward()
    for parameter in (
        model.context_question.weight_ih_l0,
        model.context_question.weight_hh_l0,
        model.local_projection.weight,
        model.local_log_temperature,
        model.writer.weight_ih,
        model.question.weight_ih,
    ):
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
    model.zero_grad(set_to_none=True)
    original_features = model.local_features
    observed = []

    def tracked(content):
        result = original_features(content)
        if content != audit.QUESTION:
            result.retain_grad()
            observed.append(result)
        return result

    model.local_features = tracked
    try:
        logits, _ = model.teacher_logits(
            model.evidence(audit.SESSION, cut_credit=True),
            audit.SESSION,
            audit.QUESTION,
            audit.TARGET,
        )
        logits.square().mean().backward()
    finally:
        model.__dict__.pop("local_features", None)
    assert observed and all(t.grad is None or t.grad.abs().sum() == 0 for t in observed)
    assert model.writer.weight_ih.grad is None or model.writer.weight_ih.grad.abs().sum() == 0


def test_terminal_contexts_unicode_and_session_guards():
    model, _, _ = migrated()
    tape = model.evidence(audit.SESSION, terminal_per_source=True)
    state = model.question_state(audit.QUESTION)
    captured = []
    handle = model.context_question.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            _, alignment, _ = model.bind_question(tape, state)
        blocks, end = [], 0
        for source in tape.sources:
            length = len(source.content)
            end += length
            blocks.append(tape.states[end - 1].expand(length, -1))
        expected = alignment @ torch.cat(blocks)
        assert torch.equal(
            captured[0][
                0, :, model.config.embedding : model.config.embedding + model.config.hidden
            ],
            expected,
        )
    finally:
        handle.remove()
    with torch.no_grad():
        encoded = "材料？".encode()
        initial = model.question_state(encoded)
        _, alignment, _ = model.bind_question(tape, initial)
        sequence, hidden, confidence = model.context_sequence(tape, initial, alignment)
        assert (
            sequence.shape == (len(encoded), 32)
            and hidden.shape == (32,)
            and confidence.shape == (len(encoded),)
        )
        with pytest.raises(ValueError, match="cross sessions"):
            model.step(tape, "wrong", state, 256)
        with pytest.raises(ValueError, match="cross sessions"):
            model.step(tape, audit.SESSION, replace(state, input_tape=replace(tape)), 256)
