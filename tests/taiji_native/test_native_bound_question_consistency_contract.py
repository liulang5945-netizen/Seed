"""Gradient, masking and inference contracts for the consistency objective."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from taiji.native_bound_question_consistency import bound_question_consistency, state_consistency
from taiji.native_contextual_question_evidence_candidate import ContextualQuestionEvidenceCandidate
from taiji.native_evidence_candidate import Observation


def test_detached_reference_and_identity():
    a = torch.randn(3, 32, requires_grad=True)
    b = torch.randn(3, 32, requires_grad=True)
    reference = SimpleNamespace(hidden=a[-1], question_states=a)
    learner = SimpleNamespace(hidden=b[-1], question_states=b)
    loss = state_consistency(reference, learner)
    assert torch.isfinite(loss)
    loss.backward()
    assert a.grad is None and b.grad.abs().sum() > 0
    assert abs(float(state_consistency(reference, reference).detach())) < 1e-6
    with pytest.raises(ValueError, match="shapes"):
        state_consistency(reference, SimpleNamespace(hidden=b[-1], question_states=b[:2]))


def make_model():
    torch.manual_seed(11)
    model = ContextualQuestionEvidenceCandidate()
    with torch.no_grad():
        model.context_gate.fill_(0.15)
        model.local_fusion.weight.fill_(0.02)
    for session, name in (("reference54", "alpha"), ("learner54", "delta")):
        model.observe(session, Observation("text", f"{name} uses quartz.", 1, "one"))
        model.observe(session, Observation("text", "bravo uses copper.", 2, "two"))
    return model


def test_learning_credit_cut_and_native_neutrality():
    torch.set_num_threads(1)
    model = make_model()
    tape = model.evidence("learner54")
    q = b"material of delta?"
    before = model.teacher_logits(tape, "learner54", q, b"quartz")[0].detach()
    for cut in (False, True):
        model.zero_grad(set_to_none=True)
        loss = bound_question_consistency(
            model,
            model.evidence("reference54"),
            model.evidence("learner54", cut_credit=cut),
            b"material of alpha?",
            q,
        )
        loss.backward()
        for prefix in ("question.", "context_question.", "local_projection."):
            assert (
                sum(
                    float(p.grad.abs().sum())
                    for n, p in model.named_parameters()
                    if n.startswith(prefix) and p.grad is not None
                )
                > 0
            )
        writer = sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if n.startswith("writer.") and p.grad is not None
        )
        assert (writer == 0) if cut else (writer > 0)
    after = model.teacher_logits(model.evidence("learner54"), "learner54", q, b"quartz")[0]
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match="align"):
        bound_question_consistency(model, replace(tape, positions=()), tape, q, q)
