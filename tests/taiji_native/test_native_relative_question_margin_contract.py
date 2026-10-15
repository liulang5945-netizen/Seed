"""Relative preference and stopped references, no answer metadata."""

from types import SimpleNamespace

import pytest
import torch

from taiji.native_relative_question_margin import bound_question_margin, state_margin
from tests.taiji_native.test_native_bound_question_consistency_contract import make_model


def state(vector):
    return SimpleNamespace(hidden=vector, question_states=vector.unsqueeze(0))


def test_margin_direction_collapse_and_detached_references():
    a = torch.tensor([1.0, 0.0], requires_grad=True)
    b = torch.tensor([0.0, 1.0], requires_grad=True)
    x = torch.tensor([0.5, 0.5], requires_grad=True)
    assert float(state_margin(state(a), state(b), state(a)).detach()) == 0
    assert float(state_margin(state(b), state(a), state(a)).detach()) > 1
    assert float(state_margin(state(a), state(a), state(a)).detach()) == pytest.approx(0.2)
    loss = state_margin(state(a), state(b), state(x))
    loss.backward()
    assert a.grad is None and b.grad is None and x.grad.abs().sum() > 0


def test_two_directions_credit_cut_and_alignment():
    torch.set_num_threads(1)
    model = make_model()
    for cut in (False, True):
        model.zero_grad(set_to_none=True)
        ref = model.evidence("reference54")
        learner = model.evidence("learner54", cut_credit=cut)
        loss = bound_question_margin(
            model,
            ref,
            learner,
            (b"material of alpha?", b"material of bravo?"),
            (b"material of delta?", b"material of bravo?"),
        )
        assert torch.isfinite(loss)
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
        assert (writer == 0) if cut else writer > 0
    with pytest.raises(ValueError, match="distinct"):
        bound_question_margin(model, ref, learner, (b"same", b"same"), (b"same", b"same"))
