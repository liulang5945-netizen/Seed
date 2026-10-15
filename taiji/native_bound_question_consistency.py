"""PLAN54: train-only activation consistency; native inference stays unchanged."""

from __future__ import annotations

import torch
from torch.nn import functional as F

LOSS_CONTRACT = "bound-question-stop-reference-cosine-v1"
CONSISTENCY_WEIGHT = 0.25


def state_consistency(reference, learner):
    """Equal-weight hidden and ordered byte states, reference always detached."""
    losses = []
    for a, b in (
        (reference.hidden, learner.hidden),
        (reference.question_states, learner.question_states),
    ):
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError("aligned nonempty question state shapes required")
        losses.append(
            (1 - (F.normalize(a.detach(), dim=-1) * F.normalize(b, dim=-1)).sum(-1)).mean()
        )
    return torch.stack(losses).mean()


def bound_question_consistency(
    model, reference_tape, learner_tape, reference_question, learner_question
):
    """Caller supplies input-only tapes; no labels, entity IDs or oracle addresses."""
    if reference_tape.positions != learner_tape.positions or len(reference_question) != len(
        learner_question
    ):
        raise ValueError("source positions and question byte lengths must align")
    with torch.no_grad():
        reference, _, _ = model.bind_question(
            reference_tape, model.question_state(reference_question)
        )
    learner, _, _ = model.bind_question(learner_tape, model.question_state(learner_question))
    return state_consistency(reference, learner)
