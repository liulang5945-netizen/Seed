"""PLAN57: same-observation question negatives, input-only activation objective."""

from __future__ import annotations

import torch

from .native_bound_question_consistency import state_consistency

LOSS_CONTRACT = "bound-question-relative-cosine-margin-v1"
MARGIN = 0.2
MARGIN_WEIGHT = 0.25


def state_margin(positive, negative, learner):
    return torch.relu(
        MARGIN + state_consistency(positive, learner) - state_consistency(negative, learner)
    )


def bound_question_margin(
    model, reference_tape, learner_tape, reference_questions, learner_questions
):
    if (
        len(reference_questions) != 2
        or len(learner_questions) != 2
        or reference_questions[0] == reference_questions[1]
        or learner_questions[0] == learner_questions[1]
        or reference_tape.positions != learner_tape.positions
        or any(
            len(a) != len(b) for a, b in zip(reference_questions, learner_questions, strict=True)
        )
    ):
        raise ValueError("two distinct aligned questions and source positions required")
    with torch.no_grad():
        references = [
            model.bind_question(reference_tape, model.question_state(q))[0]
            for q in reference_questions
        ]
    learners = [
        model.bind_question(learner_tape, model.question_state(q))[0] for q in learner_questions
    ]
    return torch.stack(
        [state_margin(references[i], references[1 - i], learners[i]) for i in (0, 1)]
    ).mean()
