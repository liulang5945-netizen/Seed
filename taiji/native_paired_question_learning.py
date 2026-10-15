"""PLAN40: paired questions over immutable observations, auxiliary once."""

from __future__ import annotations

import torch

from .native_evidence_fidelity import AUXILIARY_WEIGHT, FidelityLoss, observation_logits

LOSS_CONTRACT = "taiji-paired-question-observation-fidelity-v1"


def paired_question_loss(
    model, session, question, target, *, second_question=None, second_target=None, arm="position"
):
    if arm not in ("position", "terminal", "cut") or (second_question is None) != (
        second_target is None
    ):
        raise ValueError("valid arm and complete optional second question/target required")
    cut = arm == "cut"
    tape = model.evidence(session, cut_credit=cut, terminal_per_source=arm == "terminal")
    losses = []
    questions = [(question, target)]
    if second_question is not None:
        questions.append((second_question, second_target))
    for q, answer in questions:
        logits, _ = model.teacher_logits(tape, session, q, answer)
        labels = torch.tensor([*answer, 256], device=logits.device)
        losses.append(torch.nn.functional.cross_entropy(logits, labels))
    qa = torch.stack(losses).mean()
    logits, targets = observation_logits(model, tape, session, cut_credit=cut)
    observation = torch.nn.functional.cross_entropy(logits, targets)
    return FidelityLoss(qa + AUXILIARY_WEIGHT * observation, qa, observation, tape)
