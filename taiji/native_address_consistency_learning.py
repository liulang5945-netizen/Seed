"""PLAN46: internal frozen-reader address consistency, training only."""

from __future__ import annotations

import torch

from .native_evidence_fidelity import AUXILIARY_WEIGHT, observation_logits

LOSS_CONTRACT = "taiji-self-reader-address-consistency-v1"
ADDRESS_WEIGHT = 1.0
PRESERVATION_WEIGHT = 0.5


def address_consistency_loss(model, session, questions, references, *, preserve_logits):
    """Each question is (bytes, target bytes); references contain no gold addresses."""
    if len(questions) not in (1, 2) or len(questions) != len(references):
        raise ValueError("one or two complete frozen question references required")
    tape = model.evidence(session)
    qa, address, preservation = [], [], []
    for (question, target), reference in zip(questions, references, strict=True):
        if (
            tape.positions != reference["positions"]
            or [*target, 256] != reference["target_symbols"]
        ):
            raise ValueError("reference must preserve source positions and output target")
        if reference["logits"].requires_grad or reference["addresses"].requires_grad:
            raise ValueError("frozen self-reader references must be detached")
        logits, weights = model.teacher_logits(tape, session, question, target)
        if (
            weights.shape != reference["addresses"].shape
            or logits.shape != reference["logits"].shape
        ):
            raise ValueError("reference trajectory dimensions changed")
        labels = torch.tensor([*target, 256], device=logits.device)
        qa.append(torch.nn.functional.cross_entropy(logits, labels))
        tiny = torch.finfo(weights.dtype).tiny
        address.append(
            torch.nn.functional.kl_div(
                weights.clamp_min(tiny).log(), reference["addresses"], reduction="sum"
            )
            / len(labels)
        )
        preservation.append(
            torch.nn.functional.kl_div(
                logits.log_softmax(-1), reference["logits"].softmax(-1), reduction="sum"
            )
            / len(labels)
            if preserve_logits
            else logits.new_zeros(())
        )
    qa, address, preservation = [
        torch.stack(values).mean() for values in (qa, address, preservation)
    ]
    logits, labels = observation_logits(model, tape, session)
    observation = torch.nn.functional.cross_entropy(logits, labels)
    return dict(
        total=qa
        + AUXILIARY_WEIGHT * observation
        + ADDRESS_WEIGHT * address
        + PRESERVATION_WEIGHT * preservation,
        qa=qa,
        observation=observation,
        address=address,
        preservation=preservation,
    )
