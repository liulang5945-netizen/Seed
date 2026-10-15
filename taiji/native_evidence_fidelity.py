"""Versioned observation-only auxiliary learning for the opt-in native candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .native_evidence_candidate import NativeEvidenceCandidate
from .position_evidence_reference import EvidenceTape

LOSS_CONTRACT = "taiji-observation-byte-fidelity-v1"
AUXILIARY_WEIGHT = 0.25


def observation_logits(
    model: NativeEvidenceCandidate, tape: EvidenceTape, session: str, *, cut_credit=False
):
    """Targets derive exclusively from this session's observed external bytes."""
    if tape.session != session or tape.sources != model.sources(session):
        raise ValueError("fidelity tape must match the current session's external observations")
    expected_positions = tuple(
        (source.source, index) for source in tape.sources for index in range(len(source.content))
    )
    if tape.positions != expected_positions or len(tape.states) != len(expected_positions):
        raise ValueError("fidelity states must align with every observed byte")
    states = torch.stack(tape.states)
    if cut_credit:
        states = states.detach()
    logits = model.readout(torch.cat((states, model.value(states)), dim=-1))
    targets = torch.tensor(
        [byte for source in tape.sources for byte in source.content],
        dtype=torch.long,
        device=logits.device,
    )
    return logits, targets


@dataclass(frozen=True)
class FidelityLoss:
    total: torch.Tensor
    qa: torch.Tensor
    observation: torch.Tensor
    tape: EvidenceTape


def learning_loss(model, session: str, question: bytes, target: bytes, *, arm="position"):
    if arm not in ("position", "terminal", "cut"):
        raise ValueError("unknown fidelity control arm")
    cut = arm == "cut"
    tape = model.evidence(session, cut_credit=cut, terminal_per_source=arm == "terminal")
    qa_logits, _ = model.teacher_logits(tape, session, question, target)
    qa_targets = torch.tensor([*target, 256], device=qa_logits.device)
    qa = torch.nn.functional.cross_entropy(qa_logits, qa_targets)
    logits, targets = observation_logits(model, tape, session, cut_credit=cut)
    observation = torch.nn.functional.cross_entropy(logits, targets)
    return FidelityLoss(qa + AUXILIARY_WEIGHT * observation, qa, observation, tape)
