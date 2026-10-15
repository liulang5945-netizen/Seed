"""PLAN37: question alignment residual with exact native content initialization."""

from __future__ import annotations

import torch

from .native_question_aligned_evidence_candidate import QuestionAlignedEvidenceCandidate
from .native_relative_evidence_candidate import RelativeEvidenceTape


class ResidualQuestionEvidenceCandidate(QuestionAlignedEvidenceCandidate):
    FORMAT = "taiji-native-residual-question-evidence-candidate-v1"
    QUESTION_MECHANISM = "all-question-byte-zero-initialized-residual-v1"

    def __init__(self, config=None):
        super().__init__(config)
        with torch.no_grad():
            self.question_fusion.weight.zero_()
            self.question_fusion.bias.zero_()

    def condition(self, tape, question_states):
        states = torch.stack(tape.states)
        if tape.terminal_per_source:
            blocks = []
            end = 0
            for source in tape.sources:
                length = len(source.content)
                end += length
                blocks.append(states[end - 1].expand(length, -1))
            states = torch.cat(blocks)
        if tape.cut_credit:
            states = states.detach()
        scores = self.key(states) @ self.question_key(question_states).T / self.config.key**0.5
        alignment = torch.softmax(scores, dim=-1)
        context = alignment @ self.question_value(question_states)
        fused = states + torch.tanh(self.question_fusion(torch.cat((states, context), dim=-1)))
        return (
            RelativeEvidenceTape(
                tape.session,
                tape.sources,
                tape.positions,
                tape.states,
                self.key(fused),
                self.value(fused),
                tape.shift_indices,
            ),
            alignment,
        )
