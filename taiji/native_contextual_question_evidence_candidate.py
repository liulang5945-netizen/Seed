"""PLAN51: ordered observation-conditioned question encoding with exact zero gate."""

from __future__ import annotations

import torch
from torch import nn

from .native_local_question_evidence_candidate import LocalQuestionEvidenceCandidate
from .native_question_aligned_evidence_candidate import QuestionReadState
from .native_relative_evidence_candidate import RelativeReadState


class ContextualQuestionEvidenceCandidate(LocalQuestionEvidenceCandidate):
    FORMAT = "taiji-native-contextual-question-evidence-candidate-v1"
    QUESTION_MECHANISM = "ordered-local-context-question-gate-v1"
    CONTEXT_MECHANISM = "byte-context-confidence-gru-zero-channel-gate-v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.context_question = nn.GRU(
            self.config.embedding + self.config.hidden + 1, self.config.hidden, batch_first=True
        )
        self.context_gate = nn.Parameter(torch.zeros(self.config.hidden))

    def context_sequence(self, tape, state, alignment):
        states = torch.stack(tape.states)
        if tape.terminal_per_source:
            end, blocks = 0, []
            for source in tape.sources:
                length = len(source.content)
                end += length
                blocks.append(states[end - 1].expand(length, -1))
            states = torch.cat(blocks)
        observed_features = torch.cat(
            [self.local_features(source.content) for source in tape.sources]
        )
        if tape.cut_credit:
            states, observed_features = states.detach(), observed_features.detach()
        confidence = (
            (self.local_features(state.question_bytes) @ observed_features.T).max(-1).values
        )
        embedded = self.embedding(
            torch.tensor(list(state.question_bytes), device=self.embedding.weight.device)
        )
        inputs = torch.cat((embedded, alignment @ states, confidence[:, None]), dim=-1)
        output, hidden = self.context_question(inputs.unsqueeze(0))
        return output[0], hidden[0, 0], confidence

    def bind_question(self, tape, state):
        base, alignment, salience = super().bind_question(tape, state)
        states, hidden, _ = self.context_sequence(tape, state, alignment)
        gate = self.context_gate.tanh()
        return (
            QuestionReadState(
                RelativeReadState(base.hidden + gate * (hidden - base.hidden)),
                base.question_states + gate * (states - base.question_states),
            ),
            alignment,
            salience,
        )

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["context_question_mechanism"] = self.CONTEXT_MECHANISM
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("context_question_mechanism") != cls.CONTEXT_MECHANISM:
            raise ValueError("explicit contextual question mechanism required")
        return super().restore_bundle(payload)
