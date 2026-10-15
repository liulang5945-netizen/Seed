"""PLAN35: learned bytewise question alignment, isolated from native defaults."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .native_relative_evidence_candidate import (
    RelativeEvidenceCandidate,
    RelativeEvidenceTape,
    RelativeReadState,
)


@dataclass(frozen=True)
class QuestionEvidenceTape(RelativeEvidenceTape):
    cut_credit: bool
    terminal_per_source: bool


@dataclass(frozen=True)
class QuestionReadState:
    relative: RelativeReadState
    question_states: torch.Tensor
    input_tape: QuestionEvidenceTape | None = None
    conditioned_tape: RelativeEvidenceTape | None = None
    alignment: torch.Tensor | None = None

    @property
    def hidden(self):
        return self.relative.hidden

    @property
    def address(self):
        return self.relative.address


class QuestionAlignedEvidenceCandidate(RelativeEvidenceCandidate):
    FORMAT = "taiji-native-question-aligned-evidence-candidate-v1"
    QUESTION_MECHANISM = "all-question-byte-soft-alignment-fusion-v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.question_key = nn.Linear(self.config.hidden, self.config.key, bias=False)
        self.question_value = nn.Linear(self.config.hidden, self.config.hidden, bias=False)
        self.question_fusion = nn.Linear(2 * self.config.hidden, self.config.hidden)

    def evidence(self, session, *, cut_credit=False, terminal_per_source=False):
        tape = super().evidence(
            session, cut_credit=cut_credit, terminal_per_source=terminal_per_source
        )
        return QuestionEvidenceTape(
            tape.session,
            tape.sources,
            tape.positions,
            tape.states,
            tape.keys,
            tape.values,
            tape.shift_indices,
            cut_credit,
            terminal_per_source,
        )

    def question_state(self, question: bytes):
        if not question:
            raise ValueError("nonempty question bytes required for alignment")
        hidden = self.zero()
        states = []
        for byte in question:
            hidden = self.question(self.symbol(byte), hidden)
            states.append(hidden)
        return QuestionReadState(RelativeReadState(hidden), torch.stack(states))

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
        fused = torch.tanh(self.question_fusion(torch.cat((states, context), dim=-1)))
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

    def step(self, tape, session, state, previous):
        if not isinstance(tape, QuestionEvidenceTape) or not isinstance(state, QuestionReadState):
            raise ValueError("question-aligned state and evidence required")
        if tape.session != session or (
            state.input_tape is not None and state.input_tape is not tape
        ):
            raise ValueError("question alignment cannot cross session or evidence tape")
        conditioned, alignment = state.conditioned_tape, state.alignment
        if conditioned is None:
            conditioned, alignment = self.condition(tape, state.question_states)
        logits, relative, weights = super().step(conditioned, session, state.relative, previous)
        return (
            logits,
            QuestionReadState(relative, state.question_states, tape, conditioned, alignment),
            weights,
        )

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["question_mechanism"] = self.QUESTION_MECHANISM
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("question_mechanism") != cls.QUESTION_MECHANISM:
            raise ValueError("explicit question alignment mechanism required")
        return super().restore_bundle(payload)
