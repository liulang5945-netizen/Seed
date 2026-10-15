"""PLAN48: learned local byte matching conditions the question on observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .native_question_aligned_evidence_candidate import QuestionEvidenceTape, QuestionReadState
from .native_relative_evidence_candidate import RelativeReadState
from .native_residual_question_evidence_candidate import ResidualQuestionEvidenceCandidate


@dataclass(frozen=True)
class LocalQuestionReadState(QuestionReadState):
    question_bytes: bytes = b""


class LocalQuestionEvidenceCandidate(ResidualQuestionEvidenceCandidate):
    FORMAT = "taiji-native-local-question-evidence-candidate-v1"
    QUESTION_MECHANISM = "local-byte-window-observation-conditioned-residual-v1"
    LOCAL_MECHANISM = "shared-five-byte-cosine-question-residual-v1"
    WINDOW = 5

    def __init__(self, config=None):
        super().__init__(config)
        self.local_projection = nn.Linear(
            self.WINDOW * self.config.embedding, self.config.key, bias=False
        )
        self.local_salience = nn.Linear(self.config.hidden + 1, 1)
        self.local_fusion = nn.Linear(2 * self.config.hidden, self.config.hidden)
        self.local_log_temperature = nn.Parameter(torch.tensor(math.log(8.0)))
        with torch.no_grad():
            self.local_fusion.weight.zero_()
            self.local_fusion.bias.zero_()

    def local_features(self, content: bytes):
        if not isinstance(content, bytes) or not content:
            raise ValueError("local byte windows require nonempty bytes")
        radius = self.WINDOW // 2
        tokens = torch.tensor(
            [*([self.BOUNDARY] * radius), *content, *([self.BOUNDARY] * radius)],
            device=self.embedding.weight.device,
        )
        embedded = self.embedding(tokens)
        windows = embedded.unfold(0, self.WINDOW, 1).permute(0, 2, 1).reshape(len(content), -1)
        return torch.nn.functional.normalize(self.local_projection(windows), dim=-1)

    def question_state(self, question: bytes):
        state = super().question_state(question)
        return LocalQuestionReadState(
            state.relative, state.question_states, question_bytes=question
        )

    def bind_question(self, tape, state):
        if not isinstance(state, LocalQuestionReadState) or not state.question_bytes:
            raise ValueError("unread question must retain its native input bytes")
        observed_features = torch.cat(
            [self.local_features(source.content) for source in tape.sources]
        )
        states = torch.stack(tape.states)
        if tape.terminal_per_source:
            end, blocks = 0, []
            for source in tape.sources:
                length = len(source.content)
                end += length
                blocks.append(states[end - 1].expand(length, -1))
            states = torch.cat(blocks)
        if tape.cut_credit:
            states, observed_features = states.detach(), observed_features.detach()
        scores = self.local_features(state.question_bytes) @ observed_features.T
        temperature = self.local_log_temperature.exp().clamp(1, 32)
        alignment = torch.softmax(scores * temperature, dim=-1)
        context = alignment @ states
        salience = torch.softmax(
            self.local_salience(
                torch.cat((state.question_states, scores.max(-1).values[:, None]), dim=-1)
            ).squeeze(-1),
            dim=0,
        )
        pooled = salience @ context
        question_states = state.question_states + torch.tanh(
            self.local_fusion(torch.cat((state.question_states, context), dim=-1))
        )
        hidden = state.hidden + torch.tanh(self.local_fusion(torch.cat((state.hidden, pooled))))
        return QuestionReadState(RelativeReadState(hidden), question_states), alignment, salience

    def step(self, tape, session, state, previous):
        if not isinstance(tape, QuestionEvidenceTape) or not isinstance(state, QuestionReadState):
            raise ValueError("local binding requires native question tape and state")
        if tape.session != session or (
            state.input_tape is not None and state.input_tape is not tape
        ):
            raise ValueError("local binding cannot cross sessions")
        if state.conditioned_tape is None:
            state, _, _ = self.bind_question(tape, state)
        return super().step(tape, session, state, previous)

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["local_question_mechanism"] = self.LOCAL_MECHANISM
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("local_question_mechanism") != cls.LOCAL_MECHANISM:
            raise ValueError("explicit local question binding mechanism required")
        return super().restore_bundle(payload)
