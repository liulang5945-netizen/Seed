"""PLAN60: source-local question relations enter the evidence key/value tape."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .native_contextual_question_evidence_candidate import ContextualQuestionEvidenceCandidate
from .native_local_question_evidence_candidate import LocalQuestionReadState
from .native_question_aligned_evidence_candidate import QuestionEvidenceTape, QuestionReadState


class LocalRelationEvidenceCandidate(ContextualQuestionEvidenceCandidate):
    FORMAT = "taiji-native-local-relation-evidence-candidate-v1"
    RELATION_MECHANISM = "source-local-match-coordinate-gru-zero-kv-v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.relation_sequence = nn.GRU(
            self.config.embedding + 2, self.config.hidden, batch_first=True
        )
        self.relation_projection = nn.Linear(
            self.config.hidden, self.config.key + self.config.hidden
        )
        with torch.no_grad():
            self.relation_projection.weight.zero_()
            self.relation_projection.bias.zero_()

    def relation_inputs(self, tape, question):
        question_features = self.local_features(question)
        coordinate = torch.linspace(0, 1, len(question), device=self.embedding.weight.device)
        temperature = self.local_log_temperature.exp().clamp(1, 32)
        blocks = []
        for source in tape.sources:
            tokens = torch.tensor(list(source.content), device=self.embedding.weight.device)
            embedded = self.embedding(tokens)
            observed = self.local_features(source.content)
            if tape.terminal_per_source:
                embedded = embedded[-1:].expand(len(tokens), -1)
                observed = observed[-1:].expand(len(tokens), -1)
            if tape.cut_credit:
                embedded, observed = embedded.detach(), observed.detach()
            similarity = observed @ question_features.T
            confidence = similarity.max(-1).values
            expected = torch.softmax(similarity * temperature, dim=-1) @ coordinate
            blocks.append(torch.cat((embedded, confidence[:, None], expected[:, None]), dim=-1))
        return blocks

    def relation_tape(self, tape, conditioned, question):
        states = [
            self.relation_sequence(block.unsqueeze(0))[0][0]
            for block in self.relation_inputs(tape, question)
        ]
        residual = self.relation_projection(torch.cat(states))
        key, value = residual.split((self.config.key, self.config.hidden), dim=-1)
        return replace(conditioned, keys=conditioned.keys + key, values=conditioned.values + value)

    def step(self, tape, session, state, previous):
        if not isinstance(tape, QuestionEvidenceTape) or not isinstance(state, QuestionReadState):
            raise ValueError("native relation tape and question state required")
        if tape.session != session or (
            state.input_tape is not None and state.input_tape is not tape
        ):
            raise ValueError("relation state cannot cross sessions or evidence")
        if state.conditioned_tape is None:
            if not isinstance(state, LocalQuestionReadState) or not state.question_bytes:
                raise ValueError("first relation step requires native question bytes")
            question = state.question_bytes
            bound, _, _ = self.bind_question(tape, state)
            conditioned, alignment = self.condition(tape, bound.question_states)
            conditioned = self.relation_tape(tape, conditioned, question)
            state = QuestionReadState(
                bound.relative, bound.question_states, tape, conditioned, alignment
            )
        return super().step(tape, session, state, previous)

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["relation_mechanism"] = self.RELATION_MECHANISM
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("relation_mechanism") != cls.RELATION_MECHANISM:
            raise ValueError("explicit relation mechanism required")
        return super().restore_bundle(payload)
