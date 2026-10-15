"""PLAN28: answer-local learned location transport with unrestricted content reads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .native_evidence_candidate import NativeEvidenceCandidate
from .native_iterative_evidence_candidate import IterativeEvidenceCandidate
from .position_evidence_reference import EvidenceTape


@dataclass(frozen=True)
class RelativeEvidenceTape(EvidenceTape):
    shift_indices: torch.Tensor


@dataclass(frozen=True)
class RelativeReadState:
    hidden: torch.Tensor
    address: torch.Tensor | None = None
    tape: EvidenceTape | None = None


class RelativeEvidenceCandidate(IterativeEvidenceCandidate):
    FORMAT = "taiji-native-relative-evidence-candidate-v1"
    ADDRESS_MECHANISM = "source-local-learned-shift-minus1-zero-plus1-clamp-v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.location = nn.Linear(self.config.hidden, 4)
        with torch.no_grad():
            self.location.weight.zero_()
            self.location.bias.zero_()
            self.location.bias[3] = -2

    def question_state(self, question: bytes):
        return RelativeReadState(super().question_state(question))

    def evidence(self, session, *, cut_credit=False, terminal_per_source=False):
        tape = super().evidence(
            session, cut_credit=cut_credit, terminal_per_source=terminal_per_source
        )
        blocks = []
        offset = 0
        for source in tape.sources:
            length = len(source.content)
            indices = torch.arange(length, device=tape.values.device)
            blocks.append(
                torch.stack([(indices + d).clamp(0, length - 1) + offset for d in (-1, 0, 1)])
            )
            offset += length
        return RelativeEvidenceTape(
            tape.session,
            tape.sources,
            tape.positions,
            tape.states,
            tape.keys,
            tape.values,
            torch.cat(blocks, dim=1),
        )

    @staticmethod
    def transport(tape, address, mixture):
        shifted = address.new_zeros((3, len(address))).scatter_add(
            1, tape.shift_indices, address.expand(3, -1)
        )
        return mixture @ shifted

    def step(self, tape, session, state, previous):
        if not isinstance(state, RelativeReadState) or not isinstance(tape, RelativeEvidenceTape):
            raise ValueError("relative candidate requires its own read state and evidence tape")
        if tape.session != session or (state.tape is not None and state.tape is not tape):
            raise ValueError("relative address state cannot cross session or evidence tape")
        # First hop always reads by content; only final-hop attention persists.
        _, hidden, _ = NativeEvidenceCandidate.step(self, tape, session, state.hidden, previous)
        weights = torch.softmax(tape.keys @ self.query(hidden) / self.config.key**0.5, dim=0)
        if state.address is not None:
            controls = self.location(hidden)
            location = self.transport(tape, state.address, torch.softmax(controls[:3], dim=0))
            gate = torch.sigmoid(controls[3])
            weights = (1 - gate) * weights + gate * location
        read = weights @ tape.values
        hidden = self.decoder(torch.cat((self.symbol(previous), read)), hidden)
        return (
            self.readout(torch.cat((hidden, read))),
            RelativeReadState(hidden, weights, tape),
            weights,
        )

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["address_mechanism"] = self.ADDRESS_MECHANISM
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("address_mechanism") != cls.ADDRESS_MECHANISM:
            raise ValueError("relative checkpoint requires its explicit address mechanism")
        return super().restore_bundle(payload)
