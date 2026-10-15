"""Opt-in Taiji-owned evidence candidate; no default runtime registration."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .contracts import Observation
from .position_evidence_reference import (
    EvidenceTape,
    ObservedSource,
    PositionEvidenceReference,
)


class NativeEvidenceCandidate(PositionEvidenceReference):
    FORMAT = "taiji-native-position-evidence-candidate-v1"

    def __init__(self, config=None):
        if config is None:
            super().__init__()
        else:
            super().__init__(config)
        self._observations: dict[str, tuple[Observation, ...]] = {}

    def observe(self, session: str, observation: Observation):
        if not isinstance(session, str) or not session:
            raise ValueError("explicit session identity required")
        if not isinstance(observation, Observation):
            raise TypeError("versioned Observation required")
        if observation.modality != "text" or observation.provenance != "external":
            raise ValueError("only external text observations enter candidate evidence")
        if not isinstance(observation.value, (str, bytes)) or not observation.value:
            raise ValueError("observation must contain nonempty immutable text")
        if not isinstance(observation.source, str) or not observation.source:
            raise ValueError("source identity required")
        if type(observation.timestamp) is not int:
            raise ValueError("integer observation timestamp required")
        current = self._observations.get(session, ())
        if current and observation.timestamp <= current[-1].timestamp:
            raise ValueError("observation timestamps must strictly increase per session")
        if any(item.source == observation.source for item in current):
            raise ValueError("duplicate source identity in session")
        self._observations[session] = (*current, observation)

    def sources(self, session: str):
        return tuple(
            ObservedSource(
                session,
                item.source,
                item.value.encode("utf-8") if isinstance(item.value, str) else item.value,
            )
            for item in self._observations.get(session, ())
        )

    def evidence(self, session: str, *, cut_credit=False, terminal_per_source=False):
        tape = self.write(self.sources(session), session, cut_credit=cut_credit)
        if not terminal_per_source:
            return tape
        # Preserve every source and the same number of addresses. Only its
        # internal position representation becomes the source's terminal state.
        blocks = []
        end = 0
        for source in tape.sources:
            end += len(source.content)
            blocks.append(tape.states[end - 1].expand(len(source.content), -1))
        states = torch.cat(blocks)
        if cut_credit:
            states = states.detach()
        return EvidenceTape(
            session,
            tape.sources,
            tape.positions,
            tape.states,
            self.key(states),
            self.value(states),
        )

    def answer(self, session: str, question: bytes, max_length=256):
        if not isinstance(question, bytes):
            raise TypeError("question must be bytes")
        with torch.no_grad():
            tape = self.evidence(session)
            return self.generate(tape, session, question, max_length)

    def bundle(self, optimizer=None):
        payload = self.checkpoint((), optimizer)
        payload["sessions"] = {
            session: [item.to_payload() for item in observations]
            for session, observations in self._observations.items()
        }
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("format") != cls.FORMAT or payload.get("sources") != []:
            raise ValueError("unknown or ambiguous native evidence candidate bundle")
        if not isinstance(payload.get("sessions"), Mapping):
            raise ValueError("candidate bundle requires explicit sessions")
        # Parent restore constructs cls and checks all model parameters strictly.
        model, _ = super().restore(payload)
        for session, observations in payload["sessions"].items():
            if not observations:
                raise ValueError("empty stored session")
            for observation in observations:
                model.observe(session, Observation.from_payload(observation))
        return model
