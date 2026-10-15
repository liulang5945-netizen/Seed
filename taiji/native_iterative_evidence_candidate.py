"""Isolated two-hop pre-emission read; runtime semantics own a new bundle format."""

from __future__ import annotations

from .native_evidence_candidate import NativeEvidenceCandidate


class IterativeEvidenceCandidate(NativeEvidenceCandidate):
    FORMAT = "taiji-native-iterative-evidence-candidate-v1"
    READ_HOPS = 2

    def step(self, tape, session, hidden, previous):
        # No intermediate sample, emission, stop test or observation write.
        for _ in range(self.READ_HOPS):
            logits, hidden, weights = super().step(tape, session, hidden, previous)
        return logits, hidden, weights

    def bundle(self, optimizer=None):
        payload = super().bundle(optimizer)
        payload["read_hops"] = self.READ_HOPS
        return payload

    @classmethod
    def restore_bundle(cls, payload):
        if payload.get("read_hops") != cls.READ_HOPS:
            raise ValueError("iterative evidence checkpoint must declare two read hops")
        return super().restore_bundle(payload)
