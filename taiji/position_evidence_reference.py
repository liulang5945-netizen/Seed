"""Isolated differentiable reference; never mounted by the native runtime.

Evidence is written at each observed byte, not reconstructed from one final
prefix state. This module supplies no learned capability or product credit claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .utf8_state import advance_utf8, trim_partial_tail, utf8_allowed


@dataclass(frozen=True)
class EvidenceReferenceConfig:
    embedding: int = 16
    hidden: int = 32
    key: int = 16

    def __post_init__(self):
        if min(self.embedding, self.hidden, self.key) < 1:
            raise ValueError("reference dimensions must be positive")


@dataclass(frozen=True)
class ObservedSource:
    session: str
    source: str
    content: bytes
    provenance: str = "external"

    def __post_init__(self):
        if (
            not isinstance(self.content, bytes)
            or not self.session
            or not self.source
            or not self.content
        ):
            raise ValueError("external evidence needs session, source and content")
        if self.provenance != "external":
            raise ValueError("only external observations may enter this reference")


@dataclass(frozen=True)
class EvidenceTape:
    session: str
    sources: tuple[ObservedSource, ...]
    positions: tuple[tuple[str, int], ...]
    states: tuple[torch.Tensor, ...]
    keys: torch.Tensor
    values: torch.Tensor


class PositionEvidenceReference(nn.Module):
    FORMAT = "taiji-position-evidence-reference-v1"
    BOUNDARY = 256

    def __init__(self, config: EvidenceReferenceConfig = EvidenceReferenceConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(257, config.embedding)
        self.writer = nn.GRUCell(config.embedding, config.hidden)
        self.question = nn.GRUCell(config.embedding, config.hidden)
        self.key = nn.Linear(config.hidden, config.key, bias=False)
        self.value = nn.Linear(config.hidden, config.hidden, bias=False)
        self.query = nn.Linear(config.hidden, config.key, bias=False)
        self.decoder = nn.GRUCell(config.embedding + config.hidden, config.hidden)
        self.readout = nn.Linear(2 * config.hidden, 257)

    def zero(self):
        return self.embedding.weight.new_zeros(self.config.hidden)

    def symbol(self, value: int):
        return self.embedding(torch.tensor(value, device=self.embedding.weight.device))

    def write(
        self,
        sources: tuple[ObservedSource, ...],
        session: str,
        *,
        cut_credit=False,
        terminal_only=False,
    ) -> EvidenceTape:
        if not sources or any(s.session != session for s in sources):
            raise ValueError("evidence pool must be nonempty and scoped to one session")
        if len({s.source for s in sources}) != len(sources):
            raise ValueError("duplicate external source")
        states, positions = [], []
        for source in sources:
            hidden = self.zero()
            for index, byte in enumerate(source.content):
                hidden = self.writer(self.symbol(byte), hidden)
                if hidden.requires_grad:
                    hidden.retain_grad()
                states.append(hidden)
                positions.append((source.source, index))
        values = torch.stack(states)
        if terminal_only:
            values = values[-1:].expand_as(values)
        if cut_credit:
            values = values.detach()
        return EvidenceTape(
            session, sources, tuple(positions), tuple(states), self.key(values), self.value(values)
        )

    def question_state(self, question: bytes):
        hidden = self.zero()
        for byte in question:
            hidden = self.question(self.symbol(byte), hidden)
        return hidden

    def step(self, tape: EvidenceTape, session: str, hidden: torch.Tensor, previous: int):
        if tape.session != session:
            raise ValueError("cross-session evidence access")
        weights = torch.softmax(tape.keys @ self.query(hidden) / self.config.key**0.5, dim=0)
        read = weights @ tape.values
        hidden = self.decoder(torch.cat((self.symbol(previous), read)), hidden)
        return self.readout(torch.cat((hidden, read))), hidden, weights

    def teacher_logits(self, tape: EvidenceTape, session: str, question: bytes, target: bytes):
        """Loss-only interface; targets are never accepted by generate()."""
        hidden, previous = self.question_state(question), self.BOUNDARY
        logits, attention = [], []
        for byte in (*target, self.BOUNDARY):
            scores, hidden, weights = self.step(tape, session, hidden, previous)
            logits.append(scores)
            attention.append(weights)
            previous = byte
        return torch.stack(logits), torch.stack(attention)

    @torch.no_grad()
    def generate(self, tape: EvidenceTape, session: str, question: bytes, max_length=256):
        if tape.session != session:
            raise ValueError("cross-session evidence access")
        if max_length < 0:
            raise ValueError("max_length cannot be negative")
        hidden, previous = self.question_state(question), self.BOUNDARY
        remaining = lead = 0
        output = bytearray()
        for _ in range(max_length):
            logits, hidden, _ = self.step(tape, session, hidden, previous)
            allowed = utf8_allowed(remaining, lead)
            if remaining == 0:
                allowed.append(self.BOUNDARY)
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = logits[allowed]
            previous = int(mask.argmax())
            if previous == self.BOUNDARY:
                break
            output.append(previous)
            remaining, lead = advance_utf8(remaining, lead, previous)
        return trim_partial_tail(bytes(output))

    def checkpoint(self, sources: tuple[ObservedSource, ...], optimizer: Any = None):
        return {
            "format": self.FORMAT,
            "config": asdict(self.config),
            "state": {k: v.detach().clone() for k, v in self.state_dict().items()},
            "sources": [asdict(s) for s in sources],
            "rng": torch.get_rng_state(),
            "optimizer": None if optimizer is None else deepcopy(optimizer.state_dict()),
        }

    @classmethod
    def restore(cls, payload):
        if payload["format"] != cls.FORMAT:
            raise ValueError("unknown evidence reference checkpoint")
        model = cls(EvidenceReferenceConfig(**payload["config"]))
        model.load_state_dict(payload["state"], strict=True)
        sources = tuple(ObservedSource(**s) for s in payload["sources"])
        torch.set_rng_state(payload["rng"])
        return model, sources
