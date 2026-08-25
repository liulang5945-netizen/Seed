"""Data-driven procedural skill consolidation for Taiji."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .contracts import EpisodicMemoryRecord
from .episodic_memory import EpisodicMemoryStore

PROCEDURAL_MEMORY_CHECKPOINT_FORMAT = "taiji-procedural-memory-v1"


class ProceduralMemoryLearner(nn.Module):
    """Consolidate cue-to-action regularities from experienced records.

    Action classes are discovered from the records supplied to
    :meth:`consolidate`; the learner does not contain a task-specific action
    table.  The resulting linear readout is a compact slow skill state that
    can be checkpointed independently from episodic recall.
    """

    def __init__(self, cue_dim: int) -> None:
        super().__init__()
        if int(cue_dim) <= 0:
            raise ValueError("procedural memory cue_dim must be positive")
        self.cue_dim = int(cue_dim)
        self.action_kinds: tuple[str, ...] = ()
        self.readout: nn.Linear | None = None
        self.consolidation_count = 0

    def _batch(
        self, records: Iterable[EpisodicMemoryRecord]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        records = tuple(records)
        if not records:
            raise ValueError("procedural consolidation needs episodic records")
        if any(record.action_intent is None for record in records):
            raise ValueError("procedural consolidation needs records with action intents")
        action_kinds = tuple(sorted({record.action_intent.kind for record in records}))
        if not action_kinds:
            raise ValueError("procedural consolidation needs at least one action kind")
        cues = torch.stack([record.cue.detach().to(dtype=torch.float32) for record in records])
        if cues.ndim != 2 or cues.shape[1] != self.cue_dim:
            raise ValueError("procedural record cue dimensions do not match the learner")
        kind_to_index = {kind: index for index, kind in enumerate(action_kinds)}
        targets = torch.tensor(
            [kind_to_index[record.action_intent.kind] for record in records],
            dtype=torch.long,
            device=cues.device,
        )
        return cues, targets, action_kinds

    def _ensure_readout(self, action_kinds: tuple[str, ...]) -> None:
        if self.readout is None:
            self.action_kinds = action_kinds
            self.readout = nn.Linear(self.cue_dim, len(action_kinds))
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.bias.zero_()
            return
        if self.action_kinds != action_kinds:
            raise ValueError("procedural action kinds changed after consolidation")

    def consolidate(
        self,
        source: EpisodicMemoryStore | Iterable[EpisodicMemoryRecord],
        *,
        epochs: int = 300,
        learning_rate: float = 0.1,
    ) -> float:
        """Replay experienced action intents into the slow procedural readout."""

        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("procedural consolidation epochs and learning_rate must be positive")
        records = source.records if isinstance(source, EpisodicMemoryStore) else tuple(source)
        cues, targets, action_kinds = self._batch(records)
        self._ensure_readout(action_kinds)
        assert self.readout is not None
        cues = cues.to(self.readout.weight.device)
        targets = targets.to(self.readout.weight.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=float(learning_rate))
        loss = torch.tensor(0.0, device=cues.device)
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(self.readout(cues), targets)
            loss.backward()
            optimizer.step()
        self.consolidation_count += 1
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, cue: torch.Tensor) -> str:
        if self.readout is None:
            raise RuntimeError("procedural memory has not been consolidated")
        if cue.ndim != 1 or cue.numel() != self.cue_dim:
            raise ValueError("procedural prediction cue dimension does not match the learner")
        logits = self.readout(cue.detach().to(self.readout.weight.device))
        return self.action_kinds[int(torch.argmax(logits).item())]

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": PROCEDURAL_MEMORY_CHECKPOINT_FORMAT,
            "cue_dim": self.cue_dim,
            "action_kinds": list(self.action_kinds),
            "consolidation_count": self.consolidation_count,
            "state_dict": None
            if self.readout is None
            else {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> ProceduralMemoryLearner:
        if payload.get("format") != PROCEDURAL_MEMORY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported procedural memory checkpoint format")
        learner = cls(int(payload["cue_dim"]))
        action_kinds = tuple(str(kind) for kind in payload.get("action_kinds", ()))
        state_dict = payload.get("state_dict")
        if action_kinds and state_dict is not None:
            learner._ensure_readout(action_kinds)
            assert learner.readout is not None
            learner.load_state_dict(state_dict)
        learner.to(device)
        learner.consolidation_count = int(payload.get("consolidation_count", 0))
        return learner
