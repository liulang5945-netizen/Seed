"""Data-driven semantic consolidation over Taiji episodic experiences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .contracts import EpisodicMemoryRecord
from .episodic_memory import EpisodicMemoryStore
from .local_learning import apply_linear_delta, freeze_parameters, mean_squared_error_delta

SEMANTIC_MEMORY_CHECKPOINT_FORMAT = "taiji-semantic-memory-v1"


class SemanticMemoryLearner(nn.Module):
    """Consolidate outcome regularities from episodic records.

    The learner receives only cue vectors and experienced scalar outcomes.  It
    does not use episode IDs or a hand-authored relation table; its weights are
    the slow semantic state produced by consolidation.
    """

    def __init__(self, cue_dim: int) -> None:
        super().__init__()
        if int(cue_dim) <= 0:
            raise ValueError("semantic memory cue_dim must be positive")
        self.cue_dim = int(cue_dim)
        self.readout = nn.Linear(self.cue_dim, 1)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.bias.zero_()
        freeze_parameters(self)
        self.consolidation_count = 0

    def _batch(self, records: Iterable[EpisodicMemoryRecord]) -> tuple[torch.Tensor, torch.Tensor]:
        records = tuple(records)
        if not records:
            raise ValueError("semantic consolidation needs episodic records")
        if any(record.outcome is None for record in records):
            raise ValueError("semantic consolidation needs records with outcomes")
        cues = torch.stack([record.cue.detach().to(dtype=torch.float32) for record in records])
        if cues.ndim != 2 or cues.shape[1] != self.cue_dim:
            raise ValueError("semantic record cue dimensions do not match the learner")
        targets = torch.tensor(
            [float(record.outcome.reward) for record in records],
            dtype=cues.dtype,
            device=cues.device,
        ).unsqueeze(-1)
        return cues, targets

    def consolidate(
        self,
        source: EpisodicMemoryStore | Iterable[EpisodicMemoryRecord],
        *,
        epochs: int = 300,
        learning_rate: float = 0.1,
    ) -> float:
        """Replay episodic outcomes into the slow semantic readout."""

        records = source.records if isinstance(source, EpisodicMemoryStore) else tuple(source)
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("semantic consolidation epochs and learning_rate must be positive")
        cues, targets = self._batch(records)
        cues = cues.to(self.readout.weight.device)
        targets = targets.to(self.readout.weight.device)
        final_loss = 0.0
        for _ in range(int(epochs)):
            with torch.no_grad():
                prediction = self.readout(cues)
                final_loss = float(torch.mean((prediction - targets) ** 2))
            apply_linear_delta(
                self.readout,
                cues,
                mean_squared_error_delta(prediction, targets),
                float(learning_rate),
            )
        self.consolidation_count += 1
        return final_loss

    @torch.no_grad()
    def predict(self, cue: torch.Tensor) -> float:
        if cue.ndim != 1 or cue.numel() != self.cue_dim:
            raise ValueError("semantic prediction cue dimension does not match the learner")
        return float(self.readout(cue.detach().to(self.readout.weight.device)).squeeze())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": SEMANTIC_MEMORY_CHECKPOINT_FORMAT,
            "cue_dim": self.cue_dim,
            "consolidation_count": self.consolidation_count,
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> SemanticMemoryLearner:
        if payload.get("format") != SEMANTIC_MEMORY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported semantic memory checkpoint format")
        learner = cls(int(payload["cue_dim"]))
        learner.load_state_dict(payload["state_dict"])
        learner.to(device)
        learner.consolidation_count = int(payload.get("consolidation_count", 0))
        return learner
