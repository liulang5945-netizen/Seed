"""Data-driven procedural skill consolidation for Taiji."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
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


SEQUENTIAL_PROCEDURAL_MEMORY_CHECKPOINT_FORMAT = "taiji-sequential-procedural-memory-v1"


class ProceduralSequenceLearner(nn.Module):
    """Consolidate ordered multi-step skills from episodic action traces.

    Records are grouped by ``episode_id`` and ordered by ``tick``.  A GRU
    carries procedural context between steps; action classes are discovered
    from ``action_intent.kind`` rather than defined by the learner.
    """

    def __init__(self, cue_dim: int, *, hidden_dim: int = 16, seed: int = 0) -> None:
        super().__init__()
        if int(cue_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("sequential procedural cue_dim and hidden_dim must be positive")
        self.cue_dim = int(cue_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.encoder = nn.GRU(self.cue_dim, self.hidden_dim, batch_first=True)
        self.action_kinds: tuple[str, ...] = ()
        self.readout: nn.Linear | None = None
        self.consolidation_count = 0

    def _episodes(
        self, records: Iterable[EpisodicMemoryRecord]
    ) -> tuple[tuple[EpisodicMemoryRecord, ...], ...]:
        grouped: dict[str, list[EpisodicMemoryRecord]] = defaultdict(list)
        for record in records:
            if record.action_intent is None:
                raise ValueError("sequential consolidation needs action intents")
            grouped[record.episode_id].append(record)
        episodes = tuple(
            tuple(sorted(items, key=lambda record: (record.tick, record.memory_id)))
            for items in grouped.values()
        )
        if not episodes or any(not episode for episode in episodes):
            raise ValueError("sequential consolidation needs non-empty episodes")
        return episodes

    def _ensure_readout(self, action_kinds: tuple[str, ...]) -> None:
        if self.readout is None:
            self.action_kinds = action_kinds
            self.readout = nn.Linear(self.hidden_dim, len(action_kinds))
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.bias.zero_()
            return
        if self.action_kinds != action_kinds:
            raise ValueError("sequential procedural action kinds changed after consolidation")

    def consolidate(
        self,
        source: EpisodicMemoryStore | Iterable[EpisodicMemoryRecord],
        *,
        epochs: int = 300,
        learning_rate: float = 0.05,
    ) -> float:
        """Replay ordered episodes into the recurrent procedural state."""

        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError(
                "sequential procedural consolidation epochs and learning_rate must be positive"
            )
        records = source.records if isinstance(source, EpisodicMemoryStore) else tuple(source)
        episodes = self._episodes(records)
        action_kinds = tuple(
            sorted(
                {
                    record.action_intent.kind
                    for episode in episodes
                    for record in episode
                    if record.action_intent is not None
                }
            )
        )
        if not action_kinds:
            raise ValueError("sequential consolidation needs action kinds")
        self._ensure_readout(action_kinds)
        assert self.readout is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        loss = torch.tensor(0.0)
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            losses = []
            for episode in episodes:
                cues = torch.stack(
                    [record.cue.detach().to(dtype=torch.float32) for record in episode]
                )
                if cues.ndim != 2 or cues.shape[1] != self.cue_dim:
                    raise ValueError(
                        "sequential record cue dimensions do not match the learner"
                    )
                targets = torch.tensor(
                    [self.action_kinds.index(record.action_intent.kind) for record in episode],
                    dtype=torch.long,
                )
                hidden, _ = self.encoder(cues.unsqueeze(0))
                losses.append(nn.functional.cross_entropy(self.readout(hidden.squeeze(0)), targets))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        self.consolidation_count += 1
        return float(loss.detach())

    @torch.no_grad()
    def predict_episode(self, cues: Sequence[torch.Tensor]) -> tuple[str, ...]:
        if self.readout is None:
            raise RuntimeError("sequential procedural memory has not been consolidated")
        if not cues:
            raise ValueError("sequential procedural prediction needs at least one cue")
        stacked = torch.stack([cue.detach().to(dtype=torch.float32) for cue in cues])
        if stacked.ndim != 2 or stacked.shape[1] != self.cue_dim:
            raise ValueError("sequential prediction cue dimensions do not match the learner")
        hidden, _ = self.encoder(stacked.unsqueeze(0))
        indices = torch.argmax(self.readout(hidden.squeeze(0)), dim=-1).tolist()
        return tuple(self.action_kinds[int(index)] for index in indices)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": SEQUENTIAL_PROCEDURAL_MEMORY_CHECKPOINT_FORMAT,
            "cue_dim": self.cue_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
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
    ) -> ProceduralSequenceLearner:
        if payload.get("format") != SEQUENTIAL_PROCEDURAL_MEMORY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported sequential procedural memory checkpoint format")
        learner = cls(
            int(payload["cue_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            seed=int(payload.get("seed", 0)),
        )
        action_kinds = tuple(str(kind) for kind in payload.get("action_kinds", ()))
        state_dict = payload.get("state_dict")
        if action_kinds and state_dict is not None:
            learner._ensure_readout(action_kinds)
            assert learner.readout is not None
            learner.load_state_dict(state_dict)
        learner.to(device)
        learner.consolidation_count = int(payload.get("consolidation_count", 0))
        return learner
