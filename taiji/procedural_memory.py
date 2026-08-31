"""Data-driven procedural skill consolidation for Taiji."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, cast

import torch
from torch import nn

from .contracts import EpisodicMemoryRecord
from .episodic_memory import EpisodicMemoryStore
from .local_learning import (
    LocalAdam,
    apply_linear_delta,
    backproject_linear,
    freeze_parameters,
    gru_forward_trace,
    gru_gradients,
    linear_gradients,
    softmax_error_delta,
)

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
        self,
        records: Iterable[EpisodicMemoryRecord],
        *,
        action_kinds: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        records = tuple(records)
        if not records:
            raise ValueError("procedural consolidation needs episodic records")
        discovered_action_kinds: set[str] = set()
        for record in records:
            action_intent = record.action_intent
            if action_intent is None:
                raise ValueError("procedural consolidation needs records with action intents")
            discovered_action_kinds.add(action_intent.kind)
        resolved_action_kinds = (
            tuple(sorted(discovered_action_kinds))
            if action_kinds is None
            else tuple(dict.fromkeys(str(kind) for kind in action_kinds))
        )
        if not resolved_action_kinds:
            raise ValueError("procedural consolidation needs at least one action kind")
        if not discovered_action_kinds.issubset(resolved_action_kinds):
            raise ValueError("procedural records contain an action kind outside the readout")
        cues = torch.stack([record.cue.detach().to(dtype=torch.float32) for record in records])
        if cues.ndim != 2 or cues.shape[1] != self.cue_dim:
            raise ValueError("procedural record cue dimensions do not match the learner")
        kind_to_index = {kind: index for index, kind in enumerate(resolved_action_kinds)}
        target_indices: list[int] = []
        for record in records:
            action_intent = record.action_intent
            if action_intent is None:
                raise ValueError("procedural consolidation needs records with action intents")
            target_indices.append(kind_to_index[action_intent.kind])
        targets = torch.tensor(target_indices, dtype=torch.long, device=cues.device)
        return cues, targets, resolved_action_kinds

    def _ensure_readout(self, action_kinds: tuple[str, ...]) -> None:
        if self.readout is None:
            self.action_kinds = action_kinds
            self.readout = nn.Linear(self.cue_dim, len(action_kinds))
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.bias.zero_()
            freeze_parameters(self.readout)
            return
        if self.action_kinds == action_kinds:
            return
        if not set(self.action_kinds).issubset(action_kinds):
            raise ValueError("procedural action kinds cannot remove an existing kind")
        assert self.readout is not None
        expanded = nn.Linear(self.cue_dim, len(action_kinds))
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.bias.zero_()
            for old_index, kind in enumerate(self.action_kinds):
                new_index = action_kinds.index(kind)
                expanded.weight[new_index].copy_(self.readout.weight[old_index])
                expanded.bias[new_index].copy_(self.readout.bias[old_index])
        freeze_parameters(expanded)
        self.action_kinds = action_kinds
        self.readout = expanded

    def prepare(self, action_kinds: Sequence[str]) -> None:
        """Prepare or expand the discovered action readout without learning."""

        resolved = tuple(dict.fromkeys(str(kind).strip() for kind in action_kinds))
        if not resolved or any(not kind for kind in resolved):
            raise ValueError("procedural prepare needs non-empty action kinds")
        self._ensure_readout(resolved)

    def consolidate(
        self,
        source: EpisodicMemoryStore | Iterable[EpisodicMemoryRecord],
        *,
        epochs: int = 300,
        learning_rate: float = 0.1,
        action_kinds: Sequence[str] | None = None,
    ) -> float:
        """Replay experienced action intents into the slow procedural readout."""

        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("procedural consolidation epochs and learning_rate must be positive")
        records = source.records if isinstance(source, EpisodicMemoryStore) else tuple(source)
        cues, targets, resolved_action_kinds = self._batch(
            records,
            action_kinds=action_kinds,
        )
        self._ensure_readout(resolved_action_kinds)
        assert self.readout is not None
        cues = cues.to(self.readout.weight.device)
        targets = targets.to(self.readout.weight.device)
        final_loss = 0.0
        for _ in range(int(epochs)):
            with torch.no_grad():
                logits = self.readout(cues)
                final_loss = float(nn.functional.cross_entropy(logits, targets))
            apply_linear_delta(
                self.readout,
                cues,
                softmax_error_delta(logits, targets),
                float(learning_rate),
            )
        self.consolidation_count += 1
        return final_loss

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
            "state_dict": (
                None
                if self.readout is None
                else {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.state_dict().items()
                }
            ),
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
        freeze_parameters(self.encoder)
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
            freeze_parameters(self.readout)
            return
        if self.action_kinds != action_kinds:
            raise ValueError("sequential procedural action kinds changed after consolidation")

    def consolidate(
        self,
        source: EpisodicMemoryStore | Iterable[EpisodicMemoryRecord],
        *,
        epochs: int = 300,
        learning_rate: float = 0.05,
        action_kinds: Sequence[str] | None = None,
    ) -> float:
        """Replay ordered episodes into the recurrent procedural state."""

        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError(
                "sequential procedural consolidation epochs and learning_rate must be positive"
            )
        records = source.records if isinstance(source, EpisodicMemoryStore) else tuple(source)
        episodes = self._episodes(records)
        discovered_action_kinds = tuple(
            sorted(
                {
                    record.action_intent.kind
                    for episode in episodes
                    for record in episode
                    if record.action_intent is not None
                }
            )
        )
        resolved_action_kinds = (
            discovered_action_kinds
            if action_kinds is None
            else tuple(dict.fromkeys(str(kind) for kind in action_kinds))
        )
        if not resolved_action_kinds:
            raise ValueError("sequential consolidation needs action kinds")
        if not set(discovered_action_kinds).issubset(resolved_action_kinds):
            raise ValueError("sequential records contain an action kind outside the readout")
        self._ensure_readout(resolved_action_kinds)
        assert self.readout is not None
        batches = []
        for episode in episodes:
            cues = torch.stack([record.cue.detach().to(dtype=torch.float32) for record in episode])
            if cues.ndim != 2 or cues.shape[1] != self.cue_dim:
                raise ValueError("sequential record cue dimensions do not match the learner")
            targets = torch.tensor(
                [
                    self.action_kinds.index(record.action_intent.kind)
                    for record in episode
                    if record.action_intent is not None
                ],
                dtype=torch.long,
            )
            batches.append((cues, targets))
        trainable = (
            cast(torch.Tensor, self.encoder.weight_ih_l0),
            cast(torch.Tensor, self.encoder.weight_hh_l0),
            cast(torch.Tensor, self.encoder.bias_ih_l0),
            cast(torch.Tensor, self.encoder.bias_hh_l0),
            self.readout.weight,
            self.readout.bias,
        )
        optimizer = LocalAdam(trainable, learning_rate=float(learning_rate))
        episode_count = float(len(batches))
        final_loss = 0.0
        for _ in range(int(epochs)):
            gradients = [torch.zeros_like(parameter) for parameter in trainable]
            total_loss = 0.0
            for cues, targets in batches:
                hidden, trace = gru_forward_trace(self.encoder, cues)
                with torch.no_grad():
                    logits = self.readout(hidden)
                    total_loss += float(nn.functional.cross_entropy(logits, targets))
                # The objective averages each episode's own mean cross entropy,
                # so the per-episode delta carries a second ``1 / episodes``
                # factor on top of the ``1 / steps`` that the mean already
                # contributes.  Every episode accumulates into one gradient
                # buffer before a single optimiser step, exactly as the shared
                # ``optimizer.zero_grad()`` boundary used to arrange.
                logit_error = softmax_error_delta(logits, targets) / episode_count
                hidden_error = backproject_linear(self.readout, logit_error)
                recurrent = gru_gradients(self.encoder, trace, hidden_error)
                readout_gradients = linear_gradients(self.readout, hidden, logit_error)
                for buffer, gradient in zip(
                    gradients, (*recurrent, *readout_gradients), strict=True
                ):
                    buffer += gradient
            final_loss = total_loss / episode_count
            optimizer.apply(gradients)
        self.consolidation_count += 1
        return final_loss

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
            "state_dict": (
                None
                if self.readout is None
                else {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.state_dict().items()
                }
            ),
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
