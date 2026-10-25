"""Trainable causal task core; no detach on the declared training path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import nn

from .task_state import TaskStateV1


@dataclass(frozen=True)
class TaskCoreConfig:
    width: int = 256
    embedding_width: int = 64
    key_width: int = 64
    slots: int = 16
    chunk_bytes: int = 32
    max_observation_bytes: int = 4096
    max_generation_bytes: int = 512
    seed: int = 169

    def __post_init__(self):
        if any(
            value <= 0
            for value in (
                self.width,
                self.embedding_width,
                self.key_width,
                self.slots,
                self.chunk_bytes,
                self.max_observation_bytes,
                self.max_generation_bytes,
            )
        ):
            raise ValueError("task core dimensions and bounds must be positive")


class TaskCoreV1(nn.Module):
    def __init__(self, config: TaskCoreConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            257, config.embedding_width, device="cpu", dtype=torch.float32
        )
        self.encoder = nn.GRU(
            config.embedding_width,
            config.width,
            batch_first=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.keys = nn.Linear(config.width, config.key_width, device="cpu", dtype=torch.float32)
        self.values = nn.Linear(config.width, config.width, device="cpu", dtype=torch.float32)

    def empty(
        self,
        *,
        bundle_id: str,
        scope_id: str,
        session_id: str,
        native_state_ref: str,
        parameter_revision: int,
    ) -> TaskStateV1:
        device = self.embedding.weight.device
        return TaskStateV1(
            bundle_id,
            scope_id,
            session_id,
            native_state_ref,
            parameter_revision,
            0,
            torch.zeros(self.config.width, device=device, dtype=torch.float32),
            torch.zeros((0, self.config.width), device=device, dtype=torch.float32),
        )

    def step_task(
        self,
        observation: bytes,
        state: TaskStateV1,
        *,
        scope_id: str,
        session_id: str,
        source_id: str,
        resource_snapshot: Mapping[str, int],
    ) -> TaskStateV1:
        if state.scope_id != scope_id or state.session_id != session_id:
            raise ValueError("task state belongs to a different scope/session")
        if (
            len(state.hidden) != self.config.width
            or len(state.workspace) > self.config.slots
            or state.hidden.device != self.embedding.weight.device
        ):
            raise ValueError("task state does not match this task core")
        if (
            not isinstance(observation, bytes)
            or not observation
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError("task step requires nonempty byte observation and source identity")
        if any(name == source_id for name, _ in state.observations):
            raise ValueError("observation source was already consumed")
        remaining = resource_snapshot.get("remaining_steps")
        limit = resource_snapshot.get("max_observation_bytes")
        if (
            not isinstance(remaining, int)
            or remaining <= 0
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise ValueError("resource snapshot lacks a valid step/observation allowance")
        if (
            len(observation) > limit
            or sum(len(raw) for _, raw in state.observations) + len(observation)
            > self.config.max_observation_bytes
        ):
            raise ValueError("task observation exceeds the declared resource/state bound")
        tokens = torch.tensor(
            list(observation), dtype=torch.long, device=self.embedding.weight.device
        ).unsqueeze(0)
        encoded, hidden = self.encoder(self.embedding(tokens), state.hidden.reshape(1, 1, -1))
        positions = list(
            range(self.config.chunk_bytes - 1, len(observation), self.config.chunk_bytes)
        )
        if not positions or positions[-1] != len(observation) - 1:
            positions.append(len(observation) - 1)
        evidence = encoded[0, positions]
        refs = tuple(
            (len(state.observations), 0 if index == 0 else positions[index - 1] + 1, end + 1)
            for index, end in enumerate(positions)
        )
        workspace = torch.cat((state.workspace, evidence), dim=0)[-self.config.slots :]
        result = replace(
            state,
            hidden=hidden[0, 0],
            workspace=workspace,
            observations=state.observations + ((source_id, observation),),
            evidence_refs=(state.evidence_refs + refs)[-self.config.slots :],
            revision=state.revision + 1,
        )
        return result.reset("turn")
