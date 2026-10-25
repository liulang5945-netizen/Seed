"""Task core v2: the shared chunk workspace plus lossless per-byte evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import nn

from .task_state_v2 import MAX_EVIDENCE_BYTES, TaskStateV2


@dataclass(frozen=True)
class TaskCoreConfigV2:
    width: int = 256
    embedding_width: int = 64
    key_width: int = 64
    slots: int = 16
    chunk_bytes: int = 32
    max_observation_bytes: int = 4096
    max_generation_bytes: int = 512
    max_evidence_bytes: int = MAX_EVIDENCE_BYTES
    seed: int = 169

    def __post_init__(self) -> None:
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
                self.max_evidence_bytes,
            )
        ):
            raise ValueError("task core dimensions and bounds must be positive")
        if self.max_evidence_bytes > MAX_EVIDENCE_BYTES:
            raise ValueError("per-byte evidence exceeds the declared 512-byte bound")


class TaskCoreV2(nn.Module):
    def __init__(self, config: TaskCoreConfigV2):
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
    ) -> TaskStateV2:
        device = self.embedding.weight.device
        return TaskStateV2(
            bundle_id,
            scope_id,
            session_id,
            native_state_ref,
            parameter_revision,
            0,
            torch.zeros(self.config.width, device=device, dtype=torch.float32),
            torch.zeros((0, self.config.width), device=device, dtype=torch.float32),
            torch.zeros((0, self.config.width), device=device, dtype=torch.float32),
        )

    def step_task(
        self,
        observation: bytes,
        state: TaskStateV2,
        *,
        scope_id: str,
        session_id: str,
        source_id: str,
        resource_snapshot: Mapping[str, int],
    ) -> TaskStateV2:
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
        source_index = len(state.observations)
        byte_refs = tuple((source_index, index, index + 1) for index in range(len(observation)))
        byte_memory = torch.cat((state.byte_memory, encoded[0]), dim=0)[
            -self.config.max_evidence_bytes :
        ]
        combined_bytes = (state.byte_refs + byte_refs)[-self.config.max_evidence_bytes :]
        result = replace(
            state,
            hidden=hidden[0, 0],
            workspace=workspace,
            byte_memory=byte_memory,
            observations=state.observations + ((source_id, observation),),
            evidence_refs=(state.evidence_refs + refs)[-self.config.slots :],
            byte_refs=combined_bytes,
            revision=state.revision + 1,
        )
        return result.reset("turn")
