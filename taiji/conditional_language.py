"""Opt-in native language candidate consuming shared task state and raw evidence.

Not a default-model replacement or the old sequence-workspace checkpoint format.
Teacher forcing and generation call the identical causal next-byte step.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .internalization import content_digest
from .persistence import atomic_save
from .task_core import TaskCoreConfig, TaskCoreV1
from .task_state import TaskStateV1
from .utf8_state import advance_utf8, utf8_allowed


@dataclass(frozen=True)
class _EvidenceCache:
    bundle_id: str
    scope_id: str
    session_id: str
    parameter_revision: int
    workspace: torch.Tensor
    hidden: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


class ConditionalLanguageCandidate(nn.Module):
    def __init__(
        self, config: TaskCoreConfig | None = None, *, bundle_id: str = "task-language-v1"
    ):
        super().__init__()
        config = TaskCoreConfig() if config is None else config
        self.config, self.bundle_id, self.parameter_revision = config, bundle_id, 0
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.seed)
            self.core = TaskCoreV1(config)
            self.query = nn.Linear(
                config.width, config.key_width, device="cpu", dtype=torch.float32
            )
            self.decoder = nn.GRUCell(
                config.embedding_width + config.width,
                config.width,
                device="cpu",
                dtype=torch.float32,
            )
            self.output = nn.Linear(2 * config.width, 257, device="cpu", dtype=torch.float32)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def new_state(self, *, scope_id: str, session_id: str, native_state_ref: str) -> TaskStateV1:
        return self.core.empty(
            bundle_id=self.bundle_id,
            scope_id=scope_id,
            session_id=session_id,
            native_state_ref=native_state_ref,
            parameter_revision=self.parameter_revision,
        )

    def validate(self, state: TaskStateV1) -> None:
        if state.bundle_id != self.bundle_id or state.parameter_revision != self.parameter_revision:
            raise ValueError("task state belongs to another bundle or parameter revision")
        if len(state.hidden) != self.config.width or state.workspace.shape[1] != self.config.width:
            raise ValueError("task state width differs from this candidate")

    def step_task(self, observation: bytes, state: TaskStateV1, **kwargs: Any) -> TaskStateV1:
        self.validate(state)
        return self.core.step_task(observation, state, **kwargs)

    def evidence_cache(self, state: TaskStateV1) -> _EvidenceCache:
        self.validate(state)
        if len(state.workspace) == 0:
            raise ValueError("language generation requires actual task evidence")
        return _EvidenceCache(
            state.bundle_id,
            state.scope_id,
            state.session_id,
            state.parameter_revision,
            state.workspace,
            state.hidden,
            self.core.keys(state.workspace),
            self.core.values(state.workspace),
        )

    def next_logits(
        self, state: TaskStateV1, *, cache: _EvidenceCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate(state)
        cache = self.evidence_cache(state) if cache is None else cache
        if (
            (cache.bundle_id, cache.scope_id, cache.session_id, cache.parameter_revision)
            != (state.bundle_id, state.scope_id, state.session_id, state.parameter_revision)
            or cache.workspace is not state.workspace
            or cache.hidden is not state.hidden
        ):
            raise ValueError("evidence cache belongs to another task state or parameter revision")
        keys, values = cache.keys, cache.values
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        attention = torch.softmax(
            keys @ self.query(hidden) / math.sqrt(self.config.key_width), dim=0
        )
        evidence = attention @ values
        previous = self.core.embedding(
            torch.tensor(state.previous_byte, dtype=torch.long, device=hidden.device)
        )
        updated = self.decoder(torch.cat((previous, evidence)), hidden)
        return self.output(torch.cat((updated, evidence))), updated

    def observe_response(
        self, state: TaskStateV1, symbol: int, decoder_hidden: torch.Tensor
    ) -> TaskStateV1:
        if type(symbol) is not int or not 0 <= symbol < 256:
            raise ValueError("a response symbol must be an actual byte")
        remaining, lead = advance_utf8(state.utf8_remaining, state.utf8_lead, symbol)
        return replace(
            state,
            decoder_hidden=decoder_hidden,
            previous_byte=symbol,
            generated=state.generated + bytes([symbol]),
            utf8_remaining=remaining,
            utf8_lead=lead,
            revision=state.revision + 1,
        )

    def teacher_forced(
        self,
        prompt: bytes,
        target: bytes,
        *,
        scope_id: str = "research",
        session_id: str = "train",
        native_state_ref: str = "research:train",
    ) -> torch.Tensor:
        if not target or len(target) > self.config.max_generation_bytes:
            raise ValueError("teacher-forced response exceeds the declared byte bound")
        state = self.new_state(
            scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        state = self.step_task(
            prompt,
            state,
            scope_id=scope_id,
            session_id=session_id,
            source_id="prompt",
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": self.config.max_observation_bytes,
            },
        )
        cache = self.evidence_cache(state)
        logits = []
        for symbol in target:
            current, decoder_hidden = self.next_logits(state, cache=cache)
            logits.append(current)
            state = self.observe_response(state, symbol, decoder_hidden)
        if not logits:
            raise ValueError("teacher forcing requires a nonempty target")
        return torch.stack(logits)

    @torch.no_grad()
    def generate(self, state: TaskStateV1, *, max_new_bytes: int = 128) -> TaskStateV1:
        if max_new_bytes <= 0:
            raise ValueError("generation byte allowance must be positive")
        if len(state.generated) + max_new_bytes > self.config.max_generation_bytes:
            raise ValueError("generation exceeds the declared task-state response bound")
        cache = self.evidence_cache(state)
        for _ in range(max_new_bytes):
            if state.generated.endswith(b"<|end|>"):
                break
            logits, hidden = self.next_logits(state, cache=cache)
            allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
            index = torch.tensor(allowed, device=logits.device)
            symbol = int(index[logits[index].argmax()])
            state = self.observe_response(state, symbol, hidden)
        return state

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ConditionalLanguageTrainer:
    FORMAT = "taiji-conditional-language-trainer-v1"

    def __init__(self, model: ConditionalLanguageCandidate, *, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        self.steps = self.target_byte_exposures = 0

    def update(self, examples: Sequence[tuple[bytes, bytes]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("update requires a declared nonempty batch")
        self.optimizer.zero_grad(set_to_none=True)
        byte_count = sum(len(target) for _, target in examples)
        loss = (
            sum(
                F.cross_entropy(
                    self.model.teacher_forced(prompt, target),
                    torch.tensor(list(target), dtype=torch.long),
                    reduction="sum",
                )
                for prompt, target in examples
            )
            / byte_count
        )
        loss.backward()
        gradients = {}
        for name, parameter in self.model.named_parameters():
            group = (
                ".".join(name.split(".")[:2]) if name.startswith("core.") else name.split(".")[0]
            )
            if parameter.grad is not None:
                gradients[group] = gradients.get(group, 0.0) + float(
                    parameter.grad.detach().square().sum()
                )
        norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        )
        self.optimizer.step()
        self.model.parameter_revision += 1
        self.steps += 1
        self.target_byte_exposures += byte_count
        return {
            "step": self.steps,
            "loss_nats": float(loss.detach()),
            "loss_bits": float(loss.detach()) / math.log(2),
            "gradient_norm_before_clip": norm,
            "gradient_group_norms": {name: value**0.5 for name, value in gradients.items()},
            "target_byte_exposures": self.target_byte_exposures,
        }

    def checkpoint(self, state: TaskStateV1, *, cursor: dict[str, Any]) -> dict[str, Any]:
        self.model.validate(state)
        payload = {
            "format": self.FORMAT,
            "version": 1,
            "config": asdict(self.model.config),
            "bundle_id": self.model.bundle_id,
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "target_byte_exposures": self.target_byte_exposures,
            "rng_state": self.model.generator.get_state().clone(),
            "task_state": state.to_payload(),
            "cursor": copy.deepcopy(cursor),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path, state: TaskStateV1, *, cursor: dict[str, Any]) -> None:
        if path.exists():
            raise ValueError("candidate checkpoint is write-once")
        atomic_save(self.checkpoint(state, cursor=cursor), path)

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any]
    ) -> tuple[ConditionalLanguageTrainer, TaskStateV1, dict[str, Any]]:
        if payload.get("format") != cls.FORMAT or payload.get("version") != 1:
            raise ValueError("unsupported conditional language checkpoint")
        if content_digest(
            {name: value for name, value in payload.items() if name != "checkpoint_digest"}
        ) != payload.get("checkpoint_digest"):
            raise ValueError("conditional language checkpoint digest differs")
        model = ConditionalLanguageCandidate(
            TaskCoreConfig(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = payload["parameter_revision"]
        trainer = cls(model, learning_rate=payload["learning_rate"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.steps, trainer.target_byte_exposures = (
            payload["steps"],
            payload["target_byte_exposures"],
        )
        model.generator.set_state(payload["rng_state"])
        state = TaskStateV1.from_payload(payload["task_state"])
        model.validate(state)
        return trainer, state, copy.deepcopy(payload["cursor"])
