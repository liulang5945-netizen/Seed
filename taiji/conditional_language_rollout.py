"""Opt-in scheduled-sampling trainer for the v2 conditional language candidate.

Only the previous byte fed to the decoder changes: with probability ``p`` the
model consumes its own greedy legal byte, otherwise the training target.  The
loss is still computed against target bytes and the sampling decision carries no
gradient.  Checkpoints use a distinct format so a rollout run can never be
mistaken for the teacher-forced N1-170/171/172 trainer.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .conditional_language_v2 import ConditionalLanguageCandidateV2
from .internalization import content_digest
from .persistence import atomic_save
from .task_core_v2 import TaskCoreConfigV2
from .task_state_v2 import TaskStateV2
from .utf8_state import utf8_allowed


def rollout_logits(
    model: ConditionalLanguageCandidateV2,
    prompt: bytes,
    target: bytes,
    *,
    probability: float,
    scope_id: str,
    session_id: str,
    native_state_ref: str,
) -> torch.Tensor:
    """Teacher-forced logits with scheduled sampling on the decoder input."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("scheduled sampling probability must lie in [0, 1]")
    if not target or len(target) > model.config.max_generation_bytes:
        raise ValueError("rollout response exceeds the declared byte bound")
    state = model.new_state(
        scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
    )
    state = model.step_task(
        prompt,
        state,
        scope_id=scope_id,
        session_id=session_id,
        source_id="prompt",
        resource_snapshot={
            "remaining_steps": 1,
            "max_observation_bytes": model.config.max_observation_bytes,
        },
    )
    cache = model.evidence_cache(state)
    logits = []
    for symbol in target:
        current, hidden = model.next_logits(state, cache=cache)
        logits.append(current)
        if probability > 0 and float(torch.rand((), generator=model.generator)) < probability:
            allowed = torch.tensor(utf8_allowed(state.utf8_remaining, state.utf8_lead))
            chosen = int(allowed[current[allowed].argmax()])
        else:
            chosen = symbol
        state = model.observe_response(state, chosen, hidden)
    if not logits:
        raise ValueError("rollout training requires a nonempty target")
    return torch.stack(logits)


class ConditionalLanguageRolloutTrainer:
    FORMAT = "taiji-conditional-language-rollout-trainer-v1"
    VERSION = 1

    def __init__(
        self,
        model: ConditionalLanguageCandidateV2,
        *,
        learning_rate: float = 0.001,
        max_probability: float = 0.5,
        schedule_steps: int = 256,
    ):
        if not 0.0 <= max_probability <= 1.0:
            raise ValueError("max_probability must lie in [0, 1]")
        if schedule_steps <= 0:
            raise ValueError("schedule_steps must be positive")
        self.model = model
        self.learning_rate = learning_rate
        self.max_probability = max_probability
        self.schedule_steps = schedule_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        self.steps = 0
        self.target_byte_exposures = 0

    def probability(self) -> float:
        return min(self.max_probability, (self.steps + 1) / self.schedule_steps)

    def update(self, examples: Sequence[tuple[bytes, bytes]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("update requires a declared nonempty batch")
        probability = self.probability()
        self.optimizer.zero_grad(set_to_none=True)
        byte_count = sum(len(target) for _, target in examples)
        terms = [
            F.cross_entropy(
                rollout_logits(
                    self.model,
                    prompt,
                    target,
                    probability=probability,
                    scope_id="research",
                    session_id="train",
                    native_state_ref="research:train",
                ),
                torch.tensor(list(target), dtype=torch.long),
                reduction="sum",
            )
            for prompt, target in examples
        ]
        loss = torch.stack(terms).sum() / byte_count
        loss.backward()
        gradients: dict[str, float] = {}
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
            "probability": probability,
            "loss_nats": float(loss.detach()),
            "loss_bits": float(loss.detach()) / math.log(2),
            "gradient_norm_before_clip": norm,
            "gradient_group_norms": {name: value**0.5 for name, value in gradients.items()},
            "target_byte_exposures": self.target_byte_exposures,
        }

    def checkpoint(self, state: TaskStateV2, *, cursor: dict[str, Any]) -> dict[str, Any]:
        self.model.validate(state)
        payload = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "config": asdict(self.model.config),
            "bundle_id": self.model.bundle_id,
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "learning_rate": self.learning_rate,
            "max_probability": self.max_probability,
            "schedule_steps": self.schedule_steps,
            "steps": self.steps,
            "target_byte_exposures": self.target_byte_exposures,
            "rng_state": self.model.generator.get_state().clone(),
            "task_state": state.to_payload(),
            "cursor": copy.deepcopy(cursor),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path, state: TaskStateV2, *, cursor: dict[str, Any]) -> None:
        if Path(path).exists():
            raise ValueError("rollout checkpoint is write-once")
        atomic_save(self.checkpoint(state, cursor=cursor), path)

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any]
    ) -> tuple[ConditionalLanguageRolloutTrainer, TaskStateV2, dict[str, Any]]:
        if payload.get("format") != cls.FORMAT or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported rollout checkpoint")
        if content_digest(
            {name: value for name, value in payload.items() if name != "checkpoint_digest"}
        ) != payload.get("checkpoint_digest"):
            raise ValueError("rollout checkpoint digest differs")
        model = ConditionalLanguageCandidateV2(
            TaskCoreConfigV2(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = payload["parameter_revision"]
        trainer = cls(
            model,
            learning_rate=payload["learning_rate"],
            max_probability=payload["max_probability"],
            schedule_steps=payload["schedule_steps"],
        )
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.steps = payload["steps"]
        trainer.target_byte_exposures = payload["target_byte_exposures"]
        model.generator.set_state(payload["rng_state"])
        state = TaskStateV2.from_payload(payload["task_state"])
        model.validate(state)
        return trainer, state, copy.deepcopy(payload["cursor"])
