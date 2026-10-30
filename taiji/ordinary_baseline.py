"""Ordinary end-to-end baseline v1: raw history text in, bytes out.

The third R2 comparison object.  It uses the same V3 byte backbone, the same
targets and the same fixed denominator, but has no hard parse gate, no discrete
store and no time-conditioned reader: the whole history is concatenated into
the prompt and the model must learn selection end to end.  At zero steps it
provides the structural readouts and the restored bundle identity that R4/R5
need, not a capability claim.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .conditional_language_v3 import ConditionalLanguageCandidateV3, TaskCoreConfigV3
from .internalization import content_digest
from .persistence import atomic_save

ORDINARY_BUNDLE_FORMAT = "taiji-ordinary-end-to-end-bundle-v1"
TIME_INTENTS = ("current", "previous", "unknown")
END_MARKER = b"<|end|>"
SEPARATOR = b"<|sep|>"


def build_prompt(history: Sequence[str], query: bytes, time_intent: str) -> bytes:
    """Concatenate the whole history and the query; no labels, no store."""

    if time_intent not in TIME_INTENTS:
        raise ValueError("ordinary prompt time intent is outside the declared set")
    if not isinstance(query, bytes):
        raise ValueError("ordinary prompt query must be bytes")
    parts = [b"<|history|>"]
    for index, text in enumerate(history):
        if index:
            parts.append(SEPARATOR)
        parts.append(str(text).encode("utf-8"))
    parts.append(b"<|query|>")
    parts.append(query)
    parts.append(b"<|time|>")
    parts.append(time_intent.encode("utf-8"))
    parts.append(b"<|answer|>")
    return b"".join(parts)


class OrdinaryEndToEndV1(ConditionalLanguageCandidateV3):
    """V3 byte model with no content structure at all; every parameter trains."""

    def __init__(
        self,
        config: TaskCoreConfigV3 | None = None,
        *,
        bundle_id: str = "ordinary-v1",
    ):
        super().__init__(config, bundle_id=bundle_id)

    def trainable_parameters(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def prompt_state(self, prompt: bytes):
        state = self.new_state(
            scope_id="ordinary", session_id="ordinary", native_state_ref="ordinary"
        )
        return self.step_task(
            prompt,
            state,
            scope_id="ordinary",
            session_id="ordinary",
            source_id="ordinary:prompt",
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": self.config.max_observation_bytes,
            },
        )

    def teacher_forced_prompt(self, prompt: bytes, target: bytes) -> torch.Tensor:
        return super().teacher_forced(prompt, target)

    @torch.no_grad()
    def answer_prompt(self, prompt: bytes, *, max_new_bytes: int = 64) -> bytes:
        if max_new_bytes <= 0 or max_new_bytes > self.config.max_generation_bytes:
            raise ValueError("ordinary generation allowance is outside the bound")
        return super().generate(self.prompt_state(prompt), max_new_bytes=max_new_bytes).generated

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class OrdinaryBundleV1:
    """Model plus raw history texts; restores at zero steps."""

    model: OrdinaryEndToEndV1
    history: tuple[str, ...]
    cursor: dict[str, Any]

    FORMAT = ORDINARY_BUNDLE_FORMAT

    @classmethod
    def create(
        cls,
        config: TaskCoreConfigV3 | None = None,
        *,
        bundle_id: str = "ordinary-v1",
    ) -> OrdinaryBundleV1:
        return cls(
            model=OrdinaryEndToEndV1(config, bundle_id=bundle_id),
            history=(),
            cursor={},
        )

    def observe(self, text: str) -> None:
        if not isinstance(text, str) or not text:
            raise ValueError("ordinary history text must be nonempty")
        self.history = (*self.history, text)

    def prompt(self, *, query: bytes, time_intent: str = "current") -> bytes:
        return build_prompt(self.history, query, time_intent)

    def answer(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        max_new_bytes: int = 64,
    ) -> bytes:
        return self.model.answer_prompt(
            self.prompt(query=query, time_intent=time_intent), max_new_bytes=max_new_bytes
        )

    def score_target(
        self,
        *,
        query: bytes,
        target: bytes,
        time_intent: str = "current",
    ) -> torch.Tensor:
        return self.model.teacher_forced_prompt(
            self.prompt(query=query, time_intent=time_intent), target
        )

    def checkpoint(self, *, state: Any | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.FORMAT,
            "version": 1,
            "bundle_id": self.model.bundle_id,
            "config": asdict(self.model.config),
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "history": list(self.history),
            "cursor": copy.deepcopy(self.cursor),
            "task_state": None if state is None else state.to_payload(),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path, *, state: Any | None = None) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("ordinary bundle checkpoint is write-once")
        atomic_save(self.checkpoint(state=state), target)
        return target

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> OrdinaryBundleV1:
        expected = {
            "format",
            "version",
            "bundle_id",
            "config",
            "parameter_revision",
            "parameters",
            "history",
            "cursor",
            "task_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("ordinary bundle payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported ordinary bundle version")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("ordinary bundle checkpoint digest differs")
        model = OrdinaryEndToEndV1(
            TaskCoreConfigV3(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = int(payload["parameter_revision"])
        return cls(
            model=model,
            history=tuple(str(item) for item in payload["history"]),
            cursor=copy.deepcopy(dict(payload["cursor"])),
        )

    @classmethod
    def load(cls, path: Path) -> OrdinaryBundleV1:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls.from_checkpoint(payload)

    def probe_digest(self, *, query: bytes, time_intent: str = "current") -> str:
        logits = self.score_target(query=query, target=END_MARKER, time_intent=time_intent)
        answer = self.answer(query=query, time_intent=time_intent)
        return content_digest(
            {
                "prompt": self.prompt(query=query, time_intent=time_intent).hex(),
                "logits": [[round(float(value), 7) for value in row] for row in logits.tolist()],
                "answer_hex": answer.hex(),
            }
        )
