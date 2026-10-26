"""Opt-in V4: explicit workspace consumption beside the existing V3 byte path."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .conditional_language_v2 import _EvidenceCache
from .conditional_language_v3 import (
    ConditionalLanguageCandidateV3,
    ConditionalLanguageTrainerV3,
    TaskCoreConfigV3,
    TaskStateV3,
)
from .internalization import content_digest


@dataclass(frozen=True)
class _WorkspaceEvidenceCache(_EvidenceCache):
    workspace: torch.Tensor
    workspace_keys: torch.Tensor
    workspace_values: torch.Tensor


class ConditionalLanguageCandidateV4(ConditionalLanguageCandidateV3):
    """The state schema stays V3; bundle ownership and trainer format distinguish V4."""

    def __init__(
        self, config: TaskCoreConfigV3 | None = None, *, bundle_id: str = "task-language-v4"
    ):
        super().__init__(config, bundle_id=bundle_id)
        with torch.random.fork_rng(devices=[]):
            self.workspace_projection = nn.Linear(
                self.config.width, self.config.width, bias=False, dtype=torch.float32
            )
        with torch.no_grad():
            self.workspace_projection.weight.copy_(torch.eye(self.config.width) * 0.01)
        # A small nonzero initial path gives workspace and its producers credit.
        # Initial parent logits are close, not guaranteed identical.
        self.workspace_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    # V3/V2 states are parallel versions; runtime validate enforces isolation.
    def evidence_cache(self, state: TaskStateV3) -> _WorkspaceEvidenceCache:  # type: ignore[override]
        base = super().evidence_cache(state)
        if len(state.evidence_refs) != len(state.workspace):
            raise ValueError("workspace evidence provenance count differs")
        offsets, total = [], 0
        for _, observation in state.observations:
            offsets.append(total)
            total += len(observation)
        positions = []
        for source, start, end in state.evidence_refs:
            if not 0 <= source < len(offsets) or not 0 <= start < end <= len(
                state.observations[source][1]
            ):
                raise ValueError("workspace source span differs from actual observations")
            position = offsets[source] + end - 1
            if position >= self.config.max_input_evidence_bytes:
                raise ValueError("workspace span left the V3 input position window")
            positions.append(position)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=state.workspace.device)
        return _WorkspaceEvidenceCache(
            base.bundle_id,
            base.scope_id,
            base.session_id,
            base.parameter_revision,
            base.byte_memory,
            base.hidden,
            base.keys,
            base.values,
            state.workspace,
            self.core.keys(state.workspace) + self.input_position(position_tensor),
            self.core.values(state.workspace),
        )

    # State V3 is a deliberately parallel version; validate rejects V2 at runtime.
    def next_logits(  # type: ignore[override]
        self, state: TaskStateV3, *, cache: _EvidenceCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate(state)
        cache = self.evidence_cache(state) if cache is None else cache
        if (
            not isinstance(cache, _WorkspaceEvidenceCache)
            or (cache.bundle_id, cache.scope_id, cache.session_id, cache.parameter_revision)
            != (state.bundle_id, state.scope_id, state.session_id, state.parameter_revision)
            or cache.byte_memory is not state.byte_memory
            or cache.hidden is not state.hidden
            or cache.workspace is not state.workspace
        ):
            raise ValueError(
                "V4 workspace cache belongs to a different state or parameter revision"
            )
        attention = self.attention_weights(state, cache=cache)
        evidence = attention @ cache.values
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        if len(state.workspace):
            query = self.query(hidden) + self.output_position.weight[len(state.generated)]
            weights = torch.softmax(
                cache.workspace_keys @ query / self.config.key_width**0.5, dim=0
            )
            workspace = weights @ cache.workspace_values
            evidence = evidence + self.workspace_gate.sigmoid() * self.workspace_projection(
                workspace
            )
        previous = self.core.embedding(
            torch.tensor(state.previous_byte, dtype=torch.long, device=hidden.device)
        )
        updated = self.decoder(torch.cat((previous, evidence)), hidden)
        return self.output(torch.cat((updated, evidence))), updated


class ConditionalLanguageTrainerV4(ConditionalLanguageTrainerV3):
    FORMAT = "taiji-conditional-language-trainer-v4"
    model: ConditionalLanguageCandidateV4

    @classmethod
    def from_v3_checkpoint(cls, payload: dict[str, Any]):
        parent, state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
        model = ConditionalLanguageCandidateV4(TaskCoreConfigV3(**payload["config"]))
        mismatch = model.load_state_dict(parent.model.state_dict(), strict=False)
        if set(mismatch.missing_keys) != {"workspace_gate", "workspace_projection.weight"} or (
            mismatch.unexpected_keys
        ):
            raise ValueError("V3 to V4 shared parameter mapping differs")
        model.parameter_revision = parent.model.parameter_revision
        model.generator.set_state(parent.model.generator.get_state())
        trainer = cls(model, learning_rate=parent.learning_rate)
        old_parameters = dict(parent.model.named_parameters())
        for name, parameter in model.named_parameters():
            if name in old_parameters and old_parameters[name] in parent.optimizer.state:
                trainer.optimizer.state[parameter] = copy.deepcopy(
                    parent.optimizer.state[old_parameters[name]]
                )
        if len(parent.optimizer.param_groups) != 1:
            raise ValueError("migration requires the original single optimizer group")
        for key, value in parent.optimizer.param_groups[0].items():
            if key != "params":
                trainer.optimizer.param_groups[0][key] = copy.deepcopy(value)
        trainer.steps = parent.steps
        trainer.target_byte_exposures = parent.target_byte_exposures
        trainer.lineage = {
            "parent_format": payload["format"],
            "parent_checkpoint_digest": payload["checkpoint_digest"],
            "parent_lineage": copy.deepcopy(getattr(parent, "lineage", {})),
            "migration_training_updates": 0,
            "state_schema": "unchanged TaskStateV3; explicit V4 bundle remap",
            "new_parameters": ["workspace_gate", "workspace_projection.weight"],
            "parent_logits_identical": False,
        }
        state = replace(state, bundle_id=model.bundle_id)
        model.validate(state)
        return trainer, state, copy.deepcopy(cursor)

    # Same intentional V3-state narrowing as the parent V3 trainer.
    def checkpoint(  # type: ignore[override]
        self, state: TaskStateV3, *, cursor: dict[str, Any]
    ) -> dict[str, Any]:
        payload = super().checkpoint(state, cursor=cursor)
        payload["format"], payload["version"] = self.FORMAT, 4
        payload["checkpoint_digest"] = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]):
        if payload.get("format") != cls.FORMAT or payload.get("version") != 4:
            raise ValueError("V4 checkpoint requires explicit V3 migration or V4 format")
        if content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        ) != payload.get("checkpoint_digest"):
            raise ValueError("V4 checkpoint digest differs")
        model = ConditionalLanguageCandidateV4(
            TaskCoreConfigV3(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = payload["parameter_revision"]
        model.generator.set_state(payload["rng_state"])
        trainer = cls(model, learning_rate=payload["learning_rate"])
        # Optimizer loading can alias input tensors; continuation must never
        # mutate the caller's durable snapshot representation in memory.
        trainer.optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
        trainer.steps = payload["steps"]
        trainer.target_byte_exposures = payload["target_byte_exposures"]
        trainer.lineage = copy.deepcopy(payload.get("lineage", {}))
        state = TaskStateV3.from_payload(payload["task_state"])
        model.validate(state)
        return trainer, state, copy.deepcopy(payload["cursor"])
