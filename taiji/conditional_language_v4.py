"""Opt-in V4: explicit workspace consumption beside the existing V3 byte path."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

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
from .utf8_state import utf8_allowed

if TYPE_CHECKING:
    from .episodic_memory import EpisodicMemoryStore
    from .joint_action import JointActionReadV1
    from .joint_task_state import JointEpisodeReadV1, JointTaskStateV1


@dataclass(frozen=True)
class _WorkspaceEvidenceCache(_EvidenceCache):
    workspace: torch.Tensor
    workspace_keys: torch.Tensor
    workspace_values: torch.Tensor
    selected_source_ids: tuple[str, ...] = ()
    joint_revision: int | None = None
    joint_owner_digest: str = ""


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

    def _joint_evidence_cache(
        self,
        joint_state: JointTaskStateV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
        include_memory_sources: bool = True,
        selected_source_ids: tuple[str, ...] | None = None,
    ) -> _WorkspaceEvidenceCache:
        """Build a source-filtered cache from selected facts in the canonical state."""
        from .joint_task_state import JointEpisodeReadV1, JointTaskStateV1

        if not isinstance(joint_state, JointTaskStateV1):
            raise TypeError("joint language consumption requires JointTaskStateV1")
        joint_state._validate_live_owner()
        state = joint_state.language_state
        self.validate(state)
        if joint_state.lifecycle != "active":
            raise ValueError("joint task lifecycle does not permit language generation")
        if state.bundle_id != self.bundle_id:
            raise ValueError("joint task belongs to another language candidate")

        if selected_source_ids is not None:
            if episode_read is not None or memory_store is not None:
                raise ValueError("explicit source set cannot be mixed with other read receipts")
            selected_sources = selected_source_ids
        elif episode_read is None:
            if memory_store is not None:
                raise ValueError("memory store requires an explicit joint episode read")
            view = joint_state.content_view()
            if not view.selected_candidates:
                raise ValueError("joint language consumption requires a selected content candidate")
            unresolved = tuple(
                f"{item.candidate_id}:{item.source_status}"
                for item in view.selected_candidates
                if item.source_status != "source_linked"
            )
            if unresolved:
                raise ValueError(
                    "selected joint content lacks exact source-linked evidence: "
                    + ", ".join(unresolved)
                )
            selected_sources = view.selected_neural_source_ids
        else:
            if not isinstance(episode_read, JointEpisodeReadV1):
                raise TypeError("memory-conditioned language requires JointEpisodeReadV1")
            if memory_store is None:
                raise ValueError("memory-conditioned language requires its source store")
            episode_read.validate_for(joint_state, memory_store)
            selected_sources = (
                episode_read.selected_neural_source_ids
                if include_memory_sources
                else (episode_read.query_neural_source_id,)
            )
        if not selected_sources:
            raise ValueError("selected joint content has no exact neural input source")

        source_indices = {
            source_id: index for index, (source_id, _) in enumerate(state.observations)
        }
        missing = tuple(
            source_id for source_id in selected_sources if source_id not in source_indices
        )
        if missing:
            raise ValueError("selected joint source is absent from the neural observation ledger")
        selected_indices = {source_indices[source_id] for source_id in selected_sources}
        byte_rows = [
            index
            for index, (source, _, _) in enumerate(state.byte_refs)
            if source in selected_indices
        ]
        if not byte_rows:
            raise ValueError("selected joint sources have no surviving byte evidence")
        workspace_rows = [
            index
            for index, (source, _, _) in enumerate(state.evidence_refs)
            if source in selected_indices
        ]

        base = self.evidence_cache(state)
        byte_index = torch.tensor(byte_rows, dtype=torch.long, device=state.byte_memory.device)
        if workspace_rows:
            workspace_index = torch.tensor(
                workspace_rows, dtype=torch.long, device=state.workspace.device
            )
            workspace_keys = base.workspace_keys.index_select(0, workspace_index)
            workspace_values = base.workspace_values.index_select(0, workspace_index)
        else:
            workspace_keys = base.workspace_keys[:0]
            workspace_values = base.workspace_values[:0]
        return _WorkspaceEvidenceCache(
            base.bundle_id,
            base.scope_id,
            base.session_id,
            base.parameter_revision,
            base.byte_memory,
            base.hidden,
            base.keys.index_select(0, byte_index),
            base.values.index_select(0, byte_index),
            base.workspace,
            workspace_keys,
            workspace_values,
            selected_sources,
            joint_state.joint_revision,
            joint_state.cognitive_state_sha256,
        )

    def _action_evidence_cache(
        self,
        joint_state: JointTaskStateV1,
        action_read: JointActionReadV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
    ) -> _WorkspaceEvidenceCache:
        from .joint_action import JointActionReadV1

        if not isinstance(action_read, JointActionReadV1):
            raise TypeError("action-conditioned language requires JointActionReadV1")
        action_read.validate_for(joint_state, episode_read=episode_read, memory_store=memory_store)
        base = self._joint_evidence_cache(
            joint_state, selected_source_ids=action_read.source_neural_ids
        )
        if len(action_read.projection_bytes) > self.config.max_observation_bytes:
            raise ValueError("joint action projection exceeds the V4 observation bound")
        action_source_id = "joint-action:" + action_read.projection_sha256[:32]
        action_state = self.new_state(
            scope_id=joint_state.scope_id,
            session_id=joint_state.session_id,
            native_state_ref=action_source_id,
        )
        action_state = cast(
            TaskStateV3,
            self.step_task(
                action_read.projection_bytes,
                action_state,
                scope_id=joint_state.scope_id,
                session_id=joint_state.session_id,
                source_id=action_source_id,
                resource_snapshot={
                    "remaining_steps": 1,
                    "max_observation_bytes": self.config.max_observation_bytes,
                },
            ),
        )
        action_cache = self.evidence_cache(action_state)
        return _WorkspaceEvidenceCache(
            base.bundle_id,
            base.scope_id,
            base.session_id,
            base.parameter_revision,
            base.byte_memory,
            base.hidden,
            torch.cat((base.keys, action_cache.keys), dim=0),
            torch.cat((base.values, action_cache.values), dim=0),
            base.workspace,
            torch.cat((base.workspace_keys, action_cache.workspace_keys), dim=0),
            torch.cat((base.workspace_values, action_cache.workspace_values), dim=0),
            (*base.selected_source_ids, action_source_id),
            joint_state.joint_revision,
            joint_state.cognitive_state_sha256,
        )

    def next_logits_from_joint(
        self, joint_state: JointTaskStateV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the real V4 next-byte decoder with canonical selected-source conditioning."""
        cache = self._joint_evidence_cache(joint_state)
        return self.next_logits(joint_state.language_state, cache=cache)

    def next_logits_from_episode(
        self,
        joint_state: JointTaskStateV1,
        episode_read: JointEpisodeReadV1,
        memory_store: EpisodicMemoryStore,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run V4 with both the current query frame and one retrieved episode."""
        cache = self._joint_evidence_cache(
            joint_state, episode_read=episode_read, memory_store=memory_store
        )
        return self.next_logits(joint_state.language_state, cache=cache)

    def next_logits_from_episode_query_only(
        self,
        joint_state: JointTaskStateV1,
        episode_read: JointEpisodeReadV1,
        memory_store: EpisodicMemoryStore,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lesion baseline: keep the current query and remove retrieved memory sources."""
        cache = self._joint_evidence_cache(
            joint_state,
            episode_read=episode_read,
            memory_store=memory_store,
            include_memory_sources=False,
        )
        return self.next_logits(joint_state.language_state, cache=cache)

    def next_logits_from_action(
        self,
        joint_state: JointTaskStateV1,
        action_read: JointActionReadV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run V4 with the selected Taiji ActionIntent and its prediction in the read path."""
        cache = self._action_evidence_cache(
            joint_state,
            action_read,
            episode_read=episode_read,
            memory_store=memory_store,
        )
        return self.next_logits(joint_state.language_state, cache=cache)

    def _generate_from_joint_cache(
        self,
        joint_state: JointTaskStateV1,
        cache: _WorkspaceEvidenceCache,
        *,
        max_new_bytes: int,
    ) -> tuple[bytes, JointTaskStateV1]:
        if max_new_bytes <= 0:
            raise ValueError("generation byte allowance must be positive")
        state = joint_state.language_state
        if len(state.generated) + max_new_bytes > self.config.max_generation_bytes:
            raise ValueError("generation exceeds the declared V3 response bound")
        start = len(state.generated)
        for _ in range(max_new_bytes):
            if state.generated.endswith(b"<|end|>"):
                break
            logits, hidden = self.next_logits(state, cache=cache)
            allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
            choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            symbol = int(choices[logits[choices].argmax()])
            state = cast(TaskStateV3, self.observe_response(state, symbol, hidden))
        updated_joint_state = joint_state.with_language_state(state)
        if state.generated.endswith(b"<|end|>"):
            updated_joint_state = updated_joint_state.complete_response()
        return state.generated[start:], updated_joint_state

    @torch.no_grad()
    def generate_from_joint(
        self, joint_state: JointTaskStateV1, *, max_new_bytes: int = 128
    ) -> tuple[bytes, JointTaskStateV1]:
        """Greedy V4 generation whose evidence keys are restricted to selected joint sources."""
        cache = self._joint_evidence_cache(joint_state)
        return self._generate_from_joint_cache(joint_state, cache, max_new_bytes=max_new_bytes)

    @torch.no_grad()
    def generate_from_episode(
        self,
        joint_state: JointTaskStateV1,
        episode_read: JointEpisodeReadV1,
        memory_store: EpisodicMemoryStore,
        *,
        max_new_bytes: int = 128,
    ) -> tuple[bytes, JointTaskStateV1]:
        """Greedy V4 generation conditioned on a source-bound episodic read."""
        cache = self._joint_evidence_cache(
            joint_state, episode_read=episode_read, memory_store=memory_store
        )
        return self._generate_from_joint_cache(joint_state, cache, max_new_bytes=max_new_bytes)

    @torch.no_grad()
    def generate_from_action(
        self,
        joint_state: JointTaskStateV1,
        action_read: JointActionReadV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
        max_new_bytes: int = 128,
    ) -> tuple[bytes, JointTaskStateV1]:
        """Greedy V4 generation conditioned on the same state's selected action evidence."""
        cache = self._action_evidence_cache(
            joint_state,
            action_read,
            episode_read=episode_read,
            memory_store=memory_store,
        )
        return self._generate_from_joint_cache(joint_state, cache, max_new_bytes=max_new_bytes)

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
        if len(cache.workspace_keys):
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
