"""Joint owner-routed trainer and single-envelope recovery for the N177 T1 slice.

This opt-in runner does not alter the default Adapter or run training by being
imported. Its update method accepts only the train split and routes semantic
span loss to the native input producer and byte-answer loss to the V4 language
candidate. Every episode is rebuilt from raw InputFrames after an update.
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .adapter import TSKV8Adapter
from .conditional_language_v4 import ConditionalLanguageTrainerV4
from .contracts import CognitiveState, NativeCheckpoint, WorkspaceSelection, WorkspaceState
from .input_boundary import InputTrace
from .internalization import content_digest
from .joint_input_producer import (
    JOINT_INPUT_PRODUCER_COMPONENT_KEY,
    JointInputProducerV1,
    JointSemanticParseV1,
)
from .joint_t1_course import (
    JOINT_T1_BATCH_SIZE,
    JOINT_T1_COURSE_FORMAT,
    JOINT_T1_UPDATE_COUNT,
    JointT1EpisodeV1,
    JointT1TurnV1,
    build_joint_t1_split,
    build_joint_t1_update_order,
    joint_t1_course_contract,
    joint_t1_split_digest,
    joint_t1_update_order_sha256,
    parser_permuted_target,
)
from .joint_t1_trace_cache import JointT1TraceCacheV1
from .joint_task_state import JOINT_NATIVE_COMPONENT, JointTaskStateV1
from .persistence import atomic_save

JOINT_T1_TRAINER_FORMAT = "taiji-joint-t1-trainer-v1"
JOINT_T1_TRAINER_COMPONENT_KEY = "joint_t1_trainer_v1"
JOINT_T1_TRAINER_COMPONENT_FORMAT = "taiji-joint-t1-trainer-component-v1"

JOINT_T1_MODES = (
    "correct",
    "parser_permuted",
    "no_update",
    "readout_only",
    "responsible_owner_frozen",
)


def _mode_policy(mode: str) -> dict[str, Any]:
    if mode not in JOINT_T1_MODES:
        raise ValueError("unsupported N177 T1 update mode")
    producer_updates = mode in {"correct", "parser_permuted"}
    language_updates = mode in {
        "correct",
        "parser_permuted",
        "readout_only",
        "responsible_owner_frozen",
    }
    return {
        "mode": mode,
        "producer_parameter_owner": "input_semantic_producer",
        "producer_updates": producer_updates,
        "language_parameter_owner": "language_candidate",
        "language_updates": language_updates,
        "language_parameter_scope": (
            "output_only" if mode == "readout_only" else ("all" if language_updates else "none")
        ),
        "cognitive_state_writes": mode
        in {
            "correct",
            "parser_permuted",
            "responsible_owner_frozen",
        },
        "parser_target": (
            "entity_attribute_swapped_on_f1" if mode == "parser_permuted" else "correct"
        ),
        "answer_target": "correct",
    }


def _component_with_digest(payload: dict[str, Any]) -> dict[str, Any]:
    component = dict(payload)
    component["component_digest"] = content_digest(component)
    return component


def _carry_t1_world_workspace(
    previous: CognitiveState,
    incoming: CognitiveState,
) -> CognitiveState:
    """Preserve model-written facts while advancing the frozen Adapter input owner."""

    if previous.episode_id != incoming.episode_id:
        raise ValueError("T1 owner-state merge crossed episode identity")
    object_map = {item.object_id: item for item in incoming.world.objects}
    object_map.update({item.object_id: item for item in previous.world.objects})
    objects = tuple(object_map[key] for key in sorted(object_map))

    relation_map = {(item[0], item[1]): item for item in incoming.world.relations}
    relation_map.update({(item[0], item[1]): item for item in previous.world.relations})
    relations = tuple(relation_map[key] for key in sorted(relation_map))

    event_map = {item.event_id: item for item in incoming.world.events}
    event_map.update({item.event_id: item for item in previous.world.events})
    world_events = tuple(sorted(event_map.values(), key=lambda item: (item.tick, item.event_id)))
    affordance_map = {item.affordance_id: item for item in incoming.world.affordances}
    affordance_map.update({item.affordance_id: item for item in previous.world.affordances})
    world = replace(
        incoming.world,
        tick=max(previous.world.tick, incoming.world.tick),
        entities=(
            tuple(item.object_id for item in objects)
            if objects
            else tuple(sorted(set(previous.world.entities) | set(incoming.world.entities)))
        ),
        relations=relations,
        objects=objects,
        events=world_events,
        affordances=tuple(affordance_map[key] for key in sorted(affordance_map)),
    )

    prior_workspace = previous.workspace
    current_workspace = incoming.workspace
    candidate_map = {item.candidate_id: item for item in current_workspace.candidates}
    candidate_map.update({item.candidate_id: item for item in prior_workspace.candidates})
    candidate_order = tuple(
        dict.fromkeys(
            item.candidate_id
            for item in (*current_workspace.candidates, *prior_workspace.candidates)
        )
    )
    capacity = max(prior_workspace.capacity, current_workspace.capacity)
    if capacity > 0:
        candidate_order = candidate_order[-capacity:]
    candidates = tuple(candidate_map[item] for item in candidate_order)
    source_selection = current_workspace.selection or prior_workspace.selection
    if source_selection is not None and not set(source_selection.selected_ids).issubset(
        set(candidate_order)
    ):
        source_selection = prior_workspace.selection
    workspace_tick = max(prior_workspace.tick, current_workspace.tick)
    selection = None
    focus: tuple[str, ...] = ()
    broadcast = current_workspace.broadcast
    if source_selection is not None:
        candidate_ids = tuple(item.candidate_id for item in candidates)
        candidate_set = set(candidate_ids)
        selected = tuple(item for item in source_selection.selected_ids if item in candidate_set)
        previous_scores = dict(
            zip(source_selection.candidate_ids, source_selection.scores, strict=True)
        )
        scores = tuple(float(previous_scores.get(item, 0.0)) for item in candidate_ids)
        broadcast = source_selection.broadcast
        selection = WorkspaceSelection(
            tick=workspace_tick,
            mode=source_selection.mode if selected else "none",
            candidate_ids=candidate_ids,
            selected_ids=selected,
            scores=scores,
            broadcast=broadcast,
            capacity=capacity,
        )
        focus = selected
    bindings = dict(current_workspace.bindings)
    bindings.update(dict(prior_workspace.bindings))
    retained_ids = set(candidate_order)
    workspace = WorkspaceState(
        tick=workspace_tick,
        focus=focus,
        bindings=tuple(
            sorted((key, value) for key, value in bindings.items() if key in retained_ids)
        ),
        broadcast=broadcast,
        capacity=capacity,
        candidates=candidates,
        selection=selection,
        percept_event_id=current_workspace.percept_event_id or prior_workspace.percept_event_id,
        percept_assembly_id=current_workspace.percept_assembly_id
        or prior_workspace.percept_assembly_id,
        percept_boundary_closed=(
            current_workspace.percept_boundary_closed or prior_workspace.percept_boundary_closed
        ),
    )
    root_events = {item.event_id: item for item in incoming.events}
    root_events.update({item.event_id: item for item in previous.events})
    assemblies = {item.assembly_id: item for item in incoming.assemblies}
    assemblies.update({item.assembly_id: item for item in previous.assemblies})
    concepts = {item.concept_id: item for item in incoming.concepts}
    concepts.update({item.concept_id: item for item in previous.concepts})
    return replace(
        incoming,
        tick=max(previous.tick, incoming.tick),
        workspace=workspace,
        world=world,
        events=tuple(
            sorted(
                root_events.values(),
                key=lambda item: (item.start_tick, item.end_tick, item.event_id),
            )
        ),
        assemblies=tuple(
            sorted(
                assemblies.values(),
                key=lambda item: (item.start_tick, item.end_tick, item.assembly_id),
            )
        ),
        concepts=tuple(sorted(concepts.values(), key=lambda item: item.concept_id)),
    )


class JointT1TrainerV1:
    """One deterministic update arm over the native T1 producer and V4 reader."""

    FORMAT = JOINT_T1_TRAINER_FORMAT
    COMPONENT_KEY = JOINT_T1_TRAINER_COMPONENT_KEY

    def __init__(
        self,
        *,
        language_trainer: ConditionalLanguageTrainerV4,
        producer: JointInputProducerV1,
        base_adapter_checkpoint: Mapping[str, Any],
        train_episodes: Sequence[JointT1EpisodeV1],
        mode: str,
        seed: int,
        parent_v4_sha256: str,
        joint_bundle_id: str | None = None,
        owner_id: str = "n177-t1-research-owner",
        producer_learning_rate: float = 0.001,
        update_cursor: int = 0,
        optimizer_state: Mapping[str, Any] | None = None,
        input_trace_cache: JointT1TraceCacheV1 | None = None,
    ) -> None:
        if not isinstance(language_trainer, ConditionalLanguageTrainerV4):
            raise TypeError("N177 T1 requires ConditionalLanguageTrainerV4")
        if not isinstance(producer, JointInputProducerV1):
            raise TypeError("N177 T1 requires JointInputProducerV1")
        if not isinstance(base_adapter_checkpoint, Mapping):
            raise TypeError("N177 T1 requires one native TSK-v8 adapter checkpoint")
        if mode not in JOINT_T1_MODES or seed not in {0, 1, 2}:
            raise ValueError("N177 T1 mode or parser seed is outside the frozen contract")
        if len(parent_v4_sha256) != 64:
            raise ValueError("N177 T1 requires the bound V4 parent file SHA-256")
        if not 0 <= update_cursor <= JOINT_T1_UPDATE_COUNT:
            raise ValueError("N177 T1 update cursor is outside the fixed run")
        if producer_learning_rate <= 0:
            raise ValueError("producer learning rate must be positive")

        native = NativeCheckpoint.from_payload(base_adapter_checkpoint)
        if native.adapter != TSKV8Adapter.ADAPTER_NAME:
            raise ValueError("N177 T1 adapter checkpoint must use TSK-v8")
        if {
            JOINT_NATIVE_COMPONENT,
            JOINT_INPUT_PRODUCER_COMPONENT_KEY,
            JOINT_T1_TRAINER_COMPONENT_KEY,
        }.intersection(native.components):
            raise ValueError("N177 T1 baseline adapter checkpoint already contains a joint run")
        if any(item.partition != "train" for item in train_episodes):
            raise ValueError("N177 T1 trainer cannot load query or holdout episodes")
        episode_ids = tuple(item.episode_id for item in train_episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("N177 T1 train split contains duplicate episode IDs")
        if len(train_episodes) != 640:
            raise ValueError("N177 T1 train split must contain exactly 640 episodes")
        if language_trainer.model.bundle_id == "":
            raise ValueError("N177 T1 V4 candidate requires an explicit bundle identity")

        self.language_trainer = language_trainer
        self.producer = producer
        self.base_adapter_checkpoint = copy.deepcopy(dict(base_adapter_checkpoint))
        self.mode = mode
        self.seed = int(seed)
        self.parent_v4_sha256 = parent_v4_sha256
        self.joint_bundle_id = joint_bundle_id or f"joint-n177-t1:{mode}:seed-{seed}"
        self.owner_id = owner_id
        self.base_adapter_sha256 = content_digest(dict(base_adapter_checkpoint))
        self.base_adapter_episode_id = native.cognitive_state.episode_id
        self.input_trace_cache = input_trace_cache
        self.policy = _mode_policy(mode)
        self.train_episodes = {item.episode_id: item for item in train_episodes}
        self.train_split_sha256 = joint_t1_split_digest(train_episodes)
        self.course_sha256 = str(joint_t1_course_contract()["course_sha256"])
        self.update_order = build_joint_t1_update_order(train_episodes)
        self.update_order_sha256 = joint_t1_update_order_sha256(self.update_order)
        if input_trace_cache is not None:
            input_trace_cache.validate_for(
                base_adapter_checkpoint=self.base_adapter_checkpoint,
                train_episodes=train_episodes,
            )
        self.update_cursor = int(update_cursor)
        self.completed_updates = self.update_cursor
        self.producer_learning_rate = float(producer_learning_rate)
        self.producer_optimizer = torch.optim.AdamW(
            producer.parameters(), lr=self.producer_learning_rate, weight_decay=0.0
        )
        if optimizer_state is not None:
            self.producer_optimizer.load_state_dict(copy.deepcopy(dict(optimizer_state)))
        self.last_receipt: dict[str, Any] | None = None
        self._validate_episode_capacity()

    @property
    def model(self):
        return self.language_trainer.model

    @property
    def cursor(self) -> dict[str, Any]:
        next_batch = (
            None
            if self.update_cursor >= JOINT_T1_UPDATE_COUNT
            else list(self.update_order[self.update_cursor])
        )
        return {
            "course_format": JOINT_T1_COURSE_FORMAT,
            "course_sha256": self.course_sha256,
            "train_split_sha256": self.train_split_sha256,
            "update_order_sha256": self.update_order_sha256,
            "input_trace_cache_sha256": (
                None if self.input_trace_cache is None else self.input_trace_cache.cache_sha256
            ),
            "update_cursor": self.update_cursor,
            "next_episode_ids": next_batch,
            "mode": self.mode,
            "seed": self.seed,
        }

    def _validate_episode_capacity(self) -> None:
        maximum = self.model.config.max_observation_bytes
        for episode in self.train_episodes.values():
            total = sum(len(bytes(turn.frame.payload)) for turn in episode.turns)
            if total > maximum:
                raise ValueError(
                    f"T1 episode {episode.episode_id} exceeds the V4 joint observation bound"
                )
            if any(
                len(turn.answer_target.target_bytes) > self.model.config.max_generation_bytes
                for turn in episode.turns
            ):
                raise ValueError(
                    f"T1 episode {episode.episode_id} exceeds the V4 answer-byte bound"
                )

    def _new_episode_state(self, episode_id: str) -> tuple[TSKV8Adapter | None, JointTaskStateV1]:
        if self.input_trace_cache is not None:
            joint = JointTaskStateV1.begin(
                self.model,
                self.input_trace_cache.get_initial_state(episode_id),
                joint_bundle_id=self.joint_bundle_id,
                owner_id=self.owner_id,
                scope_id=f"n177-t1:{self.mode}:seed-{self.seed}:{episode_id}",
                session_id=f"n177-t1:seed-{self.seed}:{episode_id}",
            )
            return None, joint
        adapter = TSKV8Adapter.from_native_checkpoint(copy.deepcopy(self.base_adapter_checkpoint))
        adapter.reset_dynamics(episode_id=episode_id)
        scope_id = f"n177-t1:{self.mode}:seed-{self.seed}:{episode_id}"
        session_id = f"n177-t1:seed-{self.seed}:{episode_id}"
        joint = JointTaskStateV1.begin(
            self.model,
            adapter.cognitive_snapshot(),
            joint_bundle_id=self.joint_bundle_id,
            owner_id=self.owner_id,
            scope_id=scope_id,
            session_id=session_id,
        )
        return adapter, joint

    def _append_turn(
        self,
        adapter: TSKV8Adapter | None,
        joint: JointTaskStateV1,
        turn: JointT1TurnV1,
    ) -> tuple[InputTrace, JointTaskStateV1]:
        if self.input_trace_cache is None:
            if adapter is None:
                raise RuntimeError("raw InputFrame replay requires a live TSK-v8 Adapter")
            trace = adapter.ingest_input(turn.frame, learn=False)
            cognitive_state = adapter.cognitive_snapshot()
        else:
            entry = self.input_trace_cache.get(turn.episode_id, turn.turn_index)
            trace = entry.trace
            cognitive_state = entry.cognitive_state
        if joint.input_bindings:
            previous_owner = joint.cognitive_state
            cognitive_state = _carry_t1_world_workspace(previous_owner, cognitive_state)
            incoming_objects = {item.object_id: item for item in cognitive_state.world.objects}
            incoming_events = {item.event_id for item in cognitive_state.world.events}
            if any(
                item.object_id not in incoming_objects
                or content_digest(incoming_objects[item.object_id].to_payload())
                != content_digest(item.to_payload())
                for item in previous_owner.world.objects
            ) or not {item.event_id for item in previous_owner.world.events}.issubset(
                incoming_events
            ):
                raise RuntimeError("cached input append discarded model-owned T1 world state")
        joint = joint.append_input(
            self.model,
            trace,
            cognitive_state,
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": self.model.config.max_observation_bytes,
            },
        )
        return trace, joint

    def _semantic_loss(
        self,
        joint: JointTaskStateV1,
        trace: InputTrace,
        turn: JointT1TurnV1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = bytes(item.value for item in trace.observations)
        target = turn.semantic_target
        if target.partition != "train":
            raise ValueError("query/holdout semantic targets cannot reach the fit API")
        if (
            target.input_id != trace.input_id
            or target.source_sha256 != hashlib.sha256(raw).hexdigest()
            or target.family_id != turn.family_id
        ):
            raise ValueError("semantic target differs from its exact train source/family")
        if self.mode == "parser_permuted" and turn.level == "F1":
            target = parser_permuted_target(target)
        return self.producer.supervised_loss(joint, trace, target)

    def _answer_loss(
        self,
        joint: JointTaskStateV1,
        turn: JointT1TurnV1,
    ) -> tuple[torch.Tensor, JointTaskStateV1, int]:
        source_binding = next(
            (
                item
                for item in reversed(joint.input_bindings)
                if item.input_id == turn.frame.input_id
            ),
            None,
        )
        if source_binding is None:
            raise ValueError("answer loss input is absent from the joint source ledger")
        raw = bytes(turn.frame.payload)
        turn.answer_target.validate_for_fit(
            input_id=turn.frame.input_id,
            input_bytes=raw,
            family_id=turn.family_id,
        )
        view = joint.content_view()
        selected = view.selected_candidates
        if selected and all(item.source_status == "source_linked" for item in selected):
            cache = self.model._joint_evidence_cache(joint)
        else:
            cache = self.model._joint_evidence_cache(
                joint, selected_source_ids=(source_binding.neural_source_id,)
            )
        state = joint.language_state
        target_bytes = turn.answer_target.target_bytes
        logits = []
        for symbol in target_bytes:
            current_logits, decoder_hidden = self.model.next_logits(state, cache=cache)
            logits.append(current_logits)
            state = self.model.observe_response(state, symbol, decoder_hidden)
        target = torch.tensor(list(target_bytes), dtype=torch.long, device=logits[0].device)
        loss = F.cross_entropy(torch.stack(logits), target, reduction="sum")
        updated_joint = joint.with_language_state(state).complete_response()
        return loss, updated_joint, len(target_bytes)

    def _run_episode(
        self,
        episode: JointT1EpisodeV1,
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[JointSemanticParseV1, ...]]:
        adapter, joint = self._new_episode_state(episode.episode_id)
        semantic_terms: list[torch.Tensor] = []
        answer_terms: list[torch.Tensor] = []
        answer_bytes = 0
        parses: list[JointSemanticParseV1] = []
        semantic_fit = self.mode in {"correct", "parser_permuted"}
        language_fit = self.policy["language_updates"]

        for turn in episode.turns:
            trace, joint = self._append_turn(adapter, joint, turn)
            parse = self.producer.predict(joint, trace)
            parses.append(parse)
            if semantic_fit:
                with torch.set_grad_enabled(semantic_fit):
                    semantic_loss, _ = self._semantic_loss(joint, trace, turn)
            else:
                with torch.no_grad():
                    semantic_loss, _ = self._semantic_loss(joint, trace, turn)
            semantic_terms.append(semantic_loss)
            if self.policy["cognitive_state_writes"]:
                joint = self.producer.commit_to_joint_state(joint, trace, parse, self.model)
            with torch.set_grad_enabled(language_fit):
                answer_loss, joint, turn_bytes = self._answer_loss(joint, turn)
            answer_terms.append(answer_loss)
            answer_bytes += turn_bytes

        parser_loss = torch.stack(semantic_terms).mean()
        language_loss = torch.stack(answer_terms).sum() / answer_bytes
        return parser_loss, language_loss, answer_bytes, tuple(parses)

    def _mask_frozen_language_gradients(self) -> None:
        scope = self.policy["language_parameter_scope"]
        if scope == "all":
            return
        for name, parameter in self.model.named_parameters():
            if scope == "none" or not name.startswith("output."):
                parameter.grad = None

    @staticmethod
    def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        active = [parameter for parameter in parameters if parameter.grad is not None]
        if not active:
            return 0.0
        norm = torch.nn.utils.clip_grad_norm_(active, 1.0, error_if_nonfinite=True)
        return float(norm.detach())

    def _verify_frozen_parameters(
        self,
        producer_before: Mapping[str, torch.Tensor],
        language_before: Mapping[str, torch.Tensor],
    ) -> None:
        if not self.policy["producer_updates"]:
            for name, value in self.producer.state_dict().items():
                if not torch.equal(value.detach().cpu(), producer_before[name]):
                    raise RuntimeError(f"frozen producer parameter changed: {name}")
        language_scope = self.policy["language_parameter_scope"]
        for name, value in self.model.state_dict().items():
            allowed = language_scope == "all" or (
                language_scope == "output_only" and name.startswith("output.")
            )
            if not allowed and not torch.equal(value.detach().cpu(), language_before[name]):
                raise RuntimeError(f"frozen V4 parameter changed: {name}")

    def update(self) -> dict[str, Any]:
        if self.input_trace_cache is None:
            raise ValueError(
                "N177 T1 optimizer updates require a verified train-only InputTrace cache"
            )
        if self.update_cursor >= JOINT_T1_UPDATE_COUNT:
            raise ValueError("N177 T1 fixed update package is already complete")
        batch_ids = self.update_order[self.update_cursor]
        if len(batch_ids) != JOINT_T1_BATCH_SIZE:
            raise RuntimeError("N177 T1 update batch differs from the frozen schedule")
        episodes = tuple(self.train_episodes[item] for item in batch_ids)
        verify_mask = self.update_cursor == 0
        producer_before = (
            {
                name: value.detach().cpu().clone()
                for name, value in self.producer.state_dict().items()
            }
            if verify_mask
            else {}
        )
        language_before = (
            {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            if verify_mask
            else {}
        )
        self.producer_optimizer.zero_grad(set_to_none=True)
        self.language_trainer.optimizer.zero_grad(set_to_none=True)
        parser_terms: list[torch.Tensor] = []
        language_terms: list[torch.Tensor] = []
        byte_count = 0
        for episode in episodes:
            parser_loss, answer_loss, episode_bytes, _ = self._run_episode(episode)
            parser_terms.append(parser_loss)
            language_terms.append(answer_loss * episode_bytes)
            byte_count += episode_bytes
        parser_loss = torch.stack(parser_terms).mean()
        language_loss = torch.stack(language_terms).sum() / byte_count

        producer_gradient_norm = 0.0
        language_gradient_norm = 0.0
        if self.policy["producer_updates"] or self.policy["language_updates"]:
            (parser_loss + language_loss).backward()
            self._mask_frozen_language_gradients()
            if not self.policy["producer_updates"]:
                for parameter in self.producer.parameters():
                    parameter.grad = None
            producer_gradient_norm = self._grad_norm(tuple(self.producer.parameters()))
            language_gradient_norm = self._grad_norm(tuple(self.model.parameters()))
            if self.policy["producer_updates"]:
                self.producer_optimizer.step()
                self.producer.parameter_revision += 1
            if self.policy["language_updates"]:
                self.language_trainer.optimizer.step()
                self.model.parameter_revision += 1
                self.language_trainer.steps += 1
                self.language_trainer.target_byte_exposures += byte_count
        if verify_mask:
            self._verify_frozen_parameters(producer_before, language_before)

        self.update_cursor += 1
        self.completed_updates = self.update_cursor
        self.last_receipt = {
            "update": self.update_cursor,
            "episode_ids": list(batch_ids),
            "parser_loss_nats": float(parser_loss.detach()),
            "answer_loss_nats_per_byte": float(language_loss.detach()),
            "answer_target_byte_exposures": byte_count,
            "producer_gradient_norm_before_clip": producer_gradient_norm,
            "language_gradient_norm_before_clip": language_gradient_norm,
            "producer_parameter_revision": self.producer.parameter_revision,
            "language_parameter_revision": self.model.parameter_revision,
            "mode": self.mode,
            "seed": self.seed,
        }
        return copy.deepcopy(self.last_receipt)

    def profile_backward_only(self, *, batch_index: int) -> dict[str, Any]:
        """Measure loss/gradient ownership without stepping either optimizer."""

        if self.input_trace_cache is None:
            raise ValueError("backward-only owner profile requires the verified train trace cache")
        if not 0 <= batch_index < len(self.update_order):
            raise ValueError("requested backward-only batch is outside the frozen schedule")
        batch_ids = self.update_order[batch_index]
        before = {
            "producer_parameters": content_digest(self.producer.state_dict()),
            "language_parameters": content_digest(self.model.state_dict()),
            "producer_optimizer": content_digest(self.producer_optimizer.state_dict()),
            "language_optimizer": content_digest(self.language_trainer.optimizer.state_dict()),
            "update_cursor": self.update_cursor,
            "producer_revision": self.producer.parameter_revision,
            "language_revision": self.model.parameter_revision,
        }
        self.producer_optimizer.zero_grad(set_to_none=True)
        self.language_trainer.optimizer.zero_grad(set_to_none=True)
        parser_terms: list[torch.Tensor] = []
        language_terms: list[torch.Tensor] = []
        answer_bytes = 0
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        for episode_id in batch_ids:
            parser_loss, answer_loss, exposure, _ = self._run_episode(
                self.train_episodes[episode_id]
            )
            parser_terms.append(parser_loss)
            language_terms.append(answer_loss * exposure)
            answer_bytes += exposure
        parser_loss = torch.stack(parser_terms).mean()
        language_loss = torch.stack(language_terms).sum() / answer_bytes
        if self.policy["producer_updates"] or self.policy["language_updates"]:
            (parser_loss + language_loss).backward()
            self._mask_frozen_language_gradients()
            if not self.policy["producer_updates"]:
                for parameter in self.producer.parameters():
                    parameter.grad = None

        producer_gradient_squares = 0.0
        producer_gradient_parameters = 0
        for parameter in self.producer.parameters():
            if parameter.grad is not None:
                producer_gradient_squares += float(parameter.grad.detach().square().sum())
                producer_gradient_parameters += 1
        language_gradient_squares: dict[str, float] = {}
        language_gradient_parameters: dict[str, int] = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                group = name.split(".", 1)[0]
                language_gradient_squares[group] = language_gradient_squares.get(
                    group, 0.0
                ) + float(parameter.grad.detach().square().sum())
                language_gradient_parameters[group] = language_gradient_parameters.get(group, 0) + 1

        self.producer_optimizer.zero_grad(set_to_none=True)
        self.language_trainer.optimizer.zero_grad(set_to_none=True)
        after = {
            "producer_parameters": content_digest(self.producer.state_dict()),
            "language_parameters": content_digest(self.model.state_dict()),
            "producer_optimizer": content_digest(self.producer_optimizer.state_dict()),
            "language_optimizer": content_digest(self.language_trainer.optimizer.state_dict()),
            "update_cursor": self.update_cursor,
            "producer_revision": self.producer.parameter_revision,
            "language_revision": self.model.parameter_revision,
        }
        if before != after:
            raise RuntimeError("backward-only profile changed model, optimizer or cursor state")

        producer_norm = producer_gradient_squares**0.5
        language_norms = {
            name: value**0.5 for name, value in sorted(language_gradient_squares.items())
        }
        if self.mode in {"correct", "parser_permuted"} and (
            producer_norm <= 0.0 or sum(language_norms.values()) <= 0.0
        ):
            raise RuntimeError("joint owner profile lacks a positive parser or V4 gradient")
        if (
            self.mode in {"no_update", "readout_only", "responsible_owner_frozen"}
            and producer_norm != 0.0
        ):
            raise RuntimeError("frozen semantic producer received a task gradient")
        if self.mode == "readout_only" and any(name != "output" for name in language_norms):
            raise RuntimeError("readout-only profile leaked gradient into a frozen V4 group")
        if self.mode == "readout_only" and language_norms.get("output", 0.0) <= 0.0:
            raise RuntimeError("readout-only profile lacks a positive V4 output gradient")
        if self.mode == "responsible_owner_frozen" and sum(language_norms.values()) <= 0.0:
            raise RuntimeError("owner-frozen profile lacks its permitted V4 task gradient")
        if self.mode == "no_update" and language_norms:
            raise RuntimeError("no-update profile produced an unexpected V4 gradient")
        return {
            "mode": self.mode,
            "seed": self.seed,
            "batch_index": batch_index,
            "episode_ids": list(batch_ids),
            "parser_loss_nats": float(parser_loss.detach()),
            "answer_loss_nats_per_byte": float(language_loss.detach()),
            "answer_target_byte_exposures": answer_bytes,
            "producer_gradient_norm": producer_norm,
            "producer_parameters_with_gradient": producer_gradient_parameters,
            "language_gradient_norms_by_group": language_norms,
            "language_parameters_with_gradient_by_group": language_gradient_parameters,
            "parameters_optimizers_and_cursor_unchanged": True,
            "optimizer_updates": 0,
            "process_cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.perf_counter() - wall_start,
        }

    def checkpoint(self) -> dict[str, Any]:
        """Build one NativeCheckpoint at an update boundary with all owners and cursors."""

        checkpoint_task_id = f"{self.joint_bundle_id}:cursor-{self.update_cursor}"
        adapter = TSKV8Adapter.from_native_checkpoint(copy.deepcopy(self.base_adapter_checkpoint))
        adapter.reset_dynamics(episode_id=checkpoint_task_id)
        scope_id = f"n177-t1:{self.mode}:seed-{self.seed}:checkpoint"
        session_id = f"n177-t1:seed-{self.seed}:checkpoint"
        joint = JointTaskStateV1.begin(
            self.model,
            adapter.cognitive_snapshot(),
            joint_bundle_id=self.joint_bundle_id,
            owner_id=self.owner_id,
            scope_id=scope_id,
            session_id=session_id,
        )
        cursor = self.cursor
        language_payload = self.language_trainer.checkpoint(joint.language_state, cursor=cursor)
        native_payload = joint.attach_to_native_checkpoint(
            adapter.native_checkpoint(), language_payload
        )
        native_payload = self.producer.attach_to_native_checkpoint(native_payload, joint, parses=())
        envelope = NativeCheckpoint.from_payload(native_payload)
        joint_component = envelope.components.get(JOINT_NATIVE_COMPONENT)
        if not isinstance(joint_component, Mapping) or not isinstance(
            joint_component.get("frame"), Mapping
        ):
            raise ValueError("joint T1 checkpoint lacks its native JointTaskState frame")
        run_component = _component_with_digest(
            {
                "format": JOINT_T1_TRAINER_COMPONENT_FORMAT,
                "version": 1,
                "mode": self.mode,
                "seed": self.seed,
                "owner_id": self.owner_id,
                "joint_bundle_id": self.joint_bundle_id,
                "joint_frame": copy.deepcopy(joint_component["frame"]),
                "base_adapter_sha256": self.base_adapter_sha256,
                "base_adapter_episode_id": self.base_adapter_episode_id,
                "input_trace_cache_sha256": (
                    None if self.input_trace_cache is None else self.input_trace_cache.cache_sha256
                ),
                "parent_v4_sha256": self.parent_v4_sha256,
                "course_sha256": self.course_sha256,
                "train_split_sha256": self.train_split_sha256,
                "update_order_sha256": self.update_order_sha256,
                "update_cursor": self.update_cursor,
                "completed_updates": self.completed_updates,
                "producer_learning_rate": self.producer_learning_rate,
                "producer_optimizer": copy.deepcopy(self.producer_optimizer.state_dict()),
                "policy": copy.deepcopy(self.policy),
                "policy_sha256": content_digest(self.policy),
                "torch_rng_state": torch.random.get_rng_state().clone(),
                "python_rng_state": random.getstate(),
                "last_receipt": copy.deepcopy(self.last_receipt),
            }
        )
        return NativeCheckpoint(
            kernel=envelope.kernel,
            cognitive_state=envelope.cognitive_state,
            components={
                **envelope.components,
                JOINT_T1_TRAINER_COMPONENT_KEY: run_component,
            },
            adapter=envelope.adapter,
        ).to_payload()

    def save_checkpoint(self, path: Path | str) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("N177 T1 checkpoint paths are write-once")
        return atomic_save(self.checkpoint(), target)

    @classmethod
    def from_native_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        input_trace_cache: JointT1TraceCacheV1 | None = None,
    ) -> JointT1TrainerV1:
        envelope = NativeCheckpoint.from_payload(payload)
        component = envelope.components.get(JOINT_T1_TRAINER_COMPONENT_KEY)
        required = {
            "format",
            "version",
            "mode",
            "seed",
            "owner_id",
            "joint_bundle_id",
            "joint_frame",
            "base_adapter_sha256",
            "base_adapter_episode_id",
            "input_trace_cache_sha256",
            "parent_v4_sha256",
            "course_sha256",
            "train_split_sha256",
            "update_order_sha256",
            "update_cursor",
            "completed_updates",
            "producer_learning_rate",
            "producer_optimizer",
            "policy",
            "policy_sha256",
            "torch_rng_state",
            "python_rng_state",
            "last_receipt",
            "component_digest",
        }
        if not isinstance(component, Mapping) or set(component) != required:
            raise ValueError("native checkpoint lacks the joint T1 trainer component")
        unsigned = {key: value for key, value in component.items() if key != "component_digest"}
        if component["component_digest"] != content_digest(unsigned):
            raise ValueError("joint T1 trainer component digest differs")
        if (component["format"], component["version"]) != (
            JOINT_T1_TRAINER_COMPONENT_FORMAT,
            1,
        ):
            raise ValueError("unsupported joint T1 trainer component version")
        mode = str(component["mode"])
        seed = int(component["seed"])
        policy = _mode_policy(mode)
        if (
            content_digest(policy) != component["policy_sha256"]
            or dict(component["policy"]) != policy
        ):
            raise ValueError("restored N177 T1 credit policy differs")
        joint_frame = component.get("joint_frame")
        if not isinstance(joint_frame, Mapping):
            raise ValueError("joint T1 trainer component lacks a bound JointTaskState frame")
        native_joint_component = envelope.components.get(JOINT_NATIVE_COMPONENT)
        if (
            not isinstance(native_joint_component, Mapping)
            or not isinstance(native_joint_component.get("frame"), Mapping)
            or dict(native_joint_component["frame"]) != dict(joint_frame)
        ):
            raise ValueError("trainer and native JointTaskState frame identities differ")
        joint, language_trainer = JointTaskStateV1.restore_native_checkpoint(
            payload,
            joint_bundle_id=str(component["joint_bundle_id"]),
            owner_id=str(component["owner_id"]),
            scope_id=str(joint_frame["scope_id"]),
            session_id=str(joint_frame["session_id"]),
        )
        producer, parse_history = JointInputProducerV1.restore_from_native_checkpoint(
            payload, joint
        )
        if parse_history:
            raise ValueError("N177 T1 update-boundary checkpoint must not retain stale parses")
        adapter = TSKV8Adapter.from_native_checkpoint(dict(payload))
        base_adapter_episode_id = str(component["base_adapter_episode_id"])
        adapter.reset_dynamics(episode_id=base_adapter_episode_id)
        base_adapter_checkpoint = adapter.native_checkpoint()
        if content_digest(base_adapter_checkpoint) != component["base_adapter_sha256"]:
            raise ValueError("restored base Adapter checkpoint differs from the N177 T1 source")
        stored_cache_sha = component["input_trace_cache_sha256"]
        if stored_cache_sha is not None:
            if input_trace_cache is None:
                raise ValueError(
                    "resuming an N177 T1 update requires its verified train trace cache"
                )
            if input_trace_cache.cache_sha256 != stored_cache_sha:
                raise ValueError("restored N177 T1 input trace cache digest differs")
        train_episodes = build_joint_t1_split("train")
        trainer = cls(
            language_trainer=language_trainer,
            producer=producer,
            base_adapter_checkpoint=base_adapter_checkpoint,
            train_episodes=train_episodes,
            mode=mode,
            seed=seed,
            parent_v4_sha256=str(component["parent_v4_sha256"]),
            joint_bundle_id=str(component["joint_bundle_id"]),
            owner_id=str(component["owner_id"]),
            producer_learning_rate=float(component["producer_learning_rate"]),
            update_cursor=int(component["update_cursor"]),
            optimizer_state=component["producer_optimizer"],
            input_trace_cache=input_trace_cache,
        )
        trainer.base_adapter_episode_id = base_adapter_episode_id
        if trainer.course_sha256 != component["course_sha256"]:
            raise ValueError("restored N177 T1 course contract differs")
        if trainer.train_split_sha256 != component["train_split_sha256"]:
            raise ValueError("restored N177 T1 train split differs")
        if trainer.update_order_sha256 != component["update_order_sha256"]:
            raise ValueError("restored N177 T1 episode order differs")
        if int(component["completed_updates"]) != trainer.update_cursor:
            raise ValueError("restored N177 T1 completed-update count differs from cursor")
        trainer.last_receipt = copy.deepcopy(component["last_receipt"])
        torch.random.set_rng_state(component["torch_rng_state"])
        random.setstate(component["python_rng_state"])
        return trainer
