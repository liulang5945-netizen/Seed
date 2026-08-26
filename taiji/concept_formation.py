"""Taiji-owned cross-experience concept formation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .contracts import (
    Concept,
    ConceptSequenceTrace,
    EpisodicMemoryRecord,
    Outcome,
    WorldState,
    WorldTransition,
)

CONCEPT_FORMATION_CHECKPOINT_FORMAT = "taiji-concept-formation-v1"


@dataclass(frozen=True)
class ConceptMatch:
    """A content-addressed concept match exposed to downstream organs."""

    concept: Concept
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("concept match score must be in [0, 1]")


class ConceptFormationOrgan:
    """Form provisional concepts from latent, world and outcome evidence.

    The organ owns concept identity and support-set growth.  It receives only
    experienced episodic records and has no fixed label or domain fact table.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        signal_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
        capacity: int = 256,
        plasticity_rate: float = 0.25,
        prune_threshold: float = 0.15,
        credit_discount: float = 0.90,
        trace_capacity: int = 32,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.signal_weights = tuple(float(weight) for weight in signal_weights)
        self.capacity = int(capacity)
        self.plasticity_rate = float(plasticity_rate)
        self.prune_threshold = float(prune_threshold)
        self.credit_discount = float(credit_discount)
        self.trace_capacity = int(trace_capacity)
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("concept similarity threshold must be in (0, 1]")
        if (
            len(self.signal_weights) != 3
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in self.signal_weights)
            or abs(sum(self.signal_weights) - 1.0) > 1e-6
        ):
            raise ValueError("concept signal weights must be three positive weights summing to 1")
        if self.capacity <= 0:
            raise ValueError("concept formation capacity must be positive")
        if not 0.0 < self.plasticity_rate <= 1.0:
            raise ValueError("concept formation plasticity rate must be in (0, 1]")
        if not 0.0 <= self.prune_threshold <= 1.0:
            raise ValueError("concept formation prune threshold must be in [0, 1]")
        if not 0.0 <= self.credit_discount <= 1.0:
            raise ValueError("concept formation credit_discount must be in [0, 1]")
        if self.trace_capacity <= 0:
            raise ValueError("concept sequence trace capacity must be positive")
        self._concepts: tuple[Concept, ...] = ()

    @property
    def concepts(self) -> tuple[Concept, ...]:
        return self._concepts

    @staticmethod
    def _concept_strength(concept: Concept) -> float:
        return (
            float(concept.confidence)
            + float(concept.stability)
            + float(concept.maturity)
        ) / 3.0

    def _prune_to_capacity(self, concepts: dict[str, Concept]) -> None:
        if len(concepts) <= self.capacity:
            return
        ranked = sorted(
            concepts.values(),
            key=lambda item: (
                self._concept_strength(item),
                item.update_count,
                item.last_updated_tick,
                item.concept_id,
            ),
        )
        for concept in ranked[: len(concepts) - self.capacity]:
            concepts.pop(concept.concept_id, None)

    def lesion(self, concept_ids: Iterable[str]) -> tuple[str, ...]:
        """Remove selected concepts and return the IDs that were present."""

        requested = tuple(dict.fromkeys(str(concept_id) for concept_id in concept_ids))
        existing = {concept.concept_id for concept in self._concepts}
        removed = tuple(concept_id for concept_id in requested if concept_id in existing)
        if removed:
            removed_set = set(removed)
            self._concepts = tuple(
                concept for concept in self._concepts if concept.concept_id not in removed_set
            )
        return removed

    def lesion_sequence_traces(self, concept_ids: Iterable[str]) -> tuple[str, ...]:
        """Disable learned sequence traces while retaining concept identity."""

        requested = tuple(dict.fromkeys(str(concept_id) for concept_id in concept_ids))
        requested_set = set(requested)
        updated: list[Concept] = []
        removed: list[str] = []
        for concept in self._concepts:
            if concept.concept_id not in requested_set:
                updated.append(concept)
                continue
            if concept.sequence_traces or not concept.sequence_traces_lesioned:
                removed.append(concept.concept_id)
            updated.append(
                replace(
                    concept,
                    sequence_traces=(),
                    sequence_traces_lesioned=True,
                    update_count=concept.update_count + 1,
                )
            )
        self._concepts = tuple(updated)
        return tuple(removed)

    def lesion_sequence_trace(
        self, concept_id: str, trace_ids: Iterable[str]
    ) -> tuple[str, ...]:
        """Remove selected sequence branches while retaining the other traces."""

        requested = tuple(dict.fromkeys(str(trace_id) for trace_id in trace_ids))
        if not requested:
            return ()
        target_index = next(
            (index for index, concept in enumerate(self._concepts) if concept.concept_id == concept_id),
            None,
        )
        if target_index is None:
            return ()
        concept = self._concepts[target_index]
        requested_set = set(requested)
        removed = tuple(trace.trace_id for trace in concept.sequence_traces if trace.trace_id in requested_set)
        if not removed:
            return ()
        remaining = tuple(
            trace for trace in concept.sequence_traces if trace.trace_id not in requested_set
        )
        updated = replace(
            concept,
            sequence_traces=remaining,
            sequence_traces_lesioned=not remaining,
            update_count=concept.update_count + 1,
        )
        concepts = list(self._concepts)
        concepts[target_index] = updated
        self._concepts = tuple(concepts)
        return removed

    def grow_sequence_trace(
        self,
        concept_id: str,
        transitions: Sequence[tuple[WorldTransition, float]],
    ) -> str | None:
        """Birth one novel online branch from a contiguous real transition chain."""

        items = tuple(transitions)
        if not items:
            return None
        target_index = next(
            (index for index, concept in enumerate(self._concepts) if concept.concept_id == concept_id),
            None,
        )
        if target_index is None:
            return None
        previous_after: WorldState | None = None
        for transition, prediction_error in items:
            if not isinstance(transition, WorldTransition):
                raise TypeError("online sequence branch transitions must be WorldTransition values")
            if not math.isfinite(float(prediction_error)) or float(prediction_error) < 0.0:
                raise ValueError("online sequence branch prediction_error must be finite and non-negative")
            if previous_after is not None and transition.before.tick != previous_after.tick:
                raise ValueError("online sequence branch transitions must be contiguous")
            previous_after = transition.after
        action_kinds = tuple(transition.action.kind for transition, _ in items)
        concept = self._concepts[target_index]
        if any(
            trace.action_kinds == action_kinds
            and self._state_similarity(trace.before_prototype, items[0][0].before.latent)
            >= self.similarity_threshold
            for trace in concept.sequence_traces
        ):
            return None
        trace = self._trace_from_transitions(items)
        updated_traces = self._prune_sequence_traces((*concept.sequence_traces, trace))
        updated = replace(
            concept,
            action_kinds=tuple(dict.fromkeys((*concept.action_kinds, *action_kinds))),
            action_sequences=tuple(dict.fromkeys((*concept.action_sequences, action_kinds))),
            sequence_traces=updated_traces,
            sequence_traces_lesioned=False,
            update_count=concept.update_count + 1,
            last_updated_tick=items[-1][0].after.tick,
        )
        concepts = list(self._concepts)
        concepts[target_index] = updated
        self._concepts = tuple(concepts)
        return (
            trace.trace_id
            if trace.trace_id in {item.trace_id for item in updated_traces}
            else None
        )

    @staticmethod
    def _world_ids_similarity(
        left_objects: Sequence[str],
        left_relations: Sequence[str],
        right_objects: Sequence[str],
        right_relations: Sequence[str],
    ) -> float:
        if not left_objects and not left_relations:
            return 0.0
        if not left_objects and not right_objects:
            object_similarity = 1.0
        elif not left_objects or not right_objects:
            object_similarity = 0.0
        else:
            count_similarity = 1.0 - abs(len(left_objects) - len(right_objects)) / max(
                len(left_objects), len(right_objects)
            )
            object_similarity = max(
                ConceptFormationOrgan._jaccard(left_objects, right_objects),
                0.5 * count_similarity,
            )
        relation_similarity = max(
            ConceptFormationOrgan._jaccard(left_relations, right_relations),
            ConceptFormationOrgan._jaccard(
                ConceptFormationOrgan._relation_shapes(left_relations),
                ConceptFormationOrgan._relation_shapes(right_relations),
            ),
        )
        return (
            max(object_similarity, relation_similarity)
            if left_relations
            else object_similarity
        )

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        object_ids: Sequence[str] = (),
        relation_ids: Sequence[str] = (),
        limit: int = 1,
    ) -> tuple[ConceptMatch, ...]:
        """Retrieve concepts by current latent and optional world evidence."""

        if cue.ndim != 1:
            raise ValueError("concept retrieval cue must be a vector")
        if int(limit) <= 0:
            return ()
        matches: list[ConceptMatch] = []
        has_world_evidence = bool(object_ids or relation_ids)
        latent_weight, world_weight, _ = self.signal_weights
        normalizer = latent_weight + (world_weight if has_world_evidence else 0.0)
        for concept in self._concepts:
            if concept.prototype.numel() != cue.numel():
                continue
            latent_similarity = float(
                torch.nn.functional.cosine_similarity(
                    concept.prototype.unsqueeze(0), cue.unsqueeze(0)
                ).item()
            )
            score = latent_weight * max(0.0, latent_similarity)
            if has_world_evidence:
                score += world_weight * self._world_ids_similarity(
                    object_ids,
                    relation_ids,
                    concept.object_ids,
                    concept.relation_ids,
                )
            score = max(0.0, min(1.0, score / normalizer))
            if score >= self.similarity_threshold:
                matches.append(ConceptMatch(concept=concept, score=score))
        matches.sort(
            key=lambda item: (
                -item.score,
                -self._concept_strength(item.concept),
                item.concept.concept_id,
            )
        )
        return tuple(matches[: int(limit)])

    @staticmethod
    def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        return 1.0 if not union else len(left_set & right_set) / len(union)

    @staticmethod
    def _relation_shapes(relation_ids: Sequence[str]) -> tuple[str, ...]:
        shapes = []
        for relation_id in relation_ids:
            parts = str(relation_id).split(":", 2)
            shapes.append(parts[1] if len(parts) == 3 else str(relation_id))
        return tuple(dict.fromkeys(shapes))

    @staticmethod
    def _sequence_similarity(
        left: Sequence[str], right: Sequence[str]
    ) -> float:
        if not left or not right:
            return 0.0
        matches = sum(left_item == right_item for left_item, right_item in zip(left, right))
        return matches / max(len(left), len(right))

    def action_sequence_affinity(
        self, concept: Concept, action_kinds: Sequence[str]
    ) -> float:
        """Return learned ordered-action affinity for one candidate rollout."""

        query = tuple(str(item) for item in action_kinds)
        if not query:
            return 0.0
        if concept.sequence_traces:
            return max(
                self._sequence_similarity(query, trace.action_kinds)
                * sum(trace.step_credit) / len(trace.step_credit)
                * (1.0 - sum(trace.prediction_errors) / len(trace.prediction_errors))
                for trace in concept.sequence_traces
            )
        sequences = concept.action_sequences or tuple((kind,) for kind in concept.action_kinds)
        return max(
            (self._sequence_similarity(query, sequence) for sequence in sequences),
            default=0.0,
        )

    @staticmethod
    def _state_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
            return 0.0
        left_norm = float(torch.linalg.vector_norm(left))
        right_norm = float(torch.linalg.vector_norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        similarity = float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())
        return max(0.0, min(1.0, similarity))

    def suffix_sequence_affinity(
        self,
        concept: Concept,
        action_kinds: Sequence[str],
        *,
        current_state: WorldState | None = None,
    ) -> float:
        """Retrieve an ordered suffix using the state at its current boundary."""

        query = tuple(str(item) for item in action_kinds)
        if not query:
            return 0.0
        if not concept.sequence_traces:
            if concept.sequence_traces_lesioned:
                return 0.0
            return self.action_sequence_affinity(concept, query)
        current_latent = None if current_state is None else current_state.latent
        candidates: list[float] = []
        for trace in concept.sequence_traces:
            if len(query) > len(trace.action_kinds):
                continue
            for start in range(len(trace.action_kinds) - len(query) + 1):
                action_similarity = self._sequence_similarity(
                    query, trace.action_kinds[start : start + len(query)]
                )
                expected_state = (
                    trace.before_prototype
                    if start == 0
                    else trace.after_prototypes[start - 1]
                )
                state_similarity = (
                    1.0
                    if current_latent is None
                    else self._state_similarity(current_latent, expected_state)
                )
                credits = trace.step_credit[start : start + len(query)]
                errors = trace.prediction_errors[start : start + len(query)]
                quality = sum(credits) / len(credits) * (1.0 - sum(errors) / len(errors))
                candidates.append(action_similarity * state_similarity * quality)
        return max(candidates, default=0.0)

    @staticmethod
    def _action_sequences(
        records: Sequence[EpisodicMemoryRecord],
    ) -> tuple[tuple[str, ...], ...]:
        grouped: dict[str, list[tuple[int, str, str]]] = {}
        for record in records:
            if record.action_intent is None:
                continue
            grouped.setdefault(record.episode_id, []).append(
                (record.tick, record.memory_id, record.action_intent.kind)
            )
        sequences: list[tuple[str, ...]] = []
        for episode_id in sorted(grouped):
            sequence = tuple(
                action_kind
                for _, _, action_kind in sorted(grouped[episode_id], key=lambda item: item[:2])
            )
            if sequence and sequence not in sequences:
                sequences.append(sequence)
        return tuple(sequences)

    @staticmethod
    def _outcome_value(outcome: Outcome) -> float:
        bounded_reward = 0.5 * (1.0 + math.tanh(float(outcome.reward)))
        return 0.5 * float(bool(outcome.success)) + 0.5 * bounded_reward

    def _trace_from_transitions(
        self, transitions: Sequence[tuple[WorldTransition, float]]
    ) -> ConceptSequenceTrace:
        first_transition = transitions[0][0]
        action_kinds = tuple(transition.action.kind for transition, _ in transitions)
        before_prototype = first_transition.before.latent
        after_prototypes = tuple(transition.after.latent for transition, _ in transitions)
        if any(
            prototype.numel() != before_prototype.numel()
            for prototype in after_prototypes
        ):
            raise ValueError("online sequence branch states must share one latent dimension")
        prediction_errors = tuple(
            max(0.0, min(1.0, float(prediction_error)))
            for _, prediction_error in transitions
        )
        outcome_scores = tuple(
            self._outcome_value(transition.outcome) for transition, _ in transitions
        )
        step_quality = tuple(
            outcome * (1.0 - error)
            for outcome, error in zip(outcome_scores, prediction_errors, strict=True)
        )
        credits = [0.0] * len(step_quality)
        running = 0.0
        for index in range(len(step_quality) - 1, -1, -1):
            running = step_quality[index] + self.credit_discount * running
            normalizer = sum(
                self.credit_discount**offset for offset in range(len(step_quality) - index)
            )
            credits[index] = running / normalizer if normalizer else 0.0
        return ConceptSequenceTrace(
            action_kinds=action_kinds,
            before_prototype=before_prototype,
            after_prototypes=after_prototypes,
            step_credit=tuple(credits),
            prediction_errors=prediction_errors,
            outcome_mean=sum(outcome_scores) / len(outcome_scores),
        )

    def _sequence_traces(
        self, records: Sequence[EpisodicMemoryRecord]
    ) -> tuple[ConceptSequenceTrace, ...]:
        grouped: dict[str, list[EpisodicMemoryRecord]] = {}
        for record in records:
            if record.action_intent is None or record.world_transition is None:
                continue
            grouped.setdefault(record.episode_id, []).append(record)
        traces: list[ConceptSequenceTrace] = []
        for episode_id in sorted(grouped):
            ordered = sorted(grouped[episode_id], key=lambda item: (item.tick, item.memory_id))
            if not ordered or any(
                item.action_intent is None or item.world_transition is None for item in ordered
            ):
                continue
            transitions = tuple(item.world_transition for item in ordered)
            if any(transition is None for transition in transitions):
                continue
            transitions = tuple(transition for transition in transitions if transition is not None)
            if not transitions:
                continue
            actions = tuple(item.action_intent.kind for item in ordered if item.action_intent is not None)
            if len(actions) != len(transitions):
                continue
            before_prototype = transitions[0].before.latent
            after_prototypes = tuple(transition.after.latent for transition in transitions)
            if any(
                prototype.numel() != before_prototype.numel()
                for prototype in (*after_prototypes,)
            ):
                continue
            prediction_errors = tuple(
                max(0.0, min(1.0, float(item.prediction_error))) for item in ordered
            )
            outcome_scores = tuple(self._outcome_score(item) for item in ordered)
            step_quality = tuple(
                outcome * (1.0 - error)
                for outcome, error in zip(outcome_scores, prediction_errors, strict=True)
            )
            credits = [0.0] * len(step_quality)
            running = 0.0
            for index in range(len(step_quality) - 1, -1, -1):
                running = step_quality[index] + self.credit_discount * running
                normalizer = sum(
                    self.credit_discount**offset
                    for offset in range(len(step_quality) - index)
                )
                credits[index] = running / normalizer if normalizer else 0.0
            traces.append(
                ConceptSequenceTrace(
                    action_kinds=actions,
                    before_prototype=before_prototype,
                    after_prototypes=after_prototypes,
                    step_credit=tuple(credits),
                    prediction_errors=prediction_errors,
                    outcome_mean=sum(outcome_scores) / len(outcome_scores),
                )
            )
        return tuple(traces)

    def _merge_sequence_traces(
        self,
        existing: Sequence[ConceptSequenceTrace],
        incoming: Sequence[ConceptSequenceTrace],
    ) -> tuple[ConceptSequenceTrace, ...]:
        merged = list(existing)
        for candidate in incoming:
            match_index = next(
                (
                    index
                    for index, trace in enumerate(merged)
                    if trace.action_kinds == candidate.action_kinds
                    and self._state_similarity(trace.before_prototype, candidate.before_prototype)
                    >= self.similarity_threshold
                ),
                None,
            )
            if match_index is None:
                merged.append(candidate)
                continue
            previous = merged[match_index]
            previous_weight = float(previous.visits)
            candidate_weight = float(candidate.visits)
            total_weight = previous_weight + candidate_weight
            after_prototypes = tuple(
                (previous_weight * left + candidate_weight * right) / total_weight
                for left, right in zip(
                    previous.after_prototypes, candidate.after_prototypes, strict=True
                )
            )
            merged[match_index] = replace(
                previous,
                before_prototype=(
                    previous_weight * previous.before_prototype
                    + candidate_weight * candidate.before_prototype
                )
                / total_weight,
                after_prototypes=after_prototypes,
                step_credit=tuple(
                    (previous_weight * left + candidate_weight * right) / total_weight
                    for left, right in zip(previous.step_credit, candidate.step_credit, strict=True)
                ),
                prediction_errors=tuple(
                    (previous_weight * left + candidate_weight * right) / total_weight
                    for left, right in zip(
                        previous.prediction_errors, candidate.prediction_errors, strict=True
                    )
                ),
                outcome_mean=(
                    previous_weight * previous.outcome_mean
                    + candidate_weight * candidate.outcome_mean
                )
                / total_weight,
                visits=previous.visits + candidate.visits,
            )
        return self._prune_sequence_traces(tuple(merged))

    @staticmethod
    def _sequence_trace_strength(trace: ConceptSequenceTrace) -> float:
        if not trace.step_credit:
            return 0.0
        credit = sum(trace.step_credit) / len(trace.step_credit)
        error = sum(trace.prediction_errors) / len(trace.prediction_errors)
        return credit * (1.0 - error) * trace.outcome_mean

    def _prune_sequence_traces(
        self, traces: Sequence[ConceptSequenceTrace]
    ) -> tuple[ConceptSequenceTrace, ...]:
        if len(traces) <= self.trace_capacity:
            return tuple(traces)
        ranked = sorted(
            traces,
            key=lambda trace: (
                self._sequence_trace_strength(trace),
                trace.visits,
                trace.trace_id,
            ),
        )
        keep = {trace.trace_id for trace in ranked[-self.trace_capacity :]}
        return tuple(trace for trace in traces if trace.trace_id in keep)

    def update_sequence_trace(
        self,
        action_kind: str,
        *,
        before_state: WorldState,
        after_state: WorldState,
        outcome: Outcome,
        prediction_error: float = 0.0,
        learning_rate: float | None = None,
    ) -> int:
        """Apply one experienced transition to the matching sequence traces."""

        action_kind = str(action_kind)
        if not action_kind:
            raise ValueError("sequence trace action_kind cannot be empty")
        if not math.isfinite(float(prediction_error)) or float(prediction_error) < 0.0:
            raise ValueError("sequence trace prediction_error must be finite and non-negative")
        rate = self.plasticity_rate if learning_rate is None else float(learning_rate)
        if not 0.0 < rate <= 1.0:
            raise ValueError("sequence trace learning_rate must be in (0, 1]")
        if after_state.tick <= before_state.tick:
            raise ValueError("sequence trace update must advance the world tick")
        bounded_error = max(0.0, min(1.0, float(prediction_error)))
        bounded_reward = 0.5 * (1.0 + math.tanh(float(outcome.reward)))
        outcome_score = 0.5 * float(bool(outcome.success)) + 0.5 * bounded_reward
        quality = outcome_score * (1.0 - bounded_error)
        candidates: list[tuple[float, int, int, int]] = []
        for concept_index, concept in enumerate(self._concepts):
            for trace_index, trace in enumerate(concept.sequence_traces):
                for start, kind in enumerate(trace.action_kinds):
                    if kind != action_kind:
                        continue
                    expected_state = (
                        trace.before_prototype
                        if start == 0
                        else trace.after_prototypes[start - 1]
                    )
                    candidates.append(
                        (
                            self._state_similarity(before_state.latent, expected_state),
                            concept_index,
                            trace_index,
                            start,
                        )
                    )
        if not candidates:
            return 0
        best_state_score = max(item[0] for item in candidates)
        if best_state_score <= 0.0:
            return 0
        concepts = list(self._concepts)
        updated: set[tuple[int, int]] = set()
        before_latent = before_state.latent.detach()
        after_latent = after_state.latent.detach()
        for state_score, concept_index, trace_index, start in candidates:
            if state_score < best_state_score:
                continue
            key = (concept_index, trace_index)
            if key in updated:
                continue
            concept = concepts[concept_index]
            trace = concept.sequence_traces[trace_index]
            before_prototype = (
                (1.0 - rate) * trace.before_prototype + rate * before_latent.to(trace.before_prototype)
                if start == 0
                else trace.before_prototype
            )
            after_prototypes = list(trace.after_prototypes)
            after_prototypes[start] = (
                (1.0 - rate) * after_prototypes[start]
                + rate * after_latent.to(after_prototypes[start])
            )
            prediction_errors = list(trace.prediction_errors)
            prediction_errors[start] = (
                (1.0 - rate) * prediction_errors[start] + rate * bounded_error
            )
            step_credit = list(trace.step_credit)
            for index in range(start + 1):
                target = quality * self.credit_discount ** (start - index)
                step_credit[index] = (
                    (1.0 - rate) * step_credit[index] + rate * target
                )
            traces = list(concept.sequence_traces)
            traces[trace_index] = replace(
                trace,
                before_prototype=before_prototype,
                after_prototypes=tuple(after_prototypes),
                step_credit=tuple(step_credit),
                prediction_errors=tuple(prediction_errors),
                visits=trace.visits + 1,
            )
            concepts[concept_index] = replace(
                concept,
                sequence_traces=tuple(traces),
                outcome_mean=(1.0 - rate) * concept.outcome_mean + rate * outcome_score,
                update_count=concept.update_count + 1,
                last_updated_tick=after_state.tick,
            )
            updated.add(key)
        self._concepts = tuple(concepts)
        return len(updated)

    @staticmethod
    def _outcome_score(record: EpisodicMemoryRecord) -> float:
        if record.outcome is None:
            return 0.0
        return ConceptFormationOrgan._outcome_value(record.outcome)

    def _world_signal_similarity(
        self, left: EpisodicMemoryRecord, right: EpisodicMemoryRecord
    ) -> float:
        if not left.object_ids and not right.object_ids:
            object_similarity = 1.0
        elif not left.object_ids or not right.object_ids:
            object_similarity = 0.0
        else:
            count_similarity = 1.0 - abs(len(left.object_ids) - len(right.object_ids)) / max(
                len(left.object_ids), len(right.object_ids)
            )
            object_similarity = max(
                self._jaccard(left.object_ids, right.object_ids),
                0.5 * count_similarity,
            )

        if not left.relation_ids and not right.relation_ids:
            relation_similarity = 1.0
        elif not left.relation_ids or not right.relation_ids:
            relation_similarity = 0.0
        else:
            relation_similarity = max(
                self._jaccard(left.relation_ids, right.relation_ids),
                self._jaccard(
                    self._relation_shapes(left.relation_ids),
                    self._relation_shapes(right.relation_ids),
                ),
            )
        return 0.5 * (object_similarity + relation_similarity)

    def _similarity(self, left: EpisodicMemoryRecord, right: EpisodicMemoryRecord) -> float:
        latent_similarity = float(
            torch.nn.functional.cosine_similarity(
                left.cue.unsqueeze(0), right.cue.unsqueeze(0)
            ).item()
        )
        world_similarity = self._world_signal_similarity(left, right)
        outcome_similarity = 1.0 - abs(self._outcome_score(left) - self._outcome_score(right))
        latent_weight, world_weight, outcome_weight = self.signal_weights
        return max(
            0.0,
            min(
                1.0,
                latent_weight * latent_similarity
                + world_weight * world_similarity
                + outcome_weight * outcome_similarity,
            ),
        )

    def consolidate(
        self,
        source: Iterable[EpisodicMemoryRecord],
        *,
        tick: int,
    ) -> tuple[Concept, ...]:
        """Update the concept registry from real episodic records."""

        records = tuple(
            record
            for record in source
            if record.outcome is not None and record.event_ids
        )
        if not records:
            return self._concepts
        cue_dim = records[0].cue.numel()
        if any(record.cue.numel() != cue_dim for record in records):
            raise ValueError("concept formation records must share one cue dimension")

        clusters: list[list[EpisodicMemoryRecord]] = []
        for record in records:
            destination: list[EpisodicMemoryRecord] | None = None
            for cluster in clusters:
                similarity = sum(self._similarity(record, item) for item in cluster) / len(cluster)
                if similarity >= self.similarity_threshold:
                    destination = cluster
                    break
            if destination is None:
                clusters.append([record])
            else:
                destination.append(record)

        concepts = {item.concept_id: item for item in self._concepts}
        for cluster in clusters:
            episode_ids = {item.episode_id for item in cluster}
            if len(episode_ids) < 2:
                continue
            event_ids = tuple(dict.fromkeys(event_id for item in cluster for event_id in item.event_ids))
            assembly_ids = tuple(
                dict.fromkeys(assembly_id for item in cluster for assembly_id in item.assembly_ids)
            )
            object_ids = tuple(dict.fromkeys(object_id for item in cluster for object_id in item.object_ids))
            relation_ids = tuple(
                dict.fromkeys(relation_id for item in cluster for relation_id in item.relation_ids)
            )
            action_kinds = tuple(
                dict.fromkeys(
                    item.action_intent.kind
                    for item in cluster
                    if item.action_intent is not None
                )
            )
            action_sequences = self._action_sequences(cluster)
            sequence_traces = self._merge_sequence_traces((), self._sequence_traces(cluster))
            prototype = torch.nn.functional.normalize(
                torch.stack([item.cue for item in cluster]).mean(dim=0), dim=0
            )
            pairwise_scores = [
                self._similarity(left, right)
                for index, left in enumerate(cluster)
                for right in cluster[index + 1 :]
            ]
            stability = sum(pairwise_scores) / len(pairwise_scores)
            outcome_scores = tuple(self._outcome_score(item) for item in cluster)
            outcome_mean = sum(outcome_scores) / len(outcome_scores)
            outcome_consistency = 1.0 - sum(
                abs(score - outcome_mean) for score in outcome_scores
            ) / len(outcome_scores)
            previous_concept = next(
                (
                    item
                    for item in concepts.values()
                    if item.prototype.numel() == prototype.numel()
                    and float(
                        torch.nn.functional.cosine_similarity(
                            item.prototype.unsqueeze(0), prototype.unsqueeze(0)
                        ).item()
                    )
                    >= self.similarity_threshold
                    and self._jaccard(
                        self._relation_shapes(item.relation_ids),
                        self._relation_shapes(relation_ids),
                    )
                    >= self.similarity_threshold
                    and 1.0 - abs(item.outcome_mean - outcome_mean)
                    >= self.similarity_threshold
                ),
                None,
            )
            if previous_concept is None:
                signature = "|".join(
                    (
                        *sorted(self._relation_shapes(relation_ids)),
                        str(len(object_ids)),
                        f"{outcome_mean:.3f}",
                        *(f"sequence:{'->'.join(sequence)}" for sequence in action_sequences),
                        *(f"{value:.4f}" for value in prototype.tolist()),
                    )
                )
                concept_id = f"concept:{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]}"
            else:
                concept_id = previous_concept.concept_id
                event_ids = tuple(dict.fromkeys((*previous_concept.support_event_ids, *event_ids)))
                assembly_ids = tuple(
                    dict.fromkeys((*previous_concept.support_assembly_ids, *assembly_ids))
                )
                object_ids = tuple(dict.fromkeys((*previous_concept.object_ids, *object_ids)))
                relation_ids = tuple(dict.fromkeys((*previous_concept.relation_ids, *relation_ids)))
                action_kinds = tuple(
                    dict.fromkeys((*previous_concept.action_kinds, *action_kinds))
                )
                action_sequences = tuple(
                    dict.fromkeys((*previous_concept.action_sequences, *action_sequences))
                )
                sequence_traces = self._merge_sequence_traces(
                    previous_concept.sequence_traces, sequence_traces
                )
                prototype = torch.nn.functional.normalize(
                    (1.0 - self.plasticity_rate) * previous_concept.prototype
                    + self.plasticity_rate * prototype,
                    dim=0,
                )
            concept = Concept(
                concept_id=concept_id,
                prototype=prototype,
                support_event_ids=event_ids,
                support_assembly_ids=assembly_ids,
                object_ids=object_ids,
                relation_ids=relation_ids,
                action_kinds=action_kinds,
                action_sequences=action_sequences,
                sequence_traces=sequence_traces,
                maturity=max(
                    previous_concept.maturity if previous_concept is not None else 0.0,
                    max(0.0, min(1.0, 1.0 - 1.0 / len(episode_ids))),
                ),
                stability=max(0.0, min(1.0, stability)),
                confidence=max(0.0, min(1.0, min(stability, outcome_consistency))),
                outcome_mean=max(0.0, min(1.0, outcome_mean)),
                outcome_consistency=max(0.0, min(1.0, outcome_consistency)),
                update_count=(1 if previous_concept is None else previous_concept.update_count + 1),
                last_updated_tick=int(tick),
                provenance="semantic-consolidation",
            )
            concepts[concept.concept_id] = concept
        self._prune_to_capacity(concepts)
        if self.prune_threshold:
            concepts = {
                concept_id: concept
                for concept_id, concept in concepts.items()
                if self._concept_strength(concept) >= self.prune_threshold
            }
        self._concepts = tuple(concepts.values())
        return self._concepts

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": CONCEPT_FORMATION_CHECKPOINT_FORMAT,
            "similarity_threshold": self.similarity_threshold,
            "signal_weights": list(self.signal_weights),
            "capacity": self.capacity,
            "plasticity_rate": self.plasticity_rate,
            "prune_threshold": self.prune_threshold,
            "credit_discount": self.credit_discount,
            "trace_capacity": self.trace_capacity,
            "concepts": [item.to_payload() for item in self._concepts],
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> ConceptFormationOrgan:
        if payload.get("format") != CONCEPT_FORMATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported concept formation checkpoint format")
        organ = cls(
            similarity_threshold=float(payload["similarity_threshold"]),
            signal_weights=tuple(float(item) for item in payload["signal_weights"]),
            capacity=int(payload.get("capacity", 256)),
            plasticity_rate=float(payload.get("plasticity_rate", 0.25)),
            prune_threshold=float(payload.get("prune_threshold", 0.15)),
            credit_discount=float(payload.get("credit_discount", 0.90)),
            trace_capacity=int(payload.get("trace_capacity", 32)),
        )
        organ._concepts = tuple(
            Concept.from_payload(item, device=device) for item in payload.get("concepts", ())
        )
        return organ
