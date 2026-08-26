"""Taiji-owned cross-experience concept formation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import Concept, EpisodicMemoryRecord

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
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.signal_weights = tuple(float(weight) for weight in signal_weights)
        self.capacity = int(capacity)
        self.plasticity_rate = float(plasticity_rate)
        self.prune_threshold = float(prune_threshold)
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
    def _outcome_score(record: EpisodicMemoryRecord) -> float:
        if record.outcome is None:
            return 0.0
        bounded_reward = 0.5 * (1.0 + math.tanh(float(record.outcome.reward)))
        return 0.5 * float(bool(record.outcome.success)) + 0.5 * bounded_reward

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
        )
        organ._concepts = tuple(
            Concept.from_payload(item, device=device) for item in payload.get("concepts", ())
        )
        return organ
