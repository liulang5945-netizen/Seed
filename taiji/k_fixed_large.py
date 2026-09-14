"""Native fixed-width K ensembles for the R6 causal control.

The fixed-large control is deliberately boring: it is a frozen arithmetic
ensemble of independent native K1/K2 workers.  It has no router, task-id
branch, provider, client, or Transformer dependency.  The ensemble averages
typed score maps and then decodes one ordinary Taiji result contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import PerceptEvent, WorldState
from .internalization import content_digest
from .semantic_training import (
    StructuredSemanticLearner,
    StructuredSemanticResult,
)
from .semantic_transition import (
    StructuredSemanticTransitionLearner,
    StructuredSemanticTransitionResult,
)

FIXED_LARGE_CHECKPOINT_FORMAT = "taiji-k-fixed-large-ensemble-v1"
FIXED_LARGE_CHECKPOINT_VERSION = 1


def _mean_score_maps(
    maps: Sequence[Mapping[str, float]],
    *,
    expected_keys: Sequence[str],
    name: str,
) -> dict[str, float]:
    """Average one typed score map and reject shape/key drift."""

    if not maps:
        return {}
    keys = tuple(str(key) for key in expected_keys)
    expected = set(keys)
    for index, score_map in enumerate(maps):
        actual = {str(key) for key in score_map}
        if actual != expected:
            raise ValueError(f"fixed-large {name} replica {index} score-map keys do not match")
    width = float(len(maps))
    return {key: sum(float(score_map[key]) for score_map in maps) / width for key in keys}


def _top_two(scores: Mapping[str, float], ordered_keys: Sequence[str]) -> tuple[str, float, float]:
    ordered = sorted(
        ((str(key), float(scores[key])) for key in ordered_keys),
        key=lambda item: (-item[1], item[0]),
    )
    if not ordered:
        raise ValueError("fixed-large score map cannot be empty")
    return ordered[0][0], ordered[0][1], ordered[1][1] if len(ordered) > 1 else 0.0


def _same_contract(values: Sequence[Sequence[str]], name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"fixed-large {name} replicas cannot be empty")
    first = tuple(str(value) for value in values[0])
    if any(tuple(str(value) for value in value_set) != first for value_set in values[1:]):
        raise ValueError(f"fixed-large {name} replica contract mismatch")
    return first


class NativeKFixedLargeEnsemble:
    """A fixed-width arithmetic ensemble over native K1 and K2 workers."""

    def __init__(
        self,
        semantic_replicas: Sequence[StructuredSemanticLearner],
        transition_replicas: Sequence[StructuredSemanticTransitionLearner],
    ) -> None:
        self.semantic_replicas = tuple(semantic_replicas)
        self.transition_replicas = tuple(transition_replicas)
        if not self.semantic_replicas or not self.transition_replicas:
            raise ValueError("fixed-large ensemble requires K1 and K2 replicas")
        if len(self.semantic_replicas) != len(self.transition_replicas):
            raise ValueError("fixed-large K1/K2 ensemble widths must match")
        self.ensemble_width = len(self.semantic_replicas)
        self.semantic_fact_keys = _same_contract(
            [replica.fact_keys for replica in self.semantic_replicas], "semantic fact"
        )
        self.semantic_goal_ids = _same_contract(
            [replica.goal_ids for replica in self.semantic_replicas], "semantic goal"
        )
        self.semantic_content_ids = _same_contract(
            [replica.content_ids for replica in self.semantic_replicas],
            "semantic content",
        )
        self.transition_fact_keys = _same_contract(
            [replica.fact_keys for replica in self.transition_replicas],
            "transition fact",
        )
        self.transition_goal_ids = _same_contract(
            [replica.goal_ids for replica in self.transition_replicas],
            "transition goal",
        )
        self.transition_content_ids = _same_contract(
            [replica.content_ids for replica in self.transition_replicas],
            "transition content",
        )
        if any(
            replica.feature_dim != self.semantic_replicas[0].feature_dim
            for replica in self.semantic_replicas
        ):
            raise ValueError("fixed-large semantic feature dimensions must match")
        if any(
            replica.event_dim != self.transition_replicas[0].event_dim
            for replica in self.transition_replicas
        ):
            raise ValueError("fixed-large transition event dimensions must match")

    @property
    def parameter_count(self) -> int:
        return sum(replica.parameter_count for replica in self.semantic_replicas) + sum(
            replica.parameter_count for replica in self.transition_replicas
        )

    @property
    def owner_digests(self) -> dict[str, str]:
        owners: dict[str, str] = {}
        for index, replica in enumerate(self.semantic_replicas):
            for name, digest in replica.owner_digests().items():
                owners[f"k1.replica.{index}.{name}"] = digest
        for index, transition_replica in enumerate(self.transition_replicas):
            for name, digest in transition_replica.owner_digests().items():
                owners[f"k2.replica.{index}.{name}"] = digest
        return owners

    @property
    def source_digest(self) -> str:
        return str(
            content_digest(
                {
                    "format": FIXED_LARGE_CHECKPOINT_FORMAT,
                    "version": FIXED_LARGE_CHECKPOINT_VERSION,
                    "ensemble_width": self.ensemble_width,
                    "semantic_sources": [
                        replica.source_digest for replica in self.semantic_replicas
                    ],
                    "transition_sources": [
                        replica.source_digest for replica in self.transition_replicas
                    ],
                }
            )
        )

    def predict_semantic(self, percept: PerceptEvent) -> StructuredSemanticResult:
        results = tuple(replica.predict(percept) for replica in self.semantic_replicas)
        representative = self.semantic_replicas[0]
        fact_scores = _mean_score_maps(
            [result.fact_scores for result in results],
            expected_keys=self.semantic_fact_keys,
            name="semantic fact",
        )
        world = representative._materialize_world(percept, fact_scores)
        if percept.confidence < representative.confidence_floor:
            return StructuredSemanticResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=float(percept.confidence),
                ambiguity=1.0,
            )
        active = {
            key for key, score in fact_scores.items() if score >= representative.fact_threshold
        }
        conflicts = any(
            len(active.intersection(group)) > 1
            or (len(group) > 1 and all(0.45 <= fact_scores[key] <= 0.55 for key in group))
            for group in representative._conflict_groups(self.semantic_fact_keys)
        )
        if conflicts:
            return StructuredSemanticResult(
                status="conflict",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        if not active:
            return StructuredSemanticResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )

        goal_scores = _mean_score_maps(
            [result.goal_scores for result in results],
            expected_keys=self.semantic_goal_ids,
            name="semantic goal",
        )
        goal_id, goal_confidence, second_goal = _top_two(goal_scores, self.semantic_goal_ids)
        goal_ambiguity = 1.0 - max(0.0, goal_confidence - second_goal)
        goal = representative._goals[goal_id]
        if (
            goal_confidence < representative.confidence_floor
            or goal_confidence - second_goal < representative.ambiguity_ceiling
        ):
            return StructuredSemanticResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores=goal_scores,
                content_scores={},
                confidence=goal_confidence,
                ambiguity=goal_ambiguity,
            )

        content_scores = _mean_score_maps(
            [result.content_scores for result in results],
            expected_keys=self.semantic_content_ids,
            name="semantic content",
        )
        content_id, content_confidence, _ = _top_two(content_scores, self.semantic_content_ids)
        content = representative._content_plans[content_id]
        if content_confidence < representative.confidence_floor:
            return StructuredSemanticResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=fact_scores,
                goal_scores=goal_scores,
                content_scores=content_scores,
                confidence=content_confidence,
                ambiguity=1.0 - content_confidence,
            )
        return StructuredSemanticResult(
            status="clarify" if content.intent_kind == "request_information" else "resolved",
            world=world,
            goal=goal,
            content_plan=content,
            fact_scores=fact_scores,
            goal_scores=goal_scores,
            content_scores=content_scores,
            confidence=min(goal_confidence, content_confidence),
            ambiguity=min(goal_ambiguity, 1.0 - content_confidence),
        )

    def predict_transition(
        self, previous: WorldState, event: PerceptEvent
    ) -> StructuredSemanticTransitionResult:
        results = tuple(replica.predict(previous, event) for replica in self.transition_replicas)
        representative = self.transition_replicas[0]
        current = representative._fact_vector(previous)
        current_scores = {
            key: float(value) for key, value in zip(self.transition_fact_keys, current, strict=True)
        }
        if event.confidence < representative.confidence_floor:
            return StructuredSemanticTransitionResult(
                status="unknown",
                world=previous,
                goal=None,
                content_plan=None,
                fact_scores=current_scores,
                delta_scores={},
                goal_scores={},
                content_scores={},
                confidence=float(event.confidence),
                ambiguity=1.0,
            )
        current_active = {
            key for key, score in current_scores.items() if score >= representative.fact_threshold
        }
        if any(
            len(current_active.intersection(group)) > 1
            for group in representative._conflict_groups(self.transition_fact_keys)
        ):
            return StructuredSemanticTransitionResult(
                status="conflict",
                world=previous,
                goal=None,
                content_plan=None,
                fact_scores=current_scores,
                delta_scores={},
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )

        delta_scores = _mean_score_maps(
            [result.delta_scores for result in results],
            expected_keys=self.transition_fact_keys,
            name="transition delta",
        )
        next_values = {
            key: max(0.0, min(1.0, current_scores[key] + delta_scores[key]))
            for key in self.transition_fact_keys
        }
        world = representative._materialize_world(previous, event, next_values)
        active = {
            key for key, score in next_values.items() if score >= representative.fact_threshold
        }
        conflicts = any(
            len(active.intersection(group)) > 1
            or (len(group) > 1 and all(0.45 <= next_values[key] <= 0.55 for key in group))
            for group in representative._conflict_groups(self.transition_fact_keys)
        )
        if conflicts:
            return StructuredSemanticTransitionResult(
                status="conflict",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=next_values,
                delta_scores=delta_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )
        if not active:
            return StructuredSemanticTransitionResult(
                status="unknown",
                world=world,
                goal=None,
                content_plan=None,
                fact_scores=next_values,
                delta_scores=delta_scores,
                goal_scores={},
                content_scores={},
                confidence=0.0,
                ambiguity=1.0,
            )

        goal_scores = _mean_score_maps(
            [result.goal_scores for result in results],
            expected_keys=self.transition_goal_ids,
            name="transition goal",
        )
        goal_id, goal_confidence, second_goal = _top_two(goal_scores, self.transition_goal_ids)
        goal_ambiguity = 1.0 - max(0.0, goal_confidence - second_goal)
        goal = representative._goals[goal_id]
        if (
            goal_confidence < representative.confidence_floor
            or goal_confidence - second_goal < representative.ambiguity_ceiling
        ):
            return StructuredSemanticTransitionResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=next_values,
                delta_scores=delta_scores,
                goal_scores=goal_scores,
                content_scores={},
                confidence=goal_confidence,
                ambiguity=goal_ambiguity,
            )

        content_scores = _mean_score_maps(
            [result.content_scores for result in results],
            expected_keys=self.transition_content_ids,
            name="transition content",
        )
        content_id, content_confidence, _ = _top_two(content_scores, self.transition_content_ids)
        content = representative._content_plans[content_id]
        if content_confidence < representative.confidence_floor:
            return StructuredSemanticTransitionResult(
                status="ambiguous",
                world=world,
                goal=goal,
                content_plan=None,
                fact_scores=next_values,
                delta_scores=delta_scores,
                goal_scores=goal_scores,
                content_scores=content_scores,
                confidence=content_confidence,
                ambiguity=1.0 - content_confidence,
            )
        return StructuredSemanticTransitionResult(
            status="clarify" if content.intent_kind == "request_information" else "resolved",
            world=world,
            goal=goal,
            content_plan=content,
            fact_scores=next_values,
            delta_scores=delta_scores,
            goal_scores=goal_scores,
            content_scores=content_scores,
            confidence=min(goal_confidence, content_confidence),
            ambiguity=min(goal_ambiguity, 1.0 - content_confidence),
        )

    def checkpoint(self) -> dict[str, Any]:
        unsigned = {
            "format": FIXED_LARGE_CHECKPOINT_FORMAT,
            "version": FIXED_LARGE_CHECKPOINT_VERSION,
            "ensemble_width": self.ensemble_width,
            "semantic_checkpoints": [replica.checkpoint() for replica in self.semantic_replicas],
            "transition_checkpoints": [
                replica.checkpoint() for replica in self.transition_replicas
            ],
            "owner_digests": self.owner_digests,
            "source_digest": self.source_digest,
        }
        return {**unsigned, "ensemble_digest": content_digest(unsigned)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeKFixedLargeEnsemble:
        if payload.get("format") != FIXED_LARGE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported fixed-large ensemble checkpoint format")
        if int(payload.get("version", -1)) != FIXED_LARGE_CHECKPOINT_VERSION:
            raise ValueError("unsupported fixed-large ensemble checkpoint version")
        unsigned = {key: value for key, value in payload.items() if key != "ensemble_digest"}
        if str(payload.get("ensemble_digest", "")) != content_digest(unsigned):
            raise ValueError("fixed-large ensemble checkpoint digest mismatch")
        semantic_payloads = payload.get("semantic_checkpoints")
        transition_payloads = payload.get("transition_checkpoints")
        if not isinstance(semantic_payloads, Sequence) or not isinstance(
            transition_payloads, Sequence
        ):
            raise ValueError("fixed-large ensemble checkpoint replicas are missing")
        semantic = tuple(
            StructuredSemanticLearner.from_checkpoint(item, device="cpu")
            for item in semantic_payloads
        )
        transition = tuple(
            StructuredSemanticTransitionLearner.from_checkpoint(item, device="cpu")
            for item in transition_payloads
        )
        restored = cls(semantic, transition)
        if int(payload.get("ensemble_width", -1)) != restored.ensemble_width:
            raise ValueError("fixed-large ensemble width mismatch")
        if str(payload.get("source_digest", "")) != restored.source_digest:
            raise ValueError("fixed-large ensemble source digest mismatch")
        if payload.get("owner_digests") != restored.owner_digests:
            raise ValueError("fixed-large ensemble owner digest mismatch")
        return restored


__all__ = [
    "FIXED_LARGE_CHECKPOINT_FORMAT",
    "FIXED_LARGE_CHECKPOINT_VERSION",
    "NativeKFixedLargeEnsemble",
]
