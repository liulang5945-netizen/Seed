"""Persistent content features and query-conditioned selection for N177 E1.

The store keeps the producer's discrete entity/attribute/value identity together
with its continuous semantic feature vector and exact source. A correction
supersedes the previous value without deleting its history. The selector is a
small trainable scorer initialised to exact entity/attribute matching; it is
the E2b interface, not a learned-task claim.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .internalization import content_digest
from .joint_state_consumption_course import E1_ATTRIBUTE, E1_COLORS, E1_ENTITIES

CONTENT_FEATURE_FORMAT = "taiji-n177-e1-content-feature-v1"
CONTENT_STORE_FORMAT = "taiji-n177-e1-content-store-v1"
CONTENT_SELECTION_FORMAT = "taiji-n177-e1-content-selection-v1"

_HEX = frozenset(string.hexdigits.lower())


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value.lower())


@dataclass(frozen=True)
class PersistentContentFeatureV1:
    """One source-bound fact with a discrete identity and a continuous vector."""

    feature_id: str
    entity: str
    attribute: str
    value: str
    source_input_id: str
    source_sha256: str
    turn_index: int
    revision: int
    valid_from: int
    superseded_by: str | None
    confidence: float
    continuous: tuple[float, ...]
    feature_sha256: str

    FORMAT = CONTENT_FEATURE_FORMAT

    def __post_init__(self) -> None:
        if not self.feature_id.startswith("e1-feature:"):
            raise ValueError("E1 content feature id is malformed")
        if self.entity not in E1_ENTITIES or self.value not in E1_COLORS:
            raise ValueError("E1 content feature is outside the frozen vocabulary")
        if self.attribute != E1_ATTRIBUTE:
            raise ValueError("E1 content feature attribute differs from the frozen course")
        if not self.source_input_id:
            raise ValueError("E1 content feature requires a source input id")
        if not _is_sha256(self.source_sha256):
            raise ValueError("E1 content feature requires a source sha256")
        if self.turn_index < 0 or self.revision < 0 or self.valid_from < 0:
            raise ValueError("E1 content feature counters cannot be negative")
        if self.superseded_by is not None and not self.superseded_by.startswith("e1-feature:"):
            raise ValueError("E1 content feature supersession id is malformed")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("E1 content feature confidence must be in [0, 1]")
        if not self.continuous or not all(math.isfinite(float(item)) for item in self.continuous):
            raise ValueError("E1 content feature requires a finite continuous vector")
        if self.feature_sha256 != content_digest(self._digest_payload()):
            raise ValueError("E1 content feature digest differs")

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "source_input_id": self.source_input_id,
            "source_sha256": self.source_sha256,
            "turn_index": self.turn_index,
            "revision": self.revision,
            "valid_from": self.valid_from,
            "confidence": float(self.confidence),
            "continuous": list(self.continuous),
        }

    @classmethod
    def create(
        cls,
        *,
        entity: str,
        attribute: str,
        value: str,
        source_input_id: str,
        source_sha256: str,
        turn_index: int,
        revision: int,
        valid_from: int,
        confidence: float,
        continuous: Sequence[float],
    ) -> PersistentContentFeatureV1:
        payload = {
            "format": cls.FORMAT,
            "version": 1,
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "source_input_id": source_input_id,
            "source_sha256": source_sha256,
            "turn_index": turn_index,
            "revision": revision,
            "valid_from": valid_from,
            "confidence": float(confidence),
            "continuous": [float(item) for item in continuous],
        }
        digest = content_digest(payload)
        return cls(
            feature_id=f"e1-feature:{digest[:20]}",
            entity=entity,
            attribute=attribute,
            value=value,
            source_input_id=source_input_id,
            source_sha256=source_sha256,
            turn_index=turn_index,
            revision=revision,
            valid_from=valid_from,
            superseded_by=None,
            confidence=float(confidence),
            continuous=tuple(float(item) for item in continuous),
            feature_sha256=digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "feature_id": self.feature_id,
            "superseded_by": self.superseded_by,
            "feature_sha256": self.feature_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PersistentContentFeatureV1:
        expected = {
            "format",
            "version",
            "feature_id",
            "entity",
            "attribute",
            "value",
            "source_input_id",
            "source_sha256",
            "turn_index",
            "revision",
            "valid_from",
            "superseded_by",
            "confidence",
            "continuous",
            "feature_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("E1 content feature payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported E1 content feature version")
        continuous = payload["continuous"]
        if not isinstance(continuous, list):
            raise ValueError("E1 content feature continuous vector must be a list")
        return cls(
            feature_id=str(payload["feature_id"]),
            entity=str(payload["entity"]),
            attribute=str(payload["attribute"]),
            value=str(payload["value"]),
            source_input_id=str(payload["source_input_id"]),
            source_sha256=str(payload["source_sha256"]),
            turn_index=int(payload["turn_index"]),
            revision=int(payload["revision"]),
            valid_from=int(payload["valid_from"]),
            superseded_by=(
                None if payload["superseded_by"] is None else str(payload["superseded_by"])
            ),
            confidence=float(payload["confidence"]),
            continuous=tuple(float(item) for item in continuous),
            feature_sha256=str(payload["feature_sha256"]),
        )


@dataclass(frozen=True)
class PersistentContentStoreV1:
    """Versioned fact store; corrections supersede but never erase history."""

    features: tuple[PersistentContentFeatureV1, ...]

    FORMAT = CONTENT_STORE_FORMAT

    def __post_init__(self) -> None:
        if len({item.feature_id for item in self.features}) != len(self.features):
            raise ValueError("E1 content store repeats a feature id")
        active = {}
        for item in self.features:
            if item.superseded_by is None:
                key = (item.entity, item.attribute)
                if key in active:
                    raise ValueError("E1 content store has two active values for one key")
                active[key] = item

    def add(self, feature: PersistentContentFeatureV1) -> PersistentContentStoreV1:
        if any(item.feature_id == feature.feature_id for item in self.features):
            raise ValueError("E1 content store already contains this feature")
        if feature.superseded_by is not None:
            raise ValueError("only an active feature can be added")
        previous = self.current(feature.entity, feature.attribute)
        if previous is not None and feature.revision <= previous.revision:
            raise ValueError("E1 content correction revision must advance")
        rows = list(self.features)
        if previous is not None:
            rows = [
                (
                    replace(item, superseded_by=feature.feature_id)
                    if item.feature_id == previous.feature_id
                    else item
                )
                for item in rows
            ]
        rows.append(feature)
        return PersistentContentStoreV1(tuple(rows))

    def current(self, entity: str, attribute: str) -> PersistentContentFeatureV1 | None:
        candidates = [
            item
            for item in self.features
            if item.entity == entity and item.attribute == attribute and item.superseded_by is None
        ]
        if len(candidates) > 1:
            raise ValueError("E1 content store has multiple active values for one key")
        return candidates[0] if candidates else None

    def history(self, entity: str, attribute: str) -> tuple[PersistentContentFeatureV1, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.features
                    if item.entity == entity and item.attribute == attribute
                ),
                key=lambda item: (item.revision, item.turn_index),
            )
        )

    def active_features(self) -> tuple[PersistentContentFeatureV1, ...]:
        return tuple(
            sorted(
                (item for item in self.features if item.superseded_by is None),
                key=lambda item: (item.entity, item.attribute),
            )
        )

    def cut(self, entity: str, attribute: str) -> PersistentContentStoreV1:
        return PersistentContentStoreV1(
            tuple(
                item
                for item in self.features
                if not (item.entity == entity and item.attribute == attribute)
            )
        )

    def swap(
        self,
        entity: str,
        attribute: str,
        replacement: PersistentContentFeatureV1,
    ) -> PersistentContentStoreV1:
        if replacement.entity != entity or replacement.attribute != attribute:
            raise ValueError("E1 content swap replacement changes the selected key")
        return self.cut(entity, attribute).add(replacement)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "features": [item.to_payload() for item in self.features],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PersistentContentStoreV1:
        required = {"format", "version", "features"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("E1 content store payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported E1 content store version")
        features = payload["features"]
        if not isinstance(features, list):
            raise ValueError("E1 content store features must be a list")
        return cls(tuple(PersistentContentFeatureV1.from_payload(item) for item in features))


@dataclass(frozen=True)
class ContentSelectionV1:
    """One query-conditioned selection over a fixed candidate set."""

    feature_ids: tuple[str, ...]
    scores: tuple[float, ...]
    weights: tuple[float, ...]
    selected_index: int
    selected_feature: PersistentContentFeatureV1

    FORMAT = CONTENT_SELECTION_FORMAT

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "feature_ids": list(self.feature_ids),
            "scores": list(self.scores),
            "weights": list(self.weights),
            "selected_index": self.selected_index,
            "selected_feature": self.selected_feature.to_payload(),
        }


class ValueTokenFeatureEncoderV1(nn.Module):
    """Trainable value-token feature; one-hot init is an interface fixture.

    The producer supplies the discrete value span. This encoder turns that
    identity into a continuous vector so the decoder path can be tested before
    the producer's raw-byte feature is made value-preserving in F.
    """

    def __init__(self, *, dim: int = 64) -> None:
        super().__init__()
        if dim < len(E1_COLORS):
            raise ValueError("value-token feature width must cover every colour")
        self.values = E1_COLORS
        self.value_to_index = {value: index for index, value in enumerate(self.values)}
        self.embedding = nn.Embedding(len(self.values), dim)
        with torch.no_grad():
            self.embedding.weight.zero_()
            for index in range(len(self.values)):
                self.embedding.weight[index, index] = 1.0

    def encode(self, value: str) -> torch.Tensor:
        if value not in self.value_to_index:
            raise ValueError("value-token feature value is outside the frozen colour set")
        return self.embedding(torch.tensor(self.value_to_index[value]))

    def forward(self, value: str) -> torch.Tensor:
        return self.encode(value)


class QueryConditionedSelectorV1(nn.Module):
    """Trainable entity/attribute scorer; one-hot init gives exact-match zero-step."""

    def __init__(self, *, dim: int | None = None) -> None:
        super().__init__()
        self.entity_to_id = {entity: index for index, entity in enumerate(E1_ENTITIES)}
        self.attribute_to_id = {E1_ATTRIBUTE: 0}
        entity_dim = len(self.entity_to_id)
        attribute_dim = len(self.attribute_to_id)
        self._entity_dim = entity_dim
        self._attribute_dim = attribute_dim
        width = entity_dim + attribute_dim if dim is None else max(dim, entity_dim + attribute_dim)
        self.entity_embedding = nn.Embedding(entity_dim, width)
        self.attribute_embedding = nn.Embedding(attribute_dim, width)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            self.entity_embedding.weight.zero_()
            for index in range(entity_dim):
                self.entity_embedding.weight[index, index] = 1.0
            self.attribute_embedding.weight.zero_()
            for index in range(attribute_dim):
                self.attribute_embedding.weight[index, entity_dim + index] = 1.0

    def _query(self, entity: str, attribute: str) -> torch.Tensor:
        if entity not in self.entity_to_id:
            raise ValueError("E1 selector query entity is outside the frozen vocabulary")
        if attribute not in self.attribute_to_id:
            raise ValueError("E1 selector query attribute is outside the frozen course")
        return self.entity_embedding(
            torch.tensor(self.entity_to_id[entity])
        ) + self.attribute_embedding(torch.tensor(self.attribute_to_id[attribute]))

    def _candidate(self, feature: PersistentContentFeatureV1) -> torch.Tensor:
        if feature.entity not in self.entity_to_id:
            raise ValueError("E1 selector candidate entity is outside the frozen vocabulary")
        if feature.attribute not in self.attribute_to_id:
            raise ValueError("E1 selector candidate attribute is outside the frozen course")
        return self.entity_embedding(
            torch.tensor(self.entity_to_id[feature.entity])
        ) + self.attribute_embedding(torch.tensor(self.attribute_to_id[feature.attribute]))

    def score_candidates(
        self,
        *,
        entity: str,
        attribute: str,
        candidates: Sequence[PersistentContentFeatureV1],
    ) -> torch.Tensor:
        """Differentiable scores for E2c task-credit experiments."""

        rows = tuple(
            item for item in candidates if item.entity == entity and item.attribute == attribute
        )
        if not rows:
            raise ValueError("E1 selector requires at least one matching candidate")
        query = self._query(entity, attribute)
        scores = torch.stack([torch.dot(query, self._candidate(item)) for item in rows])
        return scores / self.temperature

    def select(
        self,
        *,
        entity: str,
        attribute: str,
        candidates: Sequence[PersistentContentFeatureV1],
    ) -> ContentSelectionV1 | None:
        rows = tuple(
            item for item in candidates if item.entity == entity and item.attribute == attribute
        )
        if not rows:
            return None
        scores = self.score_candidates(entity=entity, attribute=attribute, candidates=rows)
        weights = torch.softmax(scores, dim=0)
        index = int(torch.argmax(scores))
        return ContentSelectionV1(
            feature_ids=tuple(item.feature_id for item in rows),
            scores=tuple(float(item) for item in scores.tolist()),
            weights=tuple(float(item) for item in weights.tolist()),
            selected_index=index,
            selected_feature=rows[index],
        )


def feature_from_parse(
    parse: Any,
    *,
    source_input_id: str,
    source_sha256: str,
    turn_index: int,
    revision: int,
    valid_from: int | None = None,
    confidence: float | None = None,
    continuous: Sequence[float] | None = None,
) -> PersistentContentFeatureV1 | None:
    """Build a feature only when the producer produced exactly one role span each."""

    spans = tuple(getattr(parse, "spans", ()))
    by_role: dict[str, list[str]] = {"entity": [], "attribute": [], "value": []}
    for span in spans:
        role = getattr(span, "role", None)
        if role in by_role:
            by_role[role].append(str(getattr(span, "text", "")))
    if any(len(by_role[role]) != 1 for role in by_role):
        return None
    if continuous is None:
        continuous = tuple(float(item) for item in getattr(parse, "semantic_features", ()))
    else:
        continuous = tuple(float(item) for item in continuous)
    if not continuous:
        return None
    return PersistentContentFeatureV1.create(
        entity=by_role["entity"][0],
        attribute=by_role["attribute"][0],
        value=by_role["value"][0],
        source_input_id=source_input_id,
        source_sha256=source_sha256,
        turn_index=turn_index,
        revision=revision,
        valid_from=turn_index if valid_from is None else valid_from,
        confidence=(
            float(getattr(parse, "confidence", 0.0)) if confidence is None else float(confidence)
        ),
        continuous=continuous,
    )
