"""Content-addressed candidate sets for the first learned G owner.

P3.3 starts with a data-signal gate.  K remains frozen and emits candidates;
G may later learn to accept, reject, or abstain.  Training labels are kept in
the fit/evaluation contract and are deliberately absent from the inference
payload.  This module contains no K parameters and never calls a K learner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import Goal
from .generation import ContentPlan
from .internalization import content_digest

TAIJI_G_SELECTION_FORMAT = "taiji-g-selection-candidate-set-v1"
TAIJI_G_SELECTION_VERSION = 1
G_SELECTION_FEATURE_NAMES = (
    "goal_score",
    "content_score",
    "joint_score",
    "confidence",
    "certainty",
    "has_goal",
    "has_content",
    "is_resolved",
    "is_clarify",
    "is_abstain",
    "is_reobserve",
    "is_proposal",
)


def _digest(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return result


def _text(value: Any, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _unit(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _optional_goal(value: Any, name: str) -> Goal | None:
    if value is None:
        return None
    if isinstance(value, Goal):
        return value
    if isinstance(value, Mapping):
        return Goal.from_payload(value)
    raise TypeError(f"{name} must be a Goal or None")


def _optional_content(value: Any, name: str) -> ContentPlan | None:
    if value is None:
        return None
    if isinstance(value, ContentPlan):
        return value
    if isinstance(value, Mapping):
        return ContentPlan.from_payload(value)
    raise TypeError(f"{name} must be a ContentPlan or None")


@dataclass(frozen=True)
class GSelectionCandidate:
    """One K-proposed selection or an explicit safe-abstention candidate."""

    candidate_id: str
    source: str
    candidate_role: str
    status: str
    goal: Goal | None
    content_plan: ContentPlan | None
    goal_score: float
    content_score: float
    confidence: float
    ambiguity: float
    candidate_digest: str
    format: str = TAIJI_G_SELECTION_FORMAT
    version: int = TAIJI_G_SELECTION_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_G_SELECTION_FORMAT:
            raise ValueError("unsupported G selection candidate format")
        if int(self.version) != TAIJI_G_SELECTION_VERSION:
            raise ValueError("unsupported G selection candidate version")
        candidate_id = _text(self.candidate_id, "candidate_id")
        source = _text(self.source, "candidate source")
        candidate_role = _text(self.candidate_role, "candidate role")
        if candidate_role not in {"proposal", "abstain", "reobserve"}:
            raise ValueError("unsupported G selection candidate role")
        status = _text(self.status, "candidate status")
        if status not in {"resolved", "clarify", "ambiguous", "unknown", "conflict", "abstained"}:
            raise ValueError("unsupported G selection candidate status")
        goal = _optional_goal(self.goal, "candidate goal")
        content = _optional_content(self.content_plan, "candidate content")
        if content is not None and goal is None:
            raise ValueError("candidate content requires a candidate goal")
        goal_score = _unit(self.goal_score, "candidate goal_score")
        content_score = _unit(self.content_score, "candidate content_score")
        confidence = _unit(self.confidence, "candidate confidence")
        ambiguity = _unit(self.ambiguity, "candidate ambiguity")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "candidate_id": candidate_id,
            "source": source,
            "candidate_role": candidate_role,
            "status": status,
            "goal": None if goal is None else goal.to_payload(),
            "content_plan": None if content is None else content.to_payload(),
            "goal_score": goal_score,
            "content_score": content_score,
            "confidence": confidence,
            "ambiguity": ambiguity,
        }
        if _digest(self.candidate_digest, "candidate_digest") != content_digest(unsigned):
            raise ValueError("G selection candidate digest mismatch")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "candidate_role", candidate_role)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "goal", goal)
        object.__setattr__(self, "content_plan", content)
        object.__setattr__(self, "goal_score", goal_score)
        object.__setattr__(self, "content_score", content_score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "candidate_digest", _digest(self.candidate_digest, "candidate_digest"))

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        source: str,
        candidate_role: str,
        status: str,
        goal: Goal | None,
        content_plan: ContentPlan | None,
        goal_score: float,
        content_score: float,
        confidence: float,
        ambiguity: float,
    ) -> GSelectionCandidate:
        unsigned = {
            "format": TAIJI_G_SELECTION_FORMAT,
            "version": TAIJI_G_SELECTION_VERSION,
            "candidate_id": str(candidate_id),
            "source": str(source),
            "candidate_role": str(candidate_role),
            "status": str(status),
            "goal": None if goal is None else goal.to_payload(),
            "content_plan": None if content_plan is None else content_plan.to_payload(),
            "goal_score": float(goal_score),
            "content_score": float(content_score),
            "confidence": float(confidence),
            "ambiguity": float(ambiguity),
        }
        return cls(candidate_digest=content_digest(unsigned), **unsigned)

    @property
    def joint_score(self) -> float:
        if self.goal is None or self.content_plan is None:
            return 0.0
        return min(self.goal_score, self.content_score)

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return (
            self.goal_score,
            self.content_score,
            self.joint_score,
            self.confidence,
            1.0 - self.ambiguity,
            float(self.goal is not None),
            float(self.content_plan is not None),
            float(self.status == "resolved"),
            float(self.status == "clarify"),
            float(self.candidate_role == "abstain"),
            float(self.candidate_role == "reobserve"),
            float(self.candidate_role == "proposal"),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "candidate_id": self.candidate_id,
            "source": self.source,
            "candidate_role": self.candidate_role,
            "status": self.status,
            "goal": None if self.goal is None else self.goal.to_payload(),
            "content_plan": None
            if self.content_plan is None
            else self.content_plan.to_payload(),
            "goal_score": self.goal_score,
            "content_score": self.content_score,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionCandidate:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            candidate_id=str(payload["candidate_id"]),
            source=str(payload["source"]),
            candidate_role=str(payload["candidate_role"]),
            status=str(payload["status"]),
            goal=_optional_goal(payload.get("goal"), "candidate goal"),
            content_plan=_optional_content(payload.get("content_plan"), "candidate content"),
            goal_score=float(payload["goal_score"]),
            content_score=float(payload["content_score"]),
            confidence=float(payload["confidence"]),
            ambiguity=float(payload["ambiguity"]),
            candidate_digest=str(payload["candidate_digest"]),
        )
        if content_digest(item._payload_without_digest()) != item.candidate_digest:
            raise ValueError("normalized G selection candidate digest mismatch")
        return item


@dataclass(frozen=True)
class GSelectionCandidateSet:
    """A train/validation candidate set with labels excluded from inference."""

    example_id: str
    family_id: str
    split: str
    project_id: str
    path: str
    input_digest: str
    candidates: tuple[GSelectionCandidate, ...]
    target_candidate_id: str
    target_kind: str
    candidate_set_digest: str
    inference_digest: str
    format: str = TAIJI_G_SELECTION_FORMAT
    version: int = TAIJI_G_SELECTION_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_G_SELECTION_FORMAT:
            raise ValueError("unsupported G selection candidate-set format")
        if int(self.version) != TAIJI_G_SELECTION_VERSION:
            raise ValueError("unsupported G selection candidate-set version")
        example_id = _text(self.example_id, "G selection example_id")
        family_id = _text(self.family_id, "G selection family_id")
        split = _text(self.split, "G selection split")
        project_id = _text(self.project_id, "G selection project_id")
        path = _text(self.path, "G selection path")
        input_digest = _digest(self.input_digest, "G selection input_digest")
        candidates = tuple(self.candidates)
        if len(candidates) < 2:
            raise ValueError("G selection candidate set requires at least two candidates")
        if any(not isinstance(item, GSelectionCandidate) for item in candidates):
            raise TypeError("G selection candidate set contains an invalid candidate")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("G selection candidate ids must be unique")
        target_candidate_id = _text(self.target_candidate_id, "G selection target_candidate_id")
        if target_candidate_id not in {item.candidate_id for item in candidates}:
            raise ValueError("G selection target candidate is not in the candidate set")
        target_kind = _text(self.target_kind, "G selection target_kind")
        if target_kind not in {"pair", "abstain", "reobserve"}:
            raise ValueError("unsupported G selection target_kind")
        unsigned = self._payload_without_digests()
        if _digest(self.candidate_set_digest, "candidate_set_digest") != content_digest(unsigned):
            raise ValueError("G selection candidate-set digest mismatch")
        inference_payload = self._inference_payload()
        if _digest(self.inference_digest, "inference_digest") != content_digest(inference_payload):
            raise ValueError("G selection inference digest mismatch")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "target_candidate_id", target_candidate_id)
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "candidate_set_digest", _digest(self.candidate_set_digest, "candidate_set_digest"))
        object.__setattr__(self, "inference_digest", _digest(self.inference_digest, "inference_digest"))

    @classmethod
    def create(
        cls,
        *,
        example_id: str,
        family_id: str,
        split: str,
        project_id: str,
        path: str,
        input_digest: str,
        candidates: Sequence[GSelectionCandidate],
        target_candidate_id: str,
        target_kind: str,
    ) -> GSelectionCandidateSet:
        unsigned = {
            "format": TAIJI_G_SELECTION_FORMAT,
            "version": TAIJI_G_SELECTION_VERSION,
            "example_id": str(example_id),
            "family_id": str(family_id),
            "split": str(split),
            "project_id": str(project_id),
            "path": str(path),
            "input_digest": str(input_digest),
            "candidates": [item.to_payload() for item in candidates],
            "target_candidate_id": str(target_candidate_id),
            "target_kind": str(target_kind),
        }
        inference_payload = {
            "format": TAIJI_G_SELECTION_FORMAT,
            "version": TAIJI_G_SELECTION_VERSION,
            "example_id": str(example_id),
            "family_id": str(family_id),
            "split": str(split),
            "project_id": str(project_id),
            "path": str(path),
            "input_digest": str(input_digest),
            "candidates": [item.to_payload() for item in candidates],
        }
        return cls(
            format=TAIJI_G_SELECTION_FORMAT,
            version=TAIJI_G_SELECTION_VERSION,
            example_id=str(example_id),
            family_id=str(family_id),
            split=str(split),
            project_id=str(project_id),
            path=str(path),
            input_digest=str(input_digest),
            candidate_set_digest=content_digest(unsigned),
            inference_digest=content_digest(inference_payload),
            candidates=tuple(candidates),
            target_candidate_id=str(target_candidate_id),
            target_kind=str(target_kind),
        )

    def _payload_without_digests(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "example_id": self.example_id,
            "family_id": self.family_id,
            "split": self.split,
            "project_id": self.project_id,
            "path": self.path,
            "input_digest": self.input_digest,
            "candidates": [item.to_payload() for item in self.candidates],
            "target_candidate_id": self.target_candidate_id,
            "target_kind": self.target_kind,
        }

    def _inference_payload(self) -> dict[str, Any]:
        payload = self._payload_without_digests()
        payload.pop("target_candidate_id")
        payload.pop("target_kind")
        return payload

    def to_inference_payload(self) -> dict[str, Any]:
        """Return the only payload allowed at runtime; no target label is present."""

        return {**self._inference_payload(), "inference_digest": self.inference_digest}

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_digests(),
            "candidate_set_digest": self.candidate_set_digest,
            "inference_digest": self.inference_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionCandidateSet:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            example_id=str(payload["example_id"]),
            family_id=str(payload["family_id"]),
            split=str(payload["split"]),
            project_id=str(payload["project_id"]),
            path=str(payload["path"]),
            input_digest=str(payload["input_digest"]),
            candidates=tuple(
                GSelectionCandidate.from_payload(item)
                for item in payload.get("candidates", ())
            ),
            target_candidate_id=str(payload["target_candidate_id"]),
            target_kind=str(payload["target_kind"]),
            candidate_set_digest=str(payload["candidate_set_digest"]),
            inference_digest=str(payload["inference_digest"]),
        )
        return item

    def target_candidate(self) -> GSelectionCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == self.target_candidate_id:
                return candidate
        raise AssertionError("validated target candidate disappeared")

    @property
    def target_available(self) -> bool:
        return self.target_candidate_id in {item.candidate_id for item in self.candidates}

    @property
    def score_margin(self) -> float:
        target = self.target_candidate().joint_score
        distractors = [
            item.joint_score for item in self.candidates if item.candidate_id != self.target_candidate_id
        ]
        return target - max(distractors, default=0.0)


__all__ = [
    "G_SELECTION_FEATURE_NAMES",
    "TAIJI_G_SELECTION_FORMAT",
    "TAIJI_G_SELECTION_VERSION",
    "GSelectionCandidate",
    "GSelectionCandidateSet",
]
