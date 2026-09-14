"""Content-addressed behavioral evidence for G candidate selection.

P3.4 keeps the behavioral label outside the runtime inference payload.  A
candidate is judged by the live Workbench/action contract and by whether its
semantic slots agree with the observed world.  Safe ``abstain`` and
``reobserve`` candidates use a separate safety/progress policy so that a
typed next step is not confused with a successful task execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

TAIJI_G_BEHAVIOR_FORMAT = "taiji-g-selection-behavior-v1"
TAIJI_G_BEHAVIOR_VERSION = 1

# These are frozen for the canary.  Proposal utility is execution-oriented;
# safe utility is safety-oriented and gives reobserve credit only when the
# observed evidence says that another observation is a valid next step.
PROPOSAL_UTILITY_WEIGHTS = {
    "snapshot_match": 0.10,
    "planner_accepted": 0.15,
    "route_valid": 0.15,
    "parameter_valid": 0.10,
    "world_consistent": 0.25,
    "execution_success": 0.25,
}
SAFE_UTILITY_WEIGHTS = {
    "snapshot_match": 0.20,
    "safe_exit_valid": 0.35,
    "safe_exit_progress": 0.45,
}


def _text(value: Any, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _digest(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return result


def _utility(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _bool(value: Any) -> bool:
    return bool(value)


def _weighted(values: Mapping[str, bool], weights: Mapping[str, float]) -> float:
    return sum(float(bool(values[key])) * float(weight) for key, weight in weights.items())


@dataclass(frozen=True)
class GSelectionBehaviorOutcome:
    """One candidate's observable action/safety result."""

    candidate_id: str
    candidate_digest: str
    candidate_role: str
    snapshot_match: bool
    planner_accepted: bool
    route_valid: bool
    parameter_valid: bool
    world_consistent: bool
    execution_success: bool
    safe_exit_valid: bool
    safe_exit_progress: bool
    utility: float
    outcome_digest: str
    format: str = TAIJI_G_BEHAVIOR_FORMAT
    version: int = TAIJI_G_BEHAVIOR_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_G_BEHAVIOR_FORMAT:
            raise ValueError("unsupported G behavior format")
        if int(self.version) != TAIJI_G_BEHAVIOR_VERSION:
            raise ValueError("unsupported G behavior version")
        candidate_id = _text(self.candidate_id, "candidate_id")
        candidate_digest = _digest(self.candidate_digest, "candidate_digest")
        candidate_role = _text(self.candidate_role, "candidate_role")
        if candidate_role not in {"proposal", "abstain", "reobserve"}:
            raise ValueError("unsupported G behavior candidate role")
        values = {
            "snapshot_match": _bool(self.snapshot_match),
            "planner_accepted": _bool(self.planner_accepted),
            "route_valid": _bool(self.route_valid),
            "parameter_valid": _bool(self.parameter_valid),
            "world_consistent": _bool(self.world_consistent),
            "execution_success": _bool(self.execution_success),
            "safe_exit_valid": _bool(self.safe_exit_valid),
            "safe_exit_progress": _bool(self.safe_exit_progress),
        }
        expected = (
            _weighted(values, PROPOSAL_UTILITY_WEIGHTS)
            if candidate_role == "proposal"
            else _weighted(values, SAFE_UTILITY_WEIGHTS)
        )
        utility = _utility(self.utility, "utility")
        if not math.isclose(utility, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("G behavior utility does not match frozen component policy")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "candidate_id": candidate_id,
            "candidate_digest": candidate_digest,
            "candidate_role": candidate_role,
            **values,
            "utility": utility,
        }
        if _digest(self.outcome_digest, "outcome_digest") != content_digest(unsigned):
            raise ValueError("G behavior outcome digest mismatch")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_digest", candidate_digest)
        object.__setattr__(self, "candidate_role", candidate_role)
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "utility", utility)
        object.__setattr__(self, "outcome_digest", _digest(self.outcome_digest, "outcome_digest"))

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        candidate_digest: str,
        candidate_role: str,
        snapshot_match: bool,
        planner_accepted: bool,
        route_valid: bool,
        parameter_valid: bool,
        world_consistent: bool,
        execution_success: bool,
        safe_exit_valid: bool,
        safe_exit_progress: bool,
    ) -> GSelectionBehaviorOutcome:
        values = {
            "snapshot_match": bool(snapshot_match),
            "planner_accepted": bool(planner_accepted),
            "route_valid": bool(route_valid),
            "parameter_valid": bool(parameter_valid),
            "world_consistent": bool(world_consistent),
            "execution_success": bool(execution_success),
            "safe_exit_valid": bool(safe_exit_valid),
            "safe_exit_progress": bool(safe_exit_progress),
        }
        weights = PROPOSAL_UTILITY_WEIGHTS if candidate_role == "proposal" else SAFE_UTILITY_WEIGHTS
        utility = _weighted(values, weights)
        unsigned = {
            "format": TAIJI_G_BEHAVIOR_FORMAT,
            "version": TAIJI_G_BEHAVIOR_VERSION,
            "candidate_id": str(candidate_id),
            "candidate_digest": str(candidate_digest),
            "candidate_role": str(candidate_role),
            **values,
            "utility": utility,
        }
        return cls.from_payload({**unsigned, "outcome_digest": content_digest(unsigned)})

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "candidate_role": self.candidate_role,
            "snapshot_match": self.snapshot_match,
            "planner_accepted": self.planner_accepted,
            "route_valid": self.route_valid,
            "parameter_valid": self.parameter_valid,
            "world_consistent": self.world_consistent,
            "execution_success": self.execution_success,
            "safe_exit_valid": self.safe_exit_valid,
            "safe_exit_progress": self.safe_exit_progress,
            "utility": self.utility,
            "outcome_digest": self.outcome_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionBehaviorOutcome:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            candidate_id=str(payload["candidate_id"]),
            candidate_digest=str(payload["candidate_digest"]),
            candidate_role=str(payload["candidate_role"]),
            snapshot_match=bool(payload["snapshot_match"]),
            planner_accepted=bool(payload["planner_accepted"]),
            route_valid=bool(payload["route_valid"]),
            parameter_valid=bool(payload["parameter_valid"]),
            world_consistent=bool(payload["world_consistent"]),
            execution_success=bool(payload["execution_success"]),
            safe_exit_valid=bool(payload["safe_exit_valid"]),
            safe_exit_progress=bool(payload["safe_exit_progress"]),
            utility=float(payload["utility"]),
            outcome_digest=str(payload["outcome_digest"]),
        )


@dataclass(frozen=True)
class GSelectionBehaviorSet:
    """Independent behavior labels for one runtime candidate set."""

    candidate_set_digest: str
    inference_digest: str
    split: str
    project_id: str
    path: str
    outcomes: tuple[GSelectionBehaviorOutcome, ...]
    behavior_target_candidate_id: str
    utility_margin: float
    behavior_digest: str
    format: str = TAIJI_G_BEHAVIOR_FORMAT
    version: int = TAIJI_G_BEHAVIOR_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_G_BEHAVIOR_FORMAT:
            raise ValueError("unsupported G behavior set format")
        if int(self.version) != TAIJI_G_BEHAVIOR_VERSION:
            raise ValueError("unsupported G behavior set version")
        candidate_set_digest = _digest(self.candidate_set_digest, "candidate_set_digest")
        inference_digest = _digest(self.inference_digest, "inference_digest")
        split = _text(self.split, "split")
        project_id = _text(self.project_id, "project_id")
        path = _text(self.path, "path")
        outcomes = tuple(self.outcomes)
        if len(outcomes) < 2:
            raise ValueError("G behavior set requires at least two outcomes")
        if len({item.candidate_id for item in outcomes}) != len(outcomes):
            raise ValueError("G behavior outcome candidate ids must be unique")
        target = _text(self.behavior_target_candidate_id, "behavior_target_candidate_id")
        if target not in {item.candidate_id for item in outcomes}:
            raise ValueError("G behavior target is not present")
        ranked = sorted(outcomes, key=lambda item: (-item.utility, item.candidate_id))
        expected_margin = ranked[0].utility - ranked[1].utility
        margin = _utility(self.utility_margin, "utility_margin")
        if not math.isclose(margin, expected_margin, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("G behavior utility margin drifted")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "candidate_set_digest": candidate_set_digest,
            "inference_digest": inference_digest,
            "split": split,
            "project_id": project_id,
            "path": path,
            "outcomes": [item.to_payload() for item in outcomes],
            "behavior_target_candidate_id": target,
            "utility_margin": margin,
        }
        if _digest(self.behavior_digest, "behavior_digest") != content_digest(unsigned):
            raise ValueError("G behavior set digest mismatch")
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "inference_digest", inference_digest)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "behavior_target_candidate_id", target)
        object.__setattr__(self, "utility_margin", margin)
        object.__setattr__(
            self, "behavior_digest", _digest(self.behavior_digest, "behavior_digest")
        )

    @classmethod
    def create(
        cls,
        *,
        candidate_set_digest: str,
        inference_digest: str,
        split: str,
        project_id: str,
        path: str,
        outcomes: Sequence[GSelectionBehaviorOutcome],
    ) -> GSelectionBehaviorSet:
        normalized = tuple(outcomes)
        if len(normalized) < 2:
            raise ValueError("G behavior set requires at least two outcomes")
        ranked = sorted(normalized, key=lambda item: (-item.utility, item.candidate_id))
        unsigned = {
            "format": TAIJI_G_BEHAVIOR_FORMAT,
            "version": TAIJI_G_BEHAVIOR_VERSION,
            "candidate_set_digest": str(candidate_set_digest),
            "inference_digest": str(inference_digest),
            "split": str(split),
            "project_id": str(project_id),
            "path": str(path),
            "outcomes": [item.to_payload() for item in normalized],
            "behavior_target_candidate_id": ranked[0].candidate_id,
            "utility_margin": ranked[0].utility - ranked[1].utility,
        }
        return cls(
            behavior_digest=content_digest(unsigned),
            candidate_set_digest=str(candidate_set_digest),
            inference_digest=str(inference_digest),
            split=str(split),
            project_id=str(project_id),
            path=str(path),
            outcomes=normalized,
            behavior_target_candidate_id=ranked[0].candidate_id,
            utility_margin=ranked[0].utility - ranked[1].utility,
        )

    def target(self) -> GSelectionBehaviorOutcome:
        return next(
            item for item in self.outcomes if item.candidate_id == self.behavior_target_candidate_id
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "candidate_set_digest": self.candidate_set_digest,
            "inference_digest": self.inference_digest,
            "split": self.split,
            "project_id": self.project_id,
            "path": self.path,
            "outcomes": [item.to_payload() for item in self.outcomes],
            "behavior_target_candidate_id": self.behavior_target_candidate_id,
            "utility_margin": self.utility_margin,
            "behavior_digest": self.behavior_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionBehaviorSet:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            candidate_set_digest=str(payload["candidate_set_digest"]),
            inference_digest=str(payload["inference_digest"]),
            split=str(payload["split"]),
            project_id=str(payload["project_id"]),
            path=str(payload["path"]),
            outcomes=tuple(
                GSelectionBehaviorOutcome.from_payload(item) for item in payload.get("outcomes", ())
            ),
            behavior_target_candidate_id=str(payload["behavior_target_candidate_id"]),
            utility_margin=float(payload["utility_margin"]),
            behavior_digest=str(payload["behavior_digest"]),
        )


__all__ = [
    "GSelectionBehaviorOutcome",
    "GSelectionBehaviorSet",
    "PROPOSAL_UTILITY_WEIGHTS",
    "SAFE_UTILITY_WEIGHTS",
    "TAIJI_G_BEHAVIOR_FORMAT",
    "TAIJI_G_BEHAVIOR_VERSION",
]
