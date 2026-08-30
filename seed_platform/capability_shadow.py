"""No-side-effect shadow comparison for validated capability candidates.

The shadow layer records comparable evidence only. It never imports an
executor, evaluates source code, or performs a file/terminal/MCP side effect.
The registry remains the owner of lifecycle state; this module only evaluates
whether a shadow observation is admissible under the current snapshot and
policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .capability_registry import CAPABILITY_SIDE_EFFECTS, CapabilityRegistry

SHADOW_COMPARISON_FORMAT = "seed-capability-shadow-comparison-v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("shadow digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported shadow digest value: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _metrics(value: Mapping[str, Any]) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise TypeError("shadow resource metrics must be a mapping")
    normalized: dict[str, float | int] = {}
    for key, item in value.items():
        name = _required_text(key, "resource metric key")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("shadow resource metrics must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ValueError("shadow resource metrics must be finite and non-negative")
        normalized[name] = item
    return {key: normalized[key] for key in sorted(normalized)}


@dataclass(frozen=True)
class CapabilityShadowObservation:
    """Digest-only comparison between baseline and candidate execution."""

    capability_id: str
    candidate_bundle_digest: str
    registry_snapshot_id: str
    input_digest: str
    baseline_output_digest: str
    candidate_output_digest: str
    baseline_after_state_digest: str
    candidate_after_state_digest: str
    baseline_resources: Mapping[str, Any] = field(default_factory=dict)
    candidate_resources: Mapping[str, Any] = field(default_factory=dict)
    policy_allowed: bool = True
    approval_id: str = ""
    side_effects_performed: bool = False
    require_output_equivalence: bool = True
    format: str = SHADOW_COMPARISON_FORMAT

    def __post_init__(self) -> None:
        if self.format != SHADOW_COMPARISON_FORMAT:
            raise ValueError("unsupported shadow comparison format")
        for name, value in (
            ("capability_id", self.capability_id),
            ("candidate_bundle_digest", self.candidate_bundle_digest),
            ("registry_snapshot_id", self.registry_snapshot_id),
            ("input_digest", self.input_digest),
            ("baseline_output_digest", self.baseline_output_digest),
            ("candidate_output_digest", self.candidate_output_digest),
            ("baseline_after_state_digest", self.baseline_after_state_digest),
            ("candidate_after_state_digest", self.candidate_after_state_digest),
        ):
            _required_text(value, name)
        object.__setattr__(self, "baseline_resources", _metrics(self.baseline_resources))
        object.__setattr__(self, "candidate_resources", _metrics(self.candidate_resources))
        object.__setattr__(self, "approval_id", str(self.approval_id).strip())

    @classmethod
    def from_execution(
        cls,
        *,
        capability_id: str,
        candidate_bundle_digest: str,
        registry_snapshot_id: str,
        input_payload: Any,
        baseline_output: Any,
        candidate_output: Any,
        baseline_after_state: Any,
        candidate_after_state: Any,
        baseline_resources: Mapping[str, Any],
        candidate_resources: Mapping[str, Any],
        policy_allowed: bool = True,
        approval_id: str = "",
        side_effects_performed: bool = False,
        require_output_equivalence: bool = True,
    ) -> CapabilityShadowObservation:
        return cls(
            capability_id=capability_id,
            candidate_bundle_digest=candidate_bundle_digest,
            registry_snapshot_id=registry_snapshot_id,
            input_digest=_digest(input_payload),
            baseline_output_digest=_digest(baseline_output),
            candidate_output_digest=_digest(candidate_output),
            baseline_after_state_digest=_digest(baseline_after_state),
            candidate_after_state_digest=_digest(candidate_after_state),
            baseline_resources=baseline_resources,
            candidate_resources=candidate_resources,
            policy_allowed=policy_allowed,
            approval_id=approval_id,
            side_effects_performed=side_effects_performed,
            require_output_equivalence=require_output_equivalence,
        )

    @property
    def observation_digest(self) -> str:
        return _digest(self._identity_payload())

    @property
    def output_equal(self) -> bool:
        return self.baseline_output_digest == self.candidate_output_digest

    @property
    def after_state_equal(self) -> bool:
        return self.baseline_after_state_digest == self.candidate_after_state_digest

    @property
    def resource_delta(self) -> dict[str, float]:
        keys = set(self.baseline_resources) | set(self.candidate_resources)
        return {
            key: float(self.candidate_resources.get(key, 0))
            - float(self.baseline_resources.get(key, 0))
            for key in sorted(keys)
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "capability_id": self.capability_id,
            "candidate_bundle_digest": self.candidate_bundle_digest,
            "registry_snapshot_id": self.registry_snapshot_id,
            "input_digest": self.input_digest,
            "baseline_output_digest": self.baseline_output_digest,
            "candidate_output_digest": self.candidate_output_digest,
            "baseline_after_state_digest": self.baseline_after_state_digest,
            "candidate_after_state_digest": self.candidate_after_state_digest,
            "baseline_resources": dict(self.baseline_resources),
            "candidate_resources": dict(self.candidate_resources),
            "policy_allowed": self.policy_allowed,
            "approval_id": self.approval_id,
            "side_effects_performed": self.side_effects_performed,
            "require_output_equivalence": self.require_output_equivalence,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "observation_digest": self.observation_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilityShadowObservation:
        observation = cls(
            capability_id=str(payload.get("capability_id", "")),
            candidate_bundle_digest=str(payload.get("candidate_bundle_digest", "")),
            registry_snapshot_id=str(payload.get("registry_snapshot_id", "")),
            input_digest=str(payload.get("input_digest", "")),
            baseline_output_digest=str(payload.get("baseline_output_digest", "")),
            candidate_output_digest=str(payload.get("candidate_output_digest", "")),
            baseline_after_state_digest=str(payload.get("baseline_after_state_digest", "")),
            candidate_after_state_digest=str(payload.get("candidate_after_state_digest", "")),
            baseline_resources=payload.get("baseline_resources") or {},
            candidate_resources=payload.get("candidate_resources") or {},
            policy_allowed=bool(payload.get("policy_allowed", True)),
            approval_id=str(payload.get("approval_id", "")),
            side_effects_performed=bool(payload.get("side_effects_performed", False)),
            require_output_equivalence=bool(payload.get("require_output_equivalence", True)),
            format=str(payload.get("format", "")),
        )
        if str(payload.get("observation_digest", "")) != observation.observation_digest:
            raise ValueError("shadow observation digest mismatch")
        return observation


@dataclass(frozen=True)
class CapabilityShadowGateResult:
    passed: bool
    decision: str
    reason_code: str
    observation_digest: str
    capability_id: str
    approval_required: bool
    output_equal: bool
    after_state_equal: bool
    resource_delta: Mapping[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "observation_digest": self.observation_digest,
            "capability_id": self.capability_id,
            "approval_required": self.approval_required,
            "output_equal": self.output_equal,
            "after_state_equal": self.after_state_equal,
            "resource_delta": dict(self.resource_delta),
        }


def evaluate_shadow(
    registry: CapabilityRegistry,
    observation: CapabilityShadowObservation,
) -> CapabilityShadowGateResult:
    """Evaluate one no-side-effect observation against the active registry state."""

    bundle = registry.get_bundle(observation.candidate_bundle_digest)
    record = registry.get_record(observation.candidate_bundle_digest)
    approval_required = False
    if bundle is not None:
        approval_required = (
            bundle.effect in CAPABILITY_SIDE_EFFECTS or bundle.risk in CAPABILITY_SIDE_EFFECTS
        )

    def result(passed: bool, decision: str, reason_code: str) -> CapabilityShadowGateResult:
        return CapabilityShadowGateResult(
            passed=passed,
            decision=decision,
            reason_code=reason_code,
            observation_digest=observation.observation_digest,
            capability_id=observation.capability_id,
            approval_required=approval_required,
            output_equal=observation.output_equal,
            after_state_equal=observation.after_state_equal,
            resource_delta=observation.resource_delta,
        )

    if bundle is None or record is None:
        return result(False, "deny", "candidate_unknown")
    if record.status != "shadow":
        return result(False, "deny", "candidate_not_shadow")
    if observation.capability_id != bundle.capability_id:
        return result(False, "deny", "capability_id_mismatch")
    if observation.registry_snapshot_id != registry.snapshot_id:
        return result(False, "deny", "stale_capability_registry")
    if not observation.policy_allowed:
        return result(False, "deny", "policy_denied")
    if observation.side_effects_performed:
        return result(False, "deny", "shadow_side_effect_detected")
    if approval_required and not observation.approval_id:
        return result(False, "deny", "approval_required")
    if observation.require_output_equivalence and not observation.output_equal:
        return result(False, "deny", "output_mismatch")
    if not observation.after_state_equal:
        return result(False, "deny", "after_state_mutated")
    return result(True, "allow", "shadow_equivalent")


__all__ = [
    "CapabilityShadowGateResult",
    "CapabilityShadowObservation",
    "SHADOW_COMPARISON_FORMAT",
    "evaluate_shadow",
]
