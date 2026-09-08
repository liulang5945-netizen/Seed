"""Pressure evidence for Taiji's first shadow-growth admission step.

This module is deliberately a proposal trigger, not a topology mutator.  It
records only signals available on the native predictive path: residual error,
fast/slow conflict, activity saturation, expected utility gap and resource
state.  No evaluator task identifier or semantic routing label is accepted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .internalization import content_digest

ADAPTIVE_RESIDUAL_GROWTH_FORMAT = "taiji-adaptive-residual-growth-v1"
ADAPTIVE_RESIDUAL_GROWTH_VERSION = 1


def _text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _unit(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class AdaptiveResidualGrowthPressure:
    """One content-addressed pressure observation from the native bridge."""

    bridge_id: str
    tick: int
    residual_error: float
    fast_slow_conflict: float
    activity_saturation: float
    utility_gap: float
    resource_state: float
    evidence_id: str
    parent_checkpoint_digest: str
    pressure_digest: str

    def __post_init__(self) -> None:
        _text(self.bridge_id, "adaptive residual pressure bridge_id")
        _text(self.evidence_id, "adaptive residual pressure evidence_id")
        _text(self.parent_checkpoint_digest, "adaptive residual pressure parent digest")
        if int(self.tick) <= 0:
            raise ValueError("adaptive residual pressure tick must be positive")
        for name in (
            "residual_error",
            "fast_slow_conflict",
            "activity_saturation",
            "utility_gap",
            "resource_state",
        ):
            _unit(getattr(self, name), f"adaptive residual pressure {name}")
        _text(self.pressure_digest, "adaptive residual pressure digest")
        if self.pressure_digest != content_digest(self._payload_without_digest()):
            raise ValueError("adaptive residual pressure digest mismatch")
        object.__setattr__(self, "bridge_id", str(self.bridge_id).strip())
        object.__setattr__(self, "tick", int(self.tick))
        object.__setattr__(self, "evidence_id", str(self.evidence_id).strip())
        object.__setattr__(
            self,
            "parent_checkpoint_digest",
            str(self.parent_checkpoint_digest).strip(),
        )
        object.__setattr__(self, "pressure_digest", str(self.pressure_digest))

    @property
    def pressure(self) -> float:
        """Weighted unserved-capacity pressure used by the trigger."""

        return float(
            0.30 * self.residual_error
            + 0.25 * self.fast_slow_conflict
            + 0.20 * self.activity_saturation
            + 0.25 * self.utility_gap
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_RESIDUAL_GROWTH_FORMAT,
            "version": ADAPTIVE_RESIDUAL_GROWTH_VERSION,
            "kind": "pressure",
            "bridge_id": self.bridge_id,
            "tick": int(self.tick),
            "residual_error": float(self.residual_error),
            "fast_slow_conflict": float(self.fast_slow_conflict),
            "activity_saturation": float(self.activity_saturation),
            "utility_gap": float(self.utility_gap),
            "resource_state": float(self.resource_state),
            "evidence_id": self.evidence_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "pressure": self.pressure,
            "pressure_digest": self.pressure_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        bridge_id: str,
        tick: int,
        residual_error: float,
        fast_slow_conflict: float,
        activity_saturation: float,
        utility_gap: float,
        resource_state: float,
        evidence_id: str,
        parent_checkpoint_digest: str,
    ) -> AdaptiveResidualGrowthPressure:
        identity: dict[str, Any] = {
            "format": ADAPTIVE_RESIDUAL_GROWTH_FORMAT,
            "version": ADAPTIVE_RESIDUAL_GROWTH_VERSION,
            "kind": "pressure",
            "bridge_id": str(bridge_id).strip(),
            "tick": int(tick),
            "residual_error": float(residual_error),
            "fast_slow_conflict": float(fast_slow_conflict),
            "activity_saturation": float(activity_saturation),
            "utility_gap": float(utility_gap),
            "resource_state": float(resource_state),
            "evidence_id": str(evidence_id).strip(),
            "parent_checkpoint_digest": str(parent_checkpoint_digest).strip(),
        }
        return cls(
            bridge_id=identity["bridge_id"],
            tick=identity["tick"],
            residual_error=identity["residual_error"],
            fast_slow_conflict=identity["fast_slow_conflict"],
            activity_saturation=identity["activity_saturation"],
            utility_gap=identity["utility_gap"],
            resource_state=identity["resource_state"],
            evidence_id=identity["evidence_id"],
            parent_checkpoint_digest=identity["parent_checkpoint_digest"],
            pressure_digest=content_digest(identity),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdaptiveResidualGrowthPressure:
        if payload.get("format") != ADAPTIVE_RESIDUAL_GROWTH_FORMAT:
            raise ValueError("unsupported adaptive residual pressure format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_GROWTH_VERSION:
            raise ValueError("unsupported adaptive residual pressure version")
        if payload.get("kind") != "pressure":
            raise ValueError("adaptive residual payload is not a pressure observation")
        expected = content_digest(
            {key: value for key, value in payload.items() if key not in {"pressure", "pressure_digest"}}
        )
        if str(payload.get("pressure_digest", "")) != expected:
            raise ValueError("adaptive residual pressure digest mismatch")
        return cls(
            bridge_id=str(payload["bridge_id"]),
            tick=int(payload["tick"]),
            residual_error=float(payload["residual_error"]),
            fast_slow_conflict=float(payload["fast_slow_conflict"]),
            activity_saturation=float(payload["activity_saturation"]),
            utility_gap=float(payload["utility_gap"]),
            resource_state=float(payload["resource_state"]),
            evidence_id=str(payload["evidence_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            pressure_digest=str(payload["pressure_digest"]),
        )


@dataclass(frozen=True)
class AdaptiveResidualGrowthPolicy:
    """Fail-closed persistence policy for emitting a shadow proposal signal."""

    ema_rate: float = 0.25
    minimum_pressure: float = 0.70
    minimum_residual_error: float = 0.55
    minimum_fast_slow_conflict: float = 0.40
    minimum_activity_saturation: float = 0.40
    minimum_utility_gap: float = 0.35
    minimum_resource_state: float = 0.40
    required_pressure_steps: int = 3
    growth_resource_cost: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_rate) <= 1.0:
            raise ValueError("adaptive residual growth ema_rate must be in (0, 1]")
        for name in (
            "minimum_pressure",
            "minimum_residual_error",
            "minimum_fast_slow_conflict",
            "minimum_activity_saturation",
            "minimum_utility_gap",
            "minimum_resource_state",
        ):
            _unit(getattr(self, name), f"adaptive residual growth {name}")
        if int(self.required_pressure_steps) <= 0:
            raise ValueError("adaptive residual growth required_pressure_steps must be positive")
        if int(self.growth_resource_cost) <= 0:
            raise ValueError("adaptive residual growth resource cost must be positive")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdaptiveResidualGrowthPolicy:
        return cls(**dict(payload))


@dataclass(frozen=True)
class AdaptiveResidualGrowthDecision:
    """A non-mutating proposal signal bound to pressure evidence."""

    bridge_id: str
    should_propose: bool
    proposal_ordinal: int
    pressure: float
    residual_error_ema: float
    fast_slow_conflict_ema: float
    activity_saturation_ema: float
    utility_gap_ema: float
    resource_state_ema: float
    consecutive_pressure_steps: int
    required_pressure_steps: int
    structural_budget: int
    resource_cost: int
    evidence_ids: tuple[str, ...]
    parent_checkpoint_digest: str
    reasons: tuple[str, ...]
    decision_digest: str

    def __post_init__(self) -> None:
        _text(self.bridge_id, "adaptive residual decision bridge_id")
        _text(self.parent_checkpoint_digest, "adaptive residual decision parent digest")
        for name in (
            "pressure",
            "residual_error_ema",
            "fast_slow_conflict_ema",
            "activity_saturation_ema",
            "utility_gap_ema",
            "resource_state_ema",
        ):
            _unit(getattr(self, name), f"adaptive residual decision {name}")
        if min(
            int(self.proposal_ordinal),
            int(self.consecutive_pressure_steps),
            int(self.required_pressure_steps),
            int(self.structural_budget),
            int(self.resource_cost),
        ) < 0:
            raise ValueError("adaptive residual decision counters cannot be negative")
        if int(self.required_pressure_steps) <= 0 or int(self.resource_cost) <= 0:
            raise ValueError("adaptive residual decision thresholds must be positive")
        ids = tuple(str(item) for item in self.evidence_ids)
        if not ids or len(set(ids)) != len(ids) or any(not item for item in ids):
            raise ValueError("adaptive residual decision evidence_ids must be unique and non-empty")
        reasons = tuple(str(item) for item in self.reasons)
        if any(not item for item in reasons):
            raise ValueError("adaptive residual decision reasons must not be empty")
        _text(self.decision_digest, "adaptive residual decision digest")
        object.__setattr__(self, "bridge_id", str(self.bridge_id).strip())
        object.__setattr__(self, "evidence_ids", ids)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "parent_checkpoint_digest",
            str(self.parent_checkpoint_digest).strip(),
        )
        object.__setattr__(self, "decision_digest", str(self.decision_digest))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_RESIDUAL_GROWTH_FORMAT,
            "version": ADAPTIVE_RESIDUAL_GROWTH_VERSION,
            "kind": "decision",
            "bridge_id": self.bridge_id,
            "should_propose": bool(self.should_propose),
            "proposal_ordinal": int(self.proposal_ordinal),
            "pressure": float(self.pressure),
            "residual_error_ema": float(self.residual_error_ema),
            "fast_slow_conflict_ema": float(self.fast_slow_conflict_ema),
            "activity_saturation_ema": float(self.activity_saturation_ema),
            "utility_gap_ema": float(self.utility_gap_ema),
            "resource_state_ema": float(self.resource_state_ema),
            "consecutive_pressure_steps": int(self.consecutive_pressure_steps),
            "required_pressure_steps": int(self.required_pressure_steps),
            "structural_budget": int(self.structural_budget),
            "resource_cost": int(self.resource_cost),
            "evidence_ids": list(self.evidence_ids),
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "reasons": list(self.reasons),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "decision_digest": self.decision_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AdaptiveResidualGrowthDecision:
        if payload.get("format") != ADAPTIVE_RESIDUAL_GROWTH_FORMAT:
            raise ValueError("unsupported adaptive residual decision format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_GROWTH_VERSION:
            raise ValueError("unsupported adaptive residual decision version")
        if payload.get("kind") != "decision":
            raise ValueError("adaptive residual payload is not a decision")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "decision_digest"}
        )
        if str(payload.get("decision_digest", "")) != expected:
            raise ValueError("adaptive residual decision digest mismatch")
        return cls(
            bridge_id=str(payload["bridge_id"]),
            should_propose=bool(payload["should_propose"]),
            proposal_ordinal=int(payload["proposal_ordinal"]),
            pressure=float(payload["pressure"]),
            residual_error_ema=float(payload["residual_error_ema"]),
            fast_slow_conflict_ema=float(payload["fast_slow_conflict_ema"]),
            activity_saturation_ema=float(payload["activity_saturation_ema"]),
            utility_gap_ema=float(payload["utility_gap_ema"]),
            resource_state_ema=float(payload["resource_state_ema"]),
            consecutive_pressure_steps=int(payload["consecutive_pressure_steps"]),
            required_pressure_steps=int(payload["required_pressure_steps"]),
            structural_budget=int(payload["structural_budget"]),
            resource_cost=int(payload["resource_cost"]),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
            decision_digest=str(payload["decision_digest"]),
        )


class AdaptiveResidualGrowthTrigger:
    """Accumulate native pressure without mutating the bridge topology."""

    def __init__(
        self,
        *,
        bridge_id: str,
        policy: AdaptiveResidualGrowthPolicy | None = None,
        parent_checkpoint_digest: str | None = None,
    ) -> None:
        self.bridge_id = _text(bridge_id, "adaptive residual trigger bridge_id")
        self.policy = policy or AdaptiveResidualGrowthPolicy()
        self.residual_error_ema = 0.0
        self.fast_slow_conflict_ema = 0.0
        self.activity_saturation_ema = 0.0
        self.utility_gap_ema = 0.0
        self.resource_state_ema = 1.0
        self.consecutive_pressure_steps = 0
        self.proposal_count = 0
        self.observation_count = 0
        self._evidence_ids: list[str] = []
        self._parent_checkpoint_digest = (
            ""
            if parent_checkpoint_digest is None
            else _text(parent_checkpoint_digest, "adaptive residual trigger parent digest")
        )
        self._last_pressure: AdaptiveResidualGrowthPressure | None = None
        self._last_decision: AdaptiveResidualGrowthDecision | None = None

    @property
    def parent_checkpoint_digest(self) -> str:
        return self._parent_checkpoint_digest

    @property
    def last_pressure(self) -> AdaptiveResidualGrowthPressure | None:
        return self._last_pressure

    @property
    def last_decision(self) -> AdaptiveResidualGrowthDecision | None:
        return self._last_decision

    @property
    def pressure_ema(self) -> float:
        return float(
            0.30 * self.residual_error_ema
            + 0.25 * self.fast_slow_conflict_ema
            + 0.20 * self.activity_saturation_ema
            + 0.25 * self.utility_gap_ema
        )

    def observe(
        self,
        pressure: AdaptiveResidualGrowthPressure,
        *,
        structural_budget: int,
    ) -> AdaptiveResidualGrowthDecision:
        if not isinstance(pressure, AdaptiveResidualGrowthPressure):
            raise TypeError("adaptive residual trigger expects a pressure observation")
        if pressure.bridge_id != self.bridge_id:
            raise ValueError("adaptive residual pressure targets another bridge")
        budget = int(structural_budget)
        if budget < 0:
            raise ValueError("adaptive residual structural_budget cannot be negative")
        if pressure.evidence_id in self._evidence_ids:
            raise ValueError("adaptive residual pressure evidence_id was already observed")
        if self._parent_checkpoint_digest and pressure.parent_checkpoint_digest != self._parent_checkpoint_digest:
            raise ValueError("adaptive residual pressure parent checkpoint changed")
        self._parent_checkpoint_digest = pressure.parent_checkpoint_digest
        self._last_pressure = pressure
        self._evidence_ids.append(pressure.evidence_id)
        max_evidence = max(1, int(self.policy.required_pressure_steps))
        self._evidence_ids = self._evidence_ids[-max_evidence:]
        rate = float(self.policy.ema_rate)
        self.residual_error_ema = (1.0 - rate) * self.residual_error_ema + rate * pressure.residual_error
        self.fast_slow_conflict_ema = (
            (1.0 - rate) * self.fast_slow_conflict_ema + rate * pressure.fast_slow_conflict
        )
        self.activity_saturation_ema = (
            (1.0 - rate) * self.activity_saturation_ema + rate * pressure.activity_saturation
        )
        self.utility_gap_ema = (1.0 - rate) * self.utility_gap_ema + rate * pressure.utility_gap
        self.resource_state_ema = (1.0 - rate) * self.resource_state_ema + rate * pressure.resource_state
        self.observation_count += 1
        meets_pressure = bool(
            self.pressure_ema >= float(self.policy.minimum_pressure)
            and self.residual_error_ema >= float(self.policy.minimum_residual_error)
            and self.fast_slow_conflict_ema >= float(self.policy.minimum_fast_slow_conflict)
            and self.activity_saturation_ema >= float(self.policy.minimum_activity_saturation)
            and self.utility_gap_ema >= float(self.policy.minimum_utility_gap)
            and self.resource_state_ema >= float(self.policy.minimum_resource_state)
        )
        self.consecutive_pressure_steps = (
            self.consecutive_pressure_steps + 1 if meets_pressure else 0
        )
        pressure_persistence_satisfied = (
            self.consecutive_pressure_steps >= int(self.policy.required_pressure_steps)
        )
        should_propose = bool(
            meets_pressure
            and pressure_persistence_satisfied
            and budget >= int(self.policy.growth_resource_cost)
        )
        if should_propose:
            self.proposal_count += 1
            self.consecutive_pressure_steps = 0
        reasons: list[str] = []
        if not meets_pressure:
            reasons.append("pressure_below_threshold")
        if not pressure_persistence_satisfied:
            reasons.append("pressure_persistence_below_threshold")
        if budget < int(self.policy.growth_resource_cost):
            reasons.append("structural_budget_insufficient")
        if should_propose:
            reasons.append("persistent_native_pressure")
        identity: dict[str, Any] = {
            "format": ADAPTIVE_RESIDUAL_GROWTH_FORMAT,
            "version": ADAPTIVE_RESIDUAL_GROWTH_VERSION,
            "kind": "decision",
            "bridge_id": self.bridge_id,
            "should_propose": should_propose,
            "proposal_ordinal": self.proposal_count,
            "pressure": self.pressure_ema,
            "residual_error_ema": self.residual_error_ema,
            "fast_slow_conflict_ema": self.fast_slow_conflict_ema,
            "activity_saturation_ema": self.activity_saturation_ema,
            "utility_gap_ema": self.utility_gap_ema,
            "resource_state_ema": self.resource_state_ema,
            "consecutive_pressure_steps": self.consecutive_pressure_steps,
            "required_pressure_steps": int(self.policy.required_pressure_steps),
            "structural_budget": budget,
            "resource_cost": int(self.policy.growth_resource_cost),
            "evidence_ids": list(self._evidence_ids),
            "parent_checkpoint_digest": self._parent_checkpoint_digest,
            "reasons": reasons,
        }
        decision = AdaptiveResidualGrowthDecision(
            **{key: value for key, value in identity.items() if key not in {"format", "version", "kind"}},
            decision_digest=content_digest(identity),
        )
        self._last_decision = decision
        return decision

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": ADAPTIVE_RESIDUAL_GROWTH_FORMAT,
            "version": ADAPTIVE_RESIDUAL_GROWTH_VERSION,
            "kind": "trigger",
            "bridge_id": self.bridge_id,
            "policy": self.policy.to_payload(),
            "residual_error_ema": self.residual_error_ema,
            "fast_slow_conflict_ema": self.fast_slow_conflict_ema,
            "activity_saturation_ema": self.activity_saturation_ema,
            "utility_gap_ema": self.utility_gap_ema,
            "resource_state_ema": self.resource_state_ema,
            "consecutive_pressure_steps": self.consecutive_pressure_steps,
            "proposal_count": self.proposal_count,
            "observation_count": self.observation_count,
            "evidence_ids": list(self._evidence_ids),
            "parent_checkpoint_digest": self._parent_checkpoint_digest,
        }
        if self._last_decision is not None:
            payload["last_decision"] = self._last_decision.to_payload()
        if self._last_pressure is not None:
            payload["last_pressure"] = self._last_pressure.to_payload()
        return {**payload, "checkpoint_digest": content_digest(payload)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> AdaptiveResidualGrowthTrigger:
        if payload.get("format") != ADAPTIVE_RESIDUAL_GROWTH_FORMAT:
            raise ValueError("unsupported adaptive residual trigger format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_GROWTH_VERSION:
            raise ValueError("unsupported adaptive residual trigger version")
        if payload.get("kind") != "trigger":
            raise ValueError("adaptive residual payload is not a trigger checkpoint")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("adaptive residual trigger checkpoint digest mismatch")
        trigger = cls(
            bridge_id=str(payload["bridge_id"]),
            policy=AdaptiveResidualGrowthPolicy.from_payload(payload["policy"]),
            parent_checkpoint_digest=(
                None
                if not str(payload.get("parent_checkpoint_digest", ""))
                else str(payload["parent_checkpoint_digest"])
            ),
        )
        trigger.residual_error_ema = _unit(
            float(payload["residual_error_ema"]), "adaptive residual trigger residual_error_ema"
        )
        trigger.fast_slow_conflict_ema = _unit(
            float(payload["fast_slow_conflict_ema"]),
            "adaptive residual trigger fast_slow_conflict_ema",
        )
        trigger.activity_saturation_ema = _unit(
            float(payload["activity_saturation_ema"]),
            "adaptive residual trigger activity_saturation_ema",
        )
        trigger.utility_gap_ema = _unit(
            float(payload["utility_gap_ema"]), "adaptive residual trigger utility_gap_ema"
        )
        trigger.resource_state_ema = _unit(
            float(payload["resource_state_ema"]), "adaptive residual trigger resource_state_ema"
        )
        trigger.consecutive_pressure_steps = int(payload["consecutive_pressure_steps"])
        trigger.proposal_count = int(payload["proposal_count"])
        trigger.observation_count = int(payload["observation_count"])
        if min(
            trigger.consecutive_pressure_steps,
            trigger.proposal_count,
            trigger.observation_count,
        ) < 0:
            raise ValueError("adaptive residual trigger counters cannot be negative")
        trigger._evidence_ids = [str(item) for item in payload.get("evidence_ids", ())]
        if len(set(trigger._evidence_ids)) != len(trigger._evidence_ids) or any(
            not item for item in trigger._evidence_ids
        ):
            raise ValueError("adaptive residual trigger evidence_ids must be unique and non-empty")
        trigger._parent_checkpoint_digest = str(payload.get("parent_checkpoint_digest", ""))
        last_pressure = payload.get("last_pressure")
        if last_pressure is not None:
            if not isinstance(last_pressure, Mapping):
                raise ValueError("adaptive residual trigger last_pressure must be a mapping")
            trigger._last_pressure = AdaptiveResidualGrowthPressure.from_payload(last_pressure)
            if trigger._last_pressure.bridge_id != trigger.bridge_id:
                raise ValueError("adaptive residual trigger last_pressure targets another bridge")
            if (
                trigger._parent_checkpoint_digest
                and trigger._last_pressure.parent_checkpoint_digest
                != trigger._parent_checkpoint_digest
            ):
                raise ValueError("adaptive residual trigger last_pressure parent changed")
        last_decision = payload.get("last_decision")
        if last_decision is not None:
            if not isinstance(last_decision, Mapping):
                raise ValueError("adaptive residual trigger last_decision must be a mapping")
            trigger._last_decision = AdaptiveResidualGrowthDecision.from_payload(last_decision)
            if trigger._last_decision.bridge_id != trigger.bridge_id:
                raise ValueError("adaptive residual trigger last_decision targets another bridge")
            if (
                trigger._parent_checkpoint_digest
                and trigger._last_decision.parent_checkpoint_digest
                != trigger._parent_checkpoint_digest
            ):
                raise ValueError("adaptive residual trigger last_decision parent changed")
        return trigger


__all__ = [
    "ADAPTIVE_RESIDUAL_GROWTH_FORMAT",
    "ADAPTIVE_RESIDUAL_GROWTH_VERSION",
    "AdaptiveResidualGrowthDecision",
    "AdaptiveResidualGrowthPolicy",
    "AdaptiveResidualGrowthPressure",
    "AdaptiveResidualGrowthTrigger",
]
