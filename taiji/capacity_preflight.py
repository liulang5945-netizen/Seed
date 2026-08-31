"""Fixed-capacity comparison and fail-closed structural-growth preflight."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any

from .contracts import WorldTransition
from .internalization import content_digest
from .structural_continuation import (
    StructuralRegionCapacityPressure,
    measure_structural_region_capacity_pressure,
)
from .world_evolution import NativeWorldPredictionTrainer
from .world_learning import WorldSchema

CAPACITY_PREFLIGHT_FORMAT = "taiji-native-capacity-preflight-v1"
CAPACITY_PREFLIGHT_VERSION = 1
CAPACITY_PREFLIGHT_MANIFEST_REVISION = "taiji-w7-e3-4-fixed-capacity-preflight-v1"


def _finite_nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _unit(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _transitions(
    transitions: Iterable[WorldTransition],
    *,
    partition: str,
) -> tuple[WorldTransition, ...]:
    items = tuple(transitions)
    if not items:
        raise ValueError(f"{partition} transitions must contain at least one item")
    if any(not isinstance(item, WorldTransition) for item in items):
        raise TypeError(f"{partition} transitions must contain WorldTransition values")
    action_ids = tuple(item.action.action_id for item in items)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError(f"{partition} transitions cannot contain duplicate action IDs")
    return tuple(sorted(items, key=lambda item: item.action.action_id))


@dataclass(frozen=True)
class FixedCapacitySeedResult:
    """One independent fixed-capacity learner comparison."""

    seed: int
    hidden_dim: int
    parameter_count: int
    frozen_holdout_error: float
    replay_only_holdout_error: float
    native_holdout_error: float
    frozen_retention_error: float
    native_retention_error: float
    native_training_loss: float
    admitted: bool

    def __post_init__(self) -> None:
        if int(self.hidden_dim) <= 0 or int(self.parameter_count) <= 0:
            raise ValueError("fixed-capacity learner dimensions must be positive")
        for name in (
            "frozen_holdout_error",
            "replay_only_holdout_error",
            "native_holdout_error",
            "frozen_retention_error",
            "native_retention_error",
            "native_training_loss",
        ):
            _finite_nonnegative(getattr(self, name), f"fixed-capacity {name}")

    @property
    def holdout_gain(self) -> float:
        return self.frozen_holdout_error - self.native_holdout_error

    @property
    def retention_regression(self) -> float:
        return max(0.0, self.native_retention_error - self.frozen_retention_error)

    def to_payload(self) -> dict[str, Any]:
        payload = {key: value for key, value in asdict(self).items()}
        payload.update(
            {
                "holdout_gain": self.holdout_gain,
                "retention_regression": self.retention_regression,
            }
        )
        return payload


@dataclass(frozen=True)
class CapacityGrowthTriggerPolicy:
    """Evidence policy that permits a later structural proposal."""

    ema_rate: float = 0.5
    maximum_holdout_error: float = 0.75
    maximum_retention_regression: float = 0.05
    minimum_capacity_pressure: float = 0.80
    required_failure_steps: int = 3
    resource_cost: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_rate) <= 1.0:
            raise ValueError("capacity preflight ema_rate must be in (0, 1]")
        _finite_nonnegative(self.maximum_holdout_error, "capacity preflight maximum_holdout_error")
        _unit(
            self.maximum_retention_regression,
            "capacity preflight maximum_retention_regression",
        )
        _unit(self.minimum_capacity_pressure, "capacity preflight minimum_capacity_pressure")
        if int(self.required_failure_steps) <= 0:
            raise ValueError("capacity preflight required_failure_steps must be positive")
        if int(self.resource_cost) <= 0:
            raise ValueError("capacity preflight resource_cost must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapacityGrowthTriggerPolicy:
        return cls(**dict(payload))


@dataclass(frozen=True)
class CapacityGrowthTriggerDecision:
    """A content-addressed decision; it never mutates topology."""

    region_id: str
    should_propose: bool
    capacity_pressure: float
    residual_error: float
    residual_error_ema: float
    retention_regression: float
    consecutive_failure_steps: int
    required_failure_steps: int
    minimum_capacity_pressure: float
    maximum_holdout_error: float
    maximum_retention_regression: float
    resource_cost: int
    structural_budget: int
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    decision_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(str(item) for item in self.evidence_ids))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        if not str(self.region_id):
            raise ValueError("capacity trigger region_id must not be empty")
        for name in (
            "capacity_pressure",
            "retention_regression",
            "minimum_capacity_pressure",
            "maximum_retention_regression",
        ):
            _unit(getattr(self, name), f"capacity trigger {name}")
        for name in ("residual_error", "residual_error_ema", "maximum_holdout_error"):
            _finite_nonnegative(getattr(self, name), f"capacity trigger {name}")
        if min(
            int(self.consecutive_failure_steps),
            int(self.required_failure_steps),
            int(self.resource_cost),
            int(self.structural_budget),
        ) < 0:
            raise ValueError("capacity trigger counters and budgets cannot be negative")
        if int(self.required_failure_steps) == 0 or int(self.resource_cost) == 0:
            raise ValueError("capacity trigger required steps and resource cost must be positive")
        if not self.evidence_ids or any(not str(item) for item in self.evidence_ids):
            raise ValueError("capacity trigger evidence_ids must not be empty")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("capacity trigger evidence_ids must be unique")
        if not str(self.decision_digest):
            raise ValueError("capacity trigger decision_digest must not be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CAPACITY_PREFLIGHT_FORMAT,
            "region_id": self.region_id,
            "should_propose": self.should_propose,
            "capacity_pressure": self.capacity_pressure,
            "residual_error": self.residual_error,
            "residual_error_ema": self.residual_error_ema,
            "retention_regression": self.retention_regression,
            "consecutive_failure_steps": self.consecutive_failure_steps,
            "required_failure_steps": self.required_failure_steps,
            "minimum_capacity_pressure": self.minimum_capacity_pressure,
            "maximum_holdout_error": self.maximum_holdout_error,
            "maximum_retention_regression": self.maximum_retention_regression,
            "resource_cost": self.resource_cost,
            "structural_budget": self.structural_budget,
            "evidence_ids": list(self.evidence_ids),
            "reasons": list(self.reasons),
            "decision_digest": self.decision_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapacityGrowthTriggerDecision:
        if payload.get("format") != CAPACITY_PREFLIGHT_FORMAT:
            raise ValueError("unsupported capacity trigger format")
        identity = {
            key: value for key, value in payload.items() if key != "decision_digest"
        }
        expected = content_digest(identity)
        if str(payload.get("decision_digest", "")) != expected:
            raise ValueError("capacity trigger decision digest mismatch")
        return cls(
            region_id=str(payload["region_id"]),
            should_propose=bool(payload["should_propose"]),
            capacity_pressure=float(payload["capacity_pressure"]),
            residual_error=float(payload["residual_error"]),
            residual_error_ema=float(payload["residual_error_ema"]),
            retention_regression=float(payload["retention_regression"]),
            consecutive_failure_steps=int(payload["consecutive_failure_steps"]),
            required_failure_steps=int(payload["required_failure_steps"]),
            minimum_capacity_pressure=float(payload["minimum_capacity_pressure"]),
            maximum_holdout_error=float(payload["maximum_holdout_error"]),
            maximum_retention_regression=float(payload["maximum_retention_regression"]),
            resource_cost=int(payload["resource_cost"]),
            structural_budget=int(payload["structural_budget"]),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
            decision_digest=str(payload["decision_digest"]),
        )


class CapacityGrowthTrigger:
    """Accumulate independent fixed-capacity failures before proposal permission."""

    def __init__(
        self,
        *,
        region_id: str,
        policy: CapacityGrowthTriggerPolicy | None = None,
    ) -> None:
        self.region_id = str(region_id).strip()
        if not self.region_id:
            raise ValueError("capacity trigger region_id must not be empty")
        self.policy = policy or CapacityGrowthTriggerPolicy()
        self.residual_error_ema = 0.0
        self.consecutive_failure_steps = 0
        self.observation_count = 0

    def observe(
        self,
        *,
        residual_error: float,
        retention_regression: float,
        capacity_pressure: float,
        structural_budget: int,
        evidence_ids: Sequence[str],
    ) -> CapacityGrowthTriggerDecision:
        residual = _finite_nonnegative(residual_error, "capacity trigger residual_error")
        retention = _unit(retention_regression, "capacity trigger retention_regression")
        pressure = _unit(capacity_pressure, "capacity trigger capacity_pressure")
        budget = int(structural_budget)
        if budget < 0:
            raise ValueError("capacity trigger structural_budget cannot be negative")
        ids = tuple(str(item) for item in evidence_ids)
        if not ids or any(not item for item in ids):
            raise ValueError("capacity trigger evidence_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("capacity trigger evidence_ids must be unique")

        rate = float(self.policy.ema_rate)
        self.residual_error_ema = (
            (1.0 - rate) * self.residual_error_ema + rate * residual
        )
        self.observation_count += 1
        failure = bool(
            self.residual_error_ema > float(self.policy.maximum_holdout_error)
            and retention <= float(self.policy.maximum_retention_regression)
        )
        self.consecutive_failure_steps = (
            self.consecutive_failure_steps + 1 if failure else 0
        )
        should_propose = bool(
            failure
            and self.consecutive_failure_steps >= int(self.policy.required_failure_steps)
            and pressure >= float(self.policy.minimum_capacity_pressure)
            and budget >= int(self.policy.resource_cost)
        )
        reasons: list[str] = []
        if not failure:
            reasons.append("fixed_capacity_not_persistently_failing")
        if self.consecutive_failure_steps < int(self.policy.required_failure_steps):
            reasons.append("failure_persistence_below_threshold")
        if pressure < float(self.policy.minimum_capacity_pressure):
            reasons.append("capacity_pressure_below_threshold")
        if retention > float(self.policy.maximum_retention_regression):
            reasons.append("retention_regression_above_threshold")
        if budget < int(self.policy.resource_cost):
            reasons.append("structural_budget_insufficient")
        identity = {
            "format": CAPACITY_PREFLIGHT_FORMAT,
            "region_id": self.region_id,
            "should_propose": should_propose,
            "capacity_pressure": pressure,
            "residual_error": residual,
            "residual_error_ema": self.residual_error_ema,
            "retention_regression": retention,
            "consecutive_failure_steps": self.consecutive_failure_steps,
            "required_failure_steps": int(self.policy.required_failure_steps),
            "minimum_capacity_pressure": float(self.policy.minimum_capacity_pressure),
            "maximum_holdout_error": float(self.policy.maximum_holdout_error),
            "maximum_retention_regression": float(self.policy.maximum_retention_regression),
            "resource_cost": int(self.policy.resource_cost),
            "structural_budget": budget,
            "evidence_ids": list(ids),
            "reasons": reasons,
        }
        return CapacityGrowthTriggerDecision(
            **{key: value for key, value in identity.items() if key != "format"},
            decision_digest=content_digest(identity),
        )

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": CAPACITY_PREFLIGHT_FORMAT,
            "version": CAPACITY_PREFLIGHT_VERSION,
            "region_id": self.region_id,
            "policy": self.policy.to_payload(),
            "residual_error_ema": self.residual_error_ema,
            "consecutive_failure_steps": self.consecutive_failure_steps,
            "observation_count": self.observation_count,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> CapacityGrowthTrigger:
        if payload.get("format") != CAPACITY_PREFLIGHT_FORMAT:
            raise ValueError("unsupported capacity trigger checkpoint format")
        if int(payload.get("version", -1)) != CAPACITY_PREFLIGHT_VERSION:
            raise ValueError("unsupported capacity trigger checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("capacity trigger checkpoint digest mismatch")
        trigger = cls(
            region_id=str(payload["region_id"]),
            policy=CapacityGrowthTriggerPolicy.from_payload(payload["policy"]),
        )
        trigger.residual_error_ema = _finite_nonnegative(
            float(payload.get("residual_error_ema", 0.0)),
            "capacity trigger residual_error_ema",
        )
        trigger.consecutive_failure_steps = int(payload.get("consecutive_failure_steps", 0))
        trigger.observation_count = int(payload.get("observation_count", 0))
        if min(trigger.consecutive_failure_steps, trigger.observation_count) < 0:
            raise ValueError("capacity trigger counters cannot be negative")
        return trigger


@dataclass(frozen=True)
class NativeCapacityPreflightReport:
    """Multi-seed evidence used to decide whether growth is even eligible."""

    dataset_digest: str
    schema_digest: str
    manifest_revision: str
    region_id: str
    hidden_dim: int
    capacity_limit: int
    parameter_count: int
    seed_results: tuple[FixedCapacitySeedResult, ...]
    mean_frozen_holdout_error: float
    mean_replay_only_holdout_error: float
    mean_native_holdout_error: float
    holdout_error_std: float
    mean_frozen_retention_error: float
    mean_native_retention_error: float
    maximum_retention_regression: float
    capacity_pressure: StructuralRegionCapacityPressure
    trigger_decision: CapacityGrowthTriggerDecision
    fixed_capacity_adequate: bool

    def __post_init__(self) -> None:
        if not str(self.dataset_digest) or not str(self.schema_digest):
            raise ValueError("capacity preflight dataset and schema digests are required")
        if not str(self.manifest_revision) or not str(self.region_id):
            raise ValueError("capacity preflight identity fields are required")
        if int(self.hidden_dim) <= 0 or int(self.capacity_limit) <= 0:
            raise ValueError("capacity preflight dimensions must be positive")
        if not self.seed_results:
            raise ValueError("capacity preflight requires seed results")
        for name in (
            "mean_frozen_holdout_error",
            "mean_replay_only_holdout_error",
            "mean_native_holdout_error",
            "holdout_error_std",
            "mean_frozen_retention_error",
            "mean_native_retention_error",
            "maximum_retention_regression",
        ):
            _finite_nonnegative(getattr(self, name), f"capacity preflight {name}")

    @property
    def native_holdout_gain(self) -> float:
        return self.mean_frozen_holdout_error - self.mean_native_holdout_error

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CAPACITY_PREFLIGHT_FORMAT,
            "dataset_digest": self.dataset_digest,
            "schema_digest": self.schema_digest,
            "manifest_revision": self.manifest_revision,
            "region_id": self.region_id,
            "hidden_dim": self.hidden_dim,
            "capacity_limit": self.capacity_limit,
            "parameter_count": self.parameter_count,
            "seed_results": [item.to_payload() for item in self.seed_results],
            "mean_frozen_holdout_error": self.mean_frozen_holdout_error,
            "mean_replay_only_holdout_error": self.mean_replay_only_holdout_error,
            "mean_native_holdout_error": self.mean_native_holdout_error,
            "native_holdout_gain": self.native_holdout_gain,
            "holdout_error_std": self.holdout_error_std,
            "mean_frozen_retention_error": self.mean_frozen_retention_error,
            "mean_native_retention_error": self.mean_native_retention_error,
            "maximum_retention_regression": self.maximum_retention_regression,
            "capacity_pressure": self.capacity_pressure.to_payload(),
            "trigger_decision": self.trigger_decision.to_payload(),
            "fixed_capacity_adequate": self.fixed_capacity_adequate,
        }


class NativeFixedCapacityPreflight:
    """Compare fixed-capacity local learners and accumulate growth evidence."""

    def __init__(
        self,
        schema: WorldSchema,
        *,
        hidden_dim: int = 32,
        seeds: tuple[int, ...] = (11, 29, 47),
        epochs: int = 350,
        learning_rate: float = 0.01,
        capacity_limit: int | None = None,
        region_id: str = "world.local",
        structural_budget: int = 1,
        trigger_policy: CapacityGrowthTriggerPolicy | None = None,
        manifest_revision: str = CAPACITY_PREFLIGHT_MANIFEST_REVISION,
    ) -> None:
        if not isinstance(schema, WorldSchema):
            raise TypeError("capacity preflight schema must be WorldSchema")
        if int(hidden_dim) <= 0 or int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("capacity preflight learner settings must be positive")
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("capacity preflight seeds must be non-empty and unique")
        limit = int(hidden_dim if capacity_limit is None else capacity_limit)
        if limit <= 0:
            raise ValueError("capacity preflight capacity_limit must be positive")
        if int(structural_budget) < 0:
            raise ValueError("capacity preflight structural_budget cannot be negative")
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.seeds = normalized_seeds
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.capacity_limit = limit
        self.region_id = str(region_id).strip()
        if not self.region_id:
            raise ValueError("capacity preflight region_id must not be empty")
        self.structural_budget = int(structural_budget)
        self.manifest_revision = str(manifest_revision).strip()
        if not self.manifest_revision:
            raise ValueError("capacity preflight manifest_revision cannot be empty")
        self.trigger = CapacityGrowthTrigger(
            region_id=self.region_id,
            policy=trigger_policy,
        )

    @staticmethod
    def _parameter_count(trainer: NativeWorldPredictionTrainer) -> int:
        return sum(int(parameter.numel()) for parameter in trainer.learner.parameters())

    def compare(
        self,
        train_transitions: Iterable[WorldTransition],
        *,
        holdout_transitions: Iterable[WorldTransition],
        retention_transitions: Iterable[WorldTransition],
    ) -> NativeCapacityPreflightReport:
        train = _transitions(train_transitions, partition="train")
        holdout = _transitions(holdout_transitions, partition="holdout")
        retention = _transitions(retention_transitions, partition="retention")
        seen: set[str] = set()
        for items in (train, holdout, retention):
            ids = {item.action.action_id for item in items}
            if seen.intersection(ids):
                raise ValueError("capacity preflight transition partitions must be disjoint")
            seen.update(ids)
        dataset_digest = content_digest(
            {
                "schema": self.schema.payload(),
                "train": [item.to_payload() for item in train],
                "holdout": [item.to_payload() for item in holdout],
                "retention": [item.to_payload() for item in retention],
                "hidden_dim": self.hidden_dim,
                "capacity_limit": self.capacity_limit,
                "seeds": list(self.seeds),
            }
        )
        results: list[FixedCapacitySeedResult] = []
        for seed in self.seeds:
            trainer = NativeWorldPredictionTrainer(
                self.schema,
                hidden_dim=self.hidden_dim,
                seed=seed,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
            )
            report = trainer.consolidate(
                train,
                holdout_transitions=holdout,
                retention_transitions=retention,
            )
            results.append(
                FixedCapacitySeedResult(
                    seed=seed,
                    hidden_dim=self.hidden_dim,
                    parameter_count=self._parameter_count(trainer),
                    frozen_holdout_error=report.frozen_holdout_error,
                    replay_only_holdout_error=report.replay_only_holdout_error,
                    native_holdout_error=report.native_holdout_error,
                    frozen_retention_error=report.frozen_retention_error,
                    native_retention_error=report.native_retention_error,
                    native_training_loss=report.native_training_loss,
                    admitted=report.admitted,
                )
            )
        frozen_holdout = fmean(item.frozen_holdout_error for item in results)
        replay_holdout = fmean(item.replay_only_holdout_error for item in results)
        native_holdout = fmean(item.native_holdout_error for item in results)
        frozen_retention = fmean(item.frozen_retention_error for item in results)
        native_retention = fmean(item.native_retention_error for item in results)
        maximum_retention_regression = max(item.retention_regression for item in results)
        pressure = measure_structural_region_capacity_pressure(
            region_id=self.region_id,
            unit_count=self.hidden_dim,
            capacity_limit=self.capacity_limit,
            pending_candidate_count=0,
            reserved_resource_cost=0,
            structural_budget=self.structural_budget,
        )
        decision = self.trigger.observe(
            residual_error=native_holdout,
            retention_regression=maximum_retention_regression,
            capacity_pressure=pressure.pressure,
            structural_budget=self.structural_budget,
            evidence_ids=(
                f"capacity-dataset:{dataset_digest}",
                *(f"capacity-seed:{dataset_digest}:{seed}" for seed in self.seeds),
            ),
        )
        schema_digest = content_digest(self.schema.payload())
        return NativeCapacityPreflightReport(
            dataset_digest=dataset_digest,
            schema_digest=schema_digest,
            manifest_revision=self.manifest_revision,
            region_id=self.region_id,
            hidden_dim=self.hidden_dim,
            capacity_limit=self.capacity_limit,
            parameter_count=results[0].parameter_count,
            seed_results=tuple(results),
            mean_frozen_holdout_error=frozen_holdout,
            mean_replay_only_holdout_error=replay_holdout,
            mean_native_holdout_error=native_holdout,
            holdout_error_std=pstdev(item.native_holdout_error for item in results),
            mean_frozen_retention_error=frozen_retention,
            mean_native_retention_error=native_retention,
            maximum_retention_regression=maximum_retention_regression,
            capacity_pressure=pressure,
            trigger_decision=decision,
            fixed_capacity_adequate=not decision.should_propose,
        )

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": CAPACITY_PREFLIGHT_FORMAT,
            "version": CAPACITY_PREFLIGHT_VERSION,
            "schema": self.schema.payload(),
            "hidden_dim": self.hidden_dim,
            "seeds": list(self.seeds),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "capacity_limit": self.capacity_limit,
            "region_id": self.region_id,
            "structural_budget": self.structural_budget,
            "manifest_revision": self.manifest_revision,
            "trigger": self.trigger.checkpoint(),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeFixedCapacityPreflight:
        if payload.get("format") != CAPACITY_PREFLIGHT_FORMAT:
            raise ValueError("unsupported capacity preflight checkpoint format")
        if int(payload.get("version", -1)) != CAPACITY_PREFLIGHT_VERSION:
            raise ValueError("unsupported capacity preflight checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("capacity preflight checkpoint digest mismatch")
        schema = WorldSchema.from_payload(dict(payload["schema"]))
        trigger = CapacityGrowthTrigger.from_checkpoint(payload["trigger"])
        preflight = cls(
            schema,
            hidden_dim=int(payload["hidden_dim"]),
            seeds=tuple(int(item) for item in payload["seeds"]),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            capacity_limit=int(payload["capacity_limit"]),
            region_id=str(payload["region_id"]),
            structural_budget=int(payload["structural_budget"]),
            trigger_policy=trigger.policy,
            manifest_revision=str(payload["manifest_revision"]),
        )
        if trigger.region_id != preflight.region_id:
            raise ValueError("capacity preflight trigger region drift")
        preflight.trigger = trigger
        return preflight


__all__ = [
    "CAPACITY_PREFLIGHT_FORMAT",
    "CAPACITY_PREFLIGHT_MANIFEST_REVISION",
    "CAPACITY_PREFLIGHT_VERSION",
    "CapacityGrowthTrigger",
    "CapacityGrowthTriggerDecision",
    "CapacityGrowthTriggerPolicy",
    "FixedCapacitySeedResult",
    "NativeCapacityPreflightReport",
    "NativeFixedCapacityPreflight",
]
