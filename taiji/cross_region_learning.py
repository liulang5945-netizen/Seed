"""Learning-driven competition between explicit Taiji cross-region paths."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

CROSS_REGION_LEARNING_CHECKPOINT_FORMAT = "taiji-cross-region-learning-v1"


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


@dataclass(frozen=True)
class CrossRegionLearningDynamics:
    """Configurable meta-plasticity for selecting cross-region pathways."""

    ema_rate: float = 0.25
    quality_weight: float = 0.55
    transfer_weight: float = 0.25
    resource_weight: float = 0.20
    cost_weight: float = 0.15
    exploration_weight: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_rate) <= 1.0:
            raise ValueError("cross-region ema_rate must be in (0, 1]")
        for name in (
            "quality_weight",
            "transfer_weight",
            "resource_weight",
            "cost_weight",
            "exploration_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"cross-region {name} must be finite and non-negative")

    def to_payload(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CrossRegionLearningDynamics:
        return cls(**{key: float(value) for key, value in payload.items()})


@dataclass
class CrossRegionRouteState:
    """Checkpointable evidence state for one explicit cross-region connection."""

    connection_id: str
    resource_cost: float
    prediction_error: float = 1.0
    holdout_transfer: float = 0.0
    resource_state: float = 1.0
    evidence_count: int = 0
    selection_count: int = 0

    def __post_init__(self) -> None:
        if not self.connection_id:
            raise ValueError("cross-region connection_id must not be empty")
        if not math.isfinite(float(self.resource_cost)) or float(self.resource_cost) <= 0.0:
            raise ValueError("cross-region resource_cost must be positive")
        self.prediction_error = _unit(self.prediction_error, "cross-region prediction_error")
        self.holdout_transfer = _unit(self.holdout_transfer, "cross-region holdout_transfer")
        self.resource_state = _unit(self.resource_state, "cross-region resource_state")
        if int(self.evidence_count) < 0 or int(self.selection_count) < 0:
            raise ValueError("cross-region evidence counters cannot be negative")
        self.evidence_count = int(self.evidence_count)
        self.selection_count = int(self.selection_count)

    def to_payload(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "resource_cost": self.resource_cost,
            "prediction_error": self.prediction_error,
            "holdout_transfer": self.holdout_transfer,
            "resource_state": self.resource_state,
            "evidence_count": self.evidence_count,
            "selection_count": self.selection_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CrossRegionRouteState:
        return cls(
            connection_id=str(payload["connection_id"]),
            resource_cost=float(payload["resource_cost"]),
            prediction_error=float(payload.get("prediction_error", 1.0)),
            holdout_transfer=float(payload.get("holdout_transfer", 0.0)),
            resource_state=float(payload.get("resource_state", 1.0)),
            evidence_count=int(payload.get("evidence_count", 0)),
            selection_count=int(payload.get("selection_count", 0)),
        )


class CrossRegionCooperationLearner:
    """Select pathways from local evidence instead of an action or intent table.

    The learner is deliberately route-level rather than token-level.  Each
    connection owns an exponentially smoothed estimate of three observable
    signals: prediction error (lower is better), holdout transfer (higher is
    better), and resource state (higher is better).  Selection combines those
    estimates with an explicit resource-cost penalty and an uncertainty bonus,
    so an untested route can be explored while a repeatedly useful route wins
    on unseen patterns.  All state is serializable and the dynamics are
    configurable; no action vocabulary is embedded here.
    """

    def __init__(
        self,
        *,
        dynamics: CrossRegionLearningDynamics | None = None,
    ) -> None:
        self.dynamics = dynamics or CrossRegionLearningDynamics()
        self._routes: dict[str, CrossRegionRouteState] = {}
        self.total_evidence = 0

    @property
    def routes(self) -> tuple[CrossRegionRouteState, ...]:
        return tuple(self._routes.values())

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(self._routes)

    def register_connection(self, connection_id: str, *, resource_cost: float) -> None:
        key = str(connection_id)
        if not key:
            raise ValueError("cross-region connection_id must not be empty")
        cost = float(resource_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("cross-region resource_cost must be positive")
        existing = self._routes.get(key)
        if existing is not None:
            if not math.isclose(existing.resource_cost, cost):
                raise ValueError("cross-region resource cost changed for an existing route")
            return
        self._routes[key] = CrossRegionRouteState(key, cost)

    def unregister_connection(self, connection_id: str) -> None:
        self._routes.pop(str(connection_id), None)

    def fork_connection(self, source_id: str, target_id: str) -> None:
        """Copy route evidence to a child projection created by region split."""

        source = self._route(source_id)
        child_key = str(target_id)
        if not child_key:
            raise ValueError("cross-region child connection_id must not be empty")
        if child_key in self._routes:
            raise ValueError("cross-region child connection already exists")
        self._routes[child_key] = CrossRegionRouteState(
            connection_id=child_key,
            resource_cost=source.resource_cost,
            prediction_error=source.prediction_error,
            holdout_transfer=source.holdout_transfer,
            resource_state=source.resource_state,
            evidence_count=source.evidence_count,
            selection_count=source.selection_count,
        )

    def merge_connections(
        self,
        source_ids: Sequence[str],
        target_id: str,
        *,
        resource_cost: float,
    ) -> None:
        """Aggregate route evidence when several projections become one route."""

        old_ids = tuple(str(item) for item in source_ids)
        if not old_ids or len(set(old_ids)) != len(old_ids):
            raise ValueError("cross-region merge requires unique source routes")
        routes = tuple(self._route(connection_id) for connection_id in old_ids)
        cost = float(resource_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("cross-region merged resource_cost must be positive")
        total = sum(route.evidence_count for route in routes)
        weights = (
            tuple(route.evidence_count for route in routes)
            if total > 0
            else tuple(1 for _ in routes)
        )
        denominator = float(sum(weights))
        self._routes[str(target_id)] = CrossRegionRouteState(
            connection_id=str(target_id),
            resource_cost=cost,
            prediction_error=sum(route.prediction_error * weight for route, weight in zip(routes, weights, strict=True))
            / denominator,
            holdout_transfer=sum(route.holdout_transfer * weight for route, weight in zip(routes, weights, strict=True))
            / denominator,
            resource_state=sum(route.resource_state * weight for route, weight in zip(routes, weights, strict=True))
            / denominator,
            evidence_count=sum(route.evidence_count for route in routes),
            selection_count=sum(route.selection_count for route in routes),
        )
        for connection_id in old_ids:
            if connection_id != str(target_id):
                self._routes.pop(connection_id, None)

    def resource_cost(self, connection_id: str) -> float:
        """Return the declared structural cost for one registered route."""

        return float(self._route(connection_id).resource_cost)

    def route_state(self, connection_id: str) -> CrossRegionRouteState:
        """Return the mutable evidence state owned by one registered route."""

        return self._route(connection_id)

    def _route(self, connection_id: str) -> CrossRegionRouteState:
        try:
            return self._routes[str(connection_id)]
        except KeyError as exc:
            raise ValueError(f"unknown cross-region route: {connection_id}") from exc

    def observe(
        self,
        connection_id: str,
        *,
        prediction_error: float,
        holdout_transfer: float | None = None,
        resource_state: float,
        selected: bool = True,
    ) -> float:
        """Update one route from outcome evidence and return its new score."""

        route = self._route(connection_id)
        error = _unit(prediction_error, "cross-region prediction_error")
        resource = _unit(resource_state, "cross-region resource_state")
        rate = float(self.dynamics.ema_rate)
        route.prediction_error = (1.0 - rate) * route.prediction_error + rate * error
        if holdout_transfer is not None:
            transfer = _unit(holdout_transfer, "cross-region holdout_transfer")
            route.holdout_transfer = (1.0 - rate) * route.holdout_transfer + rate * transfer
        route.resource_state = (1.0 - rate) * route.resource_state + rate * resource
        route.evidence_count += 1
        if selected:
            route.selection_count += 1
        self.total_evidence += 1
        return self.score(route.connection_id, resource_budget=1.0)

    def score(self, connection_id: str, *, resource_budget: float = 1.0) -> float:
        """Return a deterministic utility score under the current budget."""

        budget = float(resource_budget)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("cross-region resource_budget must be positive")
        route = self._route(connection_id)
        if route.resource_cost > budget:
            return float("-inf")
        dynamics = self.dynamics
        quality = route.holdout_transfer * (1.0 - route.prediction_error)
        cost_pressure = min(1.0, route.resource_cost / budget)
        uncertainty = math.sqrt(
            math.log1p(float(self.total_evidence)) / float(route.evidence_count + 1)
        )
        return float(
            dynamics.quality_weight * quality
            + dynamics.transfer_weight * route.holdout_transfer
            + dynamics.resource_weight * route.resource_state
            - dynamics.cost_weight * cost_pressure
            + dynamics.exploration_weight * uncertainty
        )

    def select(
        self,
        connection_ids: Sequence[str] | None = None,
        *,
        resource_budget: float = 1.0,
        max_connections: int = 1,
    ) -> tuple[str, ...]:
        """Select the highest-scoring feasible routes with stable tie-breaking."""

        if int(max_connections) <= 0:
            raise ValueError("cross-region max_connections must be positive")
        candidates = self.route_ids if connection_ids is None else tuple(str(item) for item in connection_ids)
        if len(set(candidates)) != len(candidates):
            raise ValueError("cross-region selection cannot contain duplicate routes")
        for connection_id in candidates:
            self._route(connection_id)
        ranked = sorted(
            (
                connection_id
                for connection_id in candidates
                if self._route(connection_id).resource_cost <= float(resource_budget)
            ),
            key=lambda connection_id: (-self.score(connection_id, resource_budget=resource_budget), connection_id),
        )
        return tuple(ranked[: int(max_connections)])

    def scores(
        self,
        connection_ids: Sequence[str] | None = None,
        *,
        resource_budget: float = 1.0,
    ) -> dict[str, float]:
        candidates = self.route_ids if connection_ids is None else tuple(str(item) for item in connection_ids)
        return {
            connection_id: self.score(connection_id, resource_budget=resource_budget)
            for connection_id in candidates
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CROSS_REGION_LEARNING_CHECKPOINT_FORMAT,
            "dynamics": self.dynamics.to_payload(),
            "total_evidence": self.total_evidence,
            "routes": {
                connection_id: route.to_payload()
                for connection_id, route in self._routes.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CrossRegionCooperationLearner:
        if payload.get("format") != CROSS_REGION_LEARNING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported cross-region learning checkpoint format")
        dynamics_payload = payload.get("dynamics", {})
        routes_payload = payload.get("routes", {})
        if not isinstance(dynamics_payload, Mapping) or not isinstance(routes_payload, Mapping):
            raise ValueError("cross-region learning checkpoint fields must be mappings")
        learner = cls(dynamics=CrossRegionLearningDynamics.from_payload(dynamics_payload))
        learner._routes = {
            str(connection_id): CrossRegionRouteState.from_payload(route_payload)
            for connection_id, route_payload in routes_payload.items()
        }
        if set(learner._routes) != set(str(key) for key in routes_payload):
            raise ValueError("cross-region learning route identities do not match")
        learner.total_evidence = int(payload.get("total_evidence", 0))
        if learner.total_evidence < 0:
            raise ValueError("cross-region total_evidence cannot be negative")
        return learner
