"""Same-parent K capability adapter boundary for the R6 preflight.

This module is intentionally a checkpoint/lineage contract only.  It does not
train a learner, dispatch a default runtime action, or create a structural
candidate.  The adapter proves that a future Taiji-owned K continuation can
bind a K3 dependency witness to one parent, stage a content-addressed trial,
fresh-restore it, and roll the trial back without losing the observed
dependency evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest
from .outcome_dependency import OutcomeDependencyProjection

TAIJI_K_CONTINUAL_ADAPTER_FORMAT = "taiji-k-continual-adapter-v1"
TAIJI_K_CONTINUAL_ADAPTER_VERSION = 1
TAIJI_K_ADAPTER_EXCHANGE_FORMAT = "taiji-k-adapter-exchange-v1"
TAIJI_K_ADAPTER_EXCHANGE_VERSION = 1


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _tags(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_text(value, f"{name} item") for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique and non-empty")
    return normalized


@dataclass(frozen=True)
class KAdapterInput:
    """Typed input crossing the native K adapter boundary."""

    episode_id: str
    parent_checkpoint_digest: str
    observation_digest: str
    world_digest: str
    goal_digest: str
    content_plan_digest: str
    source_manifest_digest: str
    tick: int
    format: str = TAIJI_K_ADAPTER_EXCHANGE_FORMAT
    version: int = TAIJI_K_ADAPTER_EXCHANGE_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_ADAPTER_EXCHANGE_FORMAT:
            raise ValueError("unsupported K adapter input format")
        if int(self.version) != TAIJI_K_ADAPTER_EXCHANGE_VERSION:
            raise ValueError("unsupported K adapter input version")
        object.__setattr__(self, "episode_id", _text(self.episode_id, "episode_id"))
        for field_name in (
            "parent_checkpoint_digest",
            "observation_digest",
            "world_digest",
            "goal_digest",
            "content_plan_digest",
            "source_manifest_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        if int(self.tick) < 0:
            raise ValueError("K adapter input tick cannot be negative")

    @property
    def input_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": int(self.version),
            "kind": "input",
            "episode_id": self.episode_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "observation_digest": self.observation_digest,
            "world_digest": self.world_digest,
            "goal_digest": self.goal_digest,
            "content_plan_digest": self.content_plan_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "tick": int(self.tick),
        }
        if include_digest:
            payload["input_digest"] = self.input_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KAdapterInput:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            episode_id=str(payload["episode_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            observation_digest=str(payload["observation_digest"]),
            world_digest=str(payload["world_digest"]),
            goal_digest=str(payload["goal_digest"]),
            content_plan_digest=str(payload["content_plan_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            tick=int(payload["tick"]),
        )
        if str(payload.get("input_digest", "")) != item.input_digest:
            raise ValueError("K adapter input digest mismatch")
        return item


@dataclass(frozen=True)
class KAdapterOutput:
    """Typed output and outcome lineage emitted by the native K boundary."""

    parent_checkpoint_digest: str
    input_digest: str
    action_digest: str
    outcome_signature: str
    dependency_digest: str
    dependency_projection_digest: str
    success: bool
    lineage: tuple[str, ...]
    format: str = TAIJI_K_ADAPTER_EXCHANGE_FORMAT
    version: int = TAIJI_K_ADAPTER_EXCHANGE_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_ADAPTER_EXCHANGE_FORMAT:
            raise ValueError("unsupported K adapter output format")
        if int(self.version) != TAIJI_K_ADAPTER_EXCHANGE_VERSION:
            raise ValueError("unsupported K adapter output version")
        for field_name in (
            "parent_checkpoint_digest",
            "input_digest",
            "action_digest",
            "outcome_signature",
            "dependency_digest",
            "dependency_projection_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        normalized_lineage = _tags(self.lineage, "K adapter output lineage")
        if self.input_digest not in normalized_lineage:
            raise ValueError("K adapter output lineage omits input digest")
        if self.dependency_digest not in normalized_lineage:
            raise ValueError("K adapter output lineage omits dependency digest")
        object.__setattr__(self, "lineage", normalized_lineage)

    @property
    def output_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": int(self.version),
            "kind": "output",
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "input_digest": self.input_digest,
            "action_digest": self.action_digest,
            "outcome_signature": self.outcome_signature,
            "dependency_digest": self.dependency_digest,
            "dependency_projection_digest": self.dependency_projection_digest,
            "success": bool(self.success),
            "lineage": list(self.lineage),
        }
        if include_digest:
            payload["output_digest"] = self.output_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KAdapterOutput:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            input_digest=str(payload["input_digest"]),
            action_digest=str(payload["action_digest"]),
            outcome_signature=str(payload["outcome_signature"]),
            dependency_digest=str(payload["dependency_digest"]),
            dependency_projection_digest=str(payload["dependency_projection_digest"]),
            success=bool(payload["success"]),
            lineage=tuple(str(value) for value in payload["lineage"]),
        )
        if str(payload.get("output_digest", "")) != item.output_digest:
            raise ValueError("K adapter output digest mismatch")
        return item


@dataclass(frozen=True)
class KAdapterExchange:
    """Content-addressed input/output pair for one controlled K episode."""

    scope_id: str
    input: KAdapterInput
    output: KAdapterOutput
    exchange_digest: str
    format: str = TAIJI_K_ADAPTER_EXCHANGE_FORMAT
    version: int = TAIJI_K_ADAPTER_EXCHANGE_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_K_ADAPTER_EXCHANGE_FORMAT:
            raise ValueError("unsupported K adapter exchange format")
        if int(self.version) != TAIJI_K_ADAPTER_EXCHANGE_VERSION:
            raise ValueError("unsupported K adapter exchange version")
        object.__setattr__(self, "scope_id", _text(self.scope_id, "K adapter scope_id"))
        if not isinstance(self.input, KAdapterInput):
            raise TypeError("K adapter exchange input is invalid")
        if not isinstance(self.output, KAdapterOutput):
            raise TypeError("K adapter exchange output is invalid")
        if self.output.parent_checkpoint_digest != self.input.parent_checkpoint_digest:
            raise ValueError("K adapter exchange crosses parent checkpoints")
        if self.output.input_digest != self.input.input_digest:
            raise ValueError("K adapter exchange output does not echo input digest")
        if self.scope_id not in self.output.lineage:
            raise ValueError("K adapter exchange lineage omits scope")
        if self.exchange_digest != content_digest(self._payload_without_digest()):
            raise ValueError("K adapter exchange digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "kind": "exchange",
            "scope_id": self.scope_id,
            "input": self.input.to_payload(),
            "output": self.output.to_payload(),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "exchange_digest": self.exchange_digest}

    @classmethod
    def create(
        cls,
        *,
        scope_id: str,
        input: KAdapterInput,
        output: KAdapterOutput,
    ) -> KAdapterExchange:
        unsigned = {
            "format": TAIJI_K_ADAPTER_EXCHANGE_FORMAT,
            "version": TAIJI_K_ADAPTER_EXCHANGE_VERSION,
            "kind": "exchange",
            "scope_id": str(scope_id),
            "input": input.to_payload(),
            "output": output.to_payload(),
        }
        return cls(
            scope_id=str(scope_id),
            input=input,
            output=output,
            exchange_digest=content_digest(unsigned),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KAdapterExchange:
        if payload.get("format") != TAIJI_K_ADAPTER_EXCHANGE_FORMAT:
            raise ValueError("unsupported K adapter exchange format")
        input_payload = payload.get("input")
        output_payload = payload.get("output")
        if not isinstance(input_payload, Mapping) or not isinstance(output_payload, Mapping):
            raise ValueError("K adapter exchange input/output is invalid")
        return cls(
            scope_id=str(payload["scope_id"]),
            input=KAdapterInput.from_payload(input_payload),
            output=KAdapterOutput.from_payload(output_payload),
            exchange_digest=str(payload["exchange_digest"]),
            format=str(payload["format"]),
            version=int(payload.get("version", -1)),
        )


@dataclass(frozen=True)
class KAdapterRollbackRecord:
    """Immutable audit record for one reverted candidate trial."""

    trial_id: str
    parent_checkpoint_digest: str
    candidate_checkpoint_digest: str
    candidate_owner_graph_digest: str
    candidate_source_manifest_digest: str
    candidate_namespace: str
    status: str
    reason: str
    record_digest: str
    version: int = TAIJI_K_CONTINUAL_ADAPTER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", _text(self.trial_id, "trial_id"))
        object.__setattr__(
            self,
            "parent_checkpoint_digest",
            _digest(self.parent_checkpoint_digest, "parent_checkpoint_digest"),
        )
        object.__setattr__(
            self,
            "candidate_checkpoint_digest",
            _digest(self.candidate_checkpoint_digest, "candidate_checkpoint_digest"),
        )
        object.__setattr__(
            self,
            "candidate_owner_graph_digest",
            _digest(self.candidate_owner_graph_digest, "candidate_owner_graph_digest"),
        )
        object.__setattr__(
            self,
            "candidate_source_manifest_digest",
            _digest(
                self.candidate_source_manifest_digest,
                "candidate_source_manifest_digest",
            ),
        )
        object.__setattr__(
            self,
            "candidate_namespace",
            _text(self.candidate_namespace, "candidate_namespace"),
        )
        if self.status != "rolled_back":
            raise ValueError("K adapter rollback record must be rolled_back")
        object.__setattr__(self, "reason", _text(self.reason, "rollback reason"))
        if int(self.version) != TAIJI_K_CONTINUAL_ADAPTER_VERSION:
            raise ValueError("unsupported K adapter rollback record version")
        if self.record_digest != content_digest(self._payload_without_digest()):
            raise ValueError("K adapter rollback record digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": TAIJI_K_CONTINUAL_ADAPTER_FORMAT,
            "version": int(self.version),
            "trial_id": self.trial_id,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "candidate_checkpoint_digest": self.candidate_checkpoint_digest,
            "candidate_owner_graph_digest": self.candidate_owner_graph_digest,
            "candidate_source_manifest_digest": self.candidate_source_manifest_digest,
            "candidate_namespace": self.candidate_namespace,
            "status": self.status,
            "reason": self.reason,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "record_digest": self.record_digest}

    @classmethod
    def create(
        cls,
        *,
        trial_id: str,
        parent_checkpoint_digest: str,
        candidate_checkpoint_digest: str,
        candidate_owner_graph_digest: str,
        candidate_source_manifest_digest: str,
        candidate_namespace: str,
        reason: str,
    ) -> KAdapterRollbackRecord:
        unsigned: dict[str, Any] = {
            "format": TAIJI_K_CONTINUAL_ADAPTER_FORMAT,
            "version": TAIJI_K_CONTINUAL_ADAPTER_VERSION,
            "trial_id": str(trial_id),
            "parent_checkpoint_digest": str(parent_checkpoint_digest),
            "candidate_checkpoint_digest": str(candidate_checkpoint_digest),
            "candidate_owner_graph_digest": str(candidate_owner_graph_digest),
            "candidate_source_manifest_digest": str(candidate_source_manifest_digest),
            "candidate_namespace": str(candidate_namespace),
            "status": "rolled_back",
            "reason": str(reason),
        }
        return cls(
            trial_id=unsigned["trial_id"],
            parent_checkpoint_digest=unsigned["parent_checkpoint_digest"],
            candidate_checkpoint_digest=unsigned["candidate_checkpoint_digest"],
            candidate_owner_graph_digest=unsigned["candidate_owner_graph_digest"],
            candidate_source_manifest_digest=unsigned[
                "candidate_source_manifest_digest"
            ],
            candidate_namespace=unsigned["candidate_namespace"],
            status=unsigned["status"],
            reason=unsigned["reason"],
            record_digest=content_digest(unsigned),
            version=unsigned["version"],
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KAdapterRollbackRecord:
        if payload.get("format") != TAIJI_K_CONTINUAL_ADAPTER_FORMAT:
            raise ValueError("unsupported K adapter rollback format")
        return cls(
            trial_id=str(payload["trial_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            candidate_checkpoint_digest=str(payload["candidate_checkpoint_digest"]),
            candidate_owner_graph_digest=str(payload["candidate_owner_graph_digest"]),
            candidate_source_manifest_digest=str(
                payload["candidate_source_manifest_digest"]
            ),
            candidate_namespace=str(payload["candidate_namespace"]),
            status=str(payload["status"]),
            reason=str(payload["reason"]),
            record_digest=str(payload["record_digest"]),
            version=int(payload.get("version", -1)),
        )


class KContinualAdapter:
    """Checkpoint and rollback boundary for a future native K continuation."""

    CHECKPOINT_FORMAT = TAIJI_K_CONTINUAL_ADAPTER_FORMAT
    CHECKPOINT_VERSION = TAIJI_K_CONTINUAL_ADAPTER_VERSION

    def __init__(
        self,
        *,
        parent_checkpoint_digest: str,
        owner_graph_digest: str,
        source_manifest_digest: str,
        resource_manifest_digest: str,
        dependency_scope_id: str,
        parent_namespace: str = "taiji:k:parent",
        candidate_namespace: str = "taiji:k:candidate",
    ) -> None:
        self.parent_checkpoint_digest = _digest(
            parent_checkpoint_digest, "parent_checkpoint_digest"
        )
        self.owner_graph_digest = _digest(owner_graph_digest, "owner_graph_digest")
        self.source_manifest_digest = _digest(
            source_manifest_digest, "source_manifest_digest"
        )
        self.resource_manifest_digest = _digest(
            resource_manifest_digest, "resource_manifest_digest"
        )
        self.dependency_scope_id = _text(dependency_scope_id, "dependency_scope_id")
        self.parent_namespace = _text(parent_namespace, "parent_namespace")
        self.candidate_namespace = _text(candidate_namespace, "candidate_namespace")
        if self.parent_namespace == self.candidate_namespace:
            raise ValueError("parent and candidate namespaces must differ")
        self._active_namespace = self.parent_namespace
        self._staged_candidate_digest = ""
        self._staged_owner_graph_digest = ""
        self._staged_source_manifest_digest = ""
        self._staged_trial_id = ""
        self._rollback_token = ""
        self._dependency_projection: OutcomeDependencyProjection | None = None
        self._last_exchange: KAdapterExchange | None = None
        self._rollback_records: list[KAdapterRollbackRecord] = []
        self._revision = 0

    @property
    def active_namespace(self) -> str:
        return self._active_namespace

    @property
    def dependency_projection(self) -> OutcomeDependencyProjection | None:
        return self._dependency_projection

    @property
    def last_exchange(self) -> KAdapterExchange | None:
        return self._last_exchange

    @property
    def rollback_records(self) -> tuple[KAdapterRollbackRecord, ...]:
        return tuple(self._rollback_records)

    @property
    def training_steps(self) -> int:
        """The preflight adapter intentionally owns no training steps."""

        return 0

    def parent_checkpoint_matches(self, parent_payload: Mapping[str, Any]) -> bool:
        return content_digest(dict(parent_payload)) == self.parent_checkpoint_digest

    def bind_dependency_projection(
        self, projection: OutcomeDependencyProjection
    ) -> None:
        if not isinstance(projection, OutcomeDependencyProjection):
            raise TypeError("K adapter dependency boundary requires a projection")
        if not projection.accepted:
            raise ValueError("K adapter cannot bind a rejected dependency projection")
        if projection.scope_id != self.dependency_scope_id:
            raise ValueError("K adapter dependency scope mismatch")
        if len(projection.lineage) != 4:
            raise ValueError("K adapter dependency lineage is incomplete")
        if self._dependency_projection is not None:
            if (
                self._dependency_projection.projection_digest
                == projection.projection_digest
            ):
                return
            raise ValueError("K adapter dependency boundary is already bound")
        self._dependency_projection = projection
        self._revision += 1

    def record_exchange(self, exchange: KAdapterExchange) -> str:
        """Record one typed K exchange without updating a learner."""

        if not isinstance(exchange, KAdapterExchange):
            raise TypeError("K adapter exchange boundary requires a KAdapterExchange")
        if self._dependency_projection is None:
            raise ValueError("K adapter exchange requires a bound dependency projection")
        if exchange.scope_id != self.dependency_scope_id:
            raise ValueError("K adapter exchange scope mismatch")
        if exchange.input.parent_checkpoint_digest != self.parent_checkpoint_digest:
            raise ValueError("K adapter exchange input crosses the parent checkpoint")
        if exchange.output.parent_checkpoint_digest != self.parent_checkpoint_digest:
            raise ValueError("K adapter exchange output crosses the parent checkpoint")
        if (
            exchange.output.dependency_digest
            != self._dependency_projection.dependency_digest
        ):
            raise ValueError("K adapter exchange dependency digest mismatch")
        if (
            exchange.output.dependency_projection_digest
            != self._dependency_projection.projection_digest
        ):
            raise ValueError("K adapter exchange projection lineage mismatch")
        if self._last_exchange is not None:
            if self._last_exchange.exchange_digest == exchange.exchange_digest:
                return exchange.exchange_digest
            raise ValueError("K adapter exchange boundary already contains another exchange")
        self._last_exchange = exchange
        self._revision += 1
        return exchange.exchange_digest

    def stage_candidate(
        self,
        *,
        candidate_checkpoint_digest: str,
        candidate_owner_graph_digest: str,
        candidate_source_manifest_digest: str,
        candidate_parent_checkpoint_digest: str,
    ) -> str:
        if self._active_namespace != self.parent_namespace:
            raise ValueError("K adapter already has an active candidate trial")
        if self._dependency_projection is None:
            raise ValueError("K adapter requires a bound dependency projection")
        if (
            _digest(candidate_parent_checkpoint_digest, "candidate parent digest")
            != self.parent_checkpoint_digest
        ):
            raise ValueError("K adapter candidate crosses the parent checkpoint")
        candidate_digest = _digest(candidate_checkpoint_digest, "candidate checkpoint digest")
        candidate_owner = _digest(candidate_owner_graph_digest, "candidate owner graph digest")
        candidate_source = _digest(
            candidate_source_manifest_digest, "candidate source manifest digest"
        )
        trial_id = content_digest(
            {
                "parent_checkpoint_digest": self.parent_checkpoint_digest,
                "candidate_checkpoint_digest": candidate_digest,
                "candidate_owner_graph_digest": candidate_owner,
                "candidate_source_manifest_digest": candidate_source,
                "candidate_namespace": self.candidate_namespace,
                "dependency_digest": self._dependency_projection.dependency_digest,
                "revision": self._revision,
            }
        )
        self._active_namespace = self.candidate_namespace
        self._staged_candidate_digest = candidate_digest
        self._staged_owner_graph_digest = candidate_owner
        self._staged_source_manifest_digest = candidate_source
        self._staged_trial_id = trial_id
        self._rollback_token = content_digest(
            {
                "trial_id": trial_id,
                "parent_checkpoint_digest": self.parent_checkpoint_digest,
                "candidate_checkpoint_digest": candidate_digest,
            }
        )
        self._revision += 1
        return self._rollback_token

    def rollback(self, rollback_token: str) -> KAdapterRollbackRecord:
        token = _digest(rollback_token, "rollback_token")
        if self._active_namespace != self.candidate_namespace:
            raise ValueError("K adapter has no active candidate to roll back")
        if token != self._rollback_token:
            raise ValueError("K adapter rollback token mismatch")
        if not self._staged_trial_id:
            raise ValueError("K adapter candidate trial id is missing")
        record = KAdapterRollbackRecord.create(
            trial_id=self._staged_trial_id,
            parent_checkpoint_digest=self.parent_checkpoint_digest,
            candidate_checkpoint_digest=self._staged_candidate_digest,
            candidate_owner_graph_digest=self._staged_owner_graph_digest,
            candidate_source_manifest_digest=self._staged_source_manifest_digest,
            candidate_namespace=self.candidate_namespace,
            reason="explicit_parent_restore",
        )
        self._rollback_records.append(record)
        self._active_namespace = self.parent_namespace
        self._staged_candidate_digest = ""
        self._staged_owner_graph_digest = ""
        self._staged_source_manifest_digest = ""
        self._staged_trial_id = ""
        self._rollback_token = ""
        self._revision += 1
        return record

    def checkpoint(self) -> dict[str, Any]:
        unsigned: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "owner_graph_digest": self.owner_graph_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "resource_manifest_digest": self.resource_manifest_digest,
            "dependency_scope_id": self.dependency_scope_id,
            "parent_namespace": self.parent_namespace,
            "candidate_namespace": self.candidate_namespace,
            "active_namespace": self._active_namespace,
            "staged_candidate_digest": self._staged_candidate_digest,
            "staged_owner_graph_digest": self._staged_owner_graph_digest,
            "staged_source_manifest_digest": self._staged_source_manifest_digest,
            "staged_trial_id": self._staged_trial_id,
            "rollback_token": self._rollback_token,
            "dependency_projection": (
                None
                if self._dependency_projection is None
                else self._dependency_projection.to_payload()
            ),
            "last_exchange": (
                None if self._last_exchange is None else self._last_exchange.to_payload()
            ),
            "rollback_records": [item.to_payload() for item in self._rollback_records],
            "revision": int(self._revision),
        }
        return {**unsigned, "checkpoint_digest": content_digest(unsigned)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> KContinualAdapter:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported K continual adapter format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported K continual adapter version")
        unsigned = {
            key: value for key, value in payload.items() if key != "checkpoint_digest"
        }
        if str(payload.get("checkpoint_digest", "")) != content_digest(unsigned):
            raise ValueError("K continual adapter checkpoint digest mismatch")
        adapter = cls(
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            owner_graph_digest=str(payload["owner_graph_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            resource_manifest_digest=str(payload["resource_manifest_digest"]),
            dependency_scope_id=str(payload["dependency_scope_id"]),
            parent_namespace=str(payload["parent_namespace"]),
            candidate_namespace=str(payload["candidate_namespace"]),
        )
        adapter._active_namespace = _text(payload["active_namespace"], "active_namespace")
        if adapter._active_namespace not in {
            adapter.parent_namespace,
            adapter.candidate_namespace,
        }:
            raise ValueError("K continual adapter active namespace is invalid")
        adapter._staged_candidate_digest = str(payload.get("staged_candidate_digest", ""))
        adapter._staged_owner_graph_digest = str(
            payload.get("staged_owner_graph_digest", "")
        )
        adapter._staged_source_manifest_digest = str(
            payload.get("staged_source_manifest_digest", "")
        )
        adapter._staged_trial_id = str(payload.get("staged_trial_id", ""))
        adapter._rollback_token = str(payload.get("rollback_token", ""))
        projection_payload = payload.get("dependency_projection")
        if projection_payload is not None:
            if not isinstance(projection_payload, Mapping):
                raise ValueError("K continual adapter dependency projection is invalid")
            adapter.bind_dependency_projection(
                OutcomeDependencyProjection.from_payload(projection_payload)
            )
        exchange_payload = payload.get("last_exchange")
        if exchange_payload is not None:
            if not isinstance(exchange_payload, Mapping):
                raise ValueError("K continual adapter exchange is invalid")
            adapter.record_exchange(KAdapterExchange.from_payload(exchange_payload))
        adapter._rollback_records = [
            KAdapterRollbackRecord.from_payload(item)
            for item in payload.get("rollback_records", ())
        ]
        record_ids = [item.record_digest for item in adapter._rollback_records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("K continual adapter rollback records contain duplicates")
        adapter._revision = int(payload.get("revision", 0))
        if adapter._revision < 0:
            raise ValueError("K continual adapter revision cannot be negative")
        if adapter._active_namespace == adapter.parent_namespace and any(
            (
                adapter._staged_candidate_digest,
                adapter._staged_owner_graph_digest,
                adapter._staged_source_manifest_digest,
                adapter._staged_trial_id,
                adapter._rollback_token,
            )
        ):
            raise ValueError("K continual adapter parent state carries staged candidate")
        if adapter._active_namespace == adapter.candidate_namespace and not all(
            (
                adapter._staged_candidate_digest,
                adapter._staged_owner_graph_digest,
                adapter._staged_source_manifest_digest,
                adapter._staged_trial_id,
                adapter._rollback_token,
            )
        ):
            raise ValueError("K continual adapter candidate state is incomplete")
        return adapter


__all__ = [
    "KAdapterExchange",
    "KAdapterInput",
    "KAdapterOutput",
    "KAdapterRollbackRecord",
    "KContinualAdapter",
    "TAIJI_K_ADAPTER_EXCHANGE_FORMAT",
    "TAIJI_K_ADAPTER_EXCHANGE_VERSION",
    "TAIJI_K_CONTINUAL_ADAPTER_FORMAT",
    "TAIJI_K_CONTINUAL_ADAPTER_VERSION",
]
