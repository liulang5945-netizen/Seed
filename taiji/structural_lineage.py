"""Content-addressed results for coordinated structural-lineage retention."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STRUCTURAL_LINEAGE_RETENTION_RESULT_FORMAT = "taiji-structural-lineage-retention-result-v1"
STRUCTURAL_LINEAGE_RETENTION_POLICY_FORMAT = "taiji-structural-lineage-retention-policy-v1"
STRUCTURAL_LINEAGE_RETENTION_POLICY_REVISION = 1
STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION = 2
STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES = (
    "active_reservation",
    "pending_candidate",
    "pending_topology_proposal",
    "rollbackable_admission",
)
STRUCTURAL_MAINTENANCE_AUDIT_FORMAT = "taiji-structural-maintenance-audit-v1"
STRUCTURAL_MAINTENANCE_STATUS_FORMAT = "taiji-structural-maintenance-status-v1"
STRUCTURAL_LINEAGE_RETENTION_POLICY_MIGRATION_FORMAT = (
    "taiji-structural-lineage-retention-policy-migration-v1"
)
STRUCTURAL_LINEAGE_RETENTION_POLICY_MODE = "terminal_only"


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuralLineageRetentionPolicy:
    """Versioned policy snapshot for safe structural-lineage retention."""

    revision: int
    max_batches: int
    protection_rules: tuple[str, ...]
    policy_digest: str
    mode: str = STRUCTURAL_LINEAGE_RETENTION_POLICY_MODE

    def __post_init__(self) -> None:
        if int(self.revision) not in {
            STRUCTURAL_LINEAGE_RETENTION_POLICY_REVISION,
            STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION,
        }:
            raise ValueError("unsupported structural lineage retention policy revision")
        if int(self.max_batches) <= 0:
            raise ValueError("structural lineage retention policy max_batches must be positive")
        rules = tuple(str(item) for item in self.protection_rules)
        required = set(STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES)
        if set(rules) != required or len(rules) != len(set(rules)):
            raise ValueError("structural lineage retention policy protection rules are invalid")
        if str(self.mode) != STRUCTURAL_LINEAGE_RETENTION_POLICY_MODE:
            raise ValueError("unsupported structural lineage retention policy mode")
        if not str(self.policy_digest):
            raise ValueError("structural lineage retention policy digest must not be empty")
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "max_batches", int(self.max_batches))
        object.__setattr__(self, "protection_rules", tuple(sorted(rules)))
        object.__setattr__(self, "mode", str(self.mode))
        expected = _digest(self._payload_without_digest())
        if str(self.policy_digest) != expected:
            raise ValueError("structural lineage retention policy digest mismatch")
        object.__setattr__(self, "policy_digest", str(self.policy_digest))

    @classmethod
    def create(
        cls,
        max_batches: int,
        *,
        revision: int = STRUCTURAL_LINEAGE_RETENTION_POLICY_REVISION,
        protection_rules: tuple[str, ...] = STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES,
        mode: str = STRUCTURAL_LINEAGE_RETENTION_POLICY_MODE,
    ) -> StructuralLineageRetentionPolicy:
        canonical_rules = tuple(sorted(str(item) for item in protection_rules))
        payload = {
            "format": STRUCTURAL_LINEAGE_RETENTION_POLICY_FORMAT,
            "revision": int(revision),
            "max_batches": int(max_batches),
            "protection_rules": list(canonical_rules),
        }
        if int(revision) == STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION:
            payload["mode"] = str(mode)
        return cls(
            revision=int(revision),
            max_batches=int(max_batches),
            protection_rules=canonical_rules,
            policy_digest=_digest(payload),
            mode=str(mode),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        payload = {
            "format": STRUCTURAL_LINEAGE_RETENTION_POLICY_FORMAT,
            "revision": self.revision,
            "max_batches": self.max_batches,
            "protection_rules": list(self.protection_rules),
        }
        if self.revision == STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION:
            payload["mode"] = self.mode
        return payload

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "policy_digest": self.policy_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralLineageRetentionPolicy:
        if payload.get("format") != STRUCTURAL_LINEAGE_RETENTION_POLICY_FORMAT:
            raise ValueError("unsupported structural lineage retention policy format")
        raw_rules = payload.get("protection_rules", ())
        if not isinstance(raw_rules, (list, tuple)):
            raise ValueError("structural lineage retention policy protection rules must be a sequence")
        return cls(
            revision=int(payload["revision"]),
            max_batches=int(payload["max_batches"]),
            protection_rules=tuple(str(item) for item in raw_rules),
            policy_digest=str(payload["policy_digest"]),
            mode=str(payload.get("mode", STRUCTURAL_LINEAGE_RETENTION_POLICY_MODE)),
        )

    def migrate_to_latest(self) -> StructuralLineageRetentionPolicy:
        """Explicitly migrate this policy without changing its safety semantics."""

        if self.revision == STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION:
            return self
        return self.create(
            self.max_batches,
            revision=STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION,
            protection_rules=self.protection_rules,
            mode=self.mode,
        )


@dataclass(frozen=True)
class StructuralLineageRetentionPolicyMigration:
    """Auditable, reversible migration between compatible policy revisions."""

    status: str
    source_policy: StructuralLineageRetentionPolicy
    target_policy: StructuralLineageRetentionPolicy
    migration_digest: str

    def __post_init__(self) -> None:
        if self.status not in {"prepared", "committed", "rolled_back"}:
            raise ValueError("unsupported structural lineage retention policy migration status")
        if not isinstance(self.source_policy, StructuralLineageRetentionPolicy):
            raise TypeError("structural lineage retention migration source is invalid")
        if not isinstance(self.target_policy, StructuralLineageRetentionPolicy):
            raise TypeError("structural lineage retention migration target is invalid")
        if self.target_policy.revision != self.source_policy.revision + 1:
            raise ValueError("structural lineage retention policy migration revision is not adjacent")
        if (
            self.target_policy.max_batches != self.source_policy.max_batches
            or self.target_policy.protection_rules != self.source_policy.protection_rules
            or self.target_policy.mode != self.source_policy.mode
        ):
            raise ValueError("structural lineage retention policy migration changes safety semantics")
        if not str(self.migration_digest):
            raise ValueError("structural lineage retention policy migration digest must not be empty")
        object.__setattr__(self, "migration_digest", str(self.migration_digest))
        expected = _digest(self._payload_without_digest())
        if str(self.migration_digest) != expected:
            raise ValueError("structural lineage retention policy migration digest mismatch")

    @classmethod
    def create(
        cls,
        source_policy: StructuralLineageRetentionPolicy,
        target_policy: StructuralLineageRetentionPolicy,
        *,
        status: str = "prepared",
    ) -> StructuralLineageRetentionPolicyMigration:
        payload = {
            "format": STRUCTURAL_LINEAGE_RETENTION_POLICY_MIGRATION_FORMAT,
            "status": status,
            "source_policy": source_policy.to_payload(),
            "target_policy": target_policy.to_payload(),
        }
        return cls(
            status=status,
            source_policy=source_policy,
            target_policy=target_policy,
            migration_digest=_digest(payload),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_LINEAGE_RETENTION_POLICY_MIGRATION_FORMAT,
            "status": self.status,
            "source_policy": self.source_policy.to_payload(),
            "target_policy": self.target_policy.to_payload(),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "migration_digest": self.migration_digest}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> StructuralLineageRetentionPolicyMigration:
        if payload.get("format") != STRUCTURAL_LINEAGE_RETENTION_POLICY_MIGRATION_FORMAT:
            raise ValueError("unsupported structural lineage retention policy migration format")
        return cls(
            status=str(payload["status"]),
            source_policy=StructuralLineageRetentionPolicy.from_payload(payload["source_policy"]),
            target_policy=StructuralLineageRetentionPolicy.from_payload(payload["target_policy"]),
            migration_digest=str(payload["migration_digest"]),
        )


def structural_lineage_retention_policy_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest for a retention-policy payload."""

    return _digest({key: value for key, value in payload.items() if key != "policy_digest"})


@dataclass(frozen=True)
class StructuralLineageRetentionResult:
    """Describe one atomic compaction over the structural lineage graph."""

    status: str
    max_batches: int
    source_checkpoint_digest: str
    target_checkpoint_digest: str
    protected_batch_ids: tuple[str, ...]
    retained_batch_ids: tuple[str, ...]
    removed_batch_ids: tuple[str, ...]
    retained_candidate_ids: tuple[str, ...]
    removed_candidate_ids: tuple[str, ...]
    removed_record_counts: tuple[tuple[str, int], ...]
    retention_pressure: bool
    result_digest: str

    def __post_init__(self) -> None:
        if self.status not in {"compacted", "nothing_to_compact"}:
            raise ValueError("unsupported structural lineage retention status")
        if int(self.max_batches) <= 0:
            raise ValueError("structural lineage retention max_batches must be positive")
        for name in ("source_checkpoint_digest", "target_checkpoint_digest", "result_digest"):
            if not str(getattr(self, name)):
                raise ValueError(f"structural lineage retention {name} must not be empty")
        for name in (
            "protected_batch_ids",
            "retained_batch_ids",
            "removed_batch_ids",
            "retained_candidate_ids",
            "removed_candidate_ids",
        ):
            values = tuple(str(item) for item in getattr(self, name))
            if any(not item for item in values) or len(set(values)) != len(values):
                raise ValueError(f"structural lineage retention {name} must be unique and non-empty")
            object.__setattr__(self, name, values)
        counts = tuple((str(key), int(value)) for key, value in self.removed_record_counts)
        if any(not key or value < 0 for key, value in counts):
            raise ValueError("structural lineage retention record counts are invalid")
        if len({key for key, _ in counts}) != len(counts):
            raise ValueError("structural lineage retention record counts must be unique")
        object.__setattr__(self, "max_batches", int(self.max_batches))
        object.__setattr__(self, "removed_record_counts", tuple(sorted(counts)))
        object.__setattr__(self, "retention_pressure", bool(self.retention_pressure))
        expected = _digest(self._payload_without_digest())
        if str(self.result_digest) != expected:
            raise ValueError("structural lineage retention result digest mismatch")
        object.__setattr__(self, "source_checkpoint_digest", str(self.source_checkpoint_digest))
        object.__setattr__(self, "target_checkpoint_digest", str(self.target_checkpoint_digest))
        object.__setattr__(self, "result_digest", str(self.result_digest))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_LINEAGE_RETENTION_RESULT_FORMAT,
            "status": self.status,
            "max_batches": self.max_batches,
            "source_checkpoint_digest": self.source_checkpoint_digest,
            "target_checkpoint_digest": self.target_checkpoint_digest,
            "protected_batch_ids": list(self.protected_batch_ids),
            "retained_batch_ids": list(self.retained_batch_ids),
            "removed_batch_ids": list(self.removed_batch_ids),
            "retained_candidate_ids": list(self.retained_candidate_ids),
            "removed_candidate_ids": list(self.removed_candidate_ids),
            "removed_record_counts": {
                key: value for key, value in self.removed_record_counts
            },
            "retention_pressure": self.retention_pressure,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "result_digest": self.result_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralLineageRetentionResult:
        if payload.get("format") != STRUCTURAL_LINEAGE_RETENTION_RESULT_FORMAT:
            raise ValueError("unsupported structural lineage retention result format")
        raw_counts = payload.get("removed_record_counts", {})
        if not isinstance(raw_counts, Mapping):
            raise ValueError("structural lineage retention record counts must be a mapping")
        return cls(
            status=str(payload["status"]),
            max_batches=int(payload["max_batches"]),
            source_checkpoint_digest=str(payload["source_checkpoint_digest"]),
            target_checkpoint_digest=str(payload["target_checkpoint_digest"]),
            protected_batch_ids=tuple(str(item) for item in payload.get("protected_batch_ids", ())),
            retained_batch_ids=tuple(str(item) for item in payload.get("retained_batch_ids", ())),
            removed_batch_ids=tuple(str(item) for item in payload.get("removed_batch_ids", ())),
            retained_candidate_ids=tuple(
                str(item) for item in payload.get("retained_candidate_ids", ())
            ),
            removed_candidate_ids=tuple(
                str(item) for item in payload.get("removed_candidate_ids", ())
            ),
            removed_record_counts=tuple(
                (str(key), int(value)) for key, value in raw_counts.items()
            ),
            retention_pressure=bool(payload.get("retention_pressure", False)),
            result_digest=str(payload["result_digest"]),
        )


def structural_lineage_retention_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest for a retention result payload."""

    return _digest(
        {key: value for key, value in payload.items() if key != "result_digest"}
    )


@dataclass(frozen=True)
class StructuralMaintenanceAudit:
    """Stable, content-addressed runtime projection of one maintenance call."""

    maintenance_results: tuple[Any, ...]
    lineage_retention: StructuralLineageRetentionResult | None
    structural_runtime_tick: int
    audit_digest: str
    retention_policy: StructuralLineageRetentionPolicy | None = None

    def __post_init__(self) -> None:
        from .structural_growth import StructuralMaintenanceResult

        results = tuple(self.maintenance_results)
        if any(not isinstance(item, StructuralMaintenanceResult) for item in results):
            raise TypeError("structural maintenance audit results are invalid")
        if self.lineage_retention is not None and not isinstance(
            self.lineage_retention, StructuralLineageRetentionResult
        ):
            raise TypeError("structural maintenance audit retention result is invalid")
        if self.retention_policy is not None and not isinstance(
            self.retention_policy, StructuralLineageRetentionPolicy
        ):
            raise TypeError("structural maintenance audit retention policy is invalid")
        if self.retention_policy is not None and self.lineage_retention is None:
            raise ValueError("structural maintenance audit policy has no retention result")
        if (
            self.retention_policy is not None
            and self.lineage_retention is not None
            and self.retention_policy.max_batches != self.lineage_retention.max_batches
        ):
            raise ValueError("structural maintenance audit policy does not match retention result")
        if int(self.structural_runtime_tick) < 0:
            raise ValueError("structural maintenance audit runtime tick must be non-negative")
        if not str(self.audit_digest):
            raise ValueError("structural maintenance audit digest must not be empty")
        object.__setattr__(self, "maintenance_results", results)
        object.__setattr__(self, "structural_runtime_tick", int(self.structural_runtime_tick))
        expected = _digest(self._payload_without_digest())
        if str(self.audit_digest) != expected:
            raise ValueError("structural maintenance audit digest mismatch")
        object.__setattr__(self, "audit_digest", str(self.audit_digest))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_MAINTENANCE_AUDIT_FORMAT,
            "maintenance_results": [item.to_payload() for item in self.maintenance_results],
            "lineage_retention": (
                None
                if self.lineage_retention is None
                else self.lineage_retention.to_payload()
            ),
            "retention_policy": (
                None
                if self.retention_policy is None
                else self.retention_policy.to_payload()
            ),
            "structural_runtime_tick": self.structural_runtime_tick,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralMaintenanceAudit:
        from .structural_growth import StructuralMaintenanceResult

        if payload.get("format") != STRUCTURAL_MAINTENANCE_AUDIT_FORMAT:
            raise ValueError("unsupported structural maintenance audit format")
        raw_results = payload.get("maintenance_results", ())
        if not isinstance(raw_results, (list, tuple)):
            raise ValueError("structural maintenance audit results must be a sequence")
        results = tuple(
            StructuralMaintenanceResult.from_payload(item)
            for item in raw_results
            if isinstance(item, Mapping)
        )
        if len(results) != len(raw_results):
            raise ValueError("structural maintenance audit result entry is invalid")
        raw_retention = payload.get("lineage_retention")
        retention = (
            None
            if raw_retention is None
            else StructuralLineageRetentionResult.from_payload(raw_retention)
        )
        raw_policy = payload.get("retention_policy")
        policy = (
            None
            if raw_policy is None
            else StructuralLineageRetentionPolicy.from_payload(raw_policy)
        )
        return cls(
            maintenance_results=results,
            lineage_retention=retention,
            structural_runtime_tick=int(payload["structural_runtime_tick"]),
            audit_digest=str(payload["audit_digest"]),
            retention_policy=policy,
        )


def structural_maintenance_audit_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest for a maintenance audit payload."""

    return _digest(
        {key: value for key, value in payload.items() if key != "audit_digest"}
    )
