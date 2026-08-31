"""Explicit policy and audit contracts for external measured artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ARTIFACT_CONSUMPTION_POLICY_FORMAT = "taiji-artifact-consumption-policy-v1"
ARTIFACT_CONSUMPTION_AUDIT_FORMAT = "taiji-artifact-consumption-audit-v1"
ARTIFACT_CONSUMPTION_POLICY_REVISION = 1
ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY = "verified-only"
ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE = "legacy-compatible"
ARTIFACT_CONSUMPTION_MODES = (
    ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY,
    ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE,
)
ARTIFACT_CONSUMPTION_STATUSES = (
    "verified",
    "legacy_unverified",
    "missing",
    "tampered",
    "rejected",
)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_consumption_policy_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest for a policy payload."""

    return _digest(
        {key: value for key, value in payload.items() if key != "policy_digest"}
    )


@dataclass(frozen=True)
class ArtifactConsumptionPolicy:
    """Content-addressed rule for which external artifacts may be consumed.

    ``verified-only`` is the default for a newly created Taiji runtime.
    ``legacy-compatible`` is intentionally explicit and is intended for
    historical checkpoints or replay data that predates measurement sidecars.
    The reason is part of the digest so a compatibility decision cannot be
    hidden behind a bare boolean.
    """

    mode: str
    reason: str
    revision: int
    policy_digest: str

    def __post_init__(self) -> None:
        if int(self.revision) != ARTIFACT_CONSUMPTION_POLICY_REVISION:
            raise ValueError("unsupported artifact consumption policy revision")
        if str(self.mode) not in ARTIFACT_CONSUMPTION_MODES:
            raise ValueError("unsupported artifact consumption policy mode")
        if not str(self.reason).strip():
            raise ValueError("artifact consumption policy reason must not be empty")
        if not str(self.policy_digest):
            raise ValueError("artifact consumption policy digest must not be empty")
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "reason", str(self.reason).strip())
        object.__setattr__(self, "revision", int(self.revision))
        expected = artifact_consumption_policy_digest(self._payload_without_digest())
        if str(self.policy_digest) != expected:
            raise ValueError("artifact consumption policy digest mismatch")
        object.__setattr__(self, "policy_digest", str(self.policy_digest))

    @classmethod
    def create(
        cls,
        mode: str,
        *,
        reason: str,
        revision: int = ARTIFACT_CONSUMPTION_POLICY_REVISION,
    ) -> ArtifactConsumptionPolicy:
        payload = {
            "format": ARTIFACT_CONSUMPTION_POLICY_FORMAT,
            "revision": int(revision),
            "mode": str(mode),
            "reason": str(reason).strip(),
        }
        return cls(
            mode=str(mode),
            reason=str(reason).strip(),
            revision=int(revision),
            policy_digest=artifact_consumption_policy_digest(payload),
        )

    @classmethod
    def verified_only(
        cls,
        *,
        reason: str = "new-growth-default",
    ) -> ArtifactConsumptionPolicy:
        return cls.create(ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY, reason=reason)

    @classmethod
    def legacy_compatible(
        cls,
        *,
        reason: str = "historical-replay-explicit-compatibility",
    ) -> ArtifactConsumptionPolicy:
        return cls.create(ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE, reason=reason)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_CONSUMPTION_POLICY_FORMAT,
            "revision": self.revision,
            "mode": self.mode,
            "reason": self.reason,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "policy_digest": self.policy_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ArtifactConsumptionPolicy:
        if payload.get("format") != ARTIFACT_CONSUMPTION_POLICY_FORMAT:
            raise ValueError("unsupported artifact consumption policy format")
        return cls(
            mode=str(payload["mode"]),
            reason=str(payload["reason"]),
            revision=int(payload["revision"]),
            policy_digest=str(payload["policy_digest"]),
        )


def artifact_consumption_audit_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest for a consumption audit payload."""

    return _digest(
        {key: value for key, value in payload.items() if key != "audit_digest"}
    )


@dataclass(frozen=True)
class ArtifactConsumptionAudit:
    """Read-only, checkpointable observation of one artifact preflight."""

    batch_id: str
    policy: ArtifactConsumptionPolicy
    artifact_statuses: tuple[tuple[str, str], ...]
    result: str
    error_code: str
    audit_digest: str

    def __post_init__(self) -> None:
        if not str(self.batch_id):
            raise ValueError("artifact consumption audit batch id must not be empty")
        if not isinstance(self.policy, ArtifactConsumptionPolicy):
            raise TypeError("artifact consumption audit policy is invalid")
        statuses = tuple(
            sorted((str(candidate_id), str(status)) for candidate_id, status in self.artifact_statuses)
        )
        if len({candidate_id for candidate_id, _ in statuses}) != len(statuses):
            raise ValueError("artifact consumption audit candidate ids must be unique")
        if any(not candidate_id for candidate_id, _ in statuses):
            raise ValueError("artifact consumption audit candidate id must not be empty")
        if any(status not in ARTIFACT_CONSUMPTION_STATUSES for _, status in statuses):
            raise ValueError("artifact consumption audit status is invalid")
        if str(self.result) not in {"consumed", "rejected"}:
            raise ValueError("artifact consumption audit result is invalid")
        object.__setattr__(self, "batch_id", str(self.batch_id))
        object.__setattr__(self, "artifact_statuses", statuses)
        object.__setattr__(self, "result", str(self.result))
        object.__setattr__(self, "error_code", str(self.error_code))
        expected = artifact_consumption_audit_digest(self._payload_without_digest())
        if str(self.audit_digest) != expected:
            raise ValueError("artifact consumption audit digest mismatch")
        object.__setattr__(self, "audit_digest", str(self.audit_digest))

    @classmethod
    def create(
        cls,
        batch_id: str,
        policy: ArtifactConsumptionPolicy,
        artifact_statuses: Mapping[str, str],
        *,
        result: str,
        error_code: str = "",
    ) -> ArtifactConsumptionAudit:
        payload = {
            "format": ARTIFACT_CONSUMPTION_AUDIT_FORMAT,
            "batch_id": str(batch_id),
            "policy": policy.to_payload(),
            "artifact_statuses": {
                str(key): str(value) for key, value in sorted(artifact_statuses.items())
            },
            "result": str(result),
            "error_code": str(error_code),
        }
        return cls(
            batch_id=str(batch_id),
            policy=policy,
            artifact_statuses=tuple(payload["artifact_statuses"].items()),
            result=str(result),
            error_code=str(error_code),
            audit_digest=artifact_consumption_audit_digest(payload),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": ARTIFACT_CONSUMPTION_AUDIT_FORMAT,
            "batch_id": self.batch_id,
            "policy": self.policy.to_payload(),
            "artifact_statuses": {
                key: value for key, value in self.artifact_statuses
            },
            "result": self.result,
            "error_code": self.error_code,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ArtifactConsumptionAudit:
        if payload.get("format") != ARTIFACT_CONSUMPTION_AUDIT_FORMAT:
            raise ValueError("unsupported artifact consumption audit format")
        raw_statuses = payload.get("artifact_statuses", {})
        if not isinstance(raw_statuses, Mapping):
            raise ValueError("artifact consumption audit statuses must be a mapping")
        raw_policy = payload.get("policy")
        if not isinstance(raw_policy, Mapping):
            raise ValueError("artifact consumption audit policy must be a mapping")
        return cls(
            batch_id=str(payload["batch_id"]),
            policy=ArtifactConsumptionPolicy.from_payload(raw_policy),
            artifact_statuses=tuple(
                (str(key), str(value)) for key, value in raw_statuses.items()
            ),
            result=str(payload["result"]),
            error_code=str(payload.get("error_code", "")),
            audit_digest=str(payload["audit_digest"]),
        )
