"""Content-addressed manifests for attaching K workers to one Taiji parent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

K_WORKER_MANIFEST_FORMAT = "taiji-k-worker-manifest-v1"
K_WORKER_MANIFEST_VERSION = 1
K_WORKER_BUNDLE_FORMAT = "taiji-k-worker-bundle-v1"
K_WORKER_BUNDLE_VERSION = 1
K_WORKER_IDS = (
    "k1.semantic",
    "k2.transition",
    "k3.outcome_projection",
)
K_WORKER_KINDS = {
    "k1.semantic": "semantic",
    "k2.transition": "predictive_transition",
    "k3.outcome_projection": "deterministic_projection",
}
K_WORKER_CONTRACT_FORMAT = "taiji-k-worker-contract-v1"
K_WORKER_CONTRACT_VERSION = 1
K_WORKER_INPUT_CONTRACT_DIGESTS = {
    worker_id: content_digest(
        {
            "format": K_WORKER_CONTRACT_FORMAT,
            "version": K_WORKER_CONTRACT_VERSION,
            "worker_id": worker_id,
            "direction": "input",
            "shape": {
                "k1.semantic": "PerceptEvent",
                "k2.transition": "WorldState+PerceptEvent",
                "k3.outcome_projection": "WorldState+WorldEvent+OutcomeDependencySpec",
            }[worker_id],
        }
    )
    for worker_id in K_WORKER_IDS
}
K_WORKER_OUTPUT_CONTRACT_DIGESTS = {
    worker_id: content_digest(
        {
            "format": K_WORKER_CONTRACT_FORMAT,
            "version": K_WORKER_CONTRACT_VERSION,
            "worker_id": worker_id,
            "direction": "output",
            "shape": {
                "k1.semantic": "StructuredSemanticResult",
                "k2.transition": "StructuredSemanticTransitionResult",
                "k3.outcome_projection": "OutcomeDependencyProjection",
            }[worker_id],
        }
    )
    for worker_id in K_WORKER_IDS
}


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


def _owner_digests(values: Sequence[Sequence[Any]]) -> tuple[tuple[str, str], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("worker owner_digests must be a sequence of pairs")
    normalized = tuple(
        (_text(value[0], "owner digest name"), _digest(value[1], "owner digest"))
        for value in values
    )
    if not normalized or len({name for name, _digest_value in normalized}) != len(normalized):
        raise ValueError("worker owner_digests must contain unique names")
    return normalized


@dataclass(frozen=True)
class KWorkerManifest:
    """One restorable K worker and its contract/lineage metadata."""

    worker_id: str
    worker_kind: str
    checkpoint_format: str
    checkpoint_version: int
    worker_checkpoint_digest: str
    owner_digests: tuple[tuple[str, str], ...]
    source_digest: str
    input_contract_digest: str
    output_contract_digest: str
    parent_checkpoint_digest: str
    candidate_namespace: str
    training_steps: int
    optimizer_state_present: bool
    manifest_digest: str
    format: str = K_WORKER_MANIFEST_FORMAT
    version: int = K_WORKER_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.format != K_WORKER_MANIFEST_FORMAT:
            raise ValueError("unsupported K worker manifest format")
        if int(self.version) != K_WORKER_MANIFEST_VERSION:
            raise ValueError("unsupported K worker manifest version")
        object.__setattr__(self, "worker_id", _text(self.worker_id, "worker_id"))
        if self.worker_id not in K_WORKER_KINDS:
            raise ValueError("unsupported K worker id")
        expected_kind = K_WORKER_KINDS[self.worker_id]
        if self.worker_kind != expected_kind:
            raise ValueError("K worker kind does not match worker id")
        object.__setattr__(
            self, "checkpoint_format", _text(self.checkpoint_format, "checkpoint_format")
        )
        if int(self.checkpoint_version) < 1:
            raise ValueError("worker checkpoint version must be positive")
        for field_name in (
            "worker_checkpoint_digest",
            "source_digest",
            "input_contract_digest",
            "output_contract_digest",
            "parent_checkpoint_digest",
            "manifest_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "owner_digests", _owner_digests(self.owner_digests))
        object.__setattr__(
            self,
            "candidate_namespace",
            _text(self.candidate_namespace, "candidate_namespace"),
        )
        if int(self.training_steps) < 0:
            raise ValueError("worker training_steps cannot be negative")
        if self.worker_id == "k3.outcome_projection" and self.optimizer_state_present:
            raise ValueError("deterministic K3 worker cannot carry optimizer state")
        if self.manifest_digest != content_digest(self._payload_without_digest()):
            raise ValueError("K worker manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "worker_id": self.worker_id,
            "worker_kind": self.worker_kind,
            "checkpoint_format": self.checkpoint_format,
            "checkpoint_version": int(self.checkpoint_version),
            "worker_checkpoint_digest": self.worker_checkpoint_digest,
            "owner_digests": [list(item) for item in self.owner_digests],
            "source_digest": self.source_digest,
            "input_contract_digest": self.input_contract_digest,
            "output_contract_digest": self.output_contract_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "candidate_namespace": self.candidate_namespace,
            "training_steps": int(self.training_steps),
            "optimizer_state_present": bool(self.optimizer_state_present),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def create(
        cls,
        *,
        worker_id: str,
        checkpoint_format: str,
        checkpoint_version: int,
        worker_checkpoint_digest: str,
        owner_digests: Sequence[Sequence[Any]],
        source_digest: str,
        input_contract_digest: str,
        output_contract_digest: str,
        parent_checkpoint_digest: str,
        candidate_namespace: str,
        training_steps: int,
        optimizer_state_present: bool,
    ) -> KWorkerManifest:
        kind = K_WORKER_KINDS.get(str(worker_id))
        if kind is None:
            raise ValueError("unsupported K worker id")
        unsigned = {
            "format": K_WORKER_MANIFEST_FORMAT,
            "version": K_WORKER_MANIFEST_VERSION,
            "worker_id": str(worker_id),
            "worker_kind": kind,
            "checkpoint_format": str(checkpoint_format),
            "checkpoint_version": int(checkpoint_version),
            "worker_checkpoint_digest": str(worker_checkpoint_digest),
            "owner_digests": [list(item) for item in owner_digests],
            "source_digest": str(source_digest),
            "input_contract_digest": str(input_contract_digest),
            "output_contract_digest": str(output_contract_digest),
            "parent_checkpoint_digest": str(parent_checkpoint_digest),
            "candidate_namespace": str(candidate_namespace),
            "training_steps": int(training_steps),
            "optimizer_state_present": bool(optimizer_state_present),
        }
        return cls(
            worker_id=str(unsigned["worker_id"]),
            worker_kind=str(unsigned["worker_kind"]),
            checkpoint_format=str(unsigned["checkpoint_format"]),
            checkpoint_version=int(str(unsigned["checkpoint_version"])),
            worker_checkpoint_digest=str(unsigned["worker_checkpoint_digest"]),
            owner_digests=tuple((str(item[0]), str(item[1])) for item in owner_digests),
            source_digest=str(unsigned["source_digest"]),
            input_contract_digest=str(unsigned["input_contract_digest"]),
            output_contract_digest=str(unsigned["output_contract_digest"]),
            parent_checkpoint_digest=str(unsigned["parent_checkpoint_digest"]),
            candidate_namespace=str(unsigned["candidate_namespace"]),
            training_steps=int(str(unsigned["training_steps"])),
            optimizer_state_present=bool(unsigned["optimizer_state_present"]),
            manifest_digest=content_digest(unsigned),
            format=str(unsigned["format"]),
            version=int(str(unsigned["version"])),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KWorkerManifest:
        if payload.get("format") != K_WORKER_MANIFEST_FORMAT:
            raise ValueError("unsupported K worker manifest format")
        return cls(
            worker_id=str(payload["worker_id"]),
            worker_kind=str(payload["worker_kind"]),
            checkpoint_format=str(payload["checkpoint_format"]),
            checkpoint_version=int(payload["checkpoint_version"]),
            worker_checkpoint_digest=str(payload["worker_checkpoint_digest"]),
            owner_digests=tuple(tuple(item) for item in payload["owner_digests"]),
            source_digest=str(payload["source_digest"]),
            input_contract_digest=str(payload["input_contract_digest"]),
            output_contract_digest=str(payload["output_contract_digest"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            candidate_namespace=str(payload["candidate_namespace"]),
            training_steps=int(payload["training_steps"]),
            optimizer_state_present=bool(payload["optimizer_state_present"]),
            manifest_digest=str(payload["manifest_digest"]),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
        )


@dataclass(frozen=True)
class KWorkerManifestBundle:
    """The atomic worker graph attached to one fixed-capacity parent."""

    parent_checkpoint_digest: str
    source_manifest_digest: str
    resource_manifest_digest: str
    candidate_namespace: str
    workers: tuple[KWorkerManifest, ...]
    owner_graph_digest: str
    bundle_digest: str
    format: str = K_WORKER_BUNDLE_FORMAT
    version: int = K_WORKER_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.format != K_WORKER_BUNDLE_FORMAT:
            raise ValueError("unsupported K worker bundle format")
        if int(self.version) != K_WORKER_BUNDLE_VERSION:
            raise ValueError("unsupported K worker bundle version")
        for field_name in (
            "parent_checkpoint_digest",
            "source_manifest_digest",
            "resource_manifest_digest",
            "owner_graph_digest",
            "bundle_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "candidate_namespace",
            _text(self.candidate_namespace, "candidate_namespace"),
        )
        workers = tuple(self.workers)
        if any(not isinstance(item, KWorkerManifest) for item in workers):
            raise TypeError("K worker bundle contains an invalid worker")
        if tuple(item.worker_id for item in workers) != tuple(
            sorted(item.worker_id for item in workers)
        ):
            raise ValueError("K worker bundle workers must be sorted")
        if tuple(item.worker_id for item in workers) != K_WORKER_IDS:
            raise ValueError("K worker bundle must contain exactly K1/K2/K3 workers")
        if any(
            item.parent_checkpoint_digest != self.parent_checkpoint_digest
            or item.candidate_namespace != self.candidate_namespace
            for item in workers
        ):
            raise ValueError("K worker bundle contains a cross-parent or cross-namespace worker")
        object.__setattr__(self, "workers", workers)
        if self.owner_graph_digest != self.compute_owner_graph_digest(
            parent_checkpoint_digest=self.parent_checkpoint_digest,
            candidate_namespace=self.candidate_namespace,
            workers=workers,
        ):
            raise ValueError("K worker bundle owner graph digest mismatch")
        if self.bundle_digest != content_digest(self._payload_without_digest()):
            raise ValueError("K worker bundle digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "resource_manifest_digest": self.resource_manifest_digest,
            "candidate_namespace": self.candidate_namespace,
            "workers": [item.to_payload() for item in self.workers],
            "owner_graph_digest": self.owner_graph_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "bundle_digest": self.bundle_digest}

    @staticmethod
    def compute_owner_graph_digest(
        *,
        parent_checkpoint_digest: str,
        candidate_namespace: str,
        workers: Sequence[KWorkerManifest],
    ) -> str:
        return content_digest(
            {
                "format": K_WORKER_BUNDLE_FORMAT,
                "version": K_WORKER_BUNDLE_VERSION,
                "parent_checkpoint_digest": parent_checkpoint_digest,
                "candidate_namespace": candidate_namespace,
                "workers": [item.manifest_digest for item in workers],
            }
        )

    @classmethod
    def create(
        cls,
        *,
        parent_checkpoint_digest: str,
        source_manifest_digest: str,
        resource_manifest_digest: str,
        candidate_namespace: str,
        workers: Sequence[KWorkerManifest],
    ) -> KWorkerManifestBundle:
        ordered = tuple(sorted(workers, key=lambda item: item.worker_id))
        owner_graph_digest = cls.compute_owner_graph_digest(
            parent_checkpoint_digest=parent_checkpoint_digest,
            candidate_namespace=candidate_namespace,
            workers=ordered,
        )
        unsigned = {
            "format": K_WORKER_BUNDLE_FORMAT,
            "version": K_WORKER_BUNDLE_VERSION,
            "parent_checkpoint_digest": str(parent_checkpoint_digest),
            "source_manifest_digest": str(source_manifest_digest),
            "resource_manifest_digest": str(resource_manifest_digest),
            "candidate_namespace": str(candidate_namespace),
            "workers": [item.to_payload() for item in ordered],
            "owner_graph_digest": owner_graph_digest,
        }
        return cls(
            parent_checkpoint_digest=str(parent_checkpoint_digest),
            source_manifest_digest=str(source_manifest_digest),
            resource_manifest_digest=str(resource_manifest_digest),
            candidate_namespace=str(candidate_namespace),
            workers=ordered,
            owner_graph_digest=owner_graph_digest,
            bundle_digest=content_digest(unsigned),
            format=K_WORKER_BUNDLE_FORMAT,
            version=K_WORKER_BUNDLE_VERSION,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> KWorkerManifestBundle:
        if payload.get("format") != K_WORKER_BUNDLE_FORMAT:
            raise ValueError("unsupported K worker bundle format")
        raw_workers = payload.get("workers")
        if not isinstance(raw_workers, Sequence) or isinstance(
            raw_workers, (str, bytes, bytearray)
        ):
            raise ValueError("K worker bundle workers are invalid")
        return cls(
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            resource_manifest_digest=str(payload["resource_manifest_digest"]),
            candidate_namespace=str(payload["candidate_namespace"]),
            workers=tuple(
                KWorkerManifest.from_payload(item)
                for item in raw_workers
                if isinstance(item, Mapping)
            ),
            owner_graph_digest=str(payload["owner_graph_digest"]),
            bundle_digest=str(payload["bundle_digest"]),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
        )


__all__ = [
    "K_WORKER_BUNDLE_FORMAT",
    "K_WORKER_BUNDLE_VERSION",
    "K_WORKER_CONTRACT_FORMAT",
    "K_WORKER_CONTRACT_VERSION",
    "K_WORKER_IDS",
    "K_WORKER_INPUT_CONTRACT_DIGESTS",
    "K_WORKER_KINDS",
    "K_WORKER_MANIFEST_FORMAT",
    "K_WORKER_MANIFEST_VERSION",
    "K_WORKER_OUTPUT_CONTRACT_DIGESTS",
    "KWorkerManifest",
    "KWorkerManifestBundle",
]
