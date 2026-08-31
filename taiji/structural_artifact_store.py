"""Immutable content-addressed storage for measured Workbench artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .structural_validation_artifact import WorkbenchStructuralValidationArtifact
from .structural_validation_measurements import StructuralValidationMeasurements

STRUCTURAL_ARTIFACT_STORE_FORMAT = "taiji-structural-artifact-store-v1"
STRUCTURAL_ARTIFACT_STORE_AUDIT_FORMAT = "taiji-structural-artifact-store-audit-v1"
STRUCTURAL_ARTIFACT_STORE_PROJECTION_FORMAT = (
    "taiji-seed-runtime-structural-artifact-store-projection-v2"
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MEASUREMENT_FILE_PREFIX = "measurement-"
_STORE_WRITE_LOCK = threading.RLock()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def structural_artifact_store_audit_digest(payload: Mapping[str, Any]) -> str:
    """Return the content digest for a read-only store audit projection."""

    without_digest = {
        key: value for key, value in payload.items() if key != "audit_digest"
    }
    return hashlib.sha256(_canonical_bytes(without_digest)).hexdigest()


class StructuralValidationArtifactStore:
    """Write and read immutable validation artifacts by their own digest.

    The store is deliberately narrower than lineage retention: it never decides
    which artifact is live and never deletes an artifact as a side effect of a
    read or write. Runtime admission remains responsible for candidate, batch,
    parent-checkpoint and replay validation.
    """

    def __init__(self, root: Path | str) -> None:
        raw_root = str(root)
        if not raw_root.strip():
            raise ValueError("structural artifact store root must not be empty")
        self.root = Path(root).expanduser()

    def path_for(self, artifact_digest: str) -> Path:
        digest = self._validate_digest(artifact_digest)
        return self.root / f"{digest}.json"

    def measurement_path_for(self, measurement_digest: str) -> Path:
        digest = self._validate_digest(measurement_digest)
        return self.root / f"{_MEASUREMENT_FILE_PREFIX}{digest}.json"

    def contains(self, artifact_digest: str) -> bool:
        return self.path_for(artifact_digest).is_file()

    def contains_measurement(self, measurement_digest: str) -> bool:
        return self.measurement_path_for(measurement_digest).is_file()

    def put_measured_artifact(
        self,
        artifact: WorkbenchStructuralValidationArtifact | Mapping[str, Any],
        measurements: StructuralValidationMeasurements | Mapping[str, Any],
    ) -> WorkbenchStructuralValidationArtifact:
        """Persist an artifact with independently verifiable measurement facts."""

        resolved_artifact = self._resolve_artifact(artifact)
        resolved_measurements = self._resolve_measurements(measurements)
        if not resolved_artifact.measurement_digest:
            raise ValueError("measured artifact must carry a measurement digest")
        if resolved_artifact.measurement_digest != resolved_measurements.measurement_digest:
            raise ValueError("artifact and measurement digests do not match")
        measurement_payload = resolved_measurements.to_payload()
        self._put_immutable_bytes(
            self.measurement_path_for(resolved_measurements.measurement_digest),
            _canonical_bytes(measurement_payload),
        )
        return self.put(resolved_artifact)

    def load_measurements(self, measurement_digest: str) -> StructuralValidationMeasurements:
        """Load and independently verify one content-addressed measurement sidecar."""

        target = self.measurement_path_for(measurement_digest)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("structural measurement payload must be a mapping")
        measurements = StructuralValidationMeasurements.from_payload(payload)
        if measurements.measurement_digest != str(measurement_digest):
            raise ValueError("structural measurement filename digest mismatch")
        if _canonical_bytes(payload) != target.read_bytes():
            raise ValueError("structural measurement bytes are not canonical")
        return measurements

    def load_verified_artifact(
        self,
        artifact_digest: str,
    ) -> WorkbenchStructuralValidationArtifact:
        """Load an artifact only when its measurement sidecar is independently verified."""

        artifact = self.load(artifact_digest)
        if not artifact.measurement_digest:
            raise ValueError("artifact has no independently verifiable measurement sidecar")
        measurements = self.load_measurements(artifact.measurement_digest)
        if measurements.measurement_digest != artifact.measurement_digest:
            raise ValueError("artifact and measurement sidecar digests do not match")
        return artifact

    def inventory(self) -> tuple[dict[str, Any], ...]:
        """Return a deterministic, read-only integrity view of the store.

        Inventory is deliberately independent from runtime lineage. Every
        committed JSON file must be digest-named and loadable as canonical
        artifact bytes; otherwise the whole audit fails closed. No file is
        repaired, removed, or made consumable by this method.
        """

        if not self.root.exists():
            return ()
        if not self.root.is_dir():
            raise ValueError("structural artifact store root must be a directory")
        with _STORE_WRITE_LOCK:
            targets = tuple(sorted(self.root.iterdir(), key=lambda item: item.name))
            artifact_targets: list[tuple[str, Path]] = []
            measurement_targets: dict[str, Path] = {}
            for target in targets:
                if not target.is_file() or target.suffix != ".json":
                    raise ValueError(
                        "structural artifact store contains an unexpected file: "
                        f"{target.name}"
                    )
                if _DIGEST_PATTERN.fullmatch(target.stem) is not None:
                    artifact_targets.append((target.stem, target))
                    continue
                if target.stem.startswith(_MEASUREMENT_FILE_PREFIX):
                    measurement_digest = self._validate_digest(
                        target.stem[len(_MEASUREMENT_FILE_PREFIX) :]
                    )
                    measurement_targets[measurement_digest] = target
                    continue
                raise ValueError(
                    "structural artifact store contains an unexpected file: "
                    f"{target.name}"
                )

            loaded_artifacts = [
                (digest, target, self.load(digest))
                for digest, target in artifact_targets
            ]
            referenced_measurement_digests = {
                artifact.measurement_digest
                for _, _, artifact in loaded_artifacts
                if artifact.measurement_digest
            }
            unreferenced_measurements = sorted(
                set(measurement_targets) - referenced_measurement_digests
            )
            if unreferenced_measurements:
                raise ValueError(
                    "structural artifact store contains an unreferenced measurement sidecar: "
                    f"{unreferenced_measurements[0]}"
                )
            records: list[dict[str, Any]] = []
            for filename_digest, _, artifact in loaded_artifacts:
                measurement_digest = artifact.measurement_digest
                if _DIGEST_PATTERN.fullmatch(measurement_digest) is None:
                    raise ValueError(
                        "structural artifact store measurement digest must be a lowercase "
                        "SHA-256 hex string"
                    )
                measurement_status = "unmeasured"
                if measurement_digest:
                    if measurement_digest in measurement_targets:
                        self.load_measurements(measurement_digest)
                        measurement_status = "verified"
                    else:
                        measurement_status = "legacy_unverified"
                records.append(
                    {
                        "artifact_digest": artifact.artifact_digest,
                        "measurement_digest": measurement_digest,
                        "measurement_status": measurement_status,
                        "candidate_id": artifact.candidate_id,
                        "network_id": artifact.network_id,
                        "region_id": artifact.region_id,
                        "task_slice_id": artifact.task_slice_id,
                        "parent_checkpoint_digest": artifact.parent_checkpoint_digest,
                        "trial_checkpoint_digest": artifact.trial_checkpoint_digest,
                        "holdout_gain": artifact.holdout_gain,
                        "retention_regression": artifact.retention_regression,
                        "lesion_effect": artifact.lesion_effect,
                        "resource_state": artifact.resource_state,
                        "resource_cost": artifact.resource_cost,
                    }
                )
            return tuple(records)

    def audit(self) -> tuple[dict[str, Any], ...]:
        """Alias for :meth:`inventory` emphasizing its integrity purpose."""

        return self.inventory()

    def put(
        self,
        artifact: WorkbenchStructuralValidationArtifact | Mapping[str, Any],
    ) -> WorkbenchStructuralValidationArtifact:
        resolved = self._resolve_artifact(artifact)
        payload = resolved.to_payload()
        encoded = _canonical_bytes(payload)
        target = self.path_for(resolved.artifact_digest)
        self._put_immutable_bytes(target, encoded)
        return resolved

    def _put_immutable_bytes(self, target: Path, encoded: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with _STORE_WRITE_LOCK:
            if target.exists():
                existing = target.read_bytes()
                if existing != encoded:
                    raise ValueError("structural artifact store content collision")
                return

            temporary = self.root / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                temporary.write_bytes(encoded)
                os.replace(temporary, target)
            except PermissionError:
                # Windows may reject a simultaneous replace after another
                # process has already installed the same immutable bytes.
                if target.is_file() and target.read_bytes() == encoded:
                    temporary.unlink(missing_ok=True)
                    return
                temporary.unlink(missing_ok=True)
                raise
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

    def load(self, artifact_digest: str) -> WorkbenchStructuralValidationArtifact:
        target = self.path_for(artifact_digest)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("structural artifact store payload must be a mapping")
        artifact = WorkbenchStructuralValidationArtifact.from_payload(payload)
        if artifact.artifact_digest != str(artifact_digest):
            raise ValueError("structural artifact store filename digest mismatch")
        if _canonical_bytes(payload) != target.read_bytes():
            raise ValueError("structural artifact store bytes are not canonical")
        return artifact

    @staticmethod
    def _validate_digest(artifact_digest: str) -> str:
        digest = str(artifact_digest)
        if _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError("structural artifact digest must be a lowercase SHA-256 hex string")
        return digest

    @staticmethod
    def _resolve_artifact(
        artifact: WorkbenchStructuralValidationArtifact | Mapping[str, Any],
    ) -> WorkbenchStructuralValidationArtifact:
        resolved = (
            WorkbenchStructuralValidationArtifact.from_payload(artifact)
            if isinstance(artifact, Mapping)
            else artifact
        )
        if not isinstance(resolved, WorkbenchStructuralValidationArtifact):
            raise TypeError("structural artifact store accepts a Workbench validation artifact")
        return resolved

    @staticmethod
    def _resolve_measurements(
        measurements: StructuralValidationMeasurements | Mapping[str, Any],
    ) -> StructuralValidationMeasurements:
        resolved = (
            StructuralValidationMeasurements.from_payload(measurements)
            if isinstance(measurements, Mapping)
            else measurements
        )
        if not isinstance(resolved, StructuralValidationMeasurements):
            raise TypeError(
                "structural artifact store accepts StructuralValidationMeasurements"
            )
        return resolved
