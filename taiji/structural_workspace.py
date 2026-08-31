"""Workspace routing whose capacity follows a live Taiji neuron region."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import WorkspaceCandidate, WorkspaceSelection
from .neuron_region import AdaptiveNeuronRegion
from .workspace import WorkspaceRouter, WorkspaceRoutingExample

STRUCTURAL_WORKSPACE_CHECKPOINT_FORMAT = "taiji-structural-workspace-router-v1"
STRUCTURAL_WORKSPACE_MODEL_REVISION = 1


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr).encode(
            "utf-8"
        )
    ).hexdigest()


class StructuralWorkspaceRouter:
    """Bind workspace slot capacity to a Taiji-owned adaptive neuron region.

    The region contributes only a bounded capacity signal.  It does not name
    tools, actions, languages, or semantic roles; the existing content router
    remains responsible for scoring candidates and selecting the broadcast.
    """

    def __init__(
        self,
        router: WorkspaceRouter,
        region: AdaptiveNeuronRegion,
        *,
        minimum_capacity: int = 1,
        selection_threshold: float = 0.5,
    ) -> None:
        if not isinstance(router, WorkspaceRouter):
            raise TypeError("structural workspace router requires a WorkspaceRouter")
        if not isinstance(region, AdaptiveNeuronRegion):
            raise TypeError("structural workspace router requires an AdaptiveNeuronRegion")
        if int(minimum_capacity) <= 0:
            raise ValueError("structural workspace minimum_capacity must be positive")
        if not 0.0 <= float(selection_threshold) <= 1.0:
            raise ValueError("structural workspace selection_threshold must be in [0, 1]")
        if region.unit_count - len(region.lesioned_unit_ids) < int(minimum_capacity):
            raise ValueError("adaptive region is below the structural workspace minimum")
        self.router = router
        self.region = region
        self.minimum_capacity = int(minimum_capacity)
        self.selection_threshold = float(selection_threshold)

    @property
    def capacity(self) -> int:
        """Return the current live neuron count used as workspace capacity."""

        return max(0, self.region.unit_count - len(self.region.lesioned_unit_ids))

    def rebind(self, region: AdaptiveNeuronRegion) -> None:
        """Rebind after a topology transaction replaces the region object.

        Structural rollback restores a checkpoint by replacing the adaptive
        region instance.  Keeping the old Python object would make the router
        report stale capacity, so rebinding is explicit and identity-checked.
        """

        if not isinstance(region, AdaptiveNeuronRegion):
            raise TypeError("structural workspace router requires an AdaptiveNeuronRegion")
        if region.region_id != self.region.region_id:
            raise ValueError("structural workspace rebind region mismatch")
        if region.unit_count - len(region.lesioned_unit_ids) < self.minimum_capacity:
            raise ValueError("rebound adaptive region is below the structural workspace minimum")
        self.region = region

    def fit(
        self,
        examples: Iterable[WorkspaceRoutingExample],
        *,
        epochs: int = 80,
        learning_rate: float = 0.1,
    ) -> float:
        """Fit the content scorer without changing structural capacity."""

        examples = tuple(examples)
        if any(len(example.relevant_ids) > self.capacity for example in examples):
            raise ValueError("workspace examples exceed live structural capacity")
        return self.router.fit(examples, epochs=epochs, learning_rate=learning_rate)

    def route(
        self,
        candidates: Iterable[WorkspaceCandidate],
        *,
        tick: int,
        mode: str = "learned",
        random_seed: int | None = None,
    ) -> WorkspaceSelection:
        """Route with the current region capacity and return an auditable selection."""

        return self.router.route(
            candidates,
            tick=tick,
            mode=mode,
            random_seed=random_seed,
            capacity=self.capacity,
            minimum_score=(
                self.selection_threshold if mode == "learned" else None
            ),
        )

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": STRUCTURAL_WORKSPACE_CHECKPOINT_FORMAT,
            "model_revision": STRUCTURAL_WORKSPACE_MODEL_REVISION,
            "minimum_capacity": self.minimum_capacity,
            "selection_threshold": self.selection_threshold,
            "region_id": self.region.region_id,
            "region_unit_ids": list(self.region.unit_ids),
            "region_lesioned_unit_ids": list(self.region.lesioned_unit_ids),
            "router": self.router.checkpoint(),
        }
        payload["checkpoint_digest"] = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        region: AdaptiveNeuronRegion,
    ) -> StructuralWorkspaceRouter:
        if payload.get("format") != STRUCTURAL_WORKSPACE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural workspace checkpoint format")
        if int(payload.get("model_revision", -1)) != STRUCTURAL_WORKSPACE_MODEL_REVISION:
            raise ValueError("unsupported structural workspace model revision")
        expected = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest")) != expected:
            raise ValueError("structural workspace checkpoint digest mismatch")
        if str(payload.get("region_id")) != region.region_id:
            raise ValueError("structural workspace checkpoint region mismatch")
        if tuple(str(item) for item in payload.get("region_unit_ids", ())) != region.unit_ids:
            raise ValueError("structural workspace checkpoint neuron identities drifted")
        if tuple(str(item) for item in payload.get("region_lesioned_unit_ids", ())) != region.lesioned_unit_ids:
            raise ValueError("structural workspace checkpoint neuron lesion state drifted")
        router_payload = payload.get("router")
        if not isinstance(router_payload, Mapping):
            raise ValueError("structural workspace checkpoint router must be a mapping")
        return cls(
            WorkspaceRouter.from_checkpoint(dict(router_payload)),
            region,
            minimum_capacity=int(payload["minimum_capacity"]),
            selection_threshold=float(payload.get("selection_threshold", 0.5)),
        )


__all__ = [
    "STRUCTURAL_WORKSPACE_CHECKPOINT_FORMAT",
    "STRUCTURAL_WORKSPACE_MODEL_REVISION",
    "StructuralWorkspaceRouter",
]
