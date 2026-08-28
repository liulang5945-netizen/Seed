"""Native artifact inventory endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from api.seed_runtime import get_seed_runtime
from seed_platform.artifacts import ARTIFACT_TYPES
from seed_platform.paths import get_external_path
from seed_platform.settings import load_settings

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
def list_artifacts():
    """List platform-owned artifacts without exposing model-market semantics."""

    settings = load_settings()
    runtime_settings = settings.get("runtime", {})
    runtime = get_seed_runtime()
    active_id = (
        runtime.checkpoint_path.name
        if runtime is not None and runtime.checkpoint_path is not None
        else ""
    )
    checkpoint_root = Path(get_external_path("checkpoints"))
    checkpoints = []
    if checkpoint_root.is_dir():
        for path in sorted(checkpoint_root.glob("*.pt")):
            checkpoints.append(
                {
                    "artifact_type": "taiji_checkpoint",
                    "artifact_id": path.name,
                    "path": path.name,
                    "active": path.name == active_id,
                }
            )

    return {
        "status": "ok",
        "artifact_types": list(ARTIFACT_TYPES),
        "artifacts": checkpoints,
        "runtime": {
            "kind": "taiji",
            "configured_checkpoint_id": runtime_settings.get("checkpoint_id", ""),
            "active_checkpoint_id": active_id,
        },
        "language_provider": (
            runtime.status().get("language_provider") if runtime is not None else None
        ),
    }
