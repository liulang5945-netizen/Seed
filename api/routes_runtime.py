"""Runtime status and Taiji activation API.

Endpoints:
- GET /api/runtime/bootstrap  — public, minimal info for unauthenticated clients
- GET /api/runtime/status     — full status, requires auth if auth is enabled
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.models_runtime import (
    BootstrapPayload,
    RuntimeActivationRequest,
    RuntimeStatusPayload,
)
from api.seed_runtime import activate_seed
from seed_platform.app_state import app_state
from seed_platform.paths import get_external_path
from seed_platform.runtime_service import get_bootstrap_status, get_runtime_status
from seed_platform.settings import update_settings

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("/bootstrap", response_model=BootstrapPayload)
async def runtime_bootstrap():
    """Public endpoint — no auth required.

    Returns minimal info so the client shell can decide whether to
    show a login screen or proceed to load the full runtime status.
    """
    return get_bootstrap_status()


@router.get("/status", response_model=RuntimeStatusPayload)
async def runtime_status(request: Request):
    """Full runtime status payload.

    If auth is enabled, the client must send a valid Bearer token.
    The auth field in the response reflects the token validity.
    """
    return get_runtime_status(request.headers.get("Authorization", ""))


def _resolve_checkpoint(checkpoint_id: str | None) -> tuple[str, str]:
    requested = str(checkpoint_id or "").strip()
    root = Path(get_external_path("checkpoints")).resolve()
    if not requested:
        return "", ""
    candidate = Path(requested)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="checkpoint_id 必须是检查点目录下的相对路径")
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="checkpoint_id 超出检查点目录")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"检查点不存在: {requested}")
    return str(resolved), candidate.as_posix()


@router.post("/activate")
def runtime_activate(req: RuntimeActivationRequest):
    """Activate the Taiji runtime from a platform-owned checkpoint."""

    checkpoint_path, checkpoint_id = _resolve_checkpoint(req.checkpoint_id)
    try:
        runtime = activate_seed(checkpoint_path or None)
        app_state.mark_started()
        update_settings({"runtime": {"kind": "taiji", "checkpoint_id": checkpoint_id}})
        return {
            "status": "ok",
            "runtime_kind": "taiji",
            "checkpoint_id": checkpoint_id
            or (runtime.checkpoint_path.name if runtime.checkpoint_path else ""),
            "active": True,
            "runtime": runtime.status(),
        }
    except Exception as exc:
        app_state.mark_startup_failed(str(exc))
        raise HTTPException(status_code=500, detail=f"Taiji runtime 激活失败: {exc}") from exc
