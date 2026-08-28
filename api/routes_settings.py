"""Settings management API routes."""

import logging

from fastapi import APIRouter, HTTPException

from api.deprecation import gone_response
from api.seed_runtime import get_seed_runtime
from seed_platform.app_state import app_state
from seed_platform.memory import force_memory_refresh, get_memory_status_dict
from seed_platform.settings import load_settings, update_settings

logger = logging.getLogger("ApiServer.Settings")
router = APIRouter()


@router.get("/api/settings")
def get_all_settings():
    """Return all persisted settings."""
    try:
        return load_settings()
    except Exception as exc:
        logger.warning(f"Failed to read settings: {exc}")
        return {}


@router.post("/api/settings")
async def save_all_settings(req: dict):
    """Merge and persist settings."""
    try:
        update_settings(req)
        return {"status": "ok", "message": "Settings saved"}
    except Exception as exc:
        logger.warning(f"Failed to save settings: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {exc}") from exc


@router.post("/api/settings/model", include_in_schema=False)
async def set_model(req: dict):
    """Retired model-name setting; use the canonical runtime setting."""
    del req
    return gone_response(
        replacement="/api/settings/runtime",
        message="model_name/model_type 已退出产品设置，请使用 Taiji runtime 配置。",
    )


@router.post("/api/settings/device", include_in_schema=False)
async def set_device(req: dict):
    """Device settings now go through the unified settings object."""
    del req
    return gone_response(
        replacement="/api/settings",
        message="device 设置已并入统一 settings 接口。",
    )


@router.post("/api/settings/quant", include_in_schema=False)
async def set_quant(req: dict):
    """Retired Transformer quantization setting."""
    del req
    return gone_response(
        replacement="/api/settings",
        message="Transformer/GGUF quantization设置已退出 Taiji 产品运行时。",
    )


@router.post("/api/settings/gguf", include_in_schema=False)
async def set_gguf_settings(req: dict):
    """Retired GGUF setting."""
    del req
    return gone_response(
        replacement="/api/artifacts",
        message="GGUF 不属于 Taiji artifact；请使用原生 checkpoint 或语言 provider artifact。",
    )


@router.get("/api/settings/gguf_models", include_in_schema=False)
async def list_gguf_models():
    """Retired GGUF inventory."""
    return gone_response(
        replacement="/api/artifacts",
        message="GGUF 模型清单已退出产品 API。",
    )


@router.post("/api/settings/download_gguf", include_in_schema=False)
async def download_gguf(req: dict):
    """Retired GGUF download."""
    del req
    return gone_response(
        replacement="/api/artifacts",
        message="GGUF 下载已退出产品 API。",
    )


@router.get("/api/settings/runtime")
def get_runtime_settings():
    """Return the persisted Taiji runtime selection."""
    settings = load_settings()
    runtime = settings.get("runtime", {})
    return {
        "status": "ok",
        "runtime_kind": "taiji",
        "checkpoint_id": runtime.get("checkpoint_id", ""),
        "schema_version": settings.get("schema_version", 2),
    }


@router.post("/api/settings/runtime")
async def set_runtime_settings(req: dict):
    """Persist a Taiji checkpoint selection without activating it."""
    checkpoint_id = str(req.get("checkpoint_id", "") or "").strip()
    if checkpoint_id and (
        checkpoint_id.startswith(("/", "\\")) or ".." in checkpoint_id.replace("\\", "/").split("/")
    ):
        raise HTTPException(status_code=400, detail="checkpoint_id 必须是相对路径")
    update_settings({"runtime": {"kind": "taiji", "checkpoint_id": checkpoint_id}})
    return {"status": "ok", "runtime_kind": "taiji", "checkpoint_id": checkpoint_id}


@router.get("/api/system/current_runtime")
def get_current_runtime():
    """Return the effective Taiji runtime and language-provider status."""
    try:
        saved = load_settings()
        configured = saved.get("runtime", {})
        runtime = get_seed_runtime()
        active_id = (
            runtime.checkpoint_path.name
            if runtime is not None and runtime.checkpoint_path is not None
            else ""
        )
        configured_id = configured.get("checkpoint_id", "")
        provider = runtime.status().get("language_provider") if runtime else None
        return {
            "status": "ok",
            "runtime_kind": "taiji",
            "active": runtime is not None,
            "loaded": runtime is not None and app_state.startup_complete,
            "checkpoint_id": active_id or configured_id,
            "configured_checkpoint_id": configured_id,
            "pending": bool(configured_id and configured_id != active_id),
            "language_provider": provider,
        }
    except Exception as exc:
        logger.warning(f"Failed to get current runtime info: {exc}")
        logger.error(f"Memory status failed: {exc}")
        return {"status": "error", "message": "内部错误，请查看日志", "loaded": False}


@router.get("/api/system/current_model", include_in_schema=False)
def get_current_model():
    """Compatibility tombstone for the removed global model contract."""
    return gone_response(
        replacement="/api/system/current_runtime",
        message="全局 model_type/model_name 合同已退出，请使用 Taiji runtime。",
    )


@router.get("/api/system/memory")
def get_memory_status():
    """Return the current memory status."""
    return get_memory_status_dict()


@router.post("/api/system/memory/refresh")
def refresh_memory_status():
    """Force-refresh memory status."""
    force_memory_refresh()
    return get_memory_status_dict()
