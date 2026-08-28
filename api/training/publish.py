"""Compatibility tombstones for retired model publishing APIs."""

from fastapi import APIRouter

from api.deprecation import gone_response

router = APIRouter()


@router.post("/api/model/publish", include_in_schema=False)
async def publish_model():
    """Compatibility tombstone for the removed model publish API."""
    return gone_response(
        replacement="/api/train/native",
        message="旧模型发布接口已退出；Taiji 训练使用 native checkpoint。",
    )


@router.get("/api/model/published", include_in_schema=False)
def list_published_models():
    """Compatibility tombstone for the removed publish catalog."""
    return gone_response(
        replacement="/api/artifacts",
        message="旧发布清单已退出；请使用 Taiji artifact inventory。",
    )


@router.post("/api/model/export_gguf", include_in_schema=False)
async def export_gguf():
    """Compatibility tombstone for the removed GGUF exporter."""
    return gone_response(
        replacement="/api/artifacts",
        message="GGUF 导出已退出 Taiji 产品 API。",
    )


@router.get("/api/model/export_gguf/options", include_in_schema=False)
def get_gguf_export_options():
    """Compatibility tombstone for the removed GGUF options."""
    return gone_response(
        replacement="/api/artifacts",
        message="GGUF 量化选项已退出 Taiji 产品 API。",
    )
