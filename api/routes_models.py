"""Compatibility tombstones for the removed model-market API.

Taiji does not expose a global model catalog, model type switch, HF download,
or GGUF lifecycle. The router remains only long enough to give existing
clients a typed migration response; every route is hidden from OpenAPI.
"""

from fastapi import APIRouter

from api.deprecation import gone_response

router = APIRouter()


def _gone():
    return gone_response(
        replacement="/api/artifacts",
        message="旧模型市场/下载 API 已退出，请使用 Taiji artifact inventory。",
    )


@router.get("/api/models/installed", include_in_schema=False)
def list_installed_models():
    return _gone()


@router.get("/api/models/list", include_in_schema=False)
def list_available_models():
    return _gone()


@router.get("/api/models/downloaded", include_in_schema=False)
def list_downloaded_models():
    return _gone()


@router.get("/api/model/gguf_quants", include_in_schema=False)
def get_gguf_quants():
    return _gone()


@router.get("/api/models/recommend", include_in_schema=False)
def recommend_models():
    return _gone()


@router.get("/api/models/tags", include_in_schema=False)
def list_model_tags():
    return _gone()


@router.get("/api/models/families", include_in_schema=False)
def list_model_families():
    return _gone()


@router.get("/api/models/info", include_in_schema=False)
def get_model_info():
    return _gone()


@router.post("/api/models/download_hf", include_in_schema=False)
def download_hf_model():
    return _gone()


@router.post("/api/models/download", include_in_schema=False)
def download_model():
    return _gone()


@router.post("/api/models/download_cancel", include_in_schema=False)
def cancel_download():
    return _gone()


@router.post("/api/models/download_pause", include_in_schema=False)
def pause_download():
    return _gone()


@router.post("/api/models/download_resume", include_in_schema=False)
def resume_download():
    return _gone()


@router.get("/api/models/download_progress", include_in_schema=False)
def get_download_progress():
    return _gone()


@router.delete("/api/models/installed", include_in_schema=False)
def delete_installed_model():
    return _gone()


@router.post("/api/models/delete", include_in_schema=False)
def delete_model():
    return _gone()


@router.post("/api/models/select", include_in_schema=False)
def select_model():
    return _gone()
