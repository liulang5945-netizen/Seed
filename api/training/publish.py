"""
Seed — 模型发布 API（精简版）
Taiji 原生检查点不使用传统 save_model，保留 published 列表查询和 GGUF 不支持消息。
"""

import json as _json
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from seed_platform.paths import get_external_path

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/model/publish")
async def publish_model():
    """保存当前模型（Taiji 使用原生 checkpoint 落盘）"""
    raise HTTPException(
        status_code=503,
        detail="Taiji 使用 seed-native checkpoint；传统模型发布接口不适用。",
    )


@router.get("/api/model/published")
def list_published_models():
    """列出已发布（保存）的模型"""
    checkpoint_dir = get_external_path("checkpoints")
    published = []
    if os.path.isdir(checkpoint_dir):
        for name in sorted(os.listdir(checkpoint_dir)):
            full = os.path.join(checkpoint_dir, name)
            if os.path.isdir(full) and name.startswith("published_"):
                config_path = os.path.join(full, "config.json")
                published.append(
                    {
                        "name": name,
                        "path": full,
                        "has_config": os.path.exists(config_path),
                    }
                )
    return {"status": "ok", "published": published}


@router.post("/api/model/export_gguf")
async def export_gguf():
    """GGUF 导出（Seed不支持）"""
    return StreamingResponse(
        _stream_error("Seed不支持 GGUF 导出"),
        media_type="text/event-stream",
    )


async def _stream_error(msg: str):
    yield f"data: {_json.dumps({'type': 'error', 'message': msg})}\n\n"


@router.get("/api/model/export_gguf/options")
def get_gguf_export_options():
    """GGUF 量化选项（Seed不支持）"""
    return {"options": [], "message": "Seed不支持 GGUF 导出"}
