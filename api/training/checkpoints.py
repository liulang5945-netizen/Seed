"""检查点列表端点。

前端训练页通过 ``GET /api/train/checkpoints`` 拉取可恢复的检查点清单
（``useTraining.loadCheckpoints``）。本端点扫描 ``checkpoints/`` 目录，
读取每个信封的 ``metadata`` 块（tick / 保存时间 / 训练画像），并按
前端期望的字段形状（filename/epoch/step/loss/num_epochs）返回。
"""

import logging
import pickle
from pathlib import Path

import torch
from fastapi import APIRouter

logger = logging.getLogger("ApiServer.Training.Checkpoints")
router = APIRouter()

_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def _describe(path: Path) -> dict:
    """读信封元数据生成列表项；读取失败只降级元信息，不影响整体列表。"""
    item = {
        "filename": path.name,
        "epoch": 0,
        "step": 0,
        "loss": None,
        "num_epochs": 0,
        "bytes": path.stat().st_size,
        "modified_utc": "",
    }
    try:
        import time

        item["modified_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)
        )
        try:
            envelope = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            logger.warning(
                "checkpoint %s 含自定义对象，以不安全模式（weights_only=False）"
                "加载受信 checkpoint",
                path.name,
            )
            envelope = torch.load(path, map_location="cpu", weights_only=False)
        metadata = (envelope or {}).get("metadata") or {}
        item["step"] = int(metadata.get("tick", 0) or 0)
        item["saved_at_utc"] = metadata.get("saved_at_utc", "")
        profile = metadata.get("profile") or {}
        if profile:
            item["profile"] = profile
    except Exception as exc:
        logger.warning(f"checkpoint metadata read failed for {path.name}: {exc}")
        item["status"] = f"metadata_unreadable: {exc}"
    return item


@router.get("/api/train/checkpoints")
def list_checkpoints():
    """列出 checkpoints/ 目录下的可用检查点（按修改时间倒序）。"""
    try:
        if not _CHECKPOINT_DIR.is_dir():
            return {"status": "ok", "checkpoints": []}
        paths = sorted(
            _CHECKPOINT_DIR.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return {"status": "ok", "checkpoints": [_describe(p) for p in paths]}
    except Exception as exc:
        logger.error(f"list_checkpoints failed: {exc}")
        return {"status": "error", "checkpoints": [], "message": str(exc)}
