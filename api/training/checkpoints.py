"""检查点端点：清单与按名删除。

前端训练页通过 ``GET /api/train/checkpoints`` 拉取可恢复的检查点清单
（``useTraining.loadCheckpoints``）。本端点扫描 ``checkpoints/`` 目录，
读取每个信封的 ``metadata`` 块（tick / 保存时间 / 训练画像），并按
前端期望的字段形状（filename/epoch/step/loss/num_epochs）返回。

``DELETE /api/train/checkpoint/{filename}`` 按名删除一枚检查点：旧检查点会
长期堆积，训练面板需要能挑着删；正在使用（活跃应答）或已配置（下次启动加载）
的检查点被拒——删掉它们会直接破坏运行中的应答与下一次启动。
"""

import logging
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException

from api.seed_runtime import get_seed_runtime
from seed_platform.settings import load_settings

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
        # Metadata listing must never execute an unsupported pickle payload.
        envelope = torch.load(path, map_location="cpu", weights_only=True)
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
        # DEBT-G15：`pathlib.glob("*.pt")` **会**匹配点前缀文件（`glob.glob` 才不匹配，实测见
        # 登记处），所以崩溃/中断留下的 `.xxx.pt` 临时件曾被当成可用基座暴露给训练面板。
        paths = sorted(
            (p for p in _CHECKPOINT_DIR.glob("*.pt") if not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return {"status": "ok", "checkpoints": [_describe(p) for p in paths]}
    except Exception as exc:
        logger.error(f"list_checkpoints failed: {exc}")
        return {"status": "error", "checkpoints": [], "message": str(exc)}


def _active_checkpoint_id() -> str:
    """当前活跃（承载应答）的检查点名；运行时未加载时为空串。"""
    runtime = get_seed_runtime()
    if runtime is None or runtime.checkpoint_path is None:
        return ""
    return runtime.checkpoint_path.name


def _configured_checkpoint_id() -> str:
    """下次启动将加载的检查点名（settings.runtime.checkpoint_id）。"""
    settings = load_settings()
    runtime_settings = settings.get("runtime", {}) if isinstance(settings, dict) else {}
    return str(runtime_settings.get("checkpoint_id", "") or "")


def _usable_checkpoint_name(filename: str) -> str:
    """校验并返回平坦的检查点文件名；目录分离、点前缀件与非法字符一律拒。"""
    if filename in ("", ".", "..") or any(sep in filename for sep in ("/", "\\", ":")):
        raise HTTPException(status_code=400, detail="检查点名称不合法")
    if not filename.endswith(".pt") or filename.startswith("."):
        raise HTTPException(status_code=400, detail="检查点名称不合法")
    return filename


@router.delete("/api/train/checkpoint/{filename}")
def delete_checkpoint(filename: str):
    """按名删除一枚检查点；活跃或已配置（下次启动加载）的检查点拒绝删除。"""
    name = _usable_checkpoint_name(filename)
    path = _CHECKPOINT_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"检查点 {name} 不存在")
    if name == _active_checkpoint_id():
        raise HTTPException(status_code=409, detail=f"检查点 {name} 正在使用（活跃模型），不能删除")
    if name == _configured_checkpoint_id():
        raise HTTPException(status_code=409, detail=f"检查点 {name} 已配置为下次启动加载，不能删除")
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info(f"checkpoint deleted: {name}")
    return {"status": "success", "message": f"检查点 {name} 已删除"}
