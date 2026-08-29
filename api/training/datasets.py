"""
数据集管理 API 路由
上传、列表、删除、预览微调数据集
"""

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from seed.datasets import inspect_native_dataset, iter_native_documents
from seed_platform.paths import get_external_path

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()

# Seed 原生可训练的文件后缀：JSON 家族按 text 字段解析，其余按整篇 UTF-8 文本。
NATIVE_DATASET_SUFFIXES = frozenset({".jsonl", ".ndjson", ".json", ".txt", ".text", ".md", ".csv"})
# data/ 顶层的平台配置文件，不是语料，必须从数据集列表中排除。
RESERVED_DATA_FILES = frozenset({"app_settings.json", "runtime_preference.json"})
# 递归扫描时跳过的目录名（缓存、版本控制、检查点等非语料目录）。
_SKIPPED_DIRS = frozenset({"__pycache__", ".git", ".ipynb_checkpoints", "checkpoints", "logs"})
_MAX_SCAN_DEPTH = 4
# 预览/校验只采样前若干条记录，避免对 GB 级语料做全量扫描导致请求超时。
PREVIEW_SCAN_RECORDS = 2_000


@router.post("/api/train/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """接收前端上传的微调数据集"""
    try:
        data_dir = get_external_path("data")
        os.makedirs(data_dir, exist_ok=True)
        upload_name = file.filename or "dataset"
        file_path = os.path.join(data_dir, os.path.basename(upload_name))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "path": f"data/{file.filename}",
            "message": f"数据集 `{file.filename}` 已成功上传并选中！",
        }
    except Exception as e:
        logger.error(f"数据集上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _get_all_data_dirs() -> list:
    """获取所有可能的数据目录（解决打包/开发环境路径不一致问题）"""
    dirs = []
    primary = get_external_path("data")
    dirs.append(primary)
    # 回退：项目根目录下的 data/
    import sys

    if getattr(sys, "frozen", False):
        project_root = os.path.dirname(sys.executable)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(project_root, "data")
    if fallback not in dirs:
        dirs.append(fallback)
    return dirs


def _scan_datasets(data_dir: str) -> list[tuple[str, int]]:
    """递归扫描单个数据目录，返回 (相对 POSIX 路径, 字节数) 列表。"""
    found: list[tuple[str, int]] = []
    root = os.path.abspath(data_dir)
    for current, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(current, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d not in _SKIPPED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in NATIVE_DATASET_SUFFIXES:
                continue
            # 平台配置文件只在 data/ 顶层保留语义，子目录同名文件视为语料。
            if depth == 0 and filename in RESERVED_DATA_FILES:
                continue
            absolute = os.path.join(current, filename)
            try:
                size = os.path.getsize(absolute)
            except OSError:
                continue
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            found.append((relative, size))
    return found


def resolve_dataset_path(relative: str) -> str | None:
    """把前端给出的相对路径解析为真实文件绝对路径（防目录穿越）。"""
    candidate_names = [relative]
    # 兼容历史行为：旧前端可能只回传 basename。
    basename = os.path.basename(relative)
    if basename and basename != relative:
        candidate_names.append(basename)
    for data_dir in _get_all_data_dirs():
        root = os.path.abspath(data_dir)
        for name in candidate_names:
            absolute = os.path.abspath(os.path.join(root, name.replace("/", os.sep)))
            if not (absolute == root or absolute.startswith(root + os.sep)):
                continue
            if os.path.isfile(absolute):
                return absolute
        # basename 未命中时，在该目录下递归找同名文件。
        for rel, _size in _scan_datasets(root):
            if os.path.basename(rel) == basename:
                return os.path.join(root, rel.replace("/", os.sep))
    return None


@router.get("/api/train/files")
def list_train_files():
    """递归列出 data 目录下的原生可训练数据集（相对路径，已排除平台配置文件）。"""
    os.makedirs(get_external_path("data"), exist_ok=True)
    sizes: dict[str, int] = {}
    for data_dir in _get_all_data_dirs():
        if not os.path.isdir(data_dir):
            continue
        for relative, size in _scan_datasets(data_dir):
            sizes.setdefault(relative, size)
    names = sorted(sizes)
    return {
        "files": names,
        "entries": [{"path": name, "size_bytes": sizes[name]} for name in names],
    }


@router.delete("/api/train/file/{filename:path}")
def delete_train_file(filename: str):
    """删除指定的数据集文件（支持 data/ 下的相对路径）"""
    try:
        resolved = resolve_dataset_path(filename)
        if resolved is None:
            return {"status": "error", "message": "文件不存在"}
        os.remove(resolved)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/train/preview/{filename:path}")
def train_preview(filename: str):
    """按 Seed 原生合同预览数据集（UTF-8 文本文档，JSON 家族取 text 字段）。"""
    resolved = resolve_dataset_path(filename)
    if resolved is None:
        return {"samples": [], "count": 0, "native_trainable": False, "report": None}
    try:
        report = inspect_native_dataset(resolved, max_records=PREVIEW_SCAN_RECORDS)
        samples = []
        for text in iter_native_documents([Path(resolved)]):
            samples.append({"text": text[:500]})
            if len(samples) >= 5:
                break
        return {
            "samples": samples,
            "count": report.documents,
            "native_trainable": report.native_trainable,
            "report": report.to_dict(),
        }
    except Exception as e:
        logger.warning(f"数据集预览失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
