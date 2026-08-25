"""Native Taiji byte-stream training endpoint."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from seed import Seed, SeedConfig
from seed.datasets import inspect_native_dataset
from seed_platform.app_state import app_state
from taiji import TaijiConfig

from .resume import (
    _CHECKPOINT_DIR,
    _DEFAULT_CORPUS,
    _resolve_datasets,
    _train_worker,
    stream_training_events,
)

router = APIRouter()


class NativeTrainRequest(BaseModel):
    datasets: list[str] | None = None
    parameter_budget: int = 300_000
    max_symbols: int | None = None
    device: str = "auto"
    seed: int = 20260822


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"无效训练设备: {value}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="原生训练设备只支持 cpu、cuda 或 auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HTTPException(
            status_code=400, detail="请求了 CUDA，但当前 PyTorch 没有可用 CUDA 设备"
        )
    return device


def _resolve_native_datasets(names: list[str] | None) -> list[Path]:
    if names:
        resolved, missing = _resolve_datasets(names)
        if missing:
            raise HTTPException(status_code=404, detail=f"数据集不存在: {', '.join(missing)}")
        paths: list[Path] = [Path(p) for p in resolved]
    else:
        if not _DEFAULT_CORPUS.is_file():
            raise HTTPException(status_code=404, detail="默认原生语料缺失，请先上传 JSONL 数据集")
        paths = [_DEFAULT_CORPUS]

    invalid = []
    for path in paths:
        report = inspect_native_dataset(path)
        if not report.native_trainable:
            invalid.append(report.to_dict())
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"message": "数据集不符合 Seed 原生 text/raw-byte 合同", "datasets": invalid},
        )
    return paths


@router.post("/api/train/native")
def train_native(req: NativeTrainRequest):
    """Start asynchronous training on the native Taiji byte stream."""

    if req.parameter_budget <= 0:
        raise HTTPException(status_code=400, detail="parameter_budget 必须为正数")
    if req.max_symbols is not None and req.max_symbols <= 0:
        raise HTTPException(status_code=400, detail="max_symbols 必须为正数")

    device = _resolve_device(req.device)
    corpus_paths = _resolve_native_datasets(req.datasets)
    reports = [inspect_native_dataset(path) for path in corpus_paths]
    total_bytes = sum(report.total_text_bytes for report in reports)

    try:
        config = SeedConfig(taiji=TaijiConfig.capacity_profile(req.parameter_budget, seed=req.seed))
        model = Seed(config, device=device, episode_id="api-native-training")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法创建原生训练基底: {exc}") from exc

    if not app_state.try_start_training():
        raise HTTPException(status_code=409, detail="已有训练任务在运行，请先停止")

    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _CHECKPOINT_DIR / "seed_native.pt"
    event_q: queue.Queue = queue.Queue(maxsize=256)
    thread = threading.Thread(
        target=_train_worker,
        args=(
            model,
            corpus_paths,
            total_bytes,
            save_path,
            event_q,
            req.max_symbols,
            "开始 Taiji 原生 byte-stream 训练",
        ),
        daemon=True,
        name="native-train",
    )
    app_state._trainer_ref = thread
    thread.start()
    return StreamingResponse(
        stream_training_events(event_q, thread), media_type="text/event-stream"
    )
