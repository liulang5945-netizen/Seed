"""检查点续训端点。

前端训练页"恢复训练"按钮（``useTraining.resumeFromCheckpoint``）POST
``{checkpoint, datasets?}`` 到 ``/api/train/resume_checkpoint``，并消费 SSE 流：
``data: {...}`` 事件（hardware_diag / progress / warning / error / completed）
与结束标记 ``data: [DONE]``。

实现为字节级流式训练循环（语义对齐 ``scripts/training/train_seed_corpus.py``）：
加载 seed-native-v1 信封 → 逐字节 ``Seed.observe`` → 周期 SSE 进度 + 原子落盘。
训练锁由 ``app_state`` 管理（与暂停/停止端点联动）；客户端断开即请求停止，
避免无人值守的后台算力消耗。续训产物写到 ``resumed_<原名>.pt``，不覆盖
原检查点（M1 长训可能正在周期性重写 ``seed_beta.pt``）。
"""

import json
import logging
import pickle
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from seed_platform.app_state import app_state
from seed import Seed
from seed.persistence import atomic_save, attach_metadata, corpus_fingerprint

from .common import collect_hardware_diag, safe_put
from .datasets import _get_all_data_dirs

logger = logging.getLogger("ApiServer.Training.Resume")
router = APIRouter()

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _ROOT / "checkpoints"
# 未选数据集时的回退语料（与 train_seed_corpus.DEFAULT_CORPUS 一致）。
_DEFAULT_CORPUS = _ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"

PROGRESS_EVERY = 2_000  # ≈311 ticks/s 下约 6.4s 一条进度
CHECKPOINT_EVERY = 250_000  # 周期落盘间隔（ticks）


class ResumeRequest(BaseModel):
    checkpoint: str
    datasets: Optional[List[str]] = None
    # 测试/冒烟用的预算上限；前端不发送，缺省跑满整个数据集一遍。
    max_ticks: Optional[int] = None


def _resolve_datasets(names: List[str]):
    """在全部数据目录中解析数据集文件名（防目录穿越，仅取 basename）。"""
    dirs = [Path(d) for d in _get_all_data_dirs()]
    resolved: List[Path] = []
    missing: List[str] = []
    for name in names:
        safe_name = Path(name).name
        hit = next((d / safe_name for d in dirs if (d / safe_name).is_file()), None)
        if hit is None:
            missing.append(safe_name)
        elif hit not in resolved:
            resolved.append(hit)
    return resolved, missing


def _row_text(obj) -> str:
    """兼容 text / instruction-output / question-answer 三类 jsonl 行。"""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    text = obj.get("text")
    if text:
        return str(text)
    parts = [
        obj.get(key)
        for key in ("instruction", "input", "prompt", "question", "output", "response", "answer")
        if obj.get(key)
    ]
    return "\n".join(str(part) for part in parts) if parts else ""


def _iter_symbols(paths: List[Path], boundary: int):
    """逐行流式产出 (符号, 该行字节数)；字节数用于进度分母统计。"""
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = _row_text(json.loads(line))
                except json.JSONDecodeError:
                    text = line
                if not text:
                    continue
                raw = text.encode("utf-8")
                yield boundary, 0
                for symbol in raw:
                    yield symbol, 1


def _train_worker(
    model: Seed,
    corpus_paths: List[Path],
    total_bytes: int,
    save_path: Path,
    event_q: "queue.Queue",
    max_ticks: Optional[int],
) -> None:
    """后台训练线程：逐字节观察语料，周期发进度与落盘，响应停止/暂停。"""
    boundary = model.config.taiji.boundary_symbol
    fingerprint = corpus_fingerprint(corpus_paths)
    start_tick = model.tick
    ticks = 0
    consumed = 0
    window_ticks = 0
    window_surprise = 0.0
    started = time.perf_counter()
    stopped = False

    safe_put(event_q, {"type": "hardware_diag", **collect_hardware_diag("cpu")})
    safe_put(
        event_q,
        {
            "type": "progress",
            "fraction": 0.0,
            "desc": f"已从检查点恢复（tick {start_tick:,}），开始流式续训…",
            "step": start_tick,
            "epoch": 1,
            "total_epochs": 1,
            "total_steps": total_bytes,
        },
    )

    def _persist() -> None:
        envelope = attach_metadata(
            model.checkpoint(),
            tick=model.tick,
            corpus_fingerprint=fingerprint,
            extra={"trainer": "resume_checkpoint"},
        )
        atomic_save(envelope, save_path)

    def _emit_progress() -> None:
        elapsed = time.perf_counter() - started
        fraction = min(consumed / max(1, total_bytes), 0.999)
        mean_surprise = window_surprise / max(1, window_ticks)
        rate = ticks / max(1e-6, elapsed)
        safe_put(
            event_q,
            {
                "type": "progress",
                "fraction": fraction,
                "desc": (
                    f"tick {model.tick:,} | 窗口平均 surprise {mean_surprise:.3f}"
                    f" | {rate:.0f} 字节/s"
                ),
                "step": model.tick,
                "loss": mean_surprise,
                "elapsed": round(elapsed, 1),
                "eta": round(elapsed * (1 - fraction) / max(fraction, 1e-6), 1),
                "epoch": 1,
                "total_epochs": 1,
                "samples_per_sec": round(rate, 1),
                "total_steps": total_bytes,
            },
        )

    try:
        for symbol, byte_len in _iter_symbols(corpus_paths, boundary):
            if app_state.stop_training_requested:
                stopped = True
                break
            while app_state.pause_training_requested:
                if app_state.stop_training_requested:
                    break
                time.sleep(0.5)
            if app_state.stop_training_requested:
                stopped = True
                break
            step = model.observe(symbol, learn=True)
            ticks += 1
            consumed += byte_len
            if step.prior_prediction is not None:
                window_ticks += 1
                window_surprise += float(step.surprise)
            if ticks % PROGRESS_EVERY == 0:
                _emit_progress()
                window_ticks = 0
                window_surprise = 0.0
            if ticks % CHECKPOINT_EVERY == 0:
                _persist()
            if max_ticks is not None and ticks >= max_ticks:
                break
        if window_ticks > 0:  # 收尾进度只在有观测窗口时发，避免 loss=0 假象
            _emit_progress()
        _persist()
        desc = (
            "⏹ 已按请求停止，进度已落盘"
            if stopped
            else f"✅ 续训完成：新增 {ticks:,} ticks（累计 {model.tick:,}）"
        )
        safe_put(
            event_q,
            {
                "type": "completed",
                "message": desc,
                "desc": desc,
                "step": model.tick,
                "checkpoint": save_path.name,
                "ticks_added": ticks,
            },
        )
        logger.info(f"resume training finished: +{ticks} ticks → {save_path.name}")
    except Exception as exc:
        logger.error(f"resume training failed: {exc}", exc_info=True)
        safe_put(event_q, {"type": "error", "message": str(exc)})
    finally:
        app_state.finish_training()
        safe_put(event_q, None)


@router.post("/api/train/resume_checkpoint")
def resume_checkpoint(req: ResumeRequest):
    """从检查点恢复训练，SSE 流式返回进度（契约见模块文档）。"""
    name = Path(req.checkpoint).name
    if not name.endswith(".pt"):
        raise HTTPException(status_code=400, detail="非法检查点文件名")
    ckpt_path = _CHECKPOINT_DIR / name
    if not ckpt_path.is_file():
        raise HTTPException(status_code=404, detail=f"检查点不存在: {name}")

    if req.datasets:
        corpus_paths, missing = _resolve_datasets(req.datasets)
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"数据集不存在: {', '.join(missing)}",
            )
    else:
        if not _DEFAULT_CORPUS.is_file():
            raise HTTPException(status_code=404, detail="默认语料缺失，请先选择数据集")
        corpus_paths = [_DEFAULT_CORPUS]

    try:
        try:
            envelope = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            logger.warning(
                f"checkpoint {name} 含自定义对象，以不安全模式"
                "（weights_only=False）加载受信 checkpoint"
            )
            envelope = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = Seed.from_checkpoint(envelope)
    except Exception as exc:
        logger.error(f"checkpoint load failed ({name}): {exc}")
        raise HTTPException(status_code=500, detail=f"检查点加载失败: {exc}")

    # 语料漂移预警：检查点元数据记录了原训练语料指纹，本次续训语料不一致时
    # 发 warning 事件（不阻断——混合/换语料续训可能是有意为之）。
    corpus_drift_warning: Optional[str] = None
    meta = envelope.get("metadata") if isinstance(envelope, dict) else None
    prev_fp = (meta or {}).get("corpus_fingerprint") if isinstance(meta, dict) else None
    if prev_fp:
        new_fp = corpus_fingerprint(corpus_paths)
        if new_fp != prev_fp:
            corpus_drift_warning = f"语料已变更：检查点原指纹 {prev_fp}，本次续训 {new_fp}"

    if not app_state.try_start_training():
        raise HTTPException(status_code=409, detail="已有训练任务在运行，请先停止")

    total_bytes = sum(p.stat().st_size for p in corpus_paths)
    save_path = _CHECKPOINT_DIR / f"resumed_{name}"
    event_q: "queue.Queue" = queue.Queue(maxsize=256)
    thread = threading.Thread(
        target=_train_worker,
        args=(model, corpus_paths, total_bytes, save_path, event_q, req.max_ticks),
        daemon=True,
        name="resume-train",
    )
    app_state._trainer_ref = thread
    thread.start()

    def stream():
        try:
            if corpus_drift_warning:
                yield f"data: {json.dumps({'type': 'warning', 'message': corpus_drift_warning}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    evt = event_q.get(timeout=20)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if evt is None:
                    break
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            # 客户端断开（含前端"停止"按钮中断请求）→ 请求停止后台训练。
            if thread.is_alive():
                app_state.stop_training_requested = True
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
