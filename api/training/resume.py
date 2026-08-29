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

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from seed import Seed, iter_native_documents
from seed.datasets import inspect_native_dataset
from seed.persistence import atomic_save, attach_metadata, corpus_fingerprint
from seed_platform.app_state import app_state

from .common import collect_hardware_diag, safe_put
from .datasets import PREVIEW_SCAN_RECORDS, resolve_dataset_path

logger = logging.getLogger("ApiServer.Training.Resume")
router = APIRouter()

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _ROOT / "checkpoints"
# 未选数据集时的回退语料（与 train_seed_corpus.DEFAULT_CORPUS 一致）。
_DEFAULT_CORPUS = _ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"

PROGRESS_EVERY = 2_000  # 实测 CPU ≈147 字节/s，约 13.6s 一条进度（2026-08-29 实测）
CHECKPOINT_EVERY = 250_000  # 周期落盘间隔（ticks）


class ResumeRequest(BaseModel):
    checkpoint: str
    datasets: list[str] | None = None
    # 测试/冒烟用的预算上限；前端不发送，缺省跑满整个数据集一遍。
    max_ticks: int | None = None


def _resolve_datasets(names: list[str]):
    """解析前端给出的数据集相对路径（防目录穿越，保序去重）。"""
    resolved: list[Path] = []
    missing: list[str] = []
    for name in names:
        hit = resolve_dataset_path(name)
        if hit is None:
            missing.append(name)
            continue
        path = Path(hit)
        if path not in resolved:
            resolved.append(path)
    return resolved, missing


def estimated_text_bytes(path: Path) -> int:
    """估算单个数据集的纯文本字节数（进度分母的唯一口径）。

    进度计数器 ``consumed`` 只累加真实文本字节（``_iter_symbols`` 对边界符
    产出 0），因此分母必须同为纯文本字节。用文件物理大小会把 JSONL 结构
    开销算进去，实测偏大 2~3%。

    原生路径 ``native._resolve_native_datasets`` 因为还要做 native_trainable
    校验，直接复用它自己拿到的 ``report``，不走这里以免重复扫描；两处口径
    必须同为 ``estimated_total_text_bytes()``，改一处要同步改另一处。
    """
    try:
        report = inspect_native_dataset(path, max_records=PREVIEW_SCAN_RECORDS)
        return report.estimated_total_text_bytes()
    except Exception:  # 探测失败时回退到物理大小，宁可略偏大也不让分母为 0
        try:
            return path.stat().st_size
        except OSError:
            return 0


def _iter_symbols(paths: list[Path], boundary: int):
    """逐文档流式产出 (符号, 该字节是否计入进度)。"""
    for text in iter_native_documents(paths):
        raw = text.encode("utf-8")
        yield boundary, 0
        for symbol in raw:
            yield symbol, 1


def _train_worker(
    model: Seed,
    corpus_paths: list[Path],
    total_bytes: int,
    save_path: Path,
    event_q: "queue.Queue",
    max_ticks: int | None,
    start_description: str = "已从检查点恢复，开始流式续训…",
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
    paused_total = 0.0
    stopped = False

    # 进度分母必须是「本次实际要处理的字节数」，而不是整个数据集大小。
    # max_ticks 会在 N 个 tick 后 break，此时只有 N 字节会被 consumed；
    # 若仍以整份数据集作分母，fraction 永远停在极小值，ETA 会被放大到数万倍。
    effective_total = total_bytes if max_ticks is None else min(total_bytes, max_ticks)
    effective_total = max(1, effective_total)

    def _elapsed() -> float:
        """真实训练耗时：扣除暂停期间的挂钟时间。"""
        return max(1e-6, time.perf_counter() - started - paused_total)

    safe_put(event_q, {"type": "hardware_diag", **collect_hardware_diag("cpu")})
    safe_put(
        event_q,
        {
            "type": "progress",
            "fraction": 0.0,
            "desc": f"{start_description}（tick {start_tick:,}）",
            "step": start_tick,
            "epoch": 1,
            "total_epochs": 1,
            "total_steps": effective_total,
            "elapsed": 0.0,
            "eta": None,
            "samples_per_sec": 0.0,
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

    def _emit_progress(final: bool = False) -> None:
        elapsed = _elapsed()
        # 收尾时进度即为 100%：语料读完或达到 max_ticks 都意味着本次工作量已完成，
        # 不能因为分母估算偏大而卡在 99% 以下。
        fraction = 1.0 if final else min(consumed / effective_total, 0.999)
        mean_surprise = window_surprise / max(1, window_ticks)
        rate = ticks / elapsed
        remaining = max(0, effective_total - consumed)
        # ETA 用实测速率直接换算剩余字节，而不是对 fraction 做除法外推：
        # 前者在分母被高估或训练被截断时仍然有界，后者会被极小的 fraction 放大。
        eta = 0.0 if final else round(remaining / max(rate, 1e-6), 1)
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
                "eta": eta,
                "epoch": 1,
                "total_epochs": 1,
                "samples_per_sec": round(rate, 1),
                "total_steps": effective_total,
            },
        )

    try:
        for symbol, byte_len in _iter_symbols(corpus_paths, boundary):
            if app_state.stop_training_requested:
                stopped = True
                break
            while app_state.pause_training_requested:
                pause_started = time.perf_counter()
                if app_state.stop_training_requested:
                    paused_total += time.perf_counter() - pause_started
                    break
                time.sleep(0.5)
                paused_total += time.perf_counter() - pause_started
            if app_state.stop_training_requested:
                stopped = True
                break
            step = model.observe(symbol, learn=True)
            ticks += 1
            consumed += byte_len
            if step.prior_prediction is not None:
                window_ticks += 1
                window_surprise += float(step.surprise or 0.0)
            if ticks % PROGRESS_EVERY == 0:
                _emit_progress()
                window_ticks = 0
                window_surprise = 0.0
            if ticks % CHECKPOINT_EVERY == 0:
                _persist()
            if max_ticks is not None and ticks >= max_ticks:
                break
        if window_ticks > 0:  # 收尾进度只在有观测窗口时发，避免 loss=0 假象
            # 用户主动停止不算完成，不能伪造 100%；正常读完或达 max_ticks 才收敛到 100%/0s。
            _emit_progress(final=not stopped)
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


def stream_training_events(
    event_q: "queue.Queue",
    thread: threading.Thread,
    initial_events: list[dict] | None = None,
):
    """Yield one shared SSE contract for native training endpoints."""

    try:
        for event in initial_events or []:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        while True:
            try:
                event = event_q.get(timeout=20)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    finally:
        if thread.is_alive():
            app_state.stop_training_requested = True
    yield "data: [DONE]\n\n"


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
        raise HTTPException(status_code=500, detail=f"检查点加载失败: {exc}") from exc

    # 语料漂移预警：检查点元数据记录了原训练语料指纹，本次续训语料不一致时
    # 发 warning 事件（不阻断——混合/换语料续训可能是有意为之）。
    corpus_drift_warning: str | None = None
    meta = envelope.get("metadata") if isinstance(envelope, dict) else None
    prev_fp = (meta or {}).get("corpus_fingerprint") if isinstance(meta, dict) else None
    if prev_fp:
        new_fp = corpus_fingerprint(corpus_paths)
        if new_fp != prev_fp:
            corpus_drift_warning = f"语料已变更：检查点原指纹 {prev_fp}，本次续训 {new_fp}"

    if not app_state.try_start_training():
        raise HTTPException(status_code=409, detail="已有训练任务在运行，请先停止")

    total_bytes = sum(estimated_text_bytes(p) for p in corpus_paths)
    save_path = _CHECKPOINT_DIR / f"resumed_{name}"
    event_q: queue.Queue = queue.Queue(maxsize=256)
    thread = threading.Thread(
        target=_train_worker,
        args=(model, corpus_paths, total_bytes, save_path, event_q, req.max_ticks),
        daemon=True,
        name="resume-train",
    )
    app_state._trainer_ref = thread
    thread.start()

    initial_events = (
        [{"type": "warning", "message": corpus_drift_warning}] if corpus_drift_warning else []
    )
    return StreamingResponse(
        stream_training_events(event_q, thread, initial_events),
        media_type="text/event-stream",
    )
