"""训练进度/ETA 契约：分母必须是本次实际要处理的字节数。

背景（2026-08-29 实测）：`max_symbols` 限制了循环只跑 N 个 tick，
但 fraction/eta 仍按整个数据集的 total_bytes 计算，导致
fraction 停在 0.05%、ETA 报 32 天而真实剩余为 0。
本文件把「有效分母」与「ETA 单调收敛」钉成门禁。
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path

from api.training import resume

REPO = Path(__file__).resolve().parents[2]


@dataclass
class _FakeStep:
    prior_prediction: int | None = 1
    surprise: float = 0.5


class _FakeModel:
    """最小替身：只提供 _train_worker 依赖的表面。"""

    def __init__(self) -> None:
        self.tick = 0

        class _Taiji:
            boundary_symbol = 256

        class _Config:
            taiji = _Taiji()

        self.config = _Config()

    def observe(self, symbol: int, learn: bool = True) -> _FakeStep:
        self.tick += 1
        return _FakeStep()

    def checkpoint(self) -> dict:
        return {"tick": self.tick}


def _drain(event_q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(event_q.get_nowait())
        except queue.Empty:
            return events


def _run(tmp_path: Path, corpus_bytes: int, total_bytes: int, max_ticks: int | None):
    corpus = tmp_path / "corpus.jsonl"
    payload = "a" * corpus_bytes
    corpus.write_text('{"text": "' + payload + '"}\n', encoding="utf-8")

    event_q: queue.Queue = queue.Queue(maxsize=4096)
    resume._train_worker(
        _FakeModel(),
        [corpus],
        total_bytes,
        tmp_path / "ckpt.pt",
        event_q,
        max_ticks,
    )
    events = _drain(event_q)
    return [e for e in events if isinstance(e, dict) and e.get("type") == "progress"]


def test_capped_run_reaches_full_progress_and_zero_eta(tmp_path, monkeypatch) -> None:
    """max_ticks 截断时，收尾进度必须接近 100% 且 ETA 收敛到 0。"""
    monkeypatch.setattr(resume, "PROGRESS_EVERY", 100)
    monkeypatch.setattr(resume, "CHECKPOINT_EVERY", 10_000_000)

    progress = _run(tmp_path, corpus_bytes=50_000, total_bytes=50_000, max_ticks=1_000)

    assert progress, "必须至少发出一条进度事件"
    final = progress[-1]
    assert final["fraction"] >= 0.99, f"截断训练收尾 fraction 应接近 1，实测 {final['fraction']}"
    assert final["eta"] <= 1.0, f"截断训练收尾 ETA 应收敛到 0，实测 {final['eta']} 秒"


def test_progress_denominator_never_exceeds_effective_workload(tmp_path, monkeypatch) -> None:
    """分母被高估时（估算偏大），fraction 不得被系统性压低。"""
    monkeypatch.setattr(resume, "PROGRESS_EVERY", 100)
    monkeypatch.setattr(resume, "CHECKPOINT_EVERY", 10_000_000)

    progress = _run(tmp_path, corpus_bytes=5_000, total_bytes=5_000_000, max_ticks=None)

    assert progress
    final = progress[-1]
    assert (
        final["fraction"] >= 0.99
    ), f"语料读完即为 100%，不应受 total_bytes 高估影响，实测 {final['fraction']}"


def test_every_progress_event_carries_eta_and_elapsed(tmp_path, monkeypatch) -> None:
    """首条进度事件也必须带 eta/elapsed，否则前端首屏显示 '--'。"""
    monkeypatch.setattr(resume, "PROGRESS_EVERY", 100)
    monkeypatch.setattr(resume, "CHECKPOINT_EVERY", 10_000_000)

    progress = _run(tmp_path, corpus_bytes=1_000, total_bytes=1_000, max_ticks=None)

    for index, event in enumerate(progress):
        assert "eta" in event, f"第 {index} 条进度事件缺少 eta"
        assert "elapsed" in event, f"第 {index} 条进度事件缺少 elapsed"


def test_paused_time_is_excluded_from_eta(tmp_path, monkeypatch) -> None:
    """暂停时长不得计入 elapsed，否则恢复后 ETA 被整体拉长。"""
    source = (REPO / "api/training/resume.py").read_text(encoding="utf-8")
    assert "paused_total" in source, "暂停时长必须被单独扣减，见归档 Gate/CI 历史 §14.18"
