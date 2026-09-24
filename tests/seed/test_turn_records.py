"""Seed 原生回合记录的回归测试。

覆盖「态极递归适配」回流侧的输入环：Seed 原生分支此前**完全不记录**，
而三处记录（策略环/任务环/生命交互）都挂在 ``chat_strategies`` 尾部，
因此 harness 的每一次回合对学习环都是隐形的，睡眠期分析恒空。

这里锁三件事：

  1. native 回合记录本身落盘（append-only jsonl）与 ``summarize`` 的计数；
  2. ``_record_*`` 在 legacy 不可用时**分发到 native 后端**，而不是直接返回；
  3. ``_record_native_turn`` 把一回合同时喂给策略环与任务环，且工具名只取真实调用。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import chat_strategies, routes_chat
from api.models import ChatRequest
from seed_platform import turn_records


@pytest.fixture()
def records_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(turn_records, "get_external_path", lambda rel: str(tmp_path / rel))
    return tmp_path / "data" / "turn_records"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_records_append_and_summarize(records_dir: Path) -> None:
    turn_records.record_strategy("prompt", "sys", "task", True, 1.0)
    turn_records.record_strategy("tool_choice", "bash", "task", True, 1.0)
    turn_records.record_task_outcome("task", True, "answer")

    summary = turn_records.summarize()
    assert summary["available"] is True
    assert summary["strategies"] == 2
    assert summary["strategies_by_type"] == {"prompt": 1, "tool_choice": 1}
    assert summary["tasks"] == 1
    assert summary["tasks_succeeded"] == 1
    assert summary["tasks_failed"] == 0
    assert summary["last_recorded_at"] > 0


def test_failed_task_records_the_error(records_dir: Path) -> None:
    turn_records.record_task_outcome("task", False, error="boom")

    records = _read(records_dir / "task_outcomes.jsonl")
    assert records[0]["success"] is False
    assert records[0]["error"] == "boom"
    assert records[0]["source"] == "seed.native.chat"
    assert records[0]["recorded_at"] > 0


def test_strategy_fields_are_clipped(records_dir: Path) -> None:
    turn_records.record_strategy("prompt", "x" * 500, "y" * 500, True, 1.0)

    record = _read(records_dir / "strategy_records.jsonl")[0]
    assert len(record["strategy_content"]) == 200
    assert len(record["task"]) == 200


def test_rings_fall_back_to_native_records(
    records_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)

    chat_strategies._record_evolution("prompt", "answer", True)
    chat_strategies._record_recursive_strategies("prompt", "system", True, 1, ["bash"])

    strategies = _read(records_dir / "strategy_records.jsonl")
    assert [record["strategy_type"] for record in strategies] == ["prompt", "tool_choice"]
    tasks = _read(records_dir / "task_outcomes.jsonl")
    assert tasks[0]["success"] is True
    assert tasks[0]["final_answer"] == "answer"


def test_reflection_strategy_only_for_multi_step(
    records_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)

    chat_strategies._record_recursive_strategies("prompt", "system", True, 1, [])
    assert [r["strategy_type"] for r in _read(records_dir / "strategy_records.jsonl")] == ["prompt"]

    chat_strategies._record_recursive_strategies("prompt", "system", True, 3, [])
    assert [r["strategy_type"] for r in _read(records_dir / "strategy_records.jsonl")] == [
        "prompt",
        "prompt",
        "reflection",
    ]


def test_native_turn_feeds_both_rings(
    records_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)
    request = ChatRequest(prompt="hello", system_prompt="policy")

    routes_chat._record_native_turn(request, "hi there", True, {"tool_call": {"name": "bash"}})

    strategies = _read(records_dir / "strategy_records.jsonl")
    assert [record["strategy_type"] for record in strategies] == ["prompt", "tool_choice"]
    assert strategies[0]["strategy_content"] == "policy"
    tasks = _read(records_dir / "task_outcomes.jsonl")
    assert len(tasks) == 1
    assert tasks[0]["task"] == "hello"


def test_tool_names_reads_only_real_calls() -> None:
    assert routes_chat._tool_names_from_workbench(None) == []
    assert routes_chat._tool_names_from_workbench({"tool_call": None}) == []
    assert routes_chat._tool_names_from_workbench({"tool_call": "bash"}) == []
    assert routes_chat._tool_names_from_workbench({"tool_call": {"action": "read"}}) == ["read"]
    assert routes_chat._tool_names_from_workbench({"tool_call": {"name": "bash"}}) == ["bash"]