"""Seed 原生回合记录的回归测试。

覆盖「态极递归适配」回流侧的输入环：Seed 原生分支此前**完全不记录**，
而三处记录（策略环/任务环/生命交互）都挂在 ``chat_strategies`` 尾部，
因此 harness 的每一次回合对学习环都是隐形的，睡眠期分析恒空。

这里锁三件事：

  1. native 回合记录本身落盘（append-only jsonl）与 ``summarize`` 的计数；
  2. ``_record_*`` 在 legacy 不可用时**分发到 native 后端**，而不是直接返回；
  3. ``_record_native_turn`` 把一回合同时喂给策略环与任务环，且工具名只取真实调用；
  4. 回合载荷里的元数据（session / purpose / offered tools）的落点——辅助模型
     调用不进学习环，会话与「提供了哪些工具」成为记录上的归因字段。
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
    turn_records._SEEN_CONSTRAINTS.clear()
    return tmp_path / "data" / "turn_records"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


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


def test_native_turn_feeds_both_rings(records_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_constraint_seed_is_collected_once_per_process(records_dir: Path) -> None:
    turn_records.record_constraint("你是Seed，一个独立的AI生命体。")
    turn_records.record_constraint("你是Seed，一个独立的AI生命体。")

    records = _read(records_dir / "constraint_seeds.jsonl")
    assert len(records) == 1
    assert records[0]["kind"] == "constraint"
    assert records[0]["sha256"]
    assert records[0]["text"] == "你是Seed，一个独立的AI生命体。"


def test_blank_constraint_is_skipped(records_dir: Path) -> None:
    turn_records.record_constraint("   ")
    assert _read(records_dir / "constraint_seeds.jsonl") == []


def test_constraints_dedupes_across_appends(records_dir: Path) -> None:
    # Two identical lines as a restarted process would leave them.
    turn_records.record_constraint("policy A")
    turn_records._SEEN_CONSTRAINTS.clear()
    turn_records.record_constraint("policy A")
    turn_records.record_constraint("policy B")

    assert len(_read(records_dir / "constraint_seeds.jsonl")) == 3
    texts = [record["text"] for record in turn_records.constraints()]
    assert sorted(texts) == ["policy A", "policy B"]


def test_summarize_counts_constraint_seeds(records_dir: Path) -> None:
    turn_records.record_constraint("policy A")

    assert turn_records.summarize()["constraint_seeds"] == 1


def test_native_turn_collects_the_system_prompt(
    records_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)
    request = ChatRequest(prompt="hello", system_prompt="可执行约束段")

    routes_chat._record_native_turn(request, "answer", True, None)

    records = _read(records_dir / "constraint_seeds.jsonl")
    assert len(records) == 1
    assert records[0]["text"] == "可执行约束段"


def test_records_carry_the_session_and_the_offered_tools(records_dir: Path) -> None:
    turn_records.record_strategy("prompt", "sys", "task", True, 1.0, "session-1")
    turn_records.record_task_outcome(
        "task", True, "answer", session_id="session-1", tools_offered=["bash", "read", "bash"]
    )

    strategies = _read(records_dir / "strategy_records.jsonl")
    assert strategies[0]["session_id"] == "session-1"
    tasks = _read(records_dir / "task_outcomes.jsonl")
    assert tasks[0]["session_id"] == "session-1"
    # Offered, not used: duplicates collapse and order is the caller's.
    assert tasks[0]["tools_offered"] == ["bash", "read"]


def test_records_omit_attribution_they_do_not_have(records_dir: Path) -> None:
    turn_records.record_strategy("prompt", "sys", "task", True, 1.0)
    turn_records.record_task_outcome("task", True, "answer", tools_offered=["", None])

    strategy = _read(records_dir / "strategy_records.jsonl")[0]
    assert "session_id" not in strategy
    task = _read(records_dir / "task_outcomes.jsonl")[0]
    assert "session_id" not in task
    assert "tools_offered" not in task


def test_native_turn_records_session_and_offered_tools(
    records_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)
    request = ChatRequest(
        prompt="hello",
        system_prompt="policy",
        session_id="session-9",
        tools=["bash", "read"],
    )

    routes_chat._record_native_turn(request, "hi there", True, None)

    tasks = _read(records_dir / "task_outcomes.jsonl")
    assert tasks[0]["session_id"] == "session-9"
    assert tasks[0]["tools_offered"] == ["bash", "read"]
    strategies = _read(records_dir / "strategy_records.jsonl")
    assert {record["session_id"] for record in strategies} == {"session-9"}


def test_auxiliary_call_enters_no_ring(records_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_strategies, "legacy_available", lambda: False)
    request = ChatRequest(
        prompt="给这段对话起个标题",
        system_prompt="你是标题生成器",
        purpose="session-title",
        session_id="session-9",
        tools=["bash"],
    )

    routes_chat._record_native_turn(request, "标题", True, None)

    assert _read(records_dir / "task_outcomes.jsonl") == []
    assert _read(records_dir / "strategy_records.jsonl") == []
    # A title generator's prompt is not a product constraint either.
    assert _read(records_dir / "constraint_seeds.jsonl") == []
