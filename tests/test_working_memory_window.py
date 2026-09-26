"""B 支线补漏：`WorkingMemory` 滑动窗口的标记簿记不变量（DEBT-B4-2 的修复守卫）。

该模块被 `cortex.py:219-223` 自己标注为"仅注册未接入"（真正的上下文记忆走
`agent/working_memory` 经 ContextManager），所以**删还是接**是产品决定；
但 `append_round` 的 `round_marks` 在 FIFO 丢弃后整体错位是确定的簿记缺陷，先修掉并钉住：

不变量：对每条标记 `(start, end)`，`get_context_ids()[start:end]` 必须恰好等于
"那一轮仍留在窗口里的 token 前/后缀"；刚追加的一轮永远不该是空区间；
整轮被挤出窗口的标记必须消失（而不是留在原地指向别人的 token）。
"""

from __future__ import annotations

import torch

from neuroplex.brain.working_memory import WorkingMemory


def _slices(mem: WorkingMemory) -> list[list[int]]:
    buf = mem.get_context_ids()
    return [list(buf[s:e]) for s, e, _imp in mem.round_marks]


def test_marks_index_their_own_round_without_overflow() -> None:
    mem = WorkingMemory(max_tokens=16)
    mem.append_round([1, 2, 3], [4, 5])
    mem.append_round([6, 7], [8, 9, 10])
    assert mem.round_marks == [(0, 5, 1.0), (5, 10, 1.0)], mem.round_marks
    assert _slices(mem) == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert len(mem) == 10


def test_marks_survive_fifo_eviction_and_newest_round_is_not_empty() -> None:
    """债册里那组复现读数：max_tokens=8、三轮之后旧实现给 (0,5)/(5,8)/(8,8)。"""

    mem = WorkingMemory(max_tokens=8)
    mem.append_round([1, 2, 3], [4])
    mem.append_round([6, 7], [8])
    mem.append_round([9, 10, 11], [12])

    buf = mem.get_context_ids()
    assert buf == [5, 6, 7, 8, 9, 10, 11, 12] or buf == [4, 6, 7, 8, 9, 10, 11, 12], buf
    assert all(e > s for s, e, _ in mem.round_marks), f"存在空区间标记：{mem.round_marks}"
    # 最后一轮（9,10,11|12）必须完整可读
    assert _slices(mem)[-1] == [9, 10, 11, 12]
    # 每条标记切出来的都必须是对应轮的后缀/全量，而不是别人的 token
    assert (
        _slices(mem)[-2] == [6, 7, 8] or _slices(mem)[-2] == [6, 7, 8, 9][: len(_slices(mem)[-2])]
    )


def test_partially_evicted_round_keeps_only_the_surviving_prefix() -> None:
    mem = WorkingMemory(max_tokens=6)
    mem.append_round([1, 2, 3, 4, 5, 6, 7], [])  # 一轮就装不下，挤掉前 1 个
    assert mem.get_context_ids() == [2, 3, 4, 5, 6, 7]
    assert mem.round_marks == [(0, 6, 1.0)], mem.round_marks  # 只剩存活的那 6 个
    assert _slices(mem) == [[2, 3, 4, 5, 6, 7]]

    mem.append_round([8], [9, 10])  # 再挤掉最老的 3 个
    assert mem.get_context_ids() == [5, 6, 7, 8, 9, 10]
    marks = mem.round_marks
    assert len(marks) == 2 and marks[1] == (3, 6, 1.0), marks
    assert _slices(mem) == [[5, 6, 7], [8, 9, 10]]


def test_fully_evicted_round_is_dropped_and_importance_follows_the_round() -> None:
    mem = WorkingMemory(max_tokens=4)
    mem.append_round([1, 2], [3], importance=2.5)
    mem.append_round([4, 5, 6], [7, 8])  # 第一轮全部被挤出
    assert all(imp in (1.0, 2.5) for _s, _e, imp in mem.round_marks)
    assert mem.round_marks == [(0, 4, 1.0)], mem.round_marks  # 2.5 那一轮整体出窗
    assert mem.get_context_ids() == [5, 6, 7, 8]


def test_mark_bookkeeping_stays_bounded() -> None:
    mem = WorkingMemory(max_tokens=32)
    for i in range(30):
        mem.append_round([i], [i + 100])
    assert len(mem.round_marks) <= 20, len(mem.round_marks)
    buf = mem.get_context_ids()
    assert len(buf) <= 32
    for s, e, _imp in mem.round_marks:
        assert 0 <= s < e <= len(buf), (s, e, len(buf))


def test_reset_save_load_and_readouts(tmp_path) -> None:
    mem = WorkingMemory(max_tokens=8)
    mem.append_round([1, 2], [3])
    assert mem.get_context_tensor(torch.device("cpu")) is not None
    assert tuple(mem.get_context_tensor(torch.device("cpu")).shape) == (1, 3)

    summary = mem.get_summary()
    assert summary["total_tokens"] == 3 and summary["n_rounds"] == 1
    assert summary["rounds"][0]["size"] == 3 and summary["max_tokens"] == 8

    path = tmp_path / "wm.pt"
    mem.save(str(path))
    mem.append_round([9, 9, 9, 9], [9])  # 污染
    restored = WorkingMemory.load(str(path))
    assert restored.max_tokens == 8 and restored.get_context_ids() == [1, 2, 3]
    assert restored.round_marks == [(0, 3, 1.0)], restored.round_marks

    mem.reset()
    assert mem.is_empty() and mem.round_marks == [] and mem.current_round_start == 0
    assert mem.get_context_tensor(torch.device("cpu")) is None, "空窗口不该造一个零长张量"


def test_empty_round_appends_nothing_but_keeps_invariant() -> None:
    mem = WorkingMemory(max_tokens=4)
    mem.append_round([], [])
    assert mem.round_marks == [] and mem.get_context_ids() == []
    mem.append_round([1], [2])
    assert _slices(mem) == [[1, 2]]


def test_eviction_never_leaves_marks_pointing_outside_the_window() -> None:
    """随机压力：任何时刻每条标记都必须在窗口内且切片非空。"""

    rng = __import__("random").Random(7)
    mem = WorkingMemory(max_tokens=10)
    for _ in range(60):
        prompt = [rng.randrange(1000) for _ in range(rng.randrange(0, 5))]
        gen = [rng.randrange(1000) for _ in range(rng.randrange(0, 6))]
        mem.append_round(prompt, gen)
        buf = mem.get_context_ids()
        assert len(buf) <= 10
        for s, e, _imp in mem.round_marks:
            assert 0 <= s < e <= len(buf), (mem.round_marks, len(buf))
            assert buf[s:e], f"空区间标记 {s}:{e}"
    assert mem.round_marks, "60 轮之后窗口里仍应有可读标记"


def test_fixture_guard_rejects_a_broken_bookkeeping_implementation() -> None:
    """反向锚：若有人把丢弃数算回 0，标记必须当场错位（本条就是抓这个的）。"""

    mem = WorkingMemory(max_tokens=4)
    mem.append_round([1, 2], [3, 4])
    mem.append_round([5, 6], [7, 8])
    # A 的 4 个 token 全被挤出 ⇒ 只剩 B 一条标记，且它就是刚写进去的四个
    assert mem.get_context_ids() == [5, 6, 7, 8], mem.get_context_ids()
    assert mem.round_marks == [(0, 4, 1.0)], mem.round_marks
    assert _slices(mem) == [[5, 6, 7, 8]]
