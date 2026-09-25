"""A2.6 分诊器与消融档的契约测试（纯函数，不加载基座）。

这个分诊器在本轮里写错过两次（把 `emission_loses` 写成兜底分支、轨迹在命中处 break），
两次都会把整条后续路线带偏，所以把决策树钉成测试，而不是靠文档里的一段话。
"""

from __future__ import annotations

import types

import torch

from scripts.training.probe_taiji_r2_a26_emission_trace import (
    _classify,
    _keep_only_target_events,
)
from taiji.copy_circuit import ToldContentStore


def _row(
    *,
    step: int,
    copy_top: int,
    target: int,
    emitted: bool,
    gate: float = 40.0,
    forced_hit: bool = False,
) -> dict[str, object]:
    return {
        "step": step,
        "copy_top_byte": copy_top,
        "target_byte": target,
        "emitted": emitted,
        "gate_value": gate,
        "forced_open_would_hit": forced_hit,
    }


def test_empty_trace_is_not_silently_triaged() -> None:
    assert _classify([]) == "no_trace"


def test_all_steps_emitted_is_flagged_as_not_a_real_failure() -> None:
    rows = [_row(step=k, copy_top=65, target=65, emitted=True) for k in range(3)]
    assert _classify(rows) == "emitted"


def test_closed_gate_with_open_would_hit_goes_to_gate_closed() -> None:
    rows = [
        _row(step=0, copy_top=65, target=66, emitted=False, gate=0.2, forced_hit=True),
        _row(step=1, copy_top=65, target=66, emitted=False, gate=0.1),
    ]
    assert _classify(rows) == "gate_closed"


def test_suppression_needs_positive_evidence() -> None:
    """回归：`emission_loses` 必须有"copy 指对了却没发出"的那一步。

    这一族行曾被兜底分支误判成发射侧问题——它们指对的步全都发出去了，症结是续接走歪。
    """
    suppressed = [
        _row(step=0, copy_top=232, target=232, emitted=True),
        _row(step=1, copy_top=139, target=139, emitted=False),  # 指对目标却没发出（词汇 logits 赢了）
    ]
    assert _classify(suppressed) == "emission_loses"
    slipping = [
        _row(step=0, copy_top=229, target=229, emitted=True),
        _row(step=1, copy_top=229, target=175, emitted=False),  # 指错，但没"指对而不发"
        _row(step=2, copy_top=188, target=158, emitted=False),
    ]
    assert _classify(slipping) == "continuation_slips"


def test_never_aimed_is_address_miss() -> None:
    rows = [_row(step=k, copy_top=229, target=232, emitted=False) for k in range(3)]
    assert _classify(rows) == "address_miss"


def test_ablation_keeps_the_original_cue_and_only_the_target() -> None:
    store = ToldContentStore(cue_dim=4, max_events=4)
    target_cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    distractor_cue = torch.tensor([0.0, 1.0, 0.0, 0.0])
    store.record("我住在宁波。".encode(), target_cue)
    store.record("冰箱里的牛奶过期了。".encode(), distractor_cue)
    circuit = types.SimpleNamespace(store=store)

    kept = _keep_only_target_events(circuit, "我住在宁波。".encode())

    assert kept == 1
    events = store.events()
    assert [bytes(event.content) for event in events] == ["我住在宁波。".encode()]
    #: cue 必须**沿用**而不是重算：重算就是在另一条链上取消融读数。
    assert bool(torch.equal(events[0].cue, target_cue))


def test_ablation_reports_when_the_target_never_entered_the_store() -> None:
    store = ToldContentStore(cue_dim=4, max_events=4)
    store.record("冰箱里的牛奶过期了。".encode(), torch.ones(4))
    circuit = types.SimpleNamespace(store=store)
    assert _keep_only_target_events(circuit, "我住在宁波。".encode()) == 0
