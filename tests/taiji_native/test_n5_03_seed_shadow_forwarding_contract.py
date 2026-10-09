"""DEBT-N5 转发格契约：`Seed.observe()` 必须能把影子参数交给 `Taiji.observe()`，且缺省时一字不加。

现状取证（本轮改动前）：`seed/model.py:74-91` 的 `observe()` 是一条**闭集**参数表，
只把 `learn/learn_motor/readout/use_memory/use_identity` 五项转给 `self.substrate.observe`，
而 `taiji/model.py:1905-1907` 的 `Taiji.observe` 原生吃
`_adaptive_residual_shadow` / `_learn_adaptive_residual_shadow` /
`_adaptive_residual_shadow_freeze_parent` ⇒ 训练链拿不到影子学习（㊵-577 记的"Seed 层挡住"）。

本测用**录制器**接住转发后的 kwargs（不建真模型，不碰权重），四条都必须能为假：

* 缺省调用：转发的 kwargs 与改动前逐字相同（五个键、无影子键）；
* 只给影子：只多 `_adaptive_residual_shadow` 一个键；
* 影子＋学习：两个键都在；`freeze_parent=False` 时第三个键才出现；
* **只给 `_learn_adaptive_residual_shadow=True` 而不给影子 ⇒ 不许转发**（否则这一层
  会把产品自己会响亮拒绝的组合变成静默半参数）。

类型与组合校验故意不在 Seed 层重做：由 `Taiji.observe` 自己拒绝（`taiji/model.py:1956-1994`）。
"""

from __future__ import annotations

from typing import Any

from seed.model import Seed

BASE_KEYS = {"learn", "learn_motor", "readout", "use_memory", "use_identity"}


class _RecordingSubstrate:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def observe(self, symbol: int, **kwargs: Any) -> str:
        self.calls.append((int(symbol), kwargs))
        return "step"


class _StubSeed:
    """只需 `.substrate`：`Seed.observe` 是普通方法，可脱实例直接调用。"""

    def __init__(self) -> None:
        self.substrate = _RecordingSubstrate()


def _forward(**kwargs: Any) -> dict[str, Any]:
    stub = _StubSeed()
    Seed.observe(stub, 0x41, **kwargs)
    assert stub.substrate.calls == [(0x41, stub.substrate.calls[0][1])]
    return stub.substrate.calls[0][1]


def test_default_call_forwards_exactly_the_pre_change_kwargs() -> None:
    forwarded = _forward()
    assert set(forwarded) == BASE_KEYS
    assert not any(key.startswith("_adaptive_residual") for key in forwarded)


def test_shadow_only_adds_the_shadow_key() -> None:
    forwarded = _forward(_adaptive_residual_shadow="SHADOW")
    assert set(forwarded) == BASE_KEYS | {"_adaptive_residual_shadow"}
    assert forwarded["_adaptive_residual_shadow"] == "SHADOW"
    assert "_learn_adaptive_residual_shadow" not in forwarded
    assert "_adaptive_residual_shadow_freeze_parent" not in forwarded


def test_learning_and_unfreezing_forward_their_own_keys() -> None:
    learning = _forward(_adaptive_residual_shadow="SHADOW", _learn_adaptive_residual_shadow=True)
    assert learning["_learn_adaptive_residual_shadow"] is True
    unfrozen = _forward(
        _adaptive_residual_shadow="SHADOW", _adaptive_residual_shadow_freeze_parent=False
    )
    assert unfrozen["_adaptive_residual_shadow_freeze_parent"] is False
    #: 默认（freeze_parent=True）时这一枚键不能出现，否则就是把产品默认搬到这一层。
    assert "_adaptive_residual_shadow_freeze_parent" not in learning


def test_learn_without_shadow_is_not_forwarded() -> None:
    #: 缺影子 ⇒ 这一层不代产品接受半参数组合（产品侧 `taiji/model.py:1993-1994` 会拒绝）。
    forwarded = _forward(_learn_adaptive_residual_shadow=True)
    assert set(forwarded) == BASE_KEYS
