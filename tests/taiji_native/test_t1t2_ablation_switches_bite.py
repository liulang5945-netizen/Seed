"""T1/T2 的开关守卫：**每一个消融开关都必须真的咬住**，否则那一条臂是在测空气。

## 为什么非有这件

T2 的全部结论都建立在"关掉某一类写入，结构就被保住了"上。如果某个开关**没生效**，
那一条臂会与对照**逐位相同**，而我会把它读成"这一类写入不是元凶"——**一个假阴性，
而且看不出来**。本仓的纪律是"守卫必须红/绿各跑一次证明能响"，这就是那次"证明"。

## 三个已实测的坑（写在这里，免得后人重踩）

1. `Seed.observe` **只转发** `learn` / `learn_motor` / `use_memory` / `use_identity`
   （`seed/model.py:78`）⇒ 传 `learn_fabric` 会直接 `TypeError`。T1/T2 驱动因此**直接调 substrate**。
2. **`readout="action"` 之下，`predictive_context` / `predictive_readout` 根本没参与前向**
   （`taiji/model.py:2027` 的分支）⇒ 在这条配方里关掉它们的写入**是空操作**。
   所以 T2 里**不能**设一条 `abl-readout` 臂：它必然与对照逐位相同。
3. 因此本文件明确覆盖 T2 实际使用的四个开关，并额外钉住"**`learn=False` ⇒ 什么都不变**"这条正对照——
   它是仪器的红/绿证明：如果连"完全不学"都保不住结构，那问题就在测量而不在训练。
"""

from __future__ import annotations

import hashlib

import torch

from seed import Seed, SeedConfig
from taiji import TaijiConfig

DATA = b"abcdefghabcdefghabcdefgh"


def _digest(value: object) -> str:
    hasher = hashlib.sha256()

    def walk(node: object) -> None:
        if isinstance(node, torch.Tensor):
            hasher.update(str(tuple(node.shape)).encode())
            hasher.update(node.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(node, dict):
            for key in sorted(node, key=str):
                hasher.update(str(key).encode())
                walk(node[key])
        elif isinstance(node, (list, tuple)):
            hasher.update(str(len(node)).encode())
            for item in node:
                walk(item)
        else:
            hasher.update(repr(node).encode())

    walk(value)
    return hasher.hexdigest()


def _run(**observe_kwargs: object) -> dict[str, str]:
    """Small model, a few hundred ticks on the action readout, per-section digests."""

    config = SeedConfig(
        taiji=TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=7)
    )
    model = Seed(config, episode_id="t1t2-guard")
    before = {key: _digest(value) for key, value in model.substrate.checkpoint().items()}
    for symbol in DATA * 8:
        model.substrate.observe(int(symbol), readout="action", **observe_kwargs)
    after = {key: _digest(value) for key, value in model.substrate.checkpoint().items()}
    return {key: ("changed" if before[key] != after[key] else "same") for key in before}


def test_the_control_recipe_actually_writes_the_two_real_surfaces() -> None:
    """绿：默认配方下**只有 fabric 与 motor** 真的被写——这两条才是可消融的臂。

    坑 #3（实测得出，不是推断）：这条配方（`readout="action"`、没有 `settle_action`/`consolidate`）
    **不写 memory**。于是 `use_memory=False` 是**空操作** ⇒ T2 里**不设 `abl-memory` 臂**。
    """

    result = _run(learn=True)
    for section in ("fabric", "motor"):
        assert result[section] == "changed", f"默认配方下 {section} 竟然没被写：这一臂没有对照价值"
    assert result["memory"] == "same", "这条配方本该不写 memory；若开始写了，坑 #3 要重新评估"


def test_memory_is_not_written_so_ablating_it_would_be_a_no_op() -> None:
    """坑 #3 的守卫：`use_memory=False` 在这条配方里**不改变任何东西** ⇒ 那条臂没有信息量。"""

    on = _run(learn=True)
    off = _run(learn=True, use_memory=False)
    assert on == off, "use_memory 在这条配方里开始有影响了——T2 的臂表需要重订"


def test_learn_false_writes_nothing_at_all() -> None:
    """红（正对照）：`learn=False` ⇒ **除状态/随机流外一个字节都不许变**。

    这是仪器的红/绿证明：连"完全不学"都不能保住结构时，问题在测量，而不在训练。
    """

    result = _run(learn=False)
    for section in ("fabric", "motor", "memory", "predictive_context", "predictive_readout"):
        assert result[section] == "same", f"learn=False 之下 {section} 仍被写入"


def test_learn_fabric_off_freezes_only_the_fabric() -> None:
    result = _run(learn=True, learn_fabric=False)
    assert result["fabric"] == "same", "learn_fabric=False 没有咬住 fabric"
    assert result["motor"] == "changed", "learn_fabric=False 不该连带冻住 motor"


def test_learn_motor_off_freezes_only_the_motor() -> None:
    result = _run(learn=True, learn_motor=False)
    assert result["motor"] == "same", "learn_motor=False 没有咬住 motor"
    assert result["fabric"] == "changed", "learn_motor=False 不该连带冻住 fabric"


def test_use_identity_false_does_not_freeze_the_identity_organ() -> None:
    """坑 #4（实测得出）：`use_identity=False` **并不**冻住 `identity_organ` 的写入。

    它管的是"读不读身份"，不是"写不写"。⇒ 想要"关掉身份侧的写入"得另找开关，
    在这条配方里**做不到**，所以 T2 也不设那一臂。
    """

    result = _run(learn=True, use_identity=False)
    assert (
        result["identity_organ"] == "changed"
    ), "use_identity=False 竟然冻住了 identity_organ——坑 #4 的结论要重写"


def test_the_predictive_organs_are_untouched_under_the_action_readout() -> None:
    """钉住坑 #2：`readout="action"` 下，predictive 两个器官**任何开关都不会**被写。

    ⇒ 在这条配方里设一条"关掉读出头"的臂是空操作；T2 因此不设那条臂。
    哪天真要测它，必须换 `readout="predictive"`，那就**换了配方**，得另立预注册。
    """

    result = _run(learn=True)
    assert result["predictive_context"] == "same"
    assert result["predictive_readout"] == "same"
