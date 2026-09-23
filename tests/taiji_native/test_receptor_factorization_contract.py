"""R2 组合绑定实验的机器闸门：分块受力面的开关、配对性与 checkpoint 往返。

所有者 2026-09-23 授权改 `taiji/organs.py` 的 `receptors` 一处。本文件是那次授权的守卫——
它要钉住三件事，缺任何一件那次改动的结论都不成立：

1. **默认路径逐位不变**：开关默认关，且关的时候只建原来那**一张** map。
2. **配对是真的**：开关只影响 `receptors` 这一处，**其它器官的拓扑逐位相同**。
   这一条最要紧——如果开/关两臂连随机初始化都不一样，那两臂的差就不能归因给这个改动。
   （实现上靠一张**独立 generator** 抽样，不消耗主 RNG 流。）
3. **能存能读**：分块变体必须能 checkpoint 往返，且**开关与载荷不一致时必须报错**，
   不许把未分块的权重当成"分块结果"静默加载。
"""

from __future__ import annotations

import hashlib

import torch

from taiji import Taiji, TaijiConfig


def _config(*, factored: bool) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
        receptors_factored=factored,
    )


def _observe(model: Taiji, data: bytes) -> None:
    for symbol in data:
        model.observe(int(symbol), learn=False, readout="predictive")


def test_the_flag_defaults_off() -> None:
    plain = TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=7)
    assert plain.receptors_factored is False
    assert Taiji(plain).predictive_context.receptors_factor is None


def test_the_flag_survives_a_config_round_trip() -> None:
    """老 checkpoint 的 config 没有这个键，取默认值即可加载；新 checkpoint 要带着它。"""

    config = _config(factored=True)
    assert TaijiConfig.from_dict(config.to_dict()).receptors_factored is True
    legacy = config.to_dict()
    del legacy["receptors_factored"]
    assert TaijiConfig.from_dict(legacy).receptors_factored is False


def _digest(value: object) -> str:
    """按结构递归地把载荷压成一个 sha256（张量按字节、标量按 repr）。"""

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


def test_factoring_leaves_every_other_organ_bit_identical() -> None:
    """配对保证：两臂的差别**只有** encode 走哪条路，别处一个字节都不许差。

    用**整份载荷的指纹**逐键比较，而不是挑几个张量看——挑着看会漏掉我没想起来的器官。
    其中 `rng_state` 相同是"独立 generator 确实没消耗主流"的直接证据。
    """

    plain = Taiji(_config(factored=False))
    factored = Taiji(_config(factored=True))
    left, right = plain.checkpoint(), factored.checkpoint()
    assert set(left) == set(right)

    skipped = {"config", "predictive_context", "identity_organ"}
    for key in sorted(left):
        if key in skipped:
            continue
        assert _digest(left[key]) == _digest(right[key]), f"载荷键 {key} 不该有差别"
    assert _digest(left["rng_state"]) == _digest(right["rng_state"]), "分块抽样不得消耗主 RNG 流"

    # identity_organ 的**拓扑**必须逐位相同；只允许 `lineage.parent_checkpoint_digest` 不同，
    # 那是由 config 差异**派生**出来的记录，而且两臂本就该有不同的血缘指纹（免得被认成同一份）。
    left_identity = dict(left["identity_organ"])
    right_identity = dict(right["identity_organ"])
    left_lineage = dict(left_identity.pop("lineage"))
    right_lineage = dict(right_identity.pop("lineage"))
    assert _digest(left_identity) == _digest(right_identity), "identity_organ 的拓扑不该有差别"
    assert set(left_lineage) == set(right_lineage)
    assert {
        key for key in left_lineage if left_lineage[key] != right_lineage[key]
    } == {"parent_checkpoint_digest"}, "lineage 只该差那枚派生指纹"

    # config 只允许差 `receptors_factored` 这一格——多差一格就说明改动溢出了
    differing = {
        key for key in set(left["config"]) | set(right["config"]) if left["config"].get(key) != right["config"].get(key)
    }
    assert differing == {"receptors_factored"}, f"config 只该差开关本身，实际差: {sorted(differing)}"

    # 反过来：`predictive_context` 必须**不同**，否则"改动生效"是空话
    assert _digest(left["predictive_context"]) != _digest(right["predictive_context"])


def test_the_factored_path_is_actually_taken() -> None:
    """开关打开后 context 必须真的不一样，否则"改动生效"是空话。"""

    plain = Taiji(_config(factored=False))
    factored = Taiji(_config(factored=True))
    assert factored.predictive_context.receptors_factor is not None
    _observe(plain, b"abcabc")
    _observe(factored, b"abcabc")
    left = plain.snapshot().motor_context
    right = factored.snapshot().motor_context
    assert left.shape == right.shape, "分块不该改变 context 的宽度（下游形状不变）"
    assert not torch.equal(left, right), "分块后 context 应当不同"


def test_the_factored_variant_round_trips_through_a_checkpoint() -> None:
    model = Taiji(_config(factored=True))
    _observe(model, b"abcabcabc")
    payload = model.checkpoint()

    restored = Taiji.from_checkpoint(payload)
    assert restored.predictive_context.receptors_factor is not None
    for original, copy in zip(
        model.predictive_context.receptors_factor,
        restored.predictive_context.receptors_factor,
        strict=True,
    ):
        assert torch.equal(original.channel, copy.channel)
        assert torch.equal(original.polarity, copy.polarity)
    # 状态也要能回来：同一串输入在两个实例上给出同样的读出
    _observe(model, b"abc")
    _observe(restored, b"abc")
    assert torch.equal(model.snapshot().motor_context, restored.snapshot().motor_context)


def test_a_factored_checkpoint_refuses_an_unfactored_architecture() -> None:
    """单边存在的开关必须报错——静默错配会把未分块的权重当成结果加下去。"""

    payload = Taiji(_config(factored=True)).checkpoint()
    payload["config"]["receptors_factored"] = False
    try:
        Taiji.from_checkpoint(payload)
    except ValueError as error:
        assert "factored" in str(error)
    else:  # pragma: no cover
        raise AssertionError("未分块的架构不该接受带分块 map 的 checkpoint")
