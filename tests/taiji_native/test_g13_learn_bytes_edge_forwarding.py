"""DEBT-G13 守卫：`Seed.learn_bytes` 必须把流沿三参数转发到底层。

历史缺口（PLAN-A-30 §6/rev49 ③ 机检在件）：门面只转发
`epochs/include_boundary/use_memory`，`include_start_boundary`/`include_end_boundary`/`reset`
在产品自学习路径上表达不出来——"每答收一次尾/在换行后收尾"因此无法从产品侧下达。
owner 裁定（2026-09-29，PLAN-A-30 §7 第 2 项）＝现在开：一行转发＋**默认面不变**守卫。

默认面不变的口径：门面新参数的默认值取底层签名原值
（`include_start_boundary=None`、`include_end_boundary=None`、`reset=True`——None＝沿用
`include_boundary` 派生），因此不传新参数时底层收到的 kwargs 与转发前逐位相同。
"""

from __future__ import annotations

import inspect
from typing import Any

from seed.config import SeedConfig
from seed.model import Seed
from taiji import TaijiConfig
from taiji.internalization import content_digest

EDGE_PARAMS = ("include_start_boundary", "include_end_boundary", "reset")


def _tiny_taiji_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        predictive_context_fan_in=2,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        identity_organ_capacity=8,
        concept_capacity=8,
        seed=101,
    )


def _tiny_seed() -> Seed:
    return Seed(SeedConfig(taiji=_tiny_taiji_config()))


class _RecordingSubstrate:
    """替身底座：只记 kwargs，不真学。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def learn_bytes(self, data: bytes, **kwargs: Any) -> dict[str, float]:
        self.calls.append({"data": data, **kwargs})
        return {"observations": float(len(data))}


def test_facade_signature_exposes_edge_split_params() -> None:
    """机检（原 probe facade_gap 的常驻化）：三参数必须能从产品门面到达。"""

    facade = inspect.signature(Seed.learn_bytes).parameters
    missing = [name for name in EDGE_PARAMS if name not in facade]
    assert missing == [], f"facade learn_bytes 缺参数: {missing}"


def test_default_call_forwards_substrate_defaults_bit_identically() -> None:
    """默认面不变：不传新参数时，底层收到的 kwargs 与转发前逐位相同。"""

    seed = _tiny_seed()
    stub = _RecordingSubstrate()
    seed.substrate = stub  # type: ignore[assignment]

    seed.learn_bytes(b"abc", epochs=2, use_memory=False)

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["data"] == b"abc"
    assert call["epochs"] == 2
    assert call["include_boundary"] is True
    assert call["use_memory"] is False
    # 底层签名原默认：None＝沿用 include_boundary 派生；reset=True＝整流默认。
    assert call["include_start_boundary"] is None
    assert call["include_end_boundary"] is None
    assert call["reset"] is True


def test_explicit_edge_params_pass_through_verbatim() -> None:
    seed = _tiny_seed()
    stub = _RecordingSubstrate()
    seed.substrate = stub  # type: ignore[assignment]

    seed.learn_bytes(
        b"abc",
        include_boundary=False,
        include_start_boundary=False,
        include_end_boundary=True,
        reset=False,
    )

    call = stub.calls[0]
    assert call["include_boundary"] is False
    assert call["include_start_boundary"] is False
    assert call["include_end_boundary"] is True
    assert call["reset"] is False


def test_default_face_matches_direct_substrate_call() -> None:
    """行为面：门面默认调用与底层默认调用在同配置同数据上产出逐位相同的结果。"""

    data = b"abcd" * 6

    facade_model = _tiny_seed()
    facade_metrics = facade_model.learn_bytes(data)

    direct_model = _tiny_seed()
    direct_metrics = direct_model.substrate.learn_bytes(data)

    assert facade_metrics == direct_metrics
    assert content_digest(facade_model.substrate.checkpoint()) == content_digest(
        direct_model.substrate.checkpoint()
    )


def test_facade_split_chunks_preserve_single_stream_semantics() -> None:
    """收益面：经门面做分块续喂（首段只带起沿、续段只带收沿且不重置）
    与整流学习逐位等价——这是转发打开之前产品侧表达不出的形状。"""

    data = b"abcd" * 6
    cut = 11

    direct = _tiny_seed()
    direct_metrics = direct.learn_bytes(data)

    split = _tiny_seed()
    first_metrics = split.learn_bytes(
        data[:cut],
        include_boundary=False,
        include_start_boundary=True,
        include_end_boundary=False,
        reset=True,
    )
    second_metrics = split.learn_bytes(
        data[cut:],
        include_boundary=False,
        include_start_boundary=False,
        include_end_boundary=True,
        reset=False,
    )

    assert content_digest(split.substrate.checkpoint()) == content_digest(
        direct.substrate.checkpoint()
    )
    assert direct_metrics["observations"] == (
        first_metrics["observations"] + second_metrics["observations"]
    )
