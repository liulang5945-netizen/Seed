"""身份器官路由键仓的**截断存盘**守卫（PLAN-A-29 乙档，owner 裁定 (i) 的第一步）。

背景（实测）：`_value_keys` 是按 `[slots × max_keys × pattern_dim]` 稠密预分配的零表
（产品基底 128×64×1152 float32 ＝ 37.75 MB），而那张仓在出厂基底上**从来没被写过**
（零面普查记过同一件事）；`to_payload()` 却把整张缓冲原样落盘 ⇒ 产品信封 87.4 MB 里
**75.5 MB 是两份空镜像的全零稠密仓**。修法＝按档里已有的 `value_counts` 只存前
`used = max(counts)` 行，还原时补回空槽（键 0／动作 −1）。

本文件钉六件事，前两条是**实测踩过的坑**：

1. **切片必须 `clone()`**：不 clone 的切片是**视图**，底层 storage 仍是整张稠密缓冲，
   而 `torch.save` 序列化的是 storage ⇒ "截断"一位字节都省不下来
   （实测：形状 `(128, 0, 1152)`、`numel()==0` 的张量仍写出 **37.75 MB**）。
2. 带内容的往返**逐位相同**（真写过路由键的器官，裁掉的区间必须本来就空）。
3. 空仓档比稠密档**小**（尺寸这条判据不许只靠推理）。
4. **旧档兼容**：没有 `value_router_used` 的整表档照旧载入（产品件就是这一档）。
5. 更旧的**无路由三件**的档也必须载入——`load_payload` 里那句
   `self._value_counts = restored_counts.clone()` 曾写在 `if` 外面，注释声称的这条兼容路
   实测 `UnboundLocalError` 直接崩（本件顺手修掉，故此条守卫是**阳性对照**）。
6. **不许静默还原成零**：counts 超过存下来的行数／`used` 与 counts 不自洽／不变量破了却
   没回退成整表 ⇒ 全部响亮拒绝或整表存盘。
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import torch

from scripts.training.eval_taiji_foundation_baseline import _memory_config
from scripts.training.eval_taiji_m1_64_foundation_memory import (
    build_foundation_delayed_memory_corpus,
)
from taiji import Taiji, TaijiConfig
from taiji.foundation_tasks import DelayedMemoryTask

SLOTS_ROW = 1  # `_value_keys` 的形状是 (capacity, max_keys, pattern_dim)


def _model(*, router_off: bool = False) -> Taiji:
    #: 与 `test_m1_66b_value_router` 同一配法——路由仓要有内容才谈得上"截断是否无损"。
    values = _memory_config(11).to_dict()
    values["identity_organ_capacity"] = 256
    if router_off:
        values["identity_organ_value_router_enabled"] = False
    return Taiji(TaijiConfig.from_dict(values), episode_id="a29-truncation")


def _trained_model() -> Taiji:
    """真往路由仓里写过键的模型（与 `test_m1_66b_value_router` 同一喂法）。"""

    model = _model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    return model


def _payload_size(tensor: torch.Tensor) -> int:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.tell()


# ---------------------------------------------------------------- 1. 视图 storage 陷阱


def test_truncated_slice_does_not_inherit_the_dense_storage() -> None:
    """截断后的张量**自己的 storage** 必须只有 `numel * itemsize` 字节。

    这条就是那个坑本身：`keys[:, :used]` 形状对、`numel()` 对，但 storage 仍是整表，
    `torch.save` 照样写 37.75 MB ⇒ 尺寸判据只能靠 storage 字节数来钉。
    """

    model = _trained_model()
    organ = model.identity_organ
    payload = organ.to_payload(parent_checkpoint_digest="f" * 64)
    stored = payload["value_keys"]
    assert stored.untyped_storage().nbytes() == stored.numel() * stored.element_size()
    #: 与"整表存盘"比才是有意义的断言——`torch.save` 的 zip/pickle 元数据本身要 ~1 KB，
    #: 拿绝对字节数当阈值会把度量仪器的开销读成模型缺陷（本仓在两处合法性口径上踩过同形坑）。
    dense = _payload_size(torch.zeros_like(organ._value_keys))
    assert _payload_size(stored) < dense / 10, (_payload_size(stored), dense)


# ---------------------------------------------------------------- 2/3. 往返与尺寸


def test_router_payload_round_trips_bitwise_with_real_writes() -> None:
    source = _trained_model()
    organ = source.identity_organ
    assert int(organ._value_counts.max().item()) > 0, "这臂必须真写过路由键"
    payload = organ.to_payload(parent_checkpoint_digest="f" * 64)
    used = int(payload["value_router_used"])
    assert used == int(organ._value_counts.max().item())
    assert tuple(payload["value_keys"].shape) == (organ.capacity, used, organ.pattern_dim)

    fresh = _model()
    fresh.identity_organ.load_payload(dict(payload))
    assert torch.equal(fresh.identity_organ._value_keys, organ._value_keys)
    assert torch.equal(fresh.identity_organ._value_actions, organ._value_actions)
    assert torch.equal(fresh.identity_organ._value_counts, organ._value_counts)
    assert fresh.identity_organ._value_keys.shape == organ._value_keys.shape


def test_empty_router_store_is_smaller_than_the_dense_one() -> None:
    fresh = _model()
    payload = fresh.identity_organ.to_payload(parent_checkpoint_digest="f" * 64)
    assert int(payload["value_router_used"]) == 0
    assert payload["value_keys"].numel() == 0
    dense = _payload_size(torch.zeros_like(fresh.identity_organ._value_keys))
    assert _payload_size(payload["value_keys"]) * 100 < dense, "省下来的必须是两个数量级"


# ---------------------------------------------------------------- 4/5. 旧档兼容


def test_legacy_dense_payload_without_marker_still_loads() -> None:
    organ = _trained_model().identity_organ
    payload: dict[str, Any] = dict(organ.to_payload(parent_checkpoint_digest="f" * 64))
    used = int(payload.pop("value_router_used"))
    padded = torch.zeros_like(organ._value_keys)
    padded[:, :used] = payload["value_keys"]
    payload["value_keys"] = padded
    padded_actions = torch.full_like(organ._value_actions, -1)
    padded_actions[:, :used] = payload["value_actions"]
    payload["value_actions"] = padded_actions

    fresh = _model()
    fresh.identity_organ.load_payload(payload)  # 无标记 ⇒ 整表分支
    assert torch.equal(fresh.identity_organ._value_keys, organ._value_keys)


def test_legacy_payload_without_the_router_at_all_loads_empty() -> None:
    """阳性对照：注释声称"旧档没有路由三件就空载"，改前实测是 `UnboundLocalError`。"""

    organ = _trained_model().identity_organ
    payload = {
        key: value
        for key, value in organ.to_payload(parent_checkpoint_digest="f" * 64).items()
        if not key.startswith("value_")
    }
    fresh = _model()
    fresh.identity_organ.load_payload(payload)
    assert int(fresh.identity_organ._value_counts.sum().item()) == 0
    assert bool((fresh.identity_organ._value_keys == 0).all())
    assert bool((fresh.identity_organ._value_actions == -1).all())


# ---------------------------------------------------------------- 6. 不许静默丢数据


def test_counts_beyond_stored_rows_are_rejected_not_zero_filled() -> None:
    organ = _trained_model().identity_organ
    payload = dict(organ.to_payload(parent_checkpoint_digest="f" * 64))
    counts = payload["value_counts"].clone()
    counts[0] = int(payload["value_router_used"]) + 1  # 声称有一行在存档之外
    payload["value_counts"] = counts
    with pytest.raises(ValueError, match="exceed stored rows"):
        _model().identity_organ.load_payload(payload)


def test_marker_inconsistent_with_counts_is_rejected() -> None:
    """档自称存了 `used+1` 行，而 counts 的上界只有 `used` ⇒ 自称与内容不自洽，拒绝。"""

    organ = _trained_model().identity_organ
    payload = dict(organ.to_payload(parent_checkpoint_digest="f" * 64))
    used = int(payload["value_router_used"])
    claimed = used + 1
    padded_keys = torch.zeros((organ.capacity, claimed, organ.pattern_dim), dtype=torch.float32)
    padded_keys[:, :used] = payload["value_keys"]
    padded_actions = torch.full((organ.capacity, claimed), -1, dtype=torch.long)
    padded_actions[:, :used] = payload["value_actions"]
    tampered = dict(
        payload,
        value_router_used=claimed,
        value_keys=padded_keys,
        value_actions=padded_actions,
    )
    with pytest.raises(ValueError, match="do not match counts"):
        _model().identity_organ.load_payload(tampered)


def test_broken_empty_region_invariant_falls_back_to_dense_storage() -> None:
    """不变量破了（存档区外却有值）⇒ **整表存盘**，宁可大也不丢。"""

    organ = _trained_model().identity_organ
    rows = int(organ._value_keys.shape[SLOTS_ROW])
    headroom = rows - int(organ._value_counts.max().item())
    assert headroom >= 2, "需要存得下的空区才谈得上回退"
    organ._value_counts[0] = 0  # 假装计数没跟上，但该区已有键
    organ._value_keys[0, rows - 1] = 1.0
    payload = organ.to_payload(parent_checkpoint_digest="f" * 64)
    assert "value_router_used" not in payload
    assert tuple(payload["value_keys"].shape) == tuple(organ._value_keys.shape)
