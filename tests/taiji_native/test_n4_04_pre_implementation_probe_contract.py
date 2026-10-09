"""PLAN-N4-04 §3 步骤 2 的反例探针（动产品码之前必须先过的三道机械检查）。

预注册把顺序写死了：**先证明 J-N4d-2『默认关时逐位不变』这条测能为假**（尺有动态范围），
**再动 `taiji/`**。这一册就是那次探针；它不改产品码，只把三件事钉成可重跑的断言：

1. **现状事实**：`TaijiConfig` 里**没有**挂载默认位字段，而字段发现机制不是瞎的
   （同一个集合里能读到已存在的学习侧字段）——否则"缺字段"这条结论只是因为看不见。
2. **动态范围**：观测摘要对**输入**与**种子**都敏感 ⇒ 同一支摘要在"默认位挪一格"时
   确实可能变红；若这里就判不出差异，J-N4d-2 会是恒真守卫，必须换测法而不是照抄实现。
3. **挂载点在哪一层**：`attach_episodic_memory` 只在底座（Taiji）上可得、
   Seed 包装层没有 ⇒ 未来那一格必须先决定"暴露在哪一层"，不能默认挂在 Seed 上就算产品默认。

三支都直接读真模型（tiny 配置，几毫秒级），不用替身。
"""

from __future__ import annotations

import dataclasses

from seed import Seed, SeedConfig
from taiji import TaijiConfig
from taiji.internalization import content_digest

PROBE_BYTES = bytes(range(32, 96))
OTHER_BYTES = bytes(range(32, 95))


def _tiny_config(seed: int = 101) -> TaijiConfig:
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
        seed=seed,
    )


def _model(seed: int = 101) -> Seed:
    return Seed(SeedConfig(taiji=_tiny_config(seed)))


def _observation_digest(model: Seed, symbols: bytes) -> str:
    rows = []
    for symbol in symbols:
        step = model.observe(symbol, learn=True, readout="predictive")
        prior = getattr(step, "prior_prediction", None)
        surprise = getattr(step, "surprise", None)
        rows.append(
            [
                None if prior is None else int(prior),
                None if surprise is None else round(float(surprise), 12),
            ]
        )
    return str(content_digest({"rows": rows}))


def test_config_still_has_no_default_mount_field_and_discovery_is_not_blind() -> None:
    names = {field.name for field in dataclasses.fields(TaijiConfig)}
    assert "episodic_memory_default_mount" not in names
    #: 同一条取法必须能读到已存在的字段——否则"没有"只是因为我的眼睛不好使。
    assert {"episodic_learning_rate", "episodic_write_repeats"} <= names


def test_observation_digest_reproduces_bit_for_bit_on_two_fresh_models() -> None:
    first = _observation_digest(_model(), PROBE_BYTES)
    second = _observation_digest(_model(), PROBE_BYTES)
    assert first == second


def test_digest_has_dynamic_range_over_input_and_seed() -> None:
    #: 反例方向：换输入或换种子都必须变。若不变，J-N4d-2 就是恒真式，
    #: 那条"默认关逐位不变"的判据就没有资格进实现格（本仓教训：恒真守卫＝零判别力）。
    base = _observation_digest(_model(), PROBE_BYTES)
    different_input = _observation_digest(_model(), OTHER_BYTES)
    different_seed = _observation_digest(_model(seed=202), PROBE_BYTES)
    assert base != different_input
    assert base != different_seed


def test_mount_accessor_lives_on_the_substrate_not_on_the_seed_wrapper() -> None:
    model = _model()
    assert hasattr(model.substrate, "attach_episodic_memory")
    assert not hasattr(model, "attach_episodic_memory")
