"""区0-only 读出的守卫：默认路径逐位不变、掩码必须咬住、config 往返要带上开关。

来历（2026-09-24）：逐区审计显示槽结构只存在于区 0（16M 时 trace 0.729 / activity 0.709），
而三区全拼接的 cue 只有 0.4197 —— 稀释发生在"读法"上。修法是 `predictive_context_region0_only`
（**掩码**而非改宽度 ⇒ `receptors` 形状与既有 checkpoint 完全不动）。

钉住四件事：

1. **默认关 ⇒ 输出与"直接 cat"逐位相同**（产品路径一字不动）；
2. **开 ⇒ 区1/2 的段恰好为 0，区 0 的段逐位保留**（掩码咬住，且不是"全变零"）；
3. **config 往返带上开关**；老 config 没有这个键 ⇒ 取默认 False（向后兼容）；
4. **checkpoint 往返**：开关开的模型存档能读回、且读数与存档前一致。
"""

from __future__ import annotations

import torch

from seed import Seed, SeedConfig
from taiji import TaijiConfig

DATA = b"abcdefgh" * 6


def _config(*, region0: bool) -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            seed=7,
            predictive_context_region0_only=region0,
        )
    )


def _run(region0: bool) -> tuple[Seed, torch.Tensor, torch.Tensor]:
    model = Seed(_config(region0=region0), episode_id="r0-mask-guard")
    for symbol in DATA:
        model.substrate.observe(int(symbol), readout="action", learn=True)
    regions = model.architecture._state.regions
    vec = model.architecture.fabric.predictive_context(regions)
    return model, vec, regions


def test_default_off_matches_plain_concat() -> None:
    """绿：默认关 ⇒ 与直接 cat 逐位相同（产品路径一字不动）。"""

    model, vec, regions = _run(region0=False)
    plain = torch.cat(
        [*(r.activity for r in regions), *(r.trace for r in regions)],
        dim=0,
    )
    assert torch.equal(vec, plain), "默认路径不该有掩码"
    assert model.architecture.config.predictive_context_region0_only is False


def test_region0_mask_zeroes_only_the_other_regions() -> None:
    _model, vec, regions = _run(region0=True)
    n0 = regions[0].activity.shape[0]
    total = sum(r.activity.shape[0] for r in regions)  # 112（64+48）
    assert vec.shape == (2 * total,)
    assert bool((vec[n0:total] == 0).all()), "区1/2 的 activity 段应为 0"
    assert bool((vec[total + n0 :] == 0).all()), "区1/2 的 trace 段应为 0"
    assert bool((vec[:n0] != 0).any()), "区 0 的 activity 段不该全零"
    assert bool((vec[total : total + n0] != 0).any()), "区 0 的 trace 段不该全零"


def test_the_flag_survives_config_and_checkpoint_round_trips() -> None:
    config = _config(region0=True).taiji
    assert TaijiConfig.from_dict(config.to_dict()).predictive_context_region0_only is True
    legacy = config.to_dict()
    del legacy["predictive_context_region0_only"]
    assert (
        TaijiConfig.from_dict(legacy).predictive_context_region0_only is False
    ), "老 config 没有这个键 ⇒ 必须取默认 False（向后兼容）"

    model = Seed(_config(region0=True), episode_id="r0-mask-ckpt")
    for symbol in DATA[:32]:
        model.substrate.observe(int(symbol), readout="action", learn=True)
    restored = Seed.from_checkpoint(model.checkpoint())
    assert restored.architecture.config.predictive_context_region0_only is True
    assert restored.architecture.fabric.config.predictive_context_region0_only is True


def test_masked_checkpoint_round_trips_and_reproduces_its_reading() -> None:
    model = Seed(_config(region0=True), episode_id="r0-mask-round")
    for symbol in DATA:
        model.substrate.observe(int(symbol), readout="action", learn=True)
    restored = Seed.from_checkpoint(model.checkpoint())
    for symbol in DATA[:16]:
        model.substrate.observe(int(symbol), readout="action", learn=False)
        restored.substrate.observe(int(symbol), readout="action", learn=False)
    assert torch.equal(
        model.snapshot().motor_context, restored.snapshot().motor_context
    ), "掩码模型存档往返后，同一输入必须给出同样的 context"
