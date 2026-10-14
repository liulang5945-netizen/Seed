"""DEBT-G140／PLAN-N5-26 §2 的载体：`config.seed` 是结构身份，不是可拨的混杂旋钮。

㊵-824 现读（先由仓外一次性探针 `seed_structural_control_824.py` 打到现产品基座：1152→96 的受体在
seed 20260822 之外换两枚种子，与在档通道映射逐位比得 1137／1152、1142／1152 位不同，
`Seed.from_checkpoint` 一律拒绝）。本册用**小剖面**把那条读数以能红的形状钉住，不依赖任何在档 `.pt`：

* `SparseReceptorBank.__init__` 用 `torch.randperm(..., generator=generator)` 与
  `torch.randint(..., generator=generator)` 画 `channel`／`polarity`（`taiji/organs.py:99-104`），
  而那枚 generator 的种子来自 `config.seed`（`taiji/model.py:115-119`）；
* `load_payload` 要求逐位相等，否则 `ValueError: receptor channel map does not match architecture`
  （`taiji/organs.py:162-165`）。

⇒ 十三裁 ①「乙：再跑一对只换混杂参数的臂」在现形态下起不来：换 seed 连基座都装不上。
噪声地板因此不是"测出来等于 0"，而是"这条管线里造不出第二个混杂档"。
"""

from __future__ import annotations

import pytest

from taiji.config import TaijiConfig
from taiji.model import Taiji

SEED_A = 111
SEED_B = 222


def _model(seed: int) -> Taiji:
    return Taiji(TaijiConfig.training_profile(seed=seed))


def test_the_same_seed_reproduces_the_receptor_map_bit_for_bit() -> None:
    """The map is a pure function of the seed: two constructions agree position by position."""

    left = _model(SEED_A).motor.receptors
    right = _model(SEED_A).motor.receptors
    assert bool((left.channel == right.channel).all())
    assert bool((left.polarity == right.polarity).all())


def test_an_alternate_seed_moves_the_receptor_map() -> None:
    """Dynamic range for the guard below: a second seed really does change the wiring."""

    left = _model(SEED_A).motor.receptors
    right = _model(SEED_B).motor.receptors
    differing = int((left.channel != right.channel).sum())
    assert differing > 0
    assert not bool((left.polarity == right.polarity).all())


def test_restore_accepts_a_checkpoint_from_the_same_seed() -> None:
    """The refusing branch must have a passing counterpart, or the guard discriminates nothing."""

    source = _model(SEED_A)
    target = _model(SEED_A)
    target.restore(source.checkpoint())
    assert bool((target.motor.receptors.channel == source.motor.receptors.channel).all())


def test_restore_refuses_a_checkpoint_built_under_a_different_seed() -> None:
    target = _model(SEED_B)
    foreign = _model(SEED_A).checkpoint()
    with pytest.raises(ValueError, match="checkpoint configuration does not match architecture"):
        target.restore(foreign)


def test_restore_refuses_the_foreign_receptor_map_even_when_the_config_matches() -> None:
    """The face the ㊵-824 probe hit: config agrees, the stored wiring came from another seed."""

    source = _model(SEED_A)
    target = _model(SEED_B)
    payload = source.checkpoint()
    payload["config"] = target.config.to_dict()
    with pytest.raises(ValueError, match="receptor channel map does not match architecture"):
        target.restore(payload)
