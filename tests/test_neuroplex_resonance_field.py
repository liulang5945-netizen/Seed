"""B-4 覆盖率第二刀：`neuroplex/resonance/field.py`（共振场＝架构的"神经语言"）。

面内此前无人单测。断言全部钉**语义不变量**（写入归一化/替换、抑制的乘法衰减与撤销、
WTA 只留最强、leave-one-out 同时撤 excitatory 与 inhibitory、批量提升与批量不符必须报错），
不做"跑通即绿"的形状测试。
"""

from __future__ import annotations

import pytest
import torch

from neuroplex.resonance.field import ResonanceField

DIM = 16


def _field(dim: int = DIM) -> ResonanceField:
    torch.manual_seed(3)
    field = ResonanceField(dim=dim)
    field.reset()
    return field


def _vec(seed: int, dim: int = DIM) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g)


def test_write_l2_normalises_then_scales_direction_kept() -> None:
    field = _field()
    v = _vec(1) * 9.0  # 幅度必须被归一化掉
    field.write("n1", v, scale=1.0)
    unit = v / v.norm()
    assert torch.allclose(field.state, unit, atol=1e-5), "写入后 state 应为单位方向向量"

    field2 = _field()
    field2.write("n1", v, scale=2.5)
    assert torch.allclose(field2.state, unit * 2.5, atol=1e-5), "scale 只调幅度、不改方向"
    assert field2.n_active == 1


def test_write_rejects_batch_mismatch_and_auto_promotes() -> None:
    field = _field()
    field.write("n1", _vec(2))
    # 已经显式声明为 2 样本的场，写 3 批必须报错（静默广播＝跨样本污染）
    field.reset(batch_size=2)
    with pytest.raises(ValueError):
        field.write("n2", torch.randn(3, DIM))

    fresh = _field()
    fresh.write("b", torch.randn(2, DIM))  # [D] 场遇 [2, D] 写入 ⇒ 提升为逐样本场
    assert fresh.state.shape == (2, DIM) and fresh.batch_size == 2
    # H2 的实质：提升后每个样本**各加自己的向量**，不是共享一份场（共享＝跨样本污染）
    probe = torch.randn(2, DIM)
    fresh2 = _field()
    fresh2.write("b", probe)
    unit = probe / probe.norm(dim=-1, keepdim=True)
    assert torch.allclose(fresh2.state[0], unit[0], atol=1e-6)
    assert torch.allclose(fresh2.state[1], unit[1], atol=1e-6)
    assert not torch.equal(fresh2.state[0], fresh2.state[1])
    # 提升后逐样本写入不再互相覆盖（用两个方向不同的样本向量，注意 L2 归一化会让
    # 全 1 与全 2 变成同一个单位向量 ⇒ 反例要真的不同向）
    fresh.write("c", torch.stack([torch.ones(DIM), torch.arange(DIM).float() + 1.0]))
    assert not torch.equal(fresh.state[0], fresh.state[1])


def test_update_replaces_instead_of_accumulating() -> None:
    """多轮共振的替换语义：update 不叠加，state 无界增长就是这里的回归。"""

    field = _field()
    a, b = _vec(4), _vec(5)
    field.write("n1", a)
    field.update("n1", b)
    unit_b = b / b.norm()
    assert torch.allclose(field.state, unit_b, atol=1e-5), "update 必须撤旧加新，而不是累加"

    # 连做 5 轮 update 后 state 仍是一个单位向量（旧实现会涨到 5 倍）
    for _ in range(5):
        field.update("n1", a)
    assert float(field.state.norm()) == pytest.approx(1.0, abs=1e-5)


def test_write_history_is_bounded_per_neuron() -> None:
    field = _field()
    for step in range(10):
        field.write("n1", _vec(step))
    assert len(field._write_history["n1"]) == ResonanceField.HISTORY_MAXLEN

    # 历史存的是单位向量（与调质幅度解耦）
    assert float(field._write_history["n1"][-1].norm()) == pytest.approx(1.0, abs=1e-5)


def test_inhibition_is_multiplicative_decay_bounded_in_zero_one() -> None:
    field = _field()
    field.write("e1", _vec(6))
    before = field.get_effective_state().clone()
    mask = field.write_inhibit("i1", _vec(7), weight=1.0)
    assert mask.min() >= 0.0 and mask.max() <= 1.0, "掩码必须留在 [0,1]（负值=语义错误）"

    after = field.get_effective_state()
    # 抑制只可能让每个维度的贡献变小，不可能放大
    assert float((after - before).abs().max()) > 1e-6, "抑制必须改变有效场"

    # 两位抑制者按乘法累积，且更强的抑制 ⇒ 更小的掩码
    field_b = _field()
    field_b.write_inhibit("i1", _vec(7), weight=0.3)
    m1 = field_b.inhibitory_mask.clone()
    field_b.write_inhibit("i2", _vec(8), weight=0.3)
    assert float(field_b.inhibitory_mask.max()) <= float(m1.max()) + 1e-6


def test_wta_keeps_only_the_strongest_inhibitor() -> None:
    field = _field()
    field.write_inhibit("weak", torch.ones(DIM), weight=0.2)
    field.write_inhibit("strong", torch.ones(DIM), weight=0.9)
    field.write_inhibit("mid", torch.ones(DIM), weight=0.5)

    kept = field.apply_inhibitory_wta(top_k=1)
    assert kept == 1
    assert set(field._inhibit_contributions) == {"strong"}, "WTA 后只剩胜者的贡献记录"
    # 胜者 decay 可解析算出：|v| 均匀 ⇒ v_abs = 1/√D ⇒ decay = 1 - w/√D
    expect = 1.0 - 0.9 / DIM**0.5
    assert torch.allclose(
        field.inhibitory_mask, torch.full((DIM,), expect), atol=1e-6
    ), field.inhibitory_mask

    # 全员抑制（top_k 覆盖全部）时不发生竞争，原样保留
    field2 = _field()
    field2.write_inhibit("a", torch.ones(DIM), weight=0.4)
    field2.write_inhibit("b", torch.ones(DIM), weight=0.6)
    assert field2.apply_inhibitory_wta(top_k=5) == 2
    assert set(field2._inhibit_contributions) == {"a", "b"}


def test_lateral_norm_caps_magnitude_but_keeps_direction() -> None:
    field = _field()
    for step in range(5):
        field.write(f"n{step}", _vec(20))  # 全部同向 ⇒ 幅度线性叠加
    before = field.state.clone()
    assert float(before.norm()) == pytest.approx(5.0, abs=1e-4)
    field.lateral_inhibition_norm()
    assert float(field.state.norm()) == pytest.approx(1.0, abs=1e-5)
    cos = float((field.state @ before) / (field.state.norm() * before.norm()))
    assert cos == pytest.approx(1.0, abs=1e-5), "归一化不得改变方向"


def test_leave_one_out_undoes_both_excitatory_and_inhibitory() -> None:
    """BioOSS 修复点：排除某神经元时必须同时撤它的抑制衰减。"""

    field = _field()
    v = _vec(9)
    field.write("mix", v)
    field.write_inhibit("mix", torch.ones(DIM), weight=0.5)

    full = field.get_effective_state()
    loo = field._leave_one_out_state("mix")
    assert float((full - loo).abs().max()) > 1e-6, "排除自己后读数必须变（否则撤销没生效）"

    # 单写入者被排除 ⇒ 激发贡献归零
    solo = _field()
    solo.write("only", v)
    assert float(solo._leave_one_out_state("only").norm()) < 1e-3

    # 排除一个从没写过的 id ⇒ 与 get_effective_state 一致（不该误撤别人）
    untouched = solo._leave_one_out_state("ghost")
    assert torch.allclose(untouched, solo.get_effective_state(), atol=1e-6)


def test_score_uses_leave_one_out_for_the_writer_itself() -> None:
    field = _field()
    v = _vec(10)
    field.write("self", v)
    including = field.score(v)
    excluding = field.score(v, neuron_id="self")
    assert including > 0.5, f"同方向应得正分，实得 {including}"
    assert excluding < including, "把自己撤掉后不能再给自己加分（H5 语义）"

    opposite = field.score(-v)
    assert opposite < including, "反方向必须得分更低"


def test_state_round_trip_save_and_load() -> None:
    field = _field()
    field.write("n1", _vec(11))
    field.write_inhibit("i1", torch.ones(DIM), weight=0.3)
    state_before = field.state.clone()
    mask_before = field.inhibitory_mask.clone()

    snap = field.save_round_state()
    assert snap, "快照不能是空容器（空容器＝save/load 形同虚设）"

    field.write("n2", _vec(12))
    assert not torch.equal(field.state, state_before), "先确认写入确实改了场，否则回读无意义"

    fresh = _field()
    fresh.load_round_state(snap)
    assert torch.allclose(fresh.state, state_before, atol=1e-6), "load 必须还原激发态"
    assert torch.allclose(fresh.inhibitory_mask, mask_before, atol=1e-6), "load 必须还原抑制掩码"


def test_reset_clears_everything_and_batch_promotion_is_explicit() -> None:
    field = _field()
    field.write("n1", _vec(13))
    field.write_inhibit("i1", torch.ones(DIM), weight=0.5)
    field.reset()
    assert float(field.state.abs().max()) == 0.0
    assert float(field.inhibitory_mask.min()) == 1.0
    assert field._contributions == {} and field._inhibit_contributions == {}
    assert field.scores == {} and field.n_active == 0

    batched = _field()
    batched.reset(batch_size=3)
    assert batched.state.shape == (3, DIM) and batched.inhibitory_mask.shape == (3, DIM)
    assert batched.batch_size == 3
    batched.write("n1", torch.randn(3, DIM))
    assert batched.state.shape == (3, DIM)
