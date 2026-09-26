"""B-4 第八刀：`ResonanceEnsemble.add_neuron`（热插拔时的跨规格投影与几何放置）。

这一段是"混合规格联合皮层"的接线处：新神经元 field_dim 与统一场不符时必须**自动补建**
跨规格投影层（前向与反向各一枚），否则推理期 `_project_vec` 走 identity ⇒ 维度错配崩；
hidden_size 不符则按 C27 阶段 3 的决定**只警告不报错**。两种语义差别必须被分别钉住。

夹具：生产 ensemble（fallback 单神经元，场 dim=4096）＋按需构造的小 field_dim 神经元。
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch

from neuroplex.resonance.config import get_domain_neuron_config
from neuroplex.resonance.neuron import ResonanceNeuron

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()


@pytest.fixture()
def ensemble(tmp_path):
    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    return built.ensemble


def _neuron(domain: str = "zh", field_dim: int | None = None, hidden: int | None = None):
    cfg = get_domain_neuron_config(domain)
    changes = {}
    if field_dim is not None:
        changes["field_dim"] = field_dim
    if hidden is not None:
        changes["hidden_size"] = hidden
    if changes:
        cfg = dataclasses.replace(cfg, **changes)
    neuron = ResonanceNeuron(cfg)
    neuron.eval()
    return neuron, cfg


def test_duplicate_id_is_rejected_before_any_mutation(ensemble) -> None:
    existing = next(iter(ensemble.neurons))
    before = dict(ensemble.neurons)
    with pytest.raises(ValueError):
        ensemble.add_neuron(existing, _neuron()[0])
    assert set(ensemble.neurons) == set(before), "被拒的添加不得改动注册表"


def test_matching_field_dim_creates_no_projectors(ensemble) -> None:
    assert ensemble.field.dim == 4096
    neuron, cfg = _neuron("zh", field_dim=ensemble.field.dim)
    ensemble.add_neuron("zh_1", neuron)
    assert "zh_1" in ensemble.neurons
    assert "zh_1" not in ensemble._cross_spec_projectors
    assert "zh_1" not in ensemble._cross_spec_back_projectors


def test_mismatched_field_dim_builds_projectors_both_directions(ensemble) -> None:
    neuron, cfg = _neuron("code", field_dim=64)
    assert cfg.field_dim == 64 != ensemble.field.dim
    ensemble.add_neuron("code_1", neuron)

    forward = ensemble._cross_spec_projectors["code_1"]
    back = ensemble._cross_spec_back_projectors["code_1"]
    # 投影层没有 in_dim/out_dim 属性，形状声明在 linear1.weight = [out, in] 上
    assert tuple(forward.linear1.weight.shape) == (ensemble.field.dim, 64), "前向投影方向反了"
    assert tuple(back.linear1.weight.shape) == (64, ensemble.field.dim), "反向投影必须是逆形状"

    # 真投影一次：维度按声明变化（不是 identity 冒充）
    probe = torch.ones(1, 64)
    out = forward(probe)
    assert out.shape[-1] == ensemble.field.dim
    assert back(out).shape[-1] == 64
    # 构造契约：linear2 零初始化 ⇒ 初始行为恰好等于 linear1(x)（与旧单层 Linear 一致）
    assert float(back.linear2.weight.abs().max()) == 0.0
    assert torch.allclose(forward(probe), forward.linear1(probe), atol=1e-6)


def test_mixed_hidden_size_is_allowed_with_warning_not_exception(ensemble) -> None:
    """C27 阶段 3：hidden 不一致由 per-neuron embed_adapter 适配 ⇒ 只警告，不 raise。"""

    anchor_hidden = next(iter(ensemble.neurons.values())).config.hidden_size
    neuron, cfg = _neuron("en", hidden=anchor_hidden + 64)
    ensemble.add_neuron("en_1", neuron)  # 旧实现在这里 raise ValueError
    assert "en_1" in ensemble.neurons
    assert (
        ensemble.neurons["en_1"].config.hidden_size == anchor_hidden + 64
    ), "混合规格必须原样登记，不能被静默改写"


def test_refractory_counter_is_moved_onto_the_registry_device(ensemble) -> None:
    anchor = next(iter(ensemble.neurons.values()))
    neuron, _cfg = _neuron("math")
    neuron.refractory_counter = torch.zeros(1, dtype=torch.long).to("cpu")
    ensemble.add_neuron("math_1", neuron)
    assert ensemble.neurons["math_1"].refractory_counter.device == anchor.refractory_counter.device


def test_geometry_placement_follows_split_and_domain_rules(ensemble) -> None:
    geometry = ensemble.geometry
    assert geometry is not None, "夹具没有几何空间 ⇒ 本件三条放置规则都成了空测"

    parent, _ = _neuron("zh")
    ensemble.add_neuron("zh_1", parent)
    parent_pos = geometry.positions["zh_1"]

    child, _ = _neuron("zh")
    ensemble.add_neuron("zh_2", child, from_split="zh_1")
    child_pos = geometry.positions["zh_2"]
    assert not torch.equal(child_pos, parent_pos), "分裂放置必须真产生新位置"
    split_dist = float((child_pos - parent_pos).norm())

    # 同域第三枚：无父 ⇒ 取同域中心附近；新域枚：随机放置。用**相对**判据，避免把
    # 几何维度写进阈值（偏移量是 0.05·‖·‖，随机放置是 0.3·√d，比例稳定而绝对值随 d 变）。
    third, _ = _neuron("zh")
    center = torch.stack([geometry.positions[n] for n in ("zh_1", "zh_2")]).mean(dim=0)
    ensemble.add_neuron("zh_3", third)
    near_center = float((geometry.positions["zh_3"] - center).norm())

    fresh, _ = _neuron("math")
    ensemble.add_neuron("math_9", fresh)
    assert "math_9" in geometry.positions
    assert geometry.positions["math_9"].shape[-1] == geometry.embedding_dim
    far = float((geometry.positions["math_9"] - center).norm())
    assert far > max(
        split_dist, near_center
    ), f"新域随机放置反而比受控放置更近（{far} vs {split_dist}/{near_center}）⇒ 三条放置规则没分流"


def test_new_projectors_join_the_aggregated_state(ensemble) -> None:
    """跨规格投影一旦补建，就必须进聚合视图——否则热插拔的投影层重启即失。"""

    neuron, _ = _neuron("en", field_dim=32)
    ensemble.add_neuron("en_7", neuron)
    state = ensemble.state_dict()
    assert "en_7" in state["cross_spec_projectors"]
    assert "en_7" in state["cross_spec_back_projectors"]
    skipped = ensemble.load_state_dict(state)
    assert skipped == [], f"干净回读不该有跳过项：{skipped}"
