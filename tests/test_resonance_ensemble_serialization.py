"""B-4 第六刀：`ResonanceEnsemble.state_dict / load_state_dict`（手写序列化代理）。

审计 M12 点名 "ResonanceEnsemble 非 nn.Module ⇒ 设备迁移/序列化全靠手写代理"。
本件不重构它，先把这层代理的契约测住：聚合哪些面、跳过什么、什么时候才该抛。

夹具用 `create_cortex()` 的生产 ensemble，再挂上线性代理件与场（真实 nn.Module 替身），
所以测的是聚合/回读**协议**，与神经元规模无关。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

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


def _attach(ens) -> dict:
    """给 ensemble 装上可识别的代理件，返回它们（便于逐件断言）。"""
    parts = {
        "field_score_proj": nn.Linear(8, 4, bias=False),
        "shared_weight_mlp": nn.Linear(4, 4, bias=False),
    }
    for name, module in parts.items():
        setattr(ens, name, module)
    ens._cross_spec_projectors = {"zh_1": nn.Linear(6, 3, bias=False)}
    ens._cross_spec_back_projectors = {"zh_1": nn.Linear(3, 6, bias=False)}
    return parts


def test_state_dict_aggregates_only_present_parts(ensemble) -> None:
    parts = _attach(ensemble)
    state = ensemble.state_dict()

    assert {"field_score_proj", "shared_weight_mlp"} <= set(state)
    assert {"cross_spec_projectors", "cross_spec_back_projectors"} <= set(state)
    assert set(state["cross_spec_projectors"]) == {"zh_1"}
    assert "field" in state, "场（W_cond + buffers）必须进聚合视图——R1 训练闭环的持久化端"
    # 未创建的件不该凭空出现
    assert (
        "gamma_oscillator" not in state or getattr(ensemble, "gamma_oscillator", None) is not None
    )
    # neuron 本体不在此聚合（per-neuron ckpt 由 loader 管）
    for nid in ensemble.neurons:
        assert nid not in state
    # 聚合到的必须是真参数张量，不是空壳
    assert any(t.numel() for t in state["field_score_proj"].values())
    assert parts["field_score_proj"].weight.shape == state["field_score_proj"]["weight"].shape


def _snapshot(ens) -> dict:
    """聚合视图的**张量级副本**。

    ⚠️ `state_dict()` 返回的是参数的引用：不 clone 就直接改权重，"改前快照"会跟着变，
    于是"回读恢复了旧值"这类断言要么恒真、要么像我第一版那样恒假。
    """

    def _copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: _copy(v) for k, v in value.items()}
        return value

    return {k: _copy(v) for k, v in ens.state_dict().items()}


def test_clean_round_trip_reports_no_skips(ensemble) -> None:
    _attach(ensemble)
    state = _snapshot(ensemble)
    assert state, "聚合视图为空 ⇒ 下面的往返断言会是空测"

    with torch.no_grad():
        ensemble.field_score_proj.weight.add_(0.5)
        ensemble._cross_spec_projectors["zh_1"].weight.add_(-0.5)
    moved = _snapshot(ensemble)
    assert not torch.equal(
        moved["field_score_proj"]["weight"], state["field_score_proj"]["weight"]
    ), "改完权重后聚合视图必须真的跟着变，否则回读断言无意义"

    skipped = ensemble.load_state_dict(state)
    assert skipped == [], f"干净回读不该有跳过项：{skipped}"
    assert torch.equal(ensemble.field_score_proj.weight, state["field_score_proj"]["weight"])
    assert torch.equal(
        ensemble._cross_spec_projectors["zh_1"].weight,
        state["cross_spec_projectors"]["zh_1"]["weight"],
    )


def test_unknown_neuron_and_missing_module_are_skipped_not_fatal(ensemble) -> None:
    _attach(ensemble)
    state = ensemble.state_dict()
    state["cross_spec_projectors"]["ghost_9"] = nn.Linear(6, 3, bias=False).state_dict()
    state["sparse_router"] = nn.Linear(2, 2, bias=False).state_dict()  # 当前 ensemble 没有这件

    skipped = ensemble.load_state_dict(state)
    joined = " | ".join(skipped)
    assert "cross_spec_projectors[ghost_9]" in joined, f"未知 nid 必须进跳过清单：{joined}"
    assert "sparse_router" in joined, f"件不在场也必须进跳过清单：{joined}"
    assert "zh_1" not in joined, "跳过清单不该误报健康的条目"


def test_shape_mismatch_is_recorded_and_leaves_the_module_untouched(ensemble) -> None:
    parts = _attach(ensemble)
    state = ensemble.state_dict()
    bad = nn.Linear(17, 9, bias=False).state_dict()
    state["field_score_proj"] = bad
    before = parts["field_score_proj"].weight.detach().clone()

    skipped = ensemble.load_state_dict(state)
    assert any("field_score_proj" in item for item in skipped), skipped
    assert torch.equal(
        ensemble.field_score_proj.weight, before
    ), "形状不符必须整件跳过，不能部分写入"


def test_field_state_round_trips_through_the_aggregate(ensemble) -> None:
    field = ensemble.field
    with torch.no_grad():
        field.W_cond.mul_(1.7)
    snapshot = _snapshot(ensemble)["field"]
    assert snapshot, "场聚合为空 ⇒ 本条是空测"

    with torch.no_grad():
        field.W_cond.mul_(0.0)
    assert float(field.W_cond.abs().max()) == 0.0

    assert ensemble.load_state_dict({"field": snapshot}) == []
    assert float((field.W_cond - snapshot["W_cond"]).abs().max()) < 1e-6


def test_strict_mode_raises_only_for_the_documented_case(ensemble) -> None:
    _attach(ensemble)
    state = _snapshot(ensemble)
    assert "field" in state

    ensemble._field = None
    # 非严格：没场时不再静默吞掉（见 ensemble.load_state_dict 的补强分支）
    assert ensemble.load_state_dict(state, strict=False) == ["field (当前 ensemble 无场)"]
    with pytest.raises(KeyError):
        ensemble.load_state_dict(state, strict=True)


def test_missing_field_is_reported_even_when_non_strict(ensemble) -> None:
    """契约补强：docstring 说"缺失的键跳过并返回跳过清单"，静默丢掉 field 状态不算报告。"""

    _attach(ensemble)
    state = _snapshot(ensemble)
    ensemble._field = None
    assert ensemble.load_state_dict(state) == ["field (当前 ensemble 无场)"]
    # 文件里本来就没有 field 时，不该虚构跳过项
    no_field = {k: v for k, v in state.items() if k != "field"}
    assert ensemble.load_state_dict(no_field) == []
