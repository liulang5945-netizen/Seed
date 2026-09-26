"""B-4 第十二刀：ensemble 的 side-channel 装配与利用率统计、几何初始化。

这三处是"神经元之间怎么连线、连线有没有被用"的记账处，此前只在整段 forward 里路过：

* `_build_side_signals`：只为**真实存在**的 excite/inhibit 通道产出信号；自己连自己不产；
  没有 router mask 时透传原向量；有 mask 时按样本逐行乘上去（top-K 之外的 pre 在该样本被置零），
  且 mask 的列序必须按 `router_active_ids.index(pre_id)` 取（顺序错配会静默张冠李戴）；
* `_update_channel_usage`：None 入参是 no-op；并发增删导致 post 已消失时**跳过而不是 KeyError**；
  未注册的通道不记账；EMA 首见记原值、再见按 alpha 混合；
* `_init_geometry`：按域分组建坐标，并把 geometry 注册给 coaction（没有该接口时不得硬调）。
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
    import os

    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    ens = built.ensemble
    # 两枚同维神经元，便于造通道
    anchor = next(iter(ens.neurons.values()))
    ens.neurons["n1"] = anchor
    ens.neurons["n2"] = anchor.__class__(anchor.config)
    ens.neurons["n3"] = anchor.__class__(anchor.config)
    for neuron in ens.neurons.values():
        neuron.eval()
    return ens


def _wire(neuron, kind: str, pre_id: str, dim: int) -> None:
    getattr(neuron, kind)[pre_id] = nn.Linear(dim, dim, bias=False)


def test_build_side_signals_only_for_existing_channels(ensemble) -> None:
    dim = int(next(iter(ensemble.neurons.values())).config.field_dim)
    _wire(ensemble.neurons["n2"], "excite_channels", "n1", dim)
    _wire(ensemble.neurons["n2"], "inhibit_channels", "n3", dim)

    vecs = {nid: torch.full((2, dim), float(i + 1)) for i, nid in enumerate(["n1", "n2", "n3"])}
    ensemble._router_hard_mask = None
    out = ensemble._build_side_signals(["n1", "n2", "n3"], ensemble.neurons, vecs, None)

    assert set(out) == {"n1", "n2", "n3"}, "每个激活神经元都要有条目（哪怕为空）"
    assert set(out["n2"]) == {"n1", "n3"}, "只该有真实存在的通道"
    assert out["n1"] == {}, "n1 没有任何入向通道"
    assert torch.equal(out["n2"]["n1"], vecs["n1"]), "无 mask 时必须原样透传"
    assert all("n2" not in out[n2] for n2 in out), "自己连自己不得产信号"


def test_build_side_signals_applies_router_mask_per_sample(ensemble) -> None:
    dim = int(next(iter(ensemble.neurons.values())).config.field_dim)
    _wire(ensemble.neurons["n2"], "excite_channels", "n1", dim)
    vecs = {"n1": torch.ones(2, dim), "n2": torch.ones(2, dim) * 2}
    ids = ["n1", "n2"]
    ensemble._router_hard_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 样本0 留 n1，样本1 留 n2

    out = ensemble._build_side_signals(ids, ensemble.neurons, vecs, ids)
    sig = out["n2"]["n1"]
    assert torch.equal(sig[0], vecs["n1"][0]), "该样本内 top-K 的 pre 必须原样通过"
    assert torch.equal(sig[1], torch.zeros(dim)), "非 top-K 的 pre 在该样本必须被置零"
    assert ensemble._router_hard_mask is not None


def test_router_mask_column_order_follows_router_ids(ensemble) -> None:
    """mask 列序按 router_active_ids 解析：换序后同一神经元取到的列必须跟着变。"""

    dim = int(next(iter(ensemble.neurons.values())).config.field_dim)
    _wire(ensemble.neurons["n2"], "excite_channels", "n1", dim)
    vecs = {"n1": torch.ones(1, dim), "n2": torch.ones(1, dim)}
    ensemble._router_hard_mask = torch.tensor([[1.0, 0.0]])  # 列序 [n2, n1] ⇒ n1 这列是 0

    out = ensemble._build_side_signals(["n1", "n2"], ensemble.neurons, vecs, ["n2", "n1"])
    assert torch.equal(
        out["n2"]["n1"], torch.zeros(1, dim)
    ), "按 router 列序 n1 那一列是 0 ⇒ 必须置零"

    # 同样数值、列序改成 [n1, n2] ⇒ n1 取到第 0 列 = 1 ⇒ 必须原样通过
    ensemble._router_hard_mask = torch.tensor([[0.0, 1.0]])
    out2 = ensemble._build_side_signals(["n1", "n2"], ensemble.neurons, vecs, ["n1", "n2"])
    assert torch.equal(
        out2["n2"]["n1"], torch.zeros(1, dim)
    ), "列序 [n1,n2] 时第 0 列属于 n1 ⇒ 应取到 0"


def test_channel_usage_ema_and_missing_targets(ensemble) -> None:
    dim = int(next(iter(ensemble.neurons.values())).config.field_dim)
    _wire(ensemble.neurons["n2"], "excite_channels", "n1", dim)
    sig = torch.ones(1, dim)

    ensemble._update_channel_usage(None, {})  # no-op，不抛

    ensemble._channel_usage.clear()
    ensemble._update_channel_usage({"n2": {"n1": sig, "ghost": sig}}, {})
    key = "n2->n1"
    assert key in ensemble._channel_usage, "已注册通道必须记到利用率"
    first = ensemble._channel_usage[key]
    assert first > 0
    assert "n2->ghost" not in ensemble._channel_usage, "未注册的通道不得凭空记账"

    ensemble._update_channel_usage({"n2": {"n1": sig * 0}}, {})
    assert ensemble._channel_usage[key] != first, "EMA 必须随新读数移动"
    assert ensemble._channel_usage[key] < first, "信号归零后利用率应下降"

    # 并发增删：post 已从注册表消失 ⇒ 跳过而不是 KeyError
    ensemble._update_channel_usage({"vanished_neuron": {"n1": sig}}, {})
    assert "vanished_neuron->n1" not in ensemble._channel_usage


class _Geo:
    """`_init_geometry` **使用**已存在的 geometry（不负责创建），故夹具要给一枚替身。"""

    embedding_dim = 8

    def __init__(self) -> None:
        self.positions: dict[str, torch.Tensor] = {}
        self.calls: list[tuple] = []

    def assign_domain_positions(self, mapping, intra_domain_radius=0.0, inter_domain_radius=0.0):
        self.calls.append((mapping, intra_domain_radius, inter_domain_radius))
        for _domain, nids in mapping.items():
            for nid in nids:
                self.positions[nid] = torch.zeros(self.embedding_dim)


def test_init_geometry_groups_by_domain_and_registers_with_coaction(ensemble) -> None:
    geo = _Geo()
    ensemble.geometry = geo
    registered: list = []

    class _Coaction:
        def register_geometry(self, geometry):
            registered.append(geometry)

    ensemble.coaction = _Coaction()
    ensemble._init_geometry()

    assert len(geo.calls) == 1
    mapping, intra, inter = geo.calls[0]
    assert set().union(*[set(v) for v in mapping.values()]) == set(
        ensemble.neurons
    ), "每个激活神经元都要被分到某个域里"
    assert (intra, inter) == (0.2, 1.0), "域内/域间半径是 RSGN 距离先验的声明值"
    assert set(geo.positions) == set(ensemble.neurons)
    assert registered == [geo], "coaction 有 register_geometry 时必须注册"

    # coaction 缺该接口时只跳过注册，不该整体失败
    ensemble.geometry = _Geo()
    ensemble.coaction = object()
    ensemble._init_geometry()
