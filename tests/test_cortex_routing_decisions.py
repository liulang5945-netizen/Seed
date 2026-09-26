"""B-4 第九刀：Cortex 的三条路由判定（executive 回合判定 / auto_topK 稀疏激活 / 实例级演化）。

审计把"上帝对象把路由揉进推理"列为 M12；这三条是路由决策的正身，此前只有
`_fingerprint_route` 有黄金件。本件用间谍喂合成分数，断言**决策规则本身**
（切换阈值、迟滞、保护下限、回退安全），不依赖真实权重。

* `_executive_route`：启发式打底；judge NLL 只在"另一个域显著更低（差 ≥1.0）"时才夺走判定；
  quality z-score 回退路径要满足 1.5× 基准 **且** 绝对差 ≥0.7；EMA 未成熟（warmup=20）时
  quality 一律不主导（回退安全）；`quality_logits` 长度与神经元数不符 ⇒ 放弃 quality；
  置信度只有 0.7 与 0.7+0.3·w 两档（有没有域分数决定）。
* `_auto_topk_route`：四条早退各自回到"全激活"。
* `_instance_route_evolve`：后验为空/无交集/域内无成员 ⇒ 原样返回；剔除要连续
  `evict_streak` 次才生效（迟滞）；leader 的 streak 归零；`min_active` 是硬保护；
  输出顺序＝原顺序＋同域新增追加＋域外（general）保留。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()


@pytest.fixture()
def cortex(tmp_path):
    import os

    built, _hub, general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    built.set_general_tokenizer(general_sp)
    return built


def _probe(cortex, monkeypatch, *, judge=None, quality=None):
    """把 cortex.think 换成固定探针结果，并装上最小 embedding 表。"""

    payload = {}
    if judge is not None:
        payload["round1_judge_logits"] = judge
    if quality is not None:
        payload["quality_logits"] = quality

    monkeypatch.setattr(cortex, "think", lambda **kw: dict(payload))
    monkeypatch.setattr(
        cortex,
        "_neuron_shared_embeddings",
        {nid: (lambda ids: ids.unsqueeze(-1).float()) for nid in cortex.neurons},
    )
    return payload


def test_executive_route_without_probe_falls_back_to_heuristic(cortex, monkeypatch) -> None:
    monkeypatch.setattr(cortex, "_neuron_shared_embeddings", {})
    domain, conf, scores = cortex._executive_route("def foo():\n    return 1")
    assert domain == "general", "fallback 夹具只有 general 神经元 ⇒ 启发式不可能给出 code"
    assert scores == {} and conf == pytest.approx(0.7), "没有域分数时置信度必须是底档"


def test_executive_route_judge_nll_switch_and_hold(cortex, monkeypatch) -> None:
    nids = list(cortex.neurons)  # fallback 只有 ['general']
    text = "hello there"
    ids = cortex._general_sp.encode(text)
    assert len(ids) >= 2, "探针文本至少两个 token，否则 target 为空、NLL 分支不会走"
    tgt = ids[1]
    vocab = max(ids) + 2

    def logits_hitting(target_id: int) -> torch.Tensor:
        # 代码里 logits[:, :-1] 预测 ids[1:] ⇒ 长度必须是 len(ids)，不是 len(ids)-1
        lg = torch.full((1, len(ids), vocab), -20.0)
        lg[0, :, target_id] = 20.0
        return lg

    base_domain, _, _ = cortex._executive_route(text, 0.4)

    # judge 与启发式同域 ⇒ 不切换，但 per_domain_scores 变成 -NLL 诊断量
    _probe(cortex, monkeypatch, judge={n: logits_hitting(tgt) for n in nids})
    domain, conf, scores = cortex._executive_route(text)
    assert domain == base_domain
    assert set(scores) == {n.split("_")[0] for n in nids}
    assert conf == pytest.approx(0.7 + 0.3 * 0.4), "有域分数 ⇒ 置信度升到上档"

    # 两域场景：zh 的 NLL 显著更低（≥1.0）⇒ 判定切到 zh；差不到 1.0 ⇒ 保留启发式
    original = dict(cortex.neurons)
    cortex.neurons["zh_1"] = original["general"]
    try:
        worse = logits_hitting(tgt - 1 if tgt > 1 else tgt + 1)
        worse[0, :, tgt] = -40.0  # 明确错配 ⇒ NLL 远高于 zh
        _probe(cortex, monkeypatch, judge={"general": worse, "zh_1": logits_hitting(tgt)})
        domain_sw, _, scores_sw = cortex._executive_route(text)
        assert domain_sw == "zh", f"judge NLL 差 ≥1.0 必须夺走判定，实得 {domain_sw}"
        assert set(scores_sw) == {"general", "zh"}

        near = logits_hitting(tgt)
        near[0, :, tgt] += 0.2  # 只差一点点 ⇒ 不显著，保留启发式
        _probe(cortex, monkeypatch, judge={"general": near, "zh_1": logits_hitting(tgt)})
        domain_hold, _, _ = cortex._executive_route(text)
        assert domain_hold == base_domain, "差不足 1.0 时不得切换（回退安全）"
    finally:
        cortex.neurons.pop("zh_1", None)


def test_executive_route_quality_only_path_requires_maturity(cortex, monkeypatch) -> None:
    nids = list(cortex.neurons)
    values = [torch.tensor([1.0]) for _ in nids]
    _probe(cortex, monkeypatch, judge=None, quality=values)

    # warmup=20：前 19 次一律"未成熟 ⇒ quality 不主导"
    for _ in range(19):
        _domain, _conf, scores = cortex._executive_route("hello there")
        assert scores == {}, "未成熟阶段不该产出域分数"
    assert cortex._quality_logit_ema[nids[0]]["count"] == 19

    _domain, conf, scores = cortex._executive_route("hello there")
    assert cortex._quality_logit_ema[nids[0]]["count"] == 20
    assert scores != {}, "成熟后 quality z-score 应可参与"
    assert conf == pytest.approx(0.7 + 0.3 * 0.4)


def test_executive_route_discards_quality_on_length_mismatch(cortex, monkeypatch) -> None:
    original = dict(cortex.neurons)
    cortex.neurons["zh_1"] = original["general"]  # 在册两枚，只喂一条 quality
    try:
        _probe(cortex, monkeypatch, judge=None, quality=[torch.tensor([1.0])])
        for _ in range(25):
            _domain, _conf, scores = cortex._executive_route("hello there")
        assert scores == {}, "quality_logits 长度与神经元数不符必须整段放弃 quality"
    finally:
        cortex.neurons.pop("zh_1", None)


def test_auto_topk_route_early_returns(cortex, monkeypatch) -> None:
    nids = list(cortex.neurons)
    assert cortex._auto_topk_route([1, 2, 3], top_k=0) == nids
    assert cortex._auto_topk_route([1, 2, 3], top_k=len(nids) + 5) == nids

    monkeypatch.setattr(cortex, "_shared_embedding", None)
    assert cortex._auto_topk_route([1, 2, 3], top_k=1) == nids, "无共享嵌入时回到全激活"

    def boom(_ids):
        raise RuntimeError("embedding 崩")

    class _Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("embedding 崩")

    monkeypatch.setattr(cortex, "_shared_embedding", _Boom())
    assert cortex._auto_topk_route([1, 2, 3], top_k=1) == nids, "异常必须静默回退全激活而不是抛出"


def _result_with(scores: dict, logits_len: int = 6) -> dict:
    return {"round1_scores": dict(scores)}


def test_instance_route_evolve_early_returns(cortex, monkeypatch) -> None:
    monkeypatch.setattr(cortex, "_rolling_nll_quality", lambda *a, **k: {})
    assert cortex._instance_route_evolve(
        ["general"],
        _result_with({"general": 1.0}),
        "文",
        "zh",
        {},
        4,
        0.5,
        2,
        0.0,
        1,
        [1, 2],
        "soft",
    ) == (["general"], {})

    monkeypatch.setattr(cortex, "_rolling_nll_quality", lambda *a, **k: {"zh_1": -2.0})
    # 后验里没有激活集成员 ⇒ 无交集，原样返回
    out, streaks = cortex._instance_route_evolve(
        ["general"],
        _result_with({"general": 1.0}),
        "文",
        "zh",
        {},
        4,
        0.5,
        2,
        0.0,
        1,
        [1, 2],
        "soft",
    )
    assert out == ["general"] and streaks == {}


def test_instance_route_evolve_hysteresis_min_active_and_ordering(cortex, monkeypatch) -> None:
    """逐条规则单独可预期：融合本身已被 `_cortex_helpers` 的守卫覆盖，这里显式桩掉，
    只测演化的五条规则（迟滞剔除 / leader 归零 / 加入 / min_active / 输出顺序）。
    """

    nids = ["general", "zh_1", "zh_2", "zh_3"]
    original = dict(cortex.neurons)
    for nid in nids[1:]:
        cortex.neurons[nid] = original["general"]
    try:
        monkeypatch.setattr(
            cortex,
            "_rolling_nll_quality",
            lambda *a, **k: {"zh_1": -9.0, "zh_2": -9.1, "zh_3": -1.0},
        )
        monkeypatch.setattr(cortex, "_probe_inactive_fused", lambda *a, **k: {})
        fused = {"general": 0.0, "zh_1": 0.1, "zh_2": 0.05, "zh_3": 1.0}
        monkeypatch.setattr(cortex, "_fuse_leader_quality", lambda r, q, alpha=0.5: dict(fused))
        result = _result_with({"general": 0.5, "zh_1": 0.4, "zh_2": 0.4, "zh_3": 0.4})

        def evolve(active, streaks, min_active=1, evict_streak=2):
            return cortex._instance_route_evolve(
                active,
                result,
                "文",
                "zh",
                streaks,
                4,
                0.5,
                evict_streak,
                1.5,
                min_active,
                [1, 2],
                "soft",
            )

        first, streaks = evolve(["general", "zh_1", "zh_2"], {})
        assert "zh_1" in first and "zh_2" in first, f"首次劣化不该立刻剔除（迟滞）：{first}"
        assert streaks["zh_1"] == 1 and streaks["zh_2"] == 1
        assert "zh_3" in first, f"后验远高于激活集最小值 ⇒ 必须加入：{first}"

        second, streaks = evolve(first, streaks)
        assert "zh_2" not in second and "zh_1" not in second, f"连续第 2 次劣化必须剔除：{second}"
        assert second == ["general", "zh_3"], f"顺序应为原顺序＋同域新增：{second}"
        assert streaks["zh_3"] == 0, "leader 的 streak 必须归零"

        # min_active=3：同域下限是硬保护，刚被剔的劣化成员会被补回最好的一枚
        third, _ = evolve(["general", "zh_1", "zh_2"], {"zh_1": 1, "zh_2": 1}, min_active=3)
        assert sum(1 for k in third if k.startswith("zh")) >= 3, f"min_active=3 未生效：{third}"

        # evict_streak=1：一次劣化即剔；但同域被清空后，加入步骤没有比较基准
        # （active_min=None）⇒ 全部同域成员被无条件补回。这条把该边界钉住，
        # 因为它是"min_active=1 时演化等于不演化"的真实原因。
        fourth, _ = evolve(["general", "zh_1", "zh_2"], {}, evict_streak=1)
        assert set(fourth) == {
            "general",
            "zh_1",
            "zh_2",
            "zh_3",
        }, f"同域清空后应无条件补回全部成员：{fourth}"
    finally:
        for nid in nids[1:]:
            cortex.neurons.pop(nid, None)


def test_instance_route_evolve_requires_domain_membership(cortex, monkeypatch) -> None:
    monkeypatch.setattr(cortex, "_rolling_nll_quality", lambda *a, **k: {"xx_1": -1.0})
    active, streaks = cortex._instance_route_evolve(
        ["general"],
        _result_with({"general": 1.0}),
        "文",
        "nope",
        {},
        4,
        0.5,
        2,
        0.0,
        1,
        [1],
        "soft",
    )
    assert active == ["general"] and streaks == {}, "域内没有任何成员时不得演化"
