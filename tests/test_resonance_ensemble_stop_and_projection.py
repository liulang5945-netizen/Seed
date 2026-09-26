"""B-4 第十一刀：ensemble 的提前停止判据、跨词表 logits 投影、神经元 tokenizer 解析。

这三处是"融合/联合训练到底在什么条件下停、跨域怎么对齐"的决策点，此前无直测：

* `_check_adaptive_stop`（C9）：两个停止信号（分数收敛 / 主导明确）各自独立成立，
  另有四条"不许停"的门（开关关闭、round < min_rounds、round >= max_rounds、样本太少），
  以及 top2≈0 时不做除法这条防除零的口子；
* `_project_logits_to_target`（缺口 M）：三条前置校验都必须响亮报错（不是静默回退），
  同词表必须原样透传，跨词表必须产出 [N, B, L, V_tgt]；解析不出源 tokenizer 时报错；
* `_get_neuron_tokenizer`：hub 缺席 ⇒ None；先按完整 nid 取，取不到再按域前缀回退。

夹具：生产构造路径的 ensemble ＋ 自装 TokenizerHub（general 256K / zh 50K 真模型）。
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


@pytest.fixture(scope="module")
def hub_and_sp():
    _cortex, hub, general_sp, zh_sp = capture.build_cortex()
    return hub, general_sp, zh_sp


@pytest.fixture()
def ensemble(tmp_path, hub_and_sp):
    import os

    built, _hub, _gsp, _zsp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    built.ensemble.set_tokenizer_hub(hub_and_sp[0])
    return built.ensemble


# ── _check_adaptive_stop ──────────────────────────────────────────────────────


def test_adaptive_stop_gate_and_round_bounds(ensemble) -> None:
    ensemble.adaptive_stop = False
    assert ensemble._check_adaptive_stop({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, 5) == (
        False,
        "",
    ), "开关关闭时必须完全不参与"

    ensemble.adaptive_stop = True
    ensemble.min_rounds = 3
    ensemble.max_rounds = 8
    converged = ({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    # min_rounds 以下不停（保证 side_signals 至少生效一次）
    assert ensemble._check_adaptive_stop(*converged, 2) == (False, "")
    # max_rounds 及以上由循环自然结束，不该报"提前停止"
    assert ensemble._check_adaptive_stop(*converged, 8) == (False, "")
    assert ensemble._check_adaptive_stop(*converged, 3)[0] is True


def test_adaptive_stop_convergence_signal(ensemble) -> None:
    ensemble.adaptive_stop = True
    ensemble.min_rounds = 2
    ensemble.max_rounds = 16
    ensemble.convergence_threshold = 0.01

    prev = {"n1": 1.0, "n2": 0.8}
    cur = {"n1": 1.004, "n2": 0.802}
    stop, reason = ensemble._check_adaptive_stop(cur, prev, 4)
    assert stop is True and reason.startswith("converged")

    moving = {"n1": 1.4, "n2": 0.2}
    stop2, reason2 = ensemble._check_adaptive_stop(moving, prev, 4)
    assert stop2 is False or reason2.startswith("dominant"), (stop2, reason2)

    # 交集不足两个键时不做收敛判定（单键均值没有意义）
    assert ensemble._check_adaptive_stop({"n3": 0.5}, {"n3": 0.5}, 4) == (False, "")
    # 没有上一轮 ⇒ 只可能靠主导信号停
    assert ensemble._check_adaptive_stop({"n1": 0.5}, None, 4) == (False, "")


def test_adaptive_stop_dominance_signal_and_zero_guard(ensemble) -> None:
    ensemble.adaptive_stop = True
    ensemble.min_rounds = 2
    ensemble.max_rounds = 16
    ensemble.convergence_threshold = 1e-9  # 关掉收敛干扰
    ensemble.dominance_ratio = 2.0

    stop, reason = ensemble._check_adaptive_stop({"n1": 0.9, "n2": 0.3}, {"n1": 0.9, "n2": 0.31}, 3)
    assert stop is True and reason.startswith("dominant"), (stop, reason)

    close = ensemble._check_adaptive_stop({"n1": 0.5, "n2": 0.45}, {"n1": 0.1, "n2": 0.1}, 3)
    assert close == (False, ""), f"top1/top2 未过阈值不该停：{close}"

    # top2≈0：跳过比值判定（否则除零 ⇒ inf ⇒ 恒"主导"）
    # 注意 prev 要与 cur 有足够差值，否则这条会先被收敛信号判停
    assert ensemble._check_adaptive_stop({"n1": 0.9, "n2": 0.0}, {"n1": 0.4, "n2": 0.0}, 3) == (
        False,
        "",
    )


# ── _get_neuron_tokenizer ─────────────────────────────────────────────────────


def test_neuron_tokenizer_resolution_falls_back_to_domain(ensemble, hub_and_sp) -> None:
    hub, general_sp, zh_sp = hub_and_sp
    # 每个夹具各自 build_cortex() ⇒ sp 是**不同对象**，只能与该 ensemble 自己 hub 里的实例比
    own_hub = ensemble._tokenizer_hub
    assert ensemble._get_neuron_tokenizer("zh_1") is own_hub.get_tokenizer(
        "zh"
    ), "完整 nid 取不到时应按域前缀回退"
    assert ensemble._get_neuron_tokenizer("zh_1") is not own_hub.get_tokenizer("general")

    ensemble._tokenizer_hub = None
    assert ensemble._get_neuron_tokenizer("zh_1") is None, "没有 hub 时返回 None 而不是抛"


# ── _project_logits_to_target ─────────────────────────────────────────────────


def test_projection_requires_hub_and_valid_target(ensemble, hub_and_sp) -> None:
    hub, general_sp, zh_sp = hub_and_sp
    logits = {"zh_1": torch.randn(1, 2, int(zh_sp.GetPieceSize()))}

    saved = ensemble._tokenizer_hub
    ensemble._tokenizer_hub = None
    try:
        with pytest.raises(RuntimeError, match="tokenizer hub"):
            ensemble._project_logits_to_target(logits, ["zh_1"], "zh")
    finally:
        ensemble._tokenizer_hub = saved

    # 目标域不在 hub ⇒ 响亮报错。⚠️ 但这条守卫在**注册了 general 的 hub 上不可达**：
    # TokenizerHub.get_tokenizer 对未知域回退到 general（translator.py:113-115），
    # 所以 bogus domain 会静默按 general 词表投影。用只有 zh、没有 general 的 hub 才测得到。
    from neuroplex.resonance.translator import TokenizerHub

    zh_only = TokenizerHub()
    zh_only.register_domain("zh", zh_sp)
    saved = ensemble._tokenizer_hub
    ensemble._tokenizer_hub = zh_only
    try:
        with pytest.raises(RuntimeError, match="不在 tokenizer hub"):
            ensemble._project_logits_to_target(logits, ["zh_1"], "not_a_domain")
        # 同一 hub 下请求 general 也取不到 ⇒ 同样报错（而不是拿 zh 顶上）
        with pytest.raises(RuntimeError, match="不在 tokenizer hub"):
            ensemble._project_logits_to_target(logits, ["zh_1"], "general")
    finally:
        ensemble._tokenizer_hub = saved

    # 在册但词表不可用（无 GetPieceSize）⇒ 也必须报错，不能拿 0 当 vocab 算下去
    class _NoVocab:
        def encode(self, _text):
            return [1, 2]

    broken = TokenizerHub(general_tokenizer=_NoVocab())
    ensemble._tokenizer_hub = broken
    try:
        with pytest.raises(RuntimeError, match="无有效 vocab"):
            ensemble._project_logits_to_target(logits, ["zh_1"], "general")
    finally:
        ensemble._tokenizer_hub = saved


def test_projection_passes_through_same_vocab_and_projects_across_vocab(
    ensemble, hub_and_sp
) -> None:
    hub, general_sp, zh_sp = hub_and_sp
    zh_vocab = int(zh_sp.GetPieceSize())
    gen_vocab = int(general_sp.GetPieceSize())

    # 同词表 ⇒ 原样透传（不得被矩阵乘法悄悄改写）
    logits = {"zh_1": torch.randn(1, 2, zh_vocab)}
    out = ensemble._project_logits_to_target(logits, ["zh_1"], "zh")
    assert out.shape == (1, 1, 2, zh_vocab), out.shape
    assert torch.equal(out[0], logits["zh_1"])

    # 跨词表 zh(50K)→general(256K)：产出 [N, B, L, V_tgt]，且词表维确实换成目标域
    cross = {"zh_1": torch.softmax(torch.randn(1, 1, zh_vocab), dim=-1)}
    projected = ensemble._project_logits_to_target(cross, ["zh_1"], "general")
    assert projected.shape == (1, 1, 1, gen_vocab), projected.shape
    assert not torch.any(torch.isnan(projected)), "投影结果出现 NaN ⇒ 对齐矩阵或 dtype 有问题"
    # 概率分布经稀疏对齐后仍应在 [0,1] 量级（矩阵元素是 0/1 映射，不该放大）
    assert float(projected.max()) <= 1.0 + 1e-5, float(projected.max())

    # 解析不出源 tokenizer 的神经元必须响亮报错，而不是静默跳过
    ensemble._tokenizer_hub = None
    with pytest.raises(RuntimeError):
        ensemble._project_logits_to_target(cross, ["general"], "zh")


def test_phase_binding_scale_tracks_oscillator(ensemble) -> None:
    ensemble.gamma_oscillator = None
    assert ensemble._phase_binding_scale() == 0.0, "无振荡器时绑定强度必须是 0（关闭）"

    class _Osc:
        binding_scale = 0.42

    ensemble.gamma_oscillator = _Osc()
    assert ensemble._phase_binding_scale() == pytest.approx(0.42)

    class _NoAttr:
        pass

    ensemble.gamma_oscillator = _NoAttr()
    assert ensemble._phase_binding_scale() == 0.0
