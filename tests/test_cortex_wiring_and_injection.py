"""B-4 第七刀：Cortex 的注入与装配面（think 记忆归一化 / gamma / 工作记忆 / 对话态 / 场读出）。

覆盖 cortex.py 里"薄但会静默错"的一段：这些函数决定**记忆能不能进生成、调质/振荡器
挂没挂上、场快照取的是哪一份**。全部用间谍（spy）替身接住 ensemble 的 kwargs，
断言的是透传协议，不依赖真实前向。

think() 的 memory_vectors 归一化（1131-1158）是 C-4 §7 清单里"最干净的单点"，
本件先把它的契约钉住：2 元组 / 3 元组 / dict / 裸向量 / vector=None 各有确定形态。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from neuroplex.brain.cortex import Cortex
from neuroplex.resonance.gamma_oscillator import GammaOscillator

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
    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    return built


class _Spy:
    """接住 ensemble.forward / continuous_forward 的 kwargs，返回一个最小 result。"""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"field_state": torch.zeros(4096), "final_scores": {}, "round1_scores": {}}

    @property
    def kwargs(self) -> dict:
        assert self.calls, "一次前向都没发生 ⇒ 后面的断言全是空的"
        return self.calls[-1]

    def only(self) -> dict:
        assert len(self.calls) == 1, f"应当恰好一次前向，实得 {len(self.calls)}"
        return self.calls[-1]


@pytest.fixture()
def spy(cortex, monkeypatch):
    forward = _Spy()
    continuous = _Spy()
    monkeypatch.setattr(cortex.ensemble, "forward", forward)
    monkeypatch.setattr(cortex.ensemble, "continuous_forward", continuous)
    forward.continuous = continuous
    return forward


def _vec(scale: float = 1.0) -> torch.Tensor:
    return torch.full((4096,), scale)


# ── think(): 记忆注入归一化 ──────────────────────────────────────────────────


def test_think_normalizes_every_memory_shapes(cortex, spy) -> None:
    v1, v2, v3, v4 = _vec(1.0), _vec(2.0), _vec(3.0), _vec(4.0)
    cortex.think(
        shared_embeddings=torch.zeros(1, 2, 8),
        memory_vectors=[
            (v1, 0.5),  # 2 元组
            (v2, 0.7, 1.25),  # 3 元组带 phase
            {"vector": v3, "weight": 0.9},  # dict 无 phase
            {"vector": v4, "weight": 0.3, "phase": 2.5},  # dict 带 phase
            v1.clone(),  # 裸向量 ⇒ weight 默认 1.0
            (None, 1.0),  # 向量为 None ⇒ 整条丢弃
        ],
    )
    seeds = spy.kwargs["seed_memories"]
    assert len(seeds) == 5, f"6 条输入里只有 vector=None 该被丢，实得 {seeds}"
    # 带 phase 的保留三元组，不带的压成二元组（"无 phase 回退峰值对齐"的编码约定）
    assert len(seeds[0]) == 2 and seeds[0][1] == pytest.approx(0.5)
    assert len(seeds[1]) == 3 and seeds[1][2] == pytest.approx(1.25)
    assert len(seeds[2]) == 2 and seeds[2][1] == pytest.approx(0.9)
    assert len(seeds[3]) == 3 and seeds[3][2] == pytest.approx(2.5)
    assert len(seeds[4]) == 2 and seeds[4][1] == pytest.approx(1.0)
    # 权重必须是 float，不能把 numpy/str 之类原样透传给 ensemble
    assert all(isinstance(item[1], float) for item in seeds)


def test_think_without_memories_does_not_invent_the_key(cortex, spy) -> None:
    cortex.think(shared_embeddings=torch.zeros(1, 2, 8))
    assert "seed_memories" not in spy.kwargs
    cortex.think(shared_embeddings=torch.zeros(1, 2, 8), memory_vectors=[])
    assert "seed_memories" not in spy.kwargs, "空列表不该写一个空 seed_memories 进前向"
    cortex.think(shared_embeddings=torch.zeros(1, 2, 8), memory_vectors=[(None, 1.0)])
    assert "seed_memories" not in spy.kwargs, "全被丢弃时同样不该留键"


def test_think_embedding_priority_and_device_transfer(cortex, spy) -> None:
    per_neuron = {nid: torch.zeros(1, 3, 8) for nid in cortex.neurons}
    cortex.think(shared_embeddings=torch.zeros(1, 9, 8), neuron_embeddings=per_neuron)
    kwargs = spy.kwargs
    assert "shared_embeddings" not in kwargs, "neuron_embeddings 优先，二者不该同时透传"
    assert set(kwargs["neuron_embeddings"]) == set(per_neuron)
    # ⚠️ Cortex.device 是**字符串**（'cpu'），不是 torch.device ⇒ torch 接受它，但比较要先归一
    target_device = torch.device(str(cortex.device))
    assert all(t.device == target_device for t in kwargs["neuron_embeddings"].values())

    cortex.think(shared_embeddings=torch.zeros(1, 9, 8))
    assert spy.continuous.calls == []
    assert spy.kwargs["shared_embeddings"].shape == (1, 9, 8)


def test_think_dispatch_and_flags(cortex, spy) -> None:
    cortex.think(shared_embeddings=torch.zeros(1, 2, 8), collab_mode="continuous")
    assert (
        len(spy.continuous.calls) == 1 and spy.calls == []
    ), "continuous 必须走 continuous_forward"
    assert spy.continuous.calls[0]["fusion_mode"] == "soft"

    cortex.think(
        shared_embeddings=torch.zeros(1, 2, 8), active_nids=["general"], fusion_mode="residual"
    )
    assert spy.kwargs["active_nids"] == ["general"]
    assert spy.kwargs["fusion_mode"] == "residual"
    assert spy.kwargs["return_logits"] is True, "think 恒要求 logits"

    cortex.think(shared_embeddings=torch.zeros(1, 2, 8), return_judge_logits=True)
    assert spy.kwargs.get("return_judge_logits") is True


# ── 振荡器 / 调质 / 记忆 / 对话态挂载 ────────────────────────────────────────


def test_gamma_oscillator_registration_respects_existing_phases(cortex) -> None:
    osc = GammaOscillator()
    cortex.set_gamma_oscillator(osc)
    assert cortex.gamma_oscillator is osc
    assert cortex.ensemble.gamma_oscillator is osc
    assert cortex.field._gamma_oscillator is osc
    # fallback 只有一个神经元 ⇒ 先验分配后 phase 数必须覆盖它
    assert len(osc.phases) == len(cortex.neurons)

    # C23-C5：已带训练相位的振荡器不得被重新分配覆盖
    trained = GammaOscillator()
    trained.assign_phase_by_domain({"zh": ["zh_1"], "en": ["en_1"]})
    before = dict(trained.phases)
    cortex.set_gamma_oscillator(trained)
    assert {k: float(v) for k, v in trained.phases.items()} == pytest.approx(
        {k: float(v) for k, v in before.items()}
    ), "已有相位被 assign 覆盖 ⇒ 训练学到的自组织丢失"

    ticks = {"n": 0}
    counting = GammaOscillator()
    original_tick = counting.tick

    def _tick(*args, **kwargs):
        ticks["n"] += 1
        return original_tick(*args, **kwargs)

    counting.tick = _tick  # type: ignore[method-assign]
    cortex.set_gamma_oscillator(counting)
    cortex.tick_gamma()
    assert ticks["n"] == 1


def test_working_memory_and_dialogue_state_registration(cortex, tmp_path) -> None:
    from neuroplex.brain.working_memory import WorkingMemory

    memory = WorkingMemory(max_tokens=8)
    cortex.set_working_memory(memory)
    assert cortex.working_memory is memory

    memory.append_round([1, 2, 3], [4])
    assert len(memory) > 0
    cortex.clear_working_memory()
    assert len(memory) == 0, "clear 必须真的重置已注册的实例"

    # 未注册时 clear 不该炸（会话初始化路径无条件调用）；裸实例连属性都没有，靠 hasattr 兜底
    fresh = Cortex.__new__(Cortex)
    Cortex.clear_working_memory(fresh)
    cortex.set_dialogue_state(None)
    assert cortex._dialogue_state is None
    cortex.clear_dialogue_state()


def test_anchor_projector_is_opt_in_and_shape_preserving(cortex) -> None:
    state = torch.arange(8.0)
    assert cortex.project_field_state(state) is state, "未挂载投影器必须原样返回（零影响）"
    assert cortex.project_field_state(None) is None

    calls: list[tuple] = []

    def projector(x):
        calls.append(tuple(x.shape))
        return x * 2.0

    cortex.set_anchor_projector(projector)
    one_d = cortex.project_field_state(state)
    assert one_d.shape == (8,) and torch.equal(one_d, state * 2.0)
    assert calls == [(1, 8)], "投影内部按 [1,D] 走，读出侧再压回 [D]"

    two_d = cortex.project_field_state(state.reshape(1, 8))
    assert two_d.shape == (1, 8)


def test_field_readout_helpers(cortex, monkeypatch) -> None:
    assert cortex.get_field_state().shape[-1] == cortex.field.dim
    assert cortex.get_last_phase() is None, "没共振过就没有相位归属"
    monkeypatch.setattr(cortex, "_last_phase_mean", 0.75, raising=False)
    assert cortex.get_last_phase() == pytest.approx(0.75)

    # 默认场为零 ⇒ 任务场快照必须判为"没有"，而不是返回一坨零当记忆
    assert cortex.get_last_field_state() is None

    live = cortex.ensemble._get_task_field()
    with torch.no_grad():
        live.state.add_(1.0)
    assert cortex.get_last_field_state() is not None

    cortex.field.scores = {"zh_1": 0.2, "code_1": 0.9, "en_1": 0.4}
    assert cortex.get_dominant_domain() == "code_1"
    cortex.field.scores = {}
    assert cortex.get_dominant_domain() is None
