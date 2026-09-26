"""B-4 覆盖率第五刀：Cortex 的可学习状态持久化（save_state / load_state / _load_neurons）。

这块是"冻结基线能否复现"的正身（审计 §5-2 说上帝对象＋全局单例使行为不可复现），
cortex.py 里也是最大的一段可确定性单测面（约 190 语句），此前全仓无直测。

钉的契约（docstring 声明的，不是"调用没炸"）：
* 目录路径 ⇒ 落成 `cortex_state.pt`；version 字段随件走；
* **per-neuron lm_head 权重往返**：改权重→save→改成别的→load ⇒ 必须拿回 save 时的值；
* shared_embedding 以 fp16 存、按目标模块 dtype 恢复（压缩是设计声明，不是巧合）；
* 文件不存在 ⇒ load_state 返回 False（不能抛，也不能静默"成功"）；
* 文件里出现当前不存在的 neuron id ⇒ 跳过该条但整体仍算恢复成功；
* neuromodulator / coaction / sleep_consolidator 三者"在场才持久化，恢复时把同一份 payload
  交回给它们"——用可计数的鸭子替身验证 get/load 各被消费一次；
* `_load_neurons` 从 neurons_dir 扫 `neuron_*.pt` 动态装配（不是硬编码名单）。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from neuroplex.brain.cortex import Cortex

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()


class _CountingState:
    """记录 get/load 次数的最小状态件（验证"在场才持久化"的接线）。

    带 dopamine/serotonin/norepinephrine 三个浮点：`set_neuromodulator` 会读它们打日志。
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.get_calls = 0
        self.loaded_with: dict | None = None
        self.dopamine = float(payload.get("dopamine", 0.5))
        self.serotonin = float(payload.get("serotonin", 0.5))
        self.norepinephrine = float(payload.get("norepinephrine", 0.5))
        # load_state 的日志会读这些（coaction / sleep_consolidator 的规模），替身必须提供
        self._slow_matrix = payload.get("pairs", 0) * [[0.0]]
        self._activation_counts = dict(payload.get("counts", {}))
        self._replay_buffer = list(payload.get("replay", []))
        self._last_consolidation_step = int(payload.get("last_step", 0))

    def get_state_dict(self) -> dict:
        self.get_calls += 1
        return dict(self.payload)

    def load_state_dict(self, state: dict) -> None:
        self.loaded_with = dict(state)


@pytest.fixture()
def cortex(tmp_path):
    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    return built


def _lm_head_vector(cortex: Cortex) -> torch.Tensor:
    for neuron in cortex.neurons.values():
        lm = getattr(neuron, "lm_head", None)
        if lm is not None:
            for tensor in lm.state_dict().values():
                if tensor.is_floating_point() and tensor.numel() > 0:
                    return tensor.view(-1)
    pytest.skip("夹具里没有可学习 lm_head 权重，本条无从适用")


def test_save_state_dir_path_and_version(cortex: Cortex, tmp_path) -> None:
    target = tmp_path / "state_dir"
    cortex.save_state(str(target) + os.sep)
    written = target / "cortex_state.pt"
    assert written.exists(), "目录路径必须落成 cortex_state.pt"

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    state = load_legacy_checkpoint(str(written), map_location="cpu")
    assert state["version"] == 3
    assert "saved_at" in state and "neurons" in state


def test_lm_head_weights_round_trip(cortex: Cortex, tmp_path) -> None:
    vector = _lm_head_vector(cortex)
    with torch.no_grad():
        vector.add_(0.37)
    saved = vector.clone()

    path = tmp_path / "cortex.pt"
    cortex.save_state(str(path))

    with torch.no_grad():
        vector.zero_()
    assert float(vector.abs().max()) == 0.0, "先确认改回零真的生效，否则往返断言是空的"

    assert cortex.load_state(str(path)) is True
    assert float((vector - saved).abs().max()) < 1e-5, "load 必须把 lm_head 恢复到 save 时的值"


def test_shared_embedding_is_stored_fp16_and_restored_to_module_dtype(
    cortex: Cortex, tmp_path
) -> None:
    # 自己装一枚小的 shared_embedding：本条测的是持久化协议（fp16 存、按目标 dtype 恢复），
    # 与嵌入表大小无关，用 16×64 就够，且比夹具的 256K 词表快两个数量级。
    embedding = torch.nn.Embedding(16, 64)
    with torch.no_grad():
        embedding.weight.normal_(0.0, 0.2)
    cortex.set_shared_embedding(embedding)

    original = cortex._shared_embedding.weight.detach().clone()
    path = tmp_path / "cortex.pt"
    cortex.save_state(str(path))

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    state = load_legacy_checkpoint(str(path), map_location="cpu")
    assert state["shared_embedding_dtype"] == "fp16", "shared_embedding 声明按 fp16 压缩存储"
    assert state["shared_embedding"]["weight"].dtype == torch.float16

    with torch.no_grad():
        cortex._shared_embedding.weight.fill_(0.0)
    assert cortex.load_state(str(path)) is True
    restored = cortex._shared_embedding.weight.detach()
    assert restored.dtype == original.dtype, "恢复必须回到目标模块 dtype，而不是留 fp16"
    assert float((restored - original).abs().max()) < 5e-4, "fp16 往返误差必须在该量级内"
    assert float(restored.abs().max()) > 0.0, "恢复后不能还是清零后的样子"


def test_load_state_missing_file_returns_false(cortex: Cortex, tmp_path) -> None:
    assert cortex.load_state(str(tmp_path / "nope.pt")) is False
    assert (
        cortex.load_state(str(tmp_path / "no_such_dir")) is False
    ), "目录里没文件也要 False，不能抛"


def test_load_state_skips_unknown_neuron_ids_but_still_succeeds(cortex: Cortex, tmp_path) -> None:
    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    path = tmp_path / "cortex.pt"
    cortex.save_state(str(path))
    state = load_legacy_checkpoint(str(path), map_location="cpu")
    known = dict(state["neurons"])
    assert known, "夹具里至少应有一个带可学习参数的神经元，否则本条是空测"
    first = next(iter(known.values()))
    assert {"lm_head", "embed_adapter"} & set(
        first
    ), f"文件里的神经元条目该带可学习参数，实得 {list(first)}"
    fresh = torch.nn.Linear(4, 4).state_dict()
    state["neurons"] = {**known, "ghost_9": {"lm_head": fresh}}
    torch.save(state, str(path))

    assert cortex.load_state(str(path)) is True
    assert "ghost_9" not in cortex.neurons, "恢复不能凭文件凭空造神经元"
    vector = _lm_head_vector(cortex)
    assert torch.isfinite(vector).all()


def test_attached_state_components_are_persisted_and_handed_back(cortex: Cortex, tmp_path) -> None:
    neuro = _CountingState({"dopamine": 0.4, "serotonin": 0.5, "norepinephrine": 0.6})
    coact = _CountingState({"pairs": 3})
    cons = _CountingState({"replay": [11, 12], "last_step": 7})
    cortex.set_neuromodulator(neuro)
    cortex.coaction = coact
    cortex.set_sleep_consolidator(cons)

    path = tmp_path / "cortex.pt"
    cortex.save_state(str(path))
    assert neuro.get_calls == 1 and coact.get_calls == 1 and cons.get_calls == 1

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    stored = load_legacy_checkpoint(str(path), map_location="cpu")
    assert {"neuromodulator", "coaction", "sleep_consolidator"} <= set(stored)
    assert stored["neuromodulator"]["dopamine"] == pytest.approx(0.4)

    fresh_neuro, fresh_coact, fresh_cons = (
        _CountingState({}),
        _CountingState({}),
        _CountingState({}),
    )
    cortex.set_neuromodulator(fresh_neuro)
    cortex.coaction = fresh_coact
    cortex.set_sleep_consolidator(fresh_cons)
    assert cortex.load_state(str(path)) is True
    assert fresh_neuro.loaded_with == stored["neuromodulator"]
    assert fresh_coact.loaded_with == stored["coaction"]
    assert fresh_cons.loaded_with == stored["sleep_consolidator"]


def test_components_absent_at_save_time_are_not_invented_at_load(cortex: Cortex, tmp_path) -> None:
    # 反注册路径本身也要能走通（旧实现在这里 AttributeError，且前两行已经写进半状态）
    cortex.set_neuromodulator(_CountingState({"dopamine": 0.9}))
    cortex.set_neuromodulator(None)
    assert cortex._neuromodulator is None and cortex.ensemble.neuromodulator is None

    path = tmp_path / "cortex.pt"
    cortex.save_state(str(path))

    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    stored = load_legacy_checkpoint(str(path), map_location="cpu")
    assert "neuromodulator" not in stored, "未注册调质时不该写这个键"

    neuro = _CountingState({"dopamine": 0.1})
    cortex.set_neuromodulator(neuro)
    assert cortex.load_state(str(path)) is True
    assert neuro.loaded_with is None, "文件里没这一节就不该给新注册件灌随机状态"


def test_load_neurons_scans_the_directory_dynamically(cortex: Cortex, tmp_path) -> None:
    from neuroplex.legacy_checkpoint import load_legacy_checkpoint

    nid = cortex.add_neuron("zh")
    ckpt_dir = Path(cortex.neurons_dir)
    assert (ckpt_dir / f"neuron_{nid}.pt").exists()

    # 多塞一个无关命名的文件，装配必须忽略它（按 neuron_*.pt 扫描，不是扫全目录）
    (ckpt_dir / "not_a_neuron.pt").write_bytes(b"")
    found = sorted(p.name for p in ckpt_dir.glob("neuron_*.pt"))
    assert found == [f"neuron_{nid}.pt"]

    before = set(cortex.neurons)
    # 摘出运行中但**保留 ckpt**（remove_neuron 默认连 ckpt 一起删，那就没东西可装配了）
    assert cortex.remove_neuron(nid, delete_ckpt=False) is True
    assert nid not in cortex.neurons and (ckpt_dir / f"neuron_{nid}.pt").exists()
    cortex._load_neurons()  # 无返回值：契约是"把盘上的 neuron_*.pt 装回 self.neurons"
    assert nid in cortex.neurons, "盘上还在的神经元必须被重新装配（H3：不能只认硬编码 5 域）"
    assert before - {nid} <= set(cortex.neurons), "重新装配不该把其他神经元丢掉"
    payload = load_legacy_checkpoint(str(ckpt_dir / f"neuron_{nid}.pt"), map_location="cpu")
    assert "neuron_config" in payload and "state_dict" in payload
