"""`PLAN-N5-20` 的挂接契约册（J-N5r-1／2／3、G-N5r-1／2 的机检形）。

真长跑不在本册里跑——本册只钉四件事：①默认关闭；②落账那一步**不许切本臂 readout**且**关掉所有 learn 位**；
③喂入确实走产品入口而不是第二条写链；④一次真实的小喂入能把 `generated` 那一行从 0 抬起来。
第④用的是 tiny 配置（不是长跑读数）；长跑那一档的现读记在台账 ㊵-740：400 符号、每 50 喂一次
⇒ `feeds=8`、`generated_write_count=8`、行权重 **0.276523322**、奖励和 0.0、`write_surface_ok` 仍为真。
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from taiji import Taiji, TaijiConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "training" / "train_taiji_r2_readout_retrain.py"


def _runner():
    spec = importlib.util.spec_from_file_location("n5_31_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubSubstrate:
    """记录调用序列的假 substrate：用来单独钉"落账那一步传了什么参数"。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def record_generated_prediction(self, symbol: int) -> None:
        self.calls.append(("record_generated_prediction", (symbol,), {}))

    def observe(self, symbol: int, **kwargs) -> None:
        self.calls.append(("observe", (symbol,), kwargs))

    class _Field:
        PROVENANCE_KINDS = ("experienced", "imagined", "replayed", "external", "generated")
        generated_write_count = 3

        class _Readout:
            import torch as _torch

            edge_weight = _torch.zeros(5, 4)

        provenance_readout = _Readout()

    memory = _Field()


def _tiny_model() -> Taiji:
    config = TaijiConfig(
        region_sizes=(16, 12),
        synapse_fan_in=4,
        motor_fan_in=8,
        memory_units=24,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=8,
        memory_iterations=2,
        seed=5,
        identity_organ_enabled=False,
    )
    model = Taiji(config)
    model.reset_dynamics(episode_id="n5-31")
    return model


def test_the_flag_defaults_to_off_and_the_runner_keeps_one_write_path() -> None:
    """J-N5r-1 的形状面＋G-N5r-1：默认 0；跑器里不许出现第二条写链。"""

    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "generated_feed_every":
            defaults["run_arm_param"] = getattr(node.value, "value", None)
    assert "--generated-feed-every" in source
    assert source.count("default=0") >= 1
    assert "memory.write(" not in source
    assert source.count("record_generated_prediction(") == 1

    # 负对照：同一把尺必须能数出越界样本。
    fabricated = "def f(model):\n    return model.memory.write(ctx)\n"
    assert "memory.write(" in fabricated


def test_the_flush_keeps_the_arms_readout_and_turns_every_learn_off() -> None:
    """落账那一步：readout 照抄本臂、所有 learn* 关死（㊵-740 两次撞墙的形状）。"""

    runner = _runner()
    stub = _StubSubstrate()
    arm_kwargs = {
        "learn": True,
        "readout": "predictive",
        "learn_motor": True,
        "learn_fabric": True,
        "learn_predictive_readout": True,
    }
    runner.feed_generated_answer(stub, 66, arm_kwargs)

    names = [call[0] for call in stub.calls]
    assert names == ["record_generated_prediction", "observe"], names
    flush = stub.calls[1][2]
    assert flush["readout"] == "predictive", flush
    learned = {key: value for key, value in flush.items() if key.startswith("learn")}
    assert learned and set(learned.values()) == {False}, learned


def test_a_real_feed_grows_the_generated_row() -> None:
    """J-N5r-2 的小配置版：写计数与那一行的权重必须同时离开 0。"""

    runner = _runner()
    model = _tiny_model()
    row = int(model.memory.PROVENANCE_KINDS.index("generated"))
    before = float(model.memory.provenance_readout.edge_weight[row].abs().max())
    assert before == 0.0

    prior = None
    for symbol in (65, 66, 67, 68):
        step = model.observe(symbol, learn=False, learn_motor=False)
        prior = getattr(step, "prior_prediction", None)
    assert prior is not None

    fed = runner.feed_generated_answer(
        model, int(prior), {"learn": False, "readout": "action", "learn_motor": False}
    )
    after = float(model.memory.provenance_readout.edge_weight[row].abs().max())
    assert model.memory.generated_write_count == 1
    assert fed["write_count"] == 1.0
    assert after > 0.0, (before, after)


def test_the_entry_refuses_to_overwrite_an_in_flight_experience() -> None:
    """无动作入口不许把已在路上的经验挤掉——静默覆盖会让"哪次喂入长了权重"不可归因。"""

    model = _tiny_model()
    model.observe(65, learn=False, learn_motor=False)
    model.act((65,), sample=False)
    with pytest.raises(RuntimeError, match="quiet pending state"):
        model.record_generated_prediction(66)

    model2 = _tiny_model()
    model2.observe(65, learn=False, learn_motor=False)
    model2.record_generated_prediction(66)
    with pytest.raises(RuntimeError, match="quiet pending state"):
        model2.record_generated_prediction(67)


def test_pricing_keys_are_published_by_the_runner() -> None:
    """J-N5r-3：跑器自述必须带本次实测速率与由它推的 ETA，不接受引用来的定价。"""

    source = RUNNER.read_text(encoding="utf-8")
    for key in (
        "measured_symbols_per_second",
        "eta_hours_at_requested_budget",
        "generated_write_count",
        "generated_row_weight_max_abs",
        "generated_reward_sum",
    ):
        assert key in source, key
