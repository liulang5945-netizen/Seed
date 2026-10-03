"""DEBT-G30 的守卫：证据门的"有效状态"要有一个**公开出口**，而且**读它不许改变任何东西**。

来历（2026-10-03，PLAN-A-30 §99 与台账 `DEBT-G30`）：`config／override` 的合成式此前在产品里内联写一份
（`taiji/model.py` 的生成路径），外面又有四处各自扒私有字段 `_copy_evidence_utf8_gate_override` 重抄一遍
（探针两处＋测试两处）。**"默认关闭 ⇒ 逐位不变"这条验收要能从产品外面证**，而外面唯一的办法是复制推导式——
复制的那一份会随产品改动过期。同族的窗口那一支早有 `copy_evidence_window_stats()`，这条补齐同族待遇。

三件事分别钉：
1. **逐值等**——六种 `override × config` 组合下公开读数与"旧的外部推导式"相等（改错式子即红）；
2. **缺省形状**——新模型没有 override、有效值等于 config；`set(..., None)` 要**回到跟随 config**；
3. **读态不改态**——同种子同字节序列两条链，一条每步前读三次状态，逐步 `prior_prediction` 必须逐位相同
   （防的是"读数顺手清了计数/推了状态"这类把观测变成干预的写法）。
"""

from __future__ import annotations

from typing import Any

from taiji import Taiji, TaijiConfig


def _config(**overrides: Any) -> TaijiConfig:
    values: dict[str, Any] = {
        "region_sizes": (64, 48),
        "synapse_fan_in": 16,
        "motor_fan_in": 48,
        "seed": 11,
    }
    values.update(overrides)
    return TaijiConfig(**values)


def _legacy_effective(model: Taiji) -> bool:
    """外面那四处原本各自重抄的推导式——留在这里当**被超越的参照**，用来证明新出口逐值相同。"""

    override = model._copy_evidence_utf8_gate_override
    return bool(model.config.copy_evidence_utf8_gate) if override is None else bool(override)


def test_the_public_readout_matches_the_single_expression_on_all_six_combos() -> None:
    for config_value in (True, False):
        model = Taiji(_config(copy_evidence_utf8_gate=config_value))
        for override in (None, True, False):
            model.set_copy_evidence_utf8_gate(override)
            state = model.copy_evidence_utf8_gate_state()
            assert set(state) == {'config', 'override', 'effective'}, state
            assert state['config'] is config_value, state
            assert state['override'] is override, state
            expected = config_value if override is None else override
            assert state['effective'] is bool(expected), (config_value, override, state)
            assert state['effective'] == _legacy_effective(model), (config_value, override, state)


def test_a_fresh_model_follows_config_and_none_restores_that_following() -> None:
    model = Taiji(_config())
    state = model.copy_evidence_utf8_gate_state()
    assert state['override'] is None, state
    assert state['config'] is bool(TaijiConfig().copy_evidence_utf8_gate), state
    assert state['effective'] == state['config'], state

    model.set_copy_evidence_utf8_gate(False)
    assert model.copy_evidence_utf8_gate_state()['effective'] is False
    model.set_copy_evidence_utf8_gate(True)
    assert model.copy_evidence_utf8_gate_state()['effective'] is True
    #: 评测期开关的语义：`None` 不是"关"，是"回到跟随 config"。
    model.set_copy_evidence_utf8_gate(None)
    restored = model.copy_evidence_utf8_gate_state()
    assert restored['override'] is None and restored['effective'] == restored['config'], restored


def test_reading_the_state_does_not_disturb_the_generation_path() -> None:
    symbols = '我叫阿岩。我的名字是什么？'.encode()

    def run(interleave: bool) -> list[Any]:
        model = Taiji(_config(copy_evidence_utf8_gate=True))
        model.reset_dynamics(episode_id='g30-purity')
        model.observe(model.config.boundary_symbol, learn=False, readout='predictive')
        predictions: list[Any] = []
        for byte in symbols:
            if interleave:
                for _ in range(3):
                    model.copy_evidence_utf8_gate_state()
                    model.copy_evidence_window_stats()
            step = model.observe(int(byte), learn=False, readout='predictive')
            predictions.append(step.prior_prediction)
        return predictions

    plain = run(False)
    assert len(plain) == len(symbols), plain
    assert plain == run(True), '读状态把生成路径改动了——观测不能是干预'


def test_the_readout_is_used_inside_the_model_rather_than_duplicated() -> None:
    """产品内不许再留第二份同一条式子（`一副档只住一处` 的产品侧版本）。"""

    import inspect

    source = inspect.getsource(Taiji)
    assert source.count('def _copy_evidence_utf8_gate_effective') == 1, source.count(
        'def _copy_evidence_utf8_gate_effective'
    )
    #: 生成路径那处必须改成调用它，而不是把三元式再抄一遍。
    assert source.count('self._copy_evidence_utf8_gate_effective()') >= 2, source.count(
        'self._copy_evidence_utf8_gate_effective()'
    )


def test_the_a30_probe_reads_the_public_accessor_too() -> None:
    """DEBT-G30 第二步（探针升 v39）：L2 仪器那两处复制推导已改读公开出口。

    钉两件事：①私有字段 `_copy_evidence_utf8_gate_override` 在这台仪器里**一次都不许再出现**
    ——它出现就意味着有人又开始复制 `config-or-override` 那条式子（v38 及以前正是两份复制）；
    ②三列与那条守卫必须共用**同一次读取**（`gate_state[` 恰好出现四次：effective／config／override
    加守卫一处），免得"改成读出口"却又在多处各读一遍、把一次读取变成四次快照。
    """

    from pathlib import Path

    probe_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "training" / "probe_taiji_a30_stop_failure.py"
    )
    probe = probe_path.read_text(encoding="utf-8")
    assert "_copy_evidence_utf8_gate_override" not in probe
    assert "gate_state = runtime.model.substrate.copy_evidence_utf8_gate_state()" in probe
    assert probe.count("gate_state[") == 4, probe.count("gate_state[")
