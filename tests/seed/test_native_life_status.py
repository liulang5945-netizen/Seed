"""原生 homeostasis 生命数据链路的回归测试。

覆盖用户报告的「生命系统显示接入原生，但没有生命数据」这条缺陷的全部四层：

  1. ``TSKV8Adapter.ensure_homeostatic_controller`` 幂等挂载——产品此前只在
     离线评测脚本里挂过稳态器官，运行时的 ``homeostasis`` 恒为全零，因此
     根本不存在生命数据；同时 ``Seed.from_checkpoint`` 先于
     ``SeedRuntime.__init__`` 执行，挂载必须不覆盖已恢复的控制器。
  2. ``SeedRuntime.homeostasis_status`` 只上报架构真的测出来的事实。
  3. ``runtime_service._life_section`` 把原生 0..1 换算成客户端合同的 0..100，
     既不发明维度也不丢弃维度。
  4. ``LifePayload`` 不得再用固定 schema 伪造 needs——旧的 ``LifeNeedsPayload``
     会把空 needs 补成四个 50.0，并静默丢掉原生实测的 ``stress``。
"""

from __future__ import annotations

from seed import Seed


def test_ensure_homeostatic_controller_is_idempotent() -> None:
    seed = Seed(episode_id="homeostasis-idempotent")
    architecture = seed.architecture

    assert architecture.homeostatic_controller_attached is False
    architecture.ensure_homeostatic_controller()
    assert architecture.homeostatic_controller_attached is True

    first = architecture._homeostatic_controller
    architecture.ensure_homeostatic_controller()
    # 二次调用必须是空操作，否则 checkpoint 恢复出来的控制器会被丢掉。
    assert architecture._homeostatic_controller is first


def test_observation_moves_homeostatic_state_off_zero() -> None:
    seed = Seed(episode_id="homeostasis-observe")
    seed.architecture.ensure_homeostatic_controller()

    before = seed.architecture.homeostatic_state()
    assert (before.tick, before.curiosity, before.fatigue, before.stress) == (0, 0.0, 0.0, 0.0)

    # observe 的契约是「单个整数符号」，所以逐字节喂进去而不是整串。
    for symbol in b"hello world hello":
        seed.observe(symbol)

    after = seed.architecture.homeostatic_state()
    assert after.tick > before.tick
    # 未训练模型的预测误差很高，所以驱动值必然离开零点——这是实测而非估算。
    assert after.curiosity > 0.0
    assert after.fatigue > 0.0


def test_runtime_homeostasis_status_reports_measured_dimensions() -> None:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime(Seed(episode_id="homeostasis-status"))
    status = runtime.homeostasis_status()

    assert status["attached"] is True
    assert set(status["needs"]) == {"curiosity", "fatigue", "stress"}
    assert set(status["drives"]) == {"exploration", "replay", "rest", "play"}
    # 原生单位是 0..1，换算留给上层，运行时自己不做展示缩放。
    for value in status["needs"].values():
        assert 0.0 <= value <= 1.0
    assert status["mode"]
    assert status["tick"] >= 0

    assert runtime.status()["homeostasis"] == runtime.homeostasis_status()


def test_life_payload_never_fabricates_needs() -> None:
    from api.models_runtime import LifePayload

    empty = LifePayload().model_dump()
    # 旧的 LifeNeedsPayload 会在这里凭空造出四个 50.0。
    assert empty["needs"] == {}

    # 原生实测的 stress 必须原样穿过，不能被固定 schema 静默丢弃。
    measured = LifePayload(
        status="seed",
        is_running=True,
        needs={"curiosity": 42.5, "fatigue": 84.75, "stress": 3.25},
    ).model_dump()
    assert measured["needs"] == {"curiosity": 42.5, "fatigue": 84.75, "stress": 3.25}


def test_native_life_section_scales_units_without_inventing_dimensions(monkeypatch) -> None:
    import api.seed_runtime as seed_runtime
    from seed_platform.runtime_service import _native_life_section

    class _Runtime:
        def homeostasis_status(self) -> dict:
            return {
                "attached": True,
                "tick": 7,
                "mode": "wake",
                "needs": {"curiosity": 0.425, "fatigue": 0.8475, "stress": 0.0},
                "drives": {},
            }

    monkeypatch.setattr(seed_runtime, "get_seed_runtime", lambda: _Runtime())

    section = _native_life_section()
    assert section is not None
    assert section["is_running"] is True
    # 唯一的变换是 0..1 → 0..100；维度集合与运行时上报的完全一致。
    assert section["needs"] == {"curiosity": 42.5, "fatigue": 84.75, "stress": 0.0}


def test_native_life_section_falls_back_when_no_native_runtime(monkeypatch) -> None:
    import api.seed_runtime as seed_runtime
    from seed_platform.runtime_service import _native_life_section

    monkeypatch.setattr(seed_runtime, "get_seed_runtime", lambda: None)

    # 返回 None 才能让 _life_section 走 Legacy 分支，而不是谎报一个空原生态。
    assert _native_life_section() is None
