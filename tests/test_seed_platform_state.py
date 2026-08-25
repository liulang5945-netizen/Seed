"""R5: seed_platform.app_state 单元测试——覆盖率爬坡（31% → 目标 80%+）。

AppState 是纯 dataclass + 细粒度锁，直接用新实例测试，不触碰全局单例。
"""

import threading

import pytest

from seed_platform.app_state import AppState


class _FakeCortex:
    """模拟 Cortex 类型名（is_taiji 按类型名判断）"""


_FakeCortex.__name__ = "Cortex"


@pytest.fixture()
def state():
    return AppState()


# ======================== 模型切换状态 ========================


def test_switch_status_lifecycle(state):
    assert state.get_switch_status()["status"] == "idle"
    state.update_switch_status("switching", "切换中", "")
    snapshot = state.get_switch_status()
    assert snapshot == {"status": "switching", "message": "切换中", "error": ""}
    state.update_switch_status("error", "", "boom")
    assert state.get_switch_status()["error"] == "boom"
    state.reset_switch_status()
    assert state.get_switch_status() == {"status": "idle", "message": "", "error": ""}


def test_update_rag_kb(state):
    assert state.rag_kb is None
    sentinel = object()
    state.update_rag_kb(sentinel)
    assert state.rag_kb is sentinel


# ======================== 模型装载 ========================


def test_get_model_info_empty(state):
    info = state.get_model_info()
    assert info["loaded"] is False
    assert info["model_name"] is None
    assert info["ready"] is False


def test_update_model_and_is_taiji(state):
    model = _FakeCortex()
    state.update_model(model, tokenizer=object(), trainer=None, model_name="cortex-test")
    assert state.get_model_info()["loaded"] is True
    assert state.get_model_info()["model_name"] == "cortex-test"
    assert state.is_taiji() is True
    assert state.get_trainer() is None
    assert state.get_tokenizer() is not None


def test_is_taiji_false_for_other_models(state):
    class NotCortex:
        pass

    state.update_model(NotCortex(), None, None, "other")
    assert state.is_taiji() is False


def test_unload_model_clears_everything(state):
    unloaded = []

    class _Trainer:
        def unload(self):
            unloaded.append(True)

    class _Model:
        def __init__(self):
            self.moved = None

        def to(self, device):
            self.moved = device

    model = _Model()
    state.update_model(model, object(), _Trainer(), "m")
    state.mark_started()
    state.unload_model()
    assert unloaded == [True]
    assert model.moved == "cpu"
    assert state.model is None
    assert state.trainer is None
    assert state.tokenizer is None
    assert state.startup_complete is False
    assert state.get_model_info()["loaded"] is False


def test_unload_model_noop_when_empty(state):
    # 空状态卸载不应抛异常
    state.unload_model()
    assert state.model is None


# ======================== 训练锁 ========================


def test_training_lock_exclusive(state):
    assert state.try_start_training() is True
    assert state.is_training is True
    # 第二次获取失败
    assert state.try_start_training() is False
    state.stop_training_requested = True
    state.finish_training()
    assert state.is_training is False
    assert state.stop_training_requested is False
    # 释放后可再次获取
    assert state.try_start_training() is True
    state.finish_training()


def test_training_context_manager(state):
    with state.training_context("Test"):
        assert state.is_training is True
    assert state.is_training is False


def test_training_context_conflict_raises(state):
    with (
        state.training_context("First"),
        pytest.raises(RuntimeError, match="无法获取训练锁"),
        state.training_context("Second"),
    ):
        pass  # pragma: no cover
    # 外层退出后锁已释放
    assert state.try_start_training() is True
    state.finish_training()


def test_finish_training_without_lock_is_safe(state):
    # 未持锁时释放不应抛异常（吞 RuntimeError）
    state.finish_training()
    assert state.is_training is False


# ======================== 发布锁 ========================


def test_publishing_lock_lifecycle(state):
    assert state.try_start_publishing() is True
    assert state.publishing is True
    assert state.try_start_publishing() is False
    state.finish_publishing()
    assert state.publishing is False
    assert state.try_start_publishing() is True
    state.finish_publishing()


def test_force_reset_publishing_releases_lock(state):
    state.try_start_publishing()
    result = state.force_reset_publishing()
    assert result == {"was_publishing": True, "lock_was_held": True}
    assert state.publishing is False
    # 锁已释放，可重新获取
    assert state.try_start_publishing() is True
    state.finish_publishing()


def test_force_reset_publishing_when_idle(state):
    result = state.force_reset_publishing()
    assert result == {"was_publishing": False, "lock_was_held": False}


# ======================== 启动状态 ========================


def test_startup_state_machine(state):
    state.mark_starting()
    assert state.startup_complete is False
    assert state.startup_error is None

    state.mark_started()
    assert state.startup_complete is True
    assert state.startup_error is None
    assert state.get_model_info()["ready"] is True

    state.mark_startup_failed("加载失败")
    assert state.startup_complete is True
    assert state.startup_error == "加载失败"
    assert state.get_model_info()["ready"] is False


def test_register_background_task(state):
    thread = threading.Thread(target=lambda: None)
    state.register_background_task(thread)
    assert thread in state.background_tasks
