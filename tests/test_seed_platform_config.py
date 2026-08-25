"""R5: seed_platform.config / memory 单元测试——覆盖率爬坡。"""

import json

from seed_platform import config as platform_config
from seed_platform.config import (
    TrainingConfig,
    apply_env_overrides,
    get_config,
    save_config,
)
from seed_platform.memory import (
    MemoryWatchdog,
    force_memory_refresh,
    get_memory_status_dict,
    memory_guarded,
)

# ======================== TrainingConfig ========================


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.device == "auto"
    assert cfg.model_type == "self"
    assert cfg.batch_size == 4
    assert cfg.n_ctx == 2048
    assert cfg.load_in_4bit is False


def test_get_config_no_args_applies_env_cache():
    cfg = get_config()
    assert isinstance(cfg, TrainingConfig)


def test_get_config_parses_key_value_forms():
    cfg = get_config(
        [
            "--device",
            "cpu",
            "batch_size=8",
            "--load-in-4bit",
            "true",
            "learning_rate=0.5",
            "unknown_key=x",
        ]
    )
    assert cfg.device == "cpu"
    assert cfg.batch_size == 8
    assert cfg.load_in_4bit is True
    # 未知键被忽略
    assert not hasattr(cfg, "unknown_key") or getattr(cfg, "unknown_key", None) != "x"


def test_get_config_ignores_invalid_numbers():
    cfg = get_config(["batch_size=not-a-number"])
    assert cfg.batch_size == 4  # 保持默认


def test_get_config_skips_dangling_flag():
    cfg = get_config(["--device"])
    assert cfg.device == "auto"


def test_save_config_roundtrip(tmp_path):
    cfg = TrainingConfig(device="cuda", model_name="unit-test")
    path = save_config(cfg, str(tmp_path / "cfg.json"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["device"] == "cuda"
    assert data["model_name"] == "unit-test"
    # 私有字段不落盘
    assert "_hw_diag" not in data


def test_apply_env_overrides(monkeypatch):
    platform_config._ENV_OVERRIDES.clear()
    monkeypatch.setenv("TAIJI_DEVICE", "mps")
    monkeypatch.setenv("TAIJI_MODEL_NAME", "env-model")
    apply_env_overrides()
    try:
        assert platform_config._ENV_OVERRIDES["device"] == "mps"
        cfg = get_config()
        assert cfg.device == "mps"
        assert cfg.model_name == "env-model"
    finally:
        platform_config._ENV_OVERRIDES.clear()


def test_get_total_ram_gb_positive():
    assert TrainingConfig.get_total_ram_gb() > 0


def test_resolve_device_returns_string():
    cfg = TrainingConfig(device="auto")
    device = cfg.resolve_device()
    assert isinstance(device, str)
    assert device in {"cpu", "cuda", "mps"} or device


# ======================== Memory watchdog stub ========================


def test_memory_watchdog_reports_healthy():
    watchdog = MemoryWatchdog()
    assert watchdog.check() is False
    assert watchdog.is_critical() is False
    ok, message = watchdog.can_proceed(min_avail_pct=0.2)
    assert ok is True
    assert "stub" in message
    ok, message = MemoryWatchdog.can_build_embeddings(100, 384)
    assert ok is True
    assert watchdog.status.level == 0


def test_memory_guarded_bare_decorator():
    @memory_guarded
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add.__name__ == "add"


def test_memory_guarded_factory_form():
    calls = []

    @memory_guarded(min_avail_pct=0.1, on_critical=lambda: calls.append(1))
    def double(x):
        return x * 2

    assert double(4) == 8
    # stub 不触发 on_critical
    assert calls == []


def test_memory_status_dict_shape():
    status = get_memory_status_dict()
    assert status["level"] == 0
    assert status["status"] == "healthy"
    assert status["pressure"] is False
    assert status["critical"] is False


def test_force_memory_refresh_matches_status():
    assert force_memory_refresh() == get_memory_status_dict()
