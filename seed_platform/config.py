"""Taiji 核心配置模块（API 集成修复，工作4）。

提供 API 服务/硬件自适应/training 公共模块依赖的配置接口：
- TrainingConfig: 训练/推理配置 dataclass
- get_config / save_config: 配置读写
- get_external_path / get_internal_path: 外部可写路径 / 内部打包路径
- get_writable_base_dir: 可写基目录（防只读安装目录）
- apply_env_overrides: 环境变量覆盖
- MODEL_LOAD_TIMEOUT: 模型加载超时

此模块是对 API 层依赖的补齐实现；taiji-neuron 核心训练（resonance）不依赖本模块。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from seed_platform.paths import (
    get_external_path as get_external_path,
)
from seed_platform.paths import (
    get_internal_path as get_internal_path,
)
from seed_platform.paths import (
    get_writable_base_dir as get_writable_base_dir,
)

logger = logging.getLogger(__name__)

# 模型加载超时（秒）
MODEL_LOAD_TIMEOUT = 600


@dataclass
class TrainingConfig:
    """训练/推理配置（API 层与硬件自适应共用）。

    字段集合来自 api/routes_settings.py、neuroplex/core/hardware.py、
    api/training/common.py 的实际引用。
    """

    device: str = "auto"  # auto / cuda / mps / cpu
    model_name: str = ""  # 模型名称或路径
    model_type: str = "self"  # self（Legacy NeuroPlex 原生单一模型类型）
    cache_dir: str = ""  # 缓存目录
    resume_from_checkpoint: str = ""  # checkpoint 路径
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_lora: bool = False
    lora_r: int = 16
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_length: int = 2048
    gguf_path: str = ""
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    # 内部附加（hw_diag 等）
    _hw_diag: dict | None = field(default=None, repr=False)

    def resolve_device(self) -> str:
        """自动判断最优运算设备（cuda > mps > cpu）。"""
        from neuroplex.core.hardware import resolve_device

        return resolve_device(self)

    @staticmethod
    def get_total_ram_gb() -> float:
        """获取系统总内存（GB），失败时保守回退 16GB。"""
        try:
            import psutil

            return float(round(psutil.virtual_memory().total / (1024**3), 1))
        except Exception:
            try:
                return max(16.0, 0.0)
            except Exception:
                return 16.0


def get_config(args: list[str] | None = None) -> TrainingConfig:
    """获取配置。

    Args:
        args: 命令行剩余参数（key=value 或 --key value 形式，可空）

    Returns:
        TrainingConfig 实例
    """
    config = TrainingConfig()
    # 合并环境变量覆盖（apply_env_overrides 写入）
    for attr, val in _ENV_OVERRIDES.items():
        if hasattr(config, attr):
            setattr(config, attr, val)
    if not args:
        return config

    # 支持 --key value 与 key=value 两种形式
    pairs: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            pairs.append(f"{a[2:]}={args[i + 1]}")
            i += 2
        elif "=" in a:
            pairs.append(a.lstrip("-"))
            i += 1
        else:
            i += 1

    for pair in pairs:
        key, _, value = pair.partition("=")
        key = key.replace("-", "_")
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        try:
            if isinstance(current, bool):
                setattr(config, key, value.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(config, key, int(value))
            elif isinstance(current, float):
                setattr(config, key, float(value))
            else:
                setattr(config, key, value)
        except (ValueError, TypeError) as e:
            logger.debug("【get_config】处理失败（非致命）: %s", e)
    return config


def save_config(config: TrainingConfig, path: str | None = None) -> str:
    """保存配置到 JSON 文件。

    Args:
        config: TrainingConfig 实例
        path: 保存路径（默认 <writable_base>/config.json）

    Returns:
        保存路径
    """
    import json
    from dataclasses import asdict

    path = path or os.path.join(get_writable_base_dir(), "config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {k: v for k, v in asdict(config).items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def apply_env_overrides() -> None:
    """从环境变量应用配置覆盖（TAIJI_* 前缀）。

    支持：TAIJI_MODEL_NAME、TAIJI_DEVICE、TAIJI_MODEL_TYPE 等。
    """
    overrides = {
        "MODEL_NAME": "model_name",
        "DEVICE": "device",
        "MODEL_TYPE": "model_type",
        "CACHE_DIR": "cache_dir",
        "CHECKPOINT": "resume_from_checkpoint",
    }
    changed = False
    for env_key, attr in overrides.items():
        val = os.environ.get(f"TAIJI_{env_key}")
        if val:
            try:
                # 写入进程级配置缓存（供 get_config 默认值使用）
                _ENV_OVERRIDES[attr] = val
                changed = True
            except Exception as e:
                logger.debug("【apply_env_overrides】处理失败（非致命）: %s", e)
    if changed:
        print("[config] 已应用环境变量覆盖", flush=True)


# 环境变量覆盖缓存（get_config 时合并）
_ENV_OVERRIDES: dict = {}
