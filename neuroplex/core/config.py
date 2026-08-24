"""Compatibility exports for the platform-owned configuration module."""

from seed_platform.config import (
    MODEL_LOAD_TIMEOUT,
    TrainingConfig,
    apply_env_overrides,
    get_config,
    get_external_path,
    get_internal_path,
    get_writable_base_dir,
    save_config,
)

__all__ = [
    "MODEL_LOAD_TIMEOUT",
    "TrainingConfig",
    "apply_env_overrides",
    "get_config",
    "get_external_path",
    "get_internal_path",
    "get_writable_base_dir",
    "save_config",
]
