"""Persistent application settings owned by the Seed platform layer.

The platform owns storage and compatibility semantics. Legacy modules may
read or update settings, but they do not define where settings live or how
they are serialized.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .paths import get_external_path


logger = logging.getLogger("SeedPlatform.Settings")
_SETTINGS_FILENAME = "app_settings.json"
_SETTINGS_LOCK = threading.RLock()


def _settings_path() -> Path:
    return Path(get_external_path("data")) / _SETTINGS_FILENAME


def _read_settings() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            logger.warning("Ignoring non-object settings file: %s", path)
            return {}
        return data
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to read settings from %s: %s", path, exc)
        return {}


def _write_settings(data: dict[str, Any]) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def get_setting(key: str, default: Any = None) -> Any:
    """Return one setting, or ``default`` when it has not been configured."""

    with _SETTINGS_LOCK:
        return _read_settings().get(key, default)


def load_settings() -> dict[str, Any]:
    """Load the complete settings object from the platform data directory."""

    with _SETTINGS_LOCK:
        return _read_settings()


def save_settings(data: dict[str, Any]) -> None:
    """Replace all settings atomically."""

    if not isinstance(data, dict):
        raise TypeError("settings data must be a dictionary")
    with _SETTINGS_LOCK:
        _write_settings(dict(data))


def update_settings(updates: dict[str, Any]) -> None:
    """Merge updates into the persisted settings atomically."""

    if not isinstance(updates, dict):
        raise TypeError("settings updates must be a dictionary")
    with _SETTINGS_LOCK:
        data = _read_settings()
        data.update(updates)
        _write_settings(data)
