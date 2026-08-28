"""Persistent application settings owned by the Seed platform layer.

The platform owns storage and compatibility semantics. Legacy modules may
read or update settings, but they do not define where settings live or how
they are serialized.
"""

from __future__ import annotations

import contextlib
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
SETTINGS_SCHEMA_VERSION = 2

_LEGACY_MODEL_SETTING_KEYS = (
    "model_type",
    "model_name",
    "gguf_path",
    "download_hf",
    "n_gpu_layers",
    "n_ctx",
    "load_in_4bit",
    "load_in_8bit",
    "use_lora",
    "lora_r",
    "lora_alpha",
)


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
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def _safe_checkpoint_id(value: object) -> str:
    """Keep only a relative checkpoint identifier during migration."""

    if not isinstance(value, str) or not value.strip():
        return ""
    candidate = Path(value.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return candidate.as_posix()


def migrate_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate persisted settings to the Taiji-owned schema.

    Old model fields are never interpreted as a provider or runtime silently.
    A clearly native ``self``/``seed`` marker may retain a safe relative
    checkpoint id; ambiguous paths and all GGUF/Cortex/LoRA fields are moved
    to an auditable quarantine entry instead.
    """

    if not isinstance(data, dict):
        raise TypeError("settings data must be a dictionary")

    source_version = data.get("schema_version", 1)
    try:
        source_version = int(source_version)
    except (TypeError, ValueError):
        source_version = 1

    migrated = dict(data)
    runtime = migrated.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("kind") != "taiji":
        runtime = {"kind": "taiji", "checkpoint_id": ""}
    else:
        runtime = {
            "kind": "taiji",
            "checkpoint_id": _safe_checkpoint_id(runtime.get("checkpoint_id")),
        }

    migration = migrated.get("migration")
    if not isinstance(migration, dict):
        migration = {}
    quarantined = migration.get("quarantined")
    if not isinstance(quarantined, dict):
        quarantined = {}
    converted = migration.get("converted")
    if not isinstance(converted, list):
        converted = []

    legacy_values = {
        key: migrated.pop(key) for key in _LEGACY_MODEL_SETTING_KEYS if key in migrated
    }
    legacy_type = str(legacy_values.get("model_type", "") or "").lower()
    legacy_checkpoint = _safe_checkpoint_id(
        migrated.pop("checkpoint", migrated.pop("checkpoint_path", ""))
    )

    if legacy_type in {"self", "seed"} and not any(
        key in legacy_values for key in ("gguf_path", "model_name")
    ):
        if legacy_checkpoint:
            runtime["checkpoint_id"] = legacy_checkpoint
        marker = "legacy_self_marker_v1"
        if marker not in converted:
            converted.append(marker)
    elif legacy_values:
        quarantined["legacy_model_settings_v1"] = {
            "source_schema_version": source_version,
            "fields": sorted(legacy_values),
            "values": legacy_values,
            "reason": "ambiguous_or_legacy_model_semantics_not_activated",
        }

    # A non-Taiji runtime is not allowed to survive as an active setting.
    if isinstance(data.get("runtime"), dict) and data["runtime"].get("kind") != "taiji":
        quarantined["legacy_runtime_v1"] = {
            "source_schema_version": source_version,
            "fields": ["runtime"],
            "values": data["runtime"],
            "reason": "only_taiji_runtime_is_product_owned",
        }

    migrated["runtime"] = runtime
    migrated["schema_version"] = SETTINGS_SCHEMA_VERSION
    if converted or quarantined:
        migration["converted"] = converted
        migration["quarantined"] = quarantined
        migration["last_source_schema_version"] = source_version
        migrated["migration"] = migration
    elif "migration" in migrated:
        migrated["migration"] = migration
    return migrated


def _load_canonical_settings(*, persist_migration: bool = True) -> dict[str, Any]:
    raw = _read_settings()
    canonical = migrate_settings(raw)
    if persist_migration and raw and canonical != raw:
        _write_settings(canonical)
    return canonical


def get_setting(key: str, default: Any = None) -> Any:
    """Return one setting, or ``default`` when it has not been configured."""

    with _SETTINGS_LOCK:
        return _load_canonical_settings().get(key, default)


def load_settings() -> dict[str, Any]:
    """Load the complete settings object from the platform data directory."""

    with _SETTINGS_LOCK:
        return _load_canonical_settings()


def save_settings(data: dict[str, Any]) -> None:
    """Replace all settings atomically."""

    if not isinstance(data, dict):
        raise TypeError("settings data must be a dictionary")
    with _SETTINGS_LOCK:
        _write_settings(migrate_settings(dict(data)))


def update_settings(updates: dict[str, Any]) -> None:
    """Merge updates into the persisted settings atomically."""

    if not isinstance(updates, dict):
        raise TypeError("settings updates must be a dictionary")
    with _SETTINGS_LOCK:
        data = _load_canonical_settings()
        data.update(updates)
        _write_settings(migrate_settings(data))
