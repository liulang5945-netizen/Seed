"""Compatibility exports for the former Neuroplex settings service."""

from seed_platform.settings import (
    get_setting,
    load_settings,
    save_settings,
    update_settings,
)

__all__ = ["get_setting", "load_settings", "save_settings", "update_settings"]
