"""Filesystem layout for source runs and packaged Seed applications.

This module deliberately has no dependency on ``neuroplex``.  The LocalAppData
directory remains named ``Taiji`` for data compatibility with existing desktop
installations; changing that storage identity requires a separate migration.
"""

from __future__ import annotations

import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR_NAME = "Taiji"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_writable_base_dir() -> str:
    """Return the writable root used for checkpoints and user data."""

    if _is_frozen():
        local = os.environ.get("LOCALAPPDATA", "")
        base = os.path.join(local, _APP_DIR_NAME) if local else os.getcwd()
    else:
        base = _PROJECT_ROOT
    os.makedirs(base, exist_ok=True)
    return base


def get_external_path(relative_path: str) -> str:
    """Resolve a writable application path, preserving the legacy layout."""

    base = get_writable_base_dir()
    path = os.path.join(base, relative_path) if relative_path else base
    if relative_path:
        os.makedirs(path, exist_ok=True)
    return path


def get_internal_path(relative_path: str) -> str:
    """Resolve a bundled read-only resource path."""

    if _is_frozen():
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    else:
        base = _PROJECT_ROOT
    return os.path.join(base, relative_path) if relative_path else base
