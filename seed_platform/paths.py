"""Filesystem layout for source runs and packaged Seed applications.

This module deliberately has no dependency on ``neuroplex``.  The LocalAppData
directory remains named ``Taiji`` for data compatibility with existing desktop
installations; changing that storage identity requires a separate migration.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR_NAME = "Taiji"
_WRITABLE_PROBE_TIMEOUT_SECONDS = 0.5
_FROZEN_BASE_CACHE_KEY: tuple[str, str, str, str] | None = None
_FROZEN_BASE_CACHE_VALUE: str | None = None


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _probe_writable_directory(candidate: str) -> bool:
    """Return whether a directory is writable without blocking app startup.

    Some Windows ACL/filter-driver combinations can block a file-create call
    instead of returning ``PermissionError``. The probe therefore runs in a
    daemon thread with a short deadline; the caller can move to the next
    storage candidate even if the OS call never returns.
    """

    outcome = {"writable": False}
    finished = threading.Event()

    def _attempt() -> None:
        probe_path: str | None = None
        try:
            os.makedirs(candidate, exist_ok=True)
            descriptor, probe_path = tempfile.mkstemp(prefix=".seed-write-", dir=candidate)
            os.close(descriptor)
            os.unlink(probe_path)
            outcome["writable"] = True
        except OSError:
            pass
        finally:
            if probe_path:
                try:
                    if os.path.exists(probe_path):
                        os.unlink(probe_path)
                except OSError:
                    pass
            finished.set()

    threading.Thread(target=_attempt, name="seed-writable-probe", daemon=True).start()
    return finished.wait(_WRITABLE_PROBE_TIMEOUT_SECONDS) and outcome["writable"]


def _frozen_base_candidates() -> list[str]:
    """Return packaged storage candidates in compatibility-first order."""

    explicit = os.environ.get("SEED_DATA_ROOT", "").strip()
    if explicit:
        candidates = [explicit]
    else:
        local = os.environ.get("LOCALAPPDATA", "").strip()
        candidates = [os.path.join(local, _APP_DIR_NAME)] if local else []
        executable_root = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(executable_root, "user_data"))
        candidates.append(os.path.join(tempfile.gettempdir(), _APP_DIR_NAME))

    unique: list[str] = []
    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(os.path.expandvars(candidate)))
        if normalized not in unique:
            unique.append(normalized)
    return unique


def get_writable_base_dir() -> str:
    """Return the writable root used for checkpoints and user data.

    Frozen applications preserve the existing ``%LOCALAPPDATA%\\Taiji`` location
    when it is writable. ``SEED_DATA_ROOT`` may explicitly select another root;
    if the preferred location is blocked by an ACL, the packaged app falls back
    to its ``user_data`` directory and then the system temp directory.
    """

    if not _is_frozen():
        base = os.environ.get("SEED_DATA_ROOT", "").strip() or _PROJECT_ROOT
        os.makedirs(base, exist_ok=True)
        return os.path.abspath(os.path.expanduser(os.path.expandvars(base)))

    global _FROZEN_BASE_CACHE_KEY, _FROZEN_BASE_CACHE_VALUE
    cache_key = (
        os.environ.get("SEED_DATA_ROOT", "").strip(),
        os.environ.get("LOCALAPPDATA", "").strip(),
        os.path.abspath(sys.executable),
        tempfile.gettempdir(),
    )
    if cache_key == _FROZEN_BASE_CACHE_KEY and _FROZEN_BASE_CACHE_VALUE:
        return _FROZEN_BASE_CACHE_VALUE

    candidates = _frozen_base_candidates()

    for candidate in candidates:
        if _probe_writable_directory(candidate):
            _FROZEN_BASE_CACHE_KEY = cache_key
            _FROZEN_BASE_CACHE_VALUE = candidate
            return candidate

    raise OSError(
        "No writable Seed data root available; checked: "
        + ", ".join(candidates)
    )


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
