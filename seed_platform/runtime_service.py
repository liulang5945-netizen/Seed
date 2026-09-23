"""Runtime status service.

Real implementation of the service previously stubbed here. Aggregates
the health / memory / auth / life / tools / training sections into the
single payload the client shell trusts (contract: ``api.models_runtime.
RuntimeStatusPayload``). Every section is collected defensively — a
failing section degrades to its default instead of taking the whole
endpoint down with a 500.
"""

import logging
import time

logger = logging.getLogger(__name__)


def _health_section() -> dict:
    from seed_platform.app_state import app_state

    health = {
        "state": "connected",
        "message": "",
        "model_loaded": False,
        "model_name": "",
        "is_taiji": False,
        "is_seed": False,
        "startup_complete": bool(getattr(app_state, "startup_complete", False)),
        "startup_error": "",
    }

    seed_active = False
    try:
        from api.seed_runtime import get_seed_runtime, is_seed_active

        seed_active = is_seed_active()
        if seed_active:
            runtime = get_seed_runtime()
            if runtime is not None:
                health["model_name"] = runtime.name
                health["language_provider"] = runtime.language_provider_status
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"runtime_service: seed status unavailable: {exc}")

    model = getattr(app_state, "model", None)
    health["model_loaded"] = model is not None or seed_active
    health["is_seed"] = seed_active
    try:
        # Seed is the product's Taiji-native runtime.  The old app_state flag
        # only describes the optional Legacy/Cortex object and therefore cannot
        # be the sole source for the public native identity.
        health["is_taiji"] = seed_active or bool(app_state.is_taiji())
    except Exception as e:
        logger.debug("【_health_section】处理失败（非致命）: %s", e)
    if not health["model_name"]:
        health["model_name"] = getattr(app_state, "_loaded_model_name", "") or ""

    if not health["startup_complete"]:
        health["state"] = "loading"
        health["message"] = "模型正在加载中..."
    elif getattr(app_state, "startup_error", None):
        health["state"] = "error"
        health["message"] = str(app_state.startup_error)
        health["startup_error"] = str(app_state.startup_error)
    elif not health["model_loaded"]:
        health["message"] = "后端在线，模型尚未装载"

    return health


def _memory_section() -> dict:
    # psutil is not a project dependency; probe the OS directly.
    try:
        import platform

        if platform.system() == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / 1024**3
            available_gb = stat.ullAvailPhys / 1024**3
            used_pct = float(stat.dwMemoryLoad)
        else:
            info = {}
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    info[key.strip()] = int(rest.split()[0])  # kB
            total_gb = info.get("MemTotal", 0) / 1024**2
            available_gb = info.get("MemAvailable", 0) / 1024**2
            used_pct = 100.0 * (1.0 - available_gb / total_gb) if total_gb else 0.0
        return {
            "status": "ok",
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "used_pct": round(used_pct, 1),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"runtime_service: memory probe failed: {exc}")
        return {"status": "unknown", "total_gb": 0.0, "available_gb": 0.0, "used_pct": 0.0}


def _auth_section(auth_header: str) -> dict:
    section = {
        "enabled": False,
        "authenticated": True,
        "token_valid": False,
        "username": "",
        "has_password": False,
    }
    try:
        from seed_platform.auth import AuthManager

        auth = AuthManager()
        section["enabled"] = bool(auth.enabled)
        section["username"] = getattr(auth, "username", "") or ""
        section["has_password"] = bool(getattr(auth, "password_hash", None))
        if auth.enabled:
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            payload = auth.verify_token(token) if token else None
            section["token_valid"] = bool(payload)
            section["authenticated"] = bool(payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"runtime_service: auth status unavailable: {exc}")
    return section


def _scaled_organ_values(values: dict | None) -> dict[str, float]:
    """Scale every 0..1 organ value by one factor to the payload's 0..100.

    Needs and drives come from the same organ in the same units, so they share
    this single transform: no dimension is invented, none is dropped, and no
    second unit enters the payload.
    """

    return {str(name): round(float(value) * 100.0, 2) for name, value in (values or {}).items()}


def _native_life_section() -> dict | None:
    """Life data measured by the Taiji homeostasis organ, or ``None``.

    Returns ``None`` when no native runtime is active so the caller can fall
    back to Legacy.  A detached organ reports nothing beyond ``status``,
    ``is_running`` and ``needs``: a tick it never reached stays a schema
    default instead of being reported as a measured zero.
    """

    try:
        from api.seed_runtime import get_seed_runtime

        runtime = get_seed_runtime()
        if runtime is None:
            return None
        homeostasis = runtime.homeostasis_status()
    except Exception as exc:  # pragma: no cover - defensive status boundary
        logger.warning(f"runtime_service: native life status unavailable: {exc}")
        return None
    if not homeostasis.get("attached"):
        return {
            "status": "seed",
            "is_running": False,
            "needs": {},
        }
    return {
        "status": "seed",
        "is_running": True,
        "needs": _scaled_organ_values(homeostasis.get("needs")),
        "drives": _scaled_organ_values(homeostasis.get("drives")),
        "mode": str(homeostasis.get("mode") or ""),
        "tick": int(homeostasis.get("tick", 0)),
    }


def _life_section() -> dict:
    from seed_platform.dependencies import legacy_requested

    native = _native_life_section()
    if native is not None:
        return native
    if not legacy_requested():
        return {
            "status": "seed",
            "is_running": False,
            "needs": {},
        }
    try:
        from neuroplex.life.life_scheduler import get_life_scheduler

        scheduler = get_life_scheduler()
        status = scheduler.get_status()
        needs = scheduler.needs.to_dict() if hasattr(scheduler, "needs") else {}
        # Report only what the scheduler tracks.  The previous shape also
        # published `total_interactions`/`uptime_seconds`, which no scheduler
        # ever measured, so both read as a permanent zero on the panel.
        return {
            "status": "ok",
            "is_running": bool(status.get("is_running", False)),
            "needs": needs,
            "life_state": str(status.get("life_state") or ""),
            "dominant_need": str(status.get("dominant_need") or ""),
            "total_heartbeats": int(status.get("total_heartbeats", 0)),
            "total_events": int(status.get("total_events", 0)),
            "last_heartbeat": status.get("last_heartbeat"),
            "last_activity": status.get("last_activity"),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"runtime_service: life status unavailable: {exc}")
        return {
            "status": "unknown",
            "is_running": False,
            "needs": {},
        }


def _tools_section() -> dict:
    """Expose the Seed capability registry, independent of Legacy tools."""

    try:
        from seed_platform.workbench import CapabilitySnapshot

        snapshot = CapabilitySnapshot.default()
        tools = [
            {
                "name": capability.capability_id,
                "description": capability.description,
                "parameters": dict(capability.parameters),
                "source": capability.source,
                "source_id": snapshot.snapshot_id,
                "category": capability.category,
                "enabled": capability.enabled,
            }
            for capability in snapshot.capabilities
        ]
        return {
            "status": "ok",
            "tools": tools,
            "count": len(tools),
            "error": "",
            "snapshot_id": snapshot.snapshot_id,
            "revision": snapshot.revision,
            "source": "seed_platform.workbench.CapabilitySnapshot",
            "owner": "Taiji native Workbench",
            "observed_at": int(time.time()),
        }
    except Exception as exc:  # pragma: no cover - defensive status boundary
        logger.warning(f"runtime_service: Seed capability status failed: {exc}")
        return {
            "status": "error",
            "tools": [],
            "count": 0,
            "error": str(exc),
            "snapshot_id": "",
            "revision": 0,
            "source": "seed_platform.workbench.CapabilitySnapshot",
            "owner": "Taiji native Workbench",
            "observed_at": int(time.time()),
        }


def _training_section() -> dict:
    section = {
        "is_training": False,
        "publishing": False,
        "pause_requested": False,
        "stop_requested": False,
    }
    try:
        from seed_platform.app_state import app_state

        section["is_training"] = bool(getattr(app_state, "is_training", False))
        section["publishing"] = bool(getattr(app_state, "publishing", False))
        # These two request flags live on the singleton under their own names;
        # the previous shape never read them, so both read as a permanent False.
        section["pause_requested"] = bool(getattr(app_state, "pause_training_requested", False))
        section["stop_requested"] = bool(getattr(app_state, "stop_training_requested", False))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【_training_section】处理失败（非致命）: %s", e)
    return section


def get_runtime_status(auth_header: str = "") -> dict:
    """Full runtime status payload (contract: RuntimeStatusPayload)."""
    return {
        "status": "ok",
        "timestamp": int(time.time()),
        "health": _health_section(),
        "memory": _memory_section(),
        "auth": _auth_section(auth_header or ""),
        "life": _life_section(),
        "tools": _tools_section(),
        "training": _training_section(),
    }


def get_bootstrap_status() -> dict:
    """Public, unauthenticated status for first contact."""
    startup_complete = False
    startup_error = ""
    try:
        from seed_platform.app_state import app_state

        startup_complete = bool(getattr(app_state, "startup_complete", False))
        startup_error = str(getattr(app_state, "startup_error", "") or "")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【get_bootstrap_status】处理失败（非致命）: %s", e)

    auth_enabled = False
    try:
        from seed_platform.auth import AuthManager

        auth_enabled = bool(AuthManager().enabled)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【get_bootstrap_status】处理失败（非致命）: %s", e)

    return {
        "alive": True,
        "auth_enabled": auth_enabled,
        "need_login": False,
        "startup_complete": startup_complete,
        "startup_error": startup_error,
    }
