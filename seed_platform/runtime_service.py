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
        health["is_taiji"] = bool(app_state.is_taiji())
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


def _life_section() -> dict:
    from seed_platform.dependencies import legacy_requested

    if not legacy_requested():
        return {
            "status": "seed",
            "is_running": False,
            "needs": {},
            "total_interactions": 0,
            "uptime_seconds": 0,
        }
    try:
        from neuroplex.life.life_scheduler import get_life_scheduler

        scheduler = get_life_scheduler()
        status = scheduler.get_status()
        needs = scheduler.needs.to_dict() if hasattr(scheduler, "needs") else {}
        return {
            "status": "ok",
            "is_running": bool(status.get("is_running", False)),
            "needs": needs,
            "total_interactions": int(status.get("total_interactions", 0)),
            "uptime_seconds": int(status.get("uptime_seconds", 0)),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"runtime_service: life status unavailable: {exc}")
        return {
            "status": "unknown",
            "is_running": False,
            "needs": {},
            "total_interactions": 0,
            "uptime_seconds": 0,
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
        return {"status": "ok", "tools": tools, "count": len(tools), "error": ""}
    except Exception as exc:  # pragma: no cover - defensive status boundary
        logger.warning(f"runtime_service: Seed capability status failed: {exc}")
        return {"status": "error", "tools": [], "count": 0, "error": str(exc)}


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
        section["publishing"] = bool(getattr(app_state, "is_publishing", False))
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
