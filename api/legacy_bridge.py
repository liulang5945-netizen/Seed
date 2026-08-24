"""Single product-shell boundary around the frozen Neuroplex/Cortex runtime.

No native Seed/Taiji module imports this bridge.  Product code may call these
functions while platform services and routes are migrated out of ``neuroplex``;
all direct Cortex lifecycle and explicit Cortex route imports stay here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI


logger = logging.getLogger("ApiServer.LegacyBridge")
_life_scheduler: Any | None = None


def get_legacy_auth_manager() -> Any:
    """Return the current auth manager until security moves to seed_platform."""

    from neuroplex.core.security import AuthManager

    return AuthManager()


def load_legacy_runtime() -> None:
    """Restore the selected runtime, falling back to the frozen Cortex path."""

    try:
        from neuroplex.tools.builtin_tools import register_all_tools

        register_all_tools()
        try:
            from neuroplex.infra.event_subscriptions import register_all_subscriptions

            register_all_subscriptions()
            logger.info("EventBus engine subscriptions registered")
        except Exception as exc:
            logger.warning("EventBus subscriptions failed: %s", exc)
    except Exception as exc:
        logger.warning("Built-in tool registration failed: %s", exc)

    try:
        from api.routes_model_switch import load_runtime_pref
        from neuroplex.core.app_state import app_state

        pref = load_runtime_pref()
        if pref.get("runtime") == "seed":
            try:
                from api.seed_runtime import activate_seed

                checkpoint = pref.get("checkpoint") or None
                if checkpoint:
                    checkpoint = (
                        Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        / "checkpoints"
                        / checkpoint
                    )
                runtime = activate_seed(checkpoint)
                app_state.mark_started()
                logger.info("Seed runtime auto-restored from preference: %s", runtime.name)
                return
            except Exception as exc:
                logger.warning("Seed preference restore failed, falling back to Cortex: %s", exc)
    except Exception as exc:
        logger.warning("Runtime preference check failed: %s", exc)

    from neuroplex.core.model_loader import load_model_on_startup

    load_model_on_startup()


def legacy_startup_download_progress() -> dict[str, Any]:
    from neuroplex.core.model_loader import startup_download_progress

    return startup_download_progress()


def start_legacy_services() -> None:
    """Start Cortex auto-reload and life scheduling services."""

    global _life_scheduler
    try:
        from neuroplex.core.model_loader import start_auto_reload

        start_auto_reload(check_interval=60)
        logger.info("Model auto reload started")
    except Exception as exc:
        logger.warning("Model auto reload startup failed: %s", exc)

    try:
        from neuroplex.life.life_scheduler import get_life_scheduler

        _life_scheduler = get_life_scheduler()
        _life_scheduler.start()
        logger.info("Life scheduler started")
    except Exception as exc:
        logger.warning("Life scheduler startup failed: %s", exc)


def stop_legacy_services() -> None:
    """Stop services previously started by :func:`start_legacy_services`."""

    global _life_scheduler
    if _life_scheduler is None:
        return
    try:
        _life_scheduler.stop()
        logger.info("Life scheduler stopped")
    except Exception as exc:
        logger.warning("Life scheduler shutdown failed: %s", exc)
    finally:
        _life_scheduler = None


def register_legacy_routers(app: FastAPI) -> None:
    """Register routes whose contracts are explicitly Cortex-specific."""

    from .routes_life import router as life_router
    from .routes_multimodal import router as multimodal_router
    from .routes_neuroplex import router as neuroplex_router
    from .routes_neuroplex_model import router as population_compat_router

    app.include_router(life_router)
    app.include_router(multimodal_router)
    app.include_router(neuroplex_router)
    app.include_router(population_compat_router)
