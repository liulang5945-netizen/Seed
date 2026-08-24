"""Taiji FastAPI application factory."""

import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from neuroplex.core.utils import get_external_path, get_internal_path

base_dir = (
    os.path.dirname(sys.executable)
    if getattr(sys, "frozen", False)
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

logger = logging.getLogger("ApiServer")

try:
    from api.middleware.security import RateLimiter, create_rate_limit_middleware

    SECURITY_MIDDLEWARE_AVAILABLE = True
except ImportError:
    logger.warning("Security middleware not available, proceeding without it")
    RateLimiter = None
    create_rate_limit_middleware = None
    SECURITY_MIDDLEWARE_AVAILABLE = False

_global_rate_limiter: Optional[Any] = None


def get_rate_limiter() -> Optional[Any]:
    """Return the shared in-memory rate limiter instance."""
    global _global_rate_limiter
    if SECURITY_MIDDLEWARE_AVAILABLE and _global_rate_limiter is None:
        # 600/min matches the read bucket of the previous in-app rate limiter:
        # the frontend polls /api/runtime/status every few seconds from several
        # components, and a tighter global cap turned its own polling into
        # cascading 429s that also blocked legitimate page requests.
        _global_rate_limiter = RateLimiter(max_requests=600, window_seconds=60)
    return _global_rate_limiter


class JWTAuthMiddleware:
    """JWT auth middleware implemented as pure ASGI."""

    PUBLIC_PATHS = {
        "/api/auth/login",
        "/api/auth/status",
        "/api/runtime/bootstrap",
        "/api/health",
        "/",
    }
    # NOTE: /workspace_data is intentionally public for static file serving.
    # If workspace privacy is required, serve files through a protected endpoint instead.
    PUBLIC_PREFIXES = ("/assets", "/workspace_data", "/ws/")

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in self.PUBLIC_PATHS or any(
            path.startswith(prefix) for prefix in self.PUBLIC_PREFIXES
        ):
            return await self.app(scope, receive, send)

        try:
            from neuroplex.core.security import AuthManager

            auth = AuthManager()
            if not auth.enabled:
                return await self.app(scope, receive, send)

            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode("utf-8")

            async def send_401(message: str):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": json.dumps({"detail": message}).encode("utf-8"),
                    }
                )

            if not auth_header.startswith("Bearer "):
                return await send_401("Unauthorized")

            token = auth_header[7:]
            payload = auth.verify_token(token)
            if not payload:
                return await send_401("Invalid or expired token")

            scope["state"] = scope.get("state", {})
            scope["state"]["user"] = payload
        except ImportError as e:
            # 刻意允许的降级：无认证模块时不启用认证
            logger.debug("【JWTAuthMiddleware.__call__】处理失败（非致命）: %s", e)
        except Exception as exc:
            # fail-closed：非预期异常（如密钥损坏/存储故障）一律拒绝，
            # 不能 warning 后放行，否则认证系统故障 = 全网开放
            logger.error(f"JWT auth exception: {exc}")
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"detail": "Authentication service unavailable"}).encode(
                        "utf-8"
                    ),
                }
            )
            return

        return await self.app(scope, receive, send)


def _load_model_background():
    """Load the model in a background thread."""
    try:
        from neuroplex.tools.builtin_tools import register_all_tools

        register_all_tools()
        # Deep Coupling: 注册引擎间事件订阅
        try:
            from neuroplex.infra.event_subscriptions import register_all_subscriptions

            register_all_subscriptions()
            logger.info("EventBus engine subscriptions registered")
        except Exception as exc:
            logger.warning(f"EventBus subscriptions failed: {exc}")
    except Exception as exc:
        logger.warning(f"Built-in tool registration failed: {exc}")

    # 运行环境选择持久化：上次选了 Seed 原生则直接恢复，不先装 Cortex。
    try:
        from api.routes_model_switch import load_runtime_pref
        from neuroplex.core.app_state import app_state as _state

        pref = load_runtime_pref()
        if pref.get("runtime") == "seed":
            try:
                from api.seed_runtime import activate_seed

                checkpoint = pref.get("checkpoint") or None
                if checkpoint:
                    from pathlib import Path as _Path

                    checkpoint = (
                        _Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        / "checkpoints"
                        / checkpoint
                    )
                runtime = activate_seed(checkpoint)
                _state.mark_started()
                logger.info(f"Seed runtime auto-restored from preference: {runtime.name}")
                return
            except Exception as exc:
                logger.warning(f"Seed preference restore failed, falling back to Cortex: {exc}")
    except Exception as exc:
        logger.warning(f"Runtime preference check failed: {exc}")

    from neuroplex.core.model_loader import load_model_on_startup

    load_model_on_startup()


def get_startup_download_progress():
    """Compatibility helper for startup download progress."""
    from neuroplex.core.model_loader import startup_download_progress

    return startup_download_progress()


def _build_lifespan(startup_tasks: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not startup_tasks:
            yield
            return

        thread = threading.Thread(target=_load_model_background, daemon=True)
        thread.start()
        logger.info("Background model loading started")

        try:
            from neuroplex.core.model_loader import start_auto_reload

            start_auto_reload(check_interval=60)
            logger.info("Model auto reload started")
        except Exception as exc:
            logger.warning(f"Model auto reload startup failed: {exc}")

        try:
            from neuroplex.life.life_scheduler import get_life_scheduler

            scheduler = get_life_scheduler()
            scheduler.start()
            logger.info("Life scheduler started")
        except Exception as exc:
            logger.warning(f"Life scheduler startup failed: {exc}")

        yield

        try:
            from neuroplex.life.life_scheduler import get_life_scheduler

            scheduler = get_life_scheduler()
            scheduler.stop()
            logger.info("Life scheduler stopped")
        except Exception as exc:
            logger.warning(f"Life scheduler shutdown failed: {exc}")

    return lifespan


def _configure_middlewares(app: FastAPI):
    allowed_origins = os.environ.get(
        "TAIJI_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if SECURITY_MIDDLEWARE_AVAILABLE:
        limiter = get_rate_limiter()
        if limiter:
            app.middleware("http")(create_rate_limit_middleware(limiter))
            logger.info("Security middleware integrated")

    app.add_middleware(JWTAuthMiddleware)


def _register_routers(app: FastAPI):
    from .routes_agent import router as agent_router
    from .routes_agent_mcp import router as agent_mcp_router
    from .routes_agent_memory import router as agent_memory_router
    from .routes_agent_workspace import router as agent_workspace_router
    from .routes_auth import router as auth_router
    from .routes_chat import router as chat_router
    from .routes_life import router as life_router
    from .routes_model_switch import router as model_switch_router
    from .routes_models import router as models_router
    from .routes_multimodal import router as multimodal_router
    from .routes_plugins import router as plugins_router
    from .routes_rag import router as rag_router
    from .routes_runtime import router as runtime_router
    from .routes_settings import router as settings_router
    from .routes_system import router as system_router
    from .routes_neuroplex import router as neuroplex_router
    from .routes_neuroplex_model import router as population_compat_router
    from .routes_terminal import router as terminal_router
    from .routes_update import router as update_router
    from .routes_workflows import router as workflows_router
    from .training import router as training_router

    app.include_router(auth_router)
    app.include_router(runtime_router)
    app.include_router(workflows_router)
    app.include_router(plugins_router)
    app.include_router(chat_router)
    app.include_router(training_router)
    app.include_router(rag_router)
    app.include_router(models_router)
    app.include_router(system_router)
    app.include_router(settings_router)
    app.include_router(update_router)
    app.include_router(model_switch_router)
    app.include_router(agent_router)
    app.include_router(agent_workspace_router)
    app.include_router(agent_mcp_router)
    app.include_router(agent_memory_router)
    app.include_router(terminal_router)
    app.include_router(life_router)
    app.include_router(multimodal_router)
    app.include_router(neuroplex_router)
    app.include_router(population_compat_router)


def _mount_static_assets(app: FastAPI):
    external_dist = get_external_path("update_frontend")
    internal_dist = get_internal_path(os.path.join("frontend", "dist"))

    workspace_dir = get_external_path("agent_workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    app.mount("/workspace_data", StaticFiles(directory=workspace_dir), name="workspace_data")

    multimodal_dir = get_external_path(os.path.join("user_data", "multimodal_uploads"))
    os.makedirs(multimodal_dir, exist_ok=True)
    app.mount("/multimodal_media", StaticFiles(directory=multimodal_dir), name="multimodal_media")

    dist_path = (
        external_dist
        if os.path.exists(os.path.join(external_dist, "index.html"))
        else internal_dist
    )
    if not os.path.exists(dist_path):
        return

    assets_path = os.path.join(dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{catchall:path}")
    async def serve_spa(catchall: str):
        if catchall.startswith("api/") or catchall.startswith("ws/"):
            return JSONResponse(
                {"status": "error", "message": f"Endpoint not found: /{catchall}"},
                status_code=404,
            )

        # 路径穿越防护：realpath 解析后必须仍位于 dist 目录内
        dist_root = os.path.realpath(dist_path)
        file_path = os.path.realpath(os.path.join(dist_root, catchall))
        if file_path.startswith(dist_root + os.sep) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_root, "index.html"))


def create_app(*, startup_tasks: bool = True) -> FastAPI:
    """Create the FastAPI application with optional startup side effects."""
    app = FastAPI(title="Taiji API", lifespan=_build_lifespan(startup_tasks))
    _configure_middlewares(app)
    _register_routers(app)
    _mount_static_assets(app)
    return app


app = create_app()
