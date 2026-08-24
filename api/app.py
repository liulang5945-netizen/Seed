"""Taiji FastAPI application factory."""

import importlib
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

from api.legacy_bridge import (
    get_legacy_auth_manager,
    legacy_available,
    legacy_startup_download_progress,
    load_legacy_runtime,
    register_legacy_routers,
    start_legacy_services,
    stop_legacy_services,
)
from seed_platform.paths import get_external_path, get_internal_path

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
            auth = get_legacy_auth_manager()
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
    """Compatibility wrapper around the single Legacy runtime boundary."""

    load_legacy_runtime()


def get_startup_download_progress():
    """Compatibility helper for startup download progress."""

    return legacy_startup_download_progress()


def _build_lifespan(startup_tasks: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not startup_tasks:
            yield
            return

        thread = threading.Thread(target=_load_model_background, daemon=True)
        thread.start()
        logger.info("Background model loading started")

        start_legacy_services()

        yield
        stop_legacy_services()

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


def _load_optional_router(module_name: str):
    """Load a Legacy-dependent router only when the plugin is available."""

    if not legacy_available():
        logger.info("Legacy plugin disabled; skipped router %s", module_name)
        return None
    try:
        module = importlib.import_module(f"api.{module_name}")
    except ModuleNotFoundError as exc:
        logger.warning("Optional router %s unavailable: %s", module_name, exc)
        return None
    return getattr(module, "router", None)


def _register_routers(app: FastAPI):
    agent_router = _load_optional_router("routes_agent")
    from .routes_agent_mcp import router as agent_mcp_router
    from .routes_agent_memory import router as agent_memory_router
    from .routes_agent_workspace import router as agent_workspace_router
    from .routes_auth import router as auth_router
    from .routes_chat import router as chat_router
    from .routes_model_switch import router as model_switch_router
    from .routes_models import router as models_router

    plugins_router = _load_optional_router("routes_plugins")
    rag_router = _load_optional_router("routes_rag")
    from .routes_runtime import router as runtime_router
    from .routes_settings import router as settings_router
    from .routes_system import router as system_router
    from .routes_terminal import router as terminal_router
    from .routes_update import router as update_router

    workflows_router = _load_optional_router("routes_workflows")
    from .training import router as training_router

    app.include_router(auth_router)
    app.include_router(runtime_router)
    for optional_router in (workflows_router, plugins_router):
        if optional_router is not None:
            app.include_router(optional_router)
    app.include_router(chat_router)
    app.include_router(training_router)
    if rag_router is not None:
        app.include_router(rag_router)
    app.include_router(models_router)
    app.include_router(system_router)
    app.include_router(settings_router)
    app.include_router(update_router)
    app.include_router(model_switch_router)
    if agent_router is not None:
        app.include_router(agent_router)
    app.include_router(agent_workspace_router)
    app.include_router(agent_mcp_router)
    app.include_router(agent_memory_router)
    app.include_router(terminal_router)
    register_legacy_routers(app)


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

    # include_in_schema=False：SPA 兜底路由不是 API 契约的一部分，且其注册取决于
    # frontend/dist 是否已构建（该目录被 gitignore）。若纳入 schema，OpenAPI 快照会
    # 随构建产物存在与否漂移——本地绿、CI 红。
    @app.get("/{catchall:path}", include_in_schema=False)
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
