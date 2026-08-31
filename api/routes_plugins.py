"""Retired compatibility surface for the former Legacy plugin manager.

Client extensions are now owned by Seed and must be changed through the
declarative, content-addressed ``/api/client-extensions`` surface. Keeping
these routes as explicit tombstones prevents old callers from silently
reaching the removed ``neuroplex`` plugin manager.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


def _retired() -> None:
    raise HTTPException(
        status_code=410,
        detail={
            "code": "legacy_plugin_surface_retired",
            "replacement": "/api/client-extensions",
            "message": "Legacy 插件入口已退役，请使用 Seed-owned client extension snapshot",
        },
    )


@router.get("/api/plugins")
def list_plugins() -> None:
    _retired()


@router.post("/api/plugins/{plugin_id}/enable")
def enable_plugin(plugin_id: str) -> None:
    del plugin_id
    _retired()


@router.post("/api/plugins/{plugin_id}/disable")
def disable_plugin(plugin_id: str) -> None:
    del plugin_id
    _retired()


@router.delete("/api/plugins/{plugin_id}")
def uninstall_plugin(plugin_id: str) -> None:
    del plugin_id
    _retired()


@router.post("/api/plugins/install")
def install_plugin() -> None:
    _retired()


@router.get("/api/plugins/marketplace")
def plugin_marketplace() -> None:
    _retired()


@router.post("/api/plugins/marketplace/refresh")
def refresh_plugin_marketplace() -> None:
    _retired()


@router.post("/api/plugins/upload")
def upload_plugin() -> None:
    _retired()
