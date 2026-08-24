from __future__ import annotations

from fastapi import FastAPI

from api import legacy_bridge


def test_legacy_plugin_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SEED_ENABLE_LEGACY", "0")

    assert legacy_bridge.legacy_available() is False

    app = FastAPI()
    route_count = len(app.routes)
    legacy_bridge.register_legacy_routers(app)
    assert len(app.routes) == route_count


def test_disabled_legacy_progress_reports_seed_runtime(monkeypatch):
    monkeypatch.setenv("SEED_ENABLE_LEGACY", "0")

    progress = legacy_bridge.legacy_startup_download_progress()

    assert progress["status"] == "seed"
    assert progress["active"] is False
