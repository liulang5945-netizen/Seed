from __future__ import annotations

import importlib.util

import pytest
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


def test_legacy_plugin_requires_runtime_dependency(monkeypatch):
    monkeypatch.setenv("SEED_ENABLE_LEGACY", "1")
    real_find_spec = importlib.util.find_spec

    def find_spec_without_sentencepiece(name, *args, **kwargs):
        if name == "sentencepiece":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(legacy_bridge.importlib.util, "find_spec", find_spec_without_sentencepiece)

    assert legacy_bridge.legacy_available() is False


def test_legacy_cli_loader_is_gated(monkeypatch):
    monkeypatch.setenv("SEED_ENABLE_LEGACY", "0")

    with pytest.raises(RuntimeError, match="Legacy Cortex is unavailable"):
        legacy_bridge.load_legacy_cortex("data/neurons", "cpu")
