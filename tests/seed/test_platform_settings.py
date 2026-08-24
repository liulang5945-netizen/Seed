from __future__ import annotations

import ast
from pathlib import Path

import pytest

import seed_platform.settings as platform_settings


def test_settings_roundtrip_is_persistent_and_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        platform_settings,
        "get_external_path",
        lambda relative_path: str(tmp_path / relative_path),
    )

    platform_settings.save_settings({"theme": "dark", "nested": {"enabled": True}})
    platform_settings.update_settings({"device": "cuda"})

    assert platform_settings.get_setting("theme") == "dark"
    assert platform_settings.get_setting("device") == "cuda"
    assert platform_settings.load_settings()["nested"] == {"enabled": True}
    assert (tmp_path / "data" / "app_settings.json").exists()
    assert not list((tmp_path / "data").glob("*.tmp"))


def test_legacy_settings_import_is_compatibility_export():
    from neuroplex.services.settings_service import load_settings as legacy_load_settings

    assert legacy_load_settings is platform_settings.load_settings


@pytest.mark.parametrize(
    "module_path",
    [
        "api/routes_agent_workspace.py",
        "api/routes_settings.py",
        "api/routes_terminal.py",
    ],
)
def test_api_routes_use_platform_settings(module_path: str):
    source = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "seed_platform.settings" in imported_names
    assert "neuroplex.services.settings_service" not in imported_names
