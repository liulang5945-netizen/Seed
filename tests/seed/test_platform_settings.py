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


def test_legacy_model_settings_are_quarantined_without_guessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        platform_settings,
        "get_external_path",
        lambda relative_path: str(tmp_path / relative_path),
    )

    platform_settings.save_settings(
        {
            "model_type": "gguf",
            "gguf_path": "models/old.gguf",
            "model_name": "ambiguous-old-model",
        }
    )

    migrated = platform_settings.load_settings()
    assert migrated["schema_version"] == 2
    assert migrated["runtime"] == {"kind": "taiji", "checkpoint_id": ""}
    assert "model_type" not in migrated
    entry = migrated["migration"]["quarantined"]["legacy_model_settings_v1"]
    assert entry["reason"] == "ambiguous_or_legacy_model_semantics_not_activated"
    assert entry["values"]["gguf_path"] == "models/old.gguf"


def test_native_marker_migrates_only_a_safe_checkpoint_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        platform_settings,
        "get_external_path",
        lambda relative_path: str(tmp_path / relative_path),
    )

    platform_settings.save_settings({"model_type": "self", "checkpoint_path": "seed_native.pt"})

    migrated = platform_settings.load_settings()
    assert migrated["runtime"] == {
        "kind": "taiji",
        "checkpoint_id": "seed_native.pt",
    }
    assert "legacy_self_marker_v1" in migrated["migration"]["converted"]


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
