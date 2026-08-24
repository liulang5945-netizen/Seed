from __future__ import annotations

from seed_platform.dependencies import (
    CORE_DEPENDENCIES,
    LEGACY_DEPENDENCIES,
    dependency_manifest,
    legacy_requested,
)


def test_core_manifest_excludes_transformer_and_legacy_packages():
    assert "transformers" not in CORE_DEPENDENCIES
    assert "transformers" in LEGACY_DEPENDENCIES
    assert "cryptography" in CORE_DEPENDENCIES
    assert not set(CORE_DEPENDENCIES).intersection(LEGACY_DEPENDENCIES)


def test_legacy_manifest_is_opt_in(monkeypatch):
    monkeypatch.setenv("SEED_ENABLE_LEGACY", "0")
    assert legacy_requested() is False
    assert set(dependency_manifest()) == set(CORE_DEPENDENCIES)

    monkeypatch.setenv("SEED_ENABLE_LEGACY", "1")
    assert legacy_requested() is True
    manifest = dependency_manifest(include_legacy=True)
    assert set(LEGACY_DEPENDENCIES).issubset(manifest)
