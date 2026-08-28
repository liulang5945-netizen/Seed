from __future__ import annotations

from pathlib import Path

from seed import LanguageProviderConfig, SeedConfig
from seed.language_provider import activate_language_provider
from taiji import TSKV8Adapter


def test_seed_config_round_trips_provider_selection_and_environment_override(monkeypatch) -> None:
    configured = SeedConfig(
        language_provider=LanguageProviderConfig(
            mode="guarded",
            model_dir="models/qwen",
            adapter_dir="providers/qwen-lora",
            safety_report="reports/safety.json",
        )
    )

    restored = SeedConfig.from_dict(configured.to_dict())
    assert restored == configured

    monkeypatch.setenv("SEED_LANGUAGE_MODE", "structured")
    selected = LanguageProviderConfig.from_environment(configured.language_provider)
    assert selected.mode == "structured"
    assert selected.model_dir == configured.language_provider.model_dir


def test_unversioned_structured_default_migrates_to_native_readable() -> None:
    restored = LanguageProviderConfig.from_dict(
        {
            "mode": "structured",
            "provider": "qwen",
            "backend_id": "qwen2.5-0.5b-instruct",
        }
    )
    explicit = LanguageProviderConfig.from_dict(
        {
            "config_version": 2,
            "mode": "structured",
        }
    )

    assert restored.config_version == 2
    assert restored.mode == "native"
    assert explicit.mode == "structured"


def test_native_readable_provider_is_the_default_and_clears_external_metadata() -> None:
    adapter = TSKV8Adapter()
    status, provider_runtime = activate_language_provider(adapter, LanguageProviderConfig())

    assert provider_runtime is None
    assert status.state == "active"
    assert status.backend_id == "native-readable"
    assert adapter.language_provider_artifact is None
    assert adapter.native_checkpoint()["components"]["language_organ"]["backend"] == (
        "native-readable"
    )


def test_missing_explicit_provider_rolls_back_and_is_observable() -> None:
    adapter = TSKV8Adapter()
    missing_root = Path(__file__).resolve().parents[2] / "__provider_missing_test_root__"
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir=str(missing_root / "missing-model"),
        adapter_dir=str(missing_root / "missing-adapter"),
        safety_report=str(missing_root / "missing-safety.json"),
    )

    status, provider_runtime = activate_language_provider(adapter, config)

    assert provider_runtime is None
    assert status.state == "fallback"
    assert status.reason_code == "provider_missing"
    assert status.rollback == "native-readable"
    assert adapter.language_provider_artifact is None
    assert adapter.native_checkpoint()["components"]["language_organ"]["backend"] == (
        "native-readable"
    )


def test_unknown_provider_is_classified_as_manifest_mismatch() -> None:
    adapter = TSKV8Adapter()
    config = LanguageProviderConfig(
        mode="raw",
        provider="unknown-decoder",
        model_dir=str(Path(__file__).resolve().parents[2]),
    )

    status, _ = activate_language_provider(adapter, config)

    assert status.state == "fallback"
    assert status.reason_code == "provider_manifest_mismatch"
