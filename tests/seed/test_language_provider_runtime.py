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


def test_structured_provider_is_the_default_and_clears_external_metadata() -> None:
    adapter = TSKV8Adapter()
    status, provider_runtime = activate_language_provider(adapter, LanguageProviderConfig())

    assert provider_runtime is None
    assert status.state == "active"
    assert status.backend_id == "structured-stub"
    assert adapter.language_provider_artifact is None
    assert adapter.native_checkpoint()["components"]["language_organ"]["backend"] == (
        "structured-stub"
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
    assert status.rollback == "structured-stub"
    assert adapter.language_provider_artifact is None
    assert adapter.native_checkpoint()["components"]["language_organ"]["backend"] == (
        "structured-stub"
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
