from __future__ import annotations

from pathlib import Path

from seed import LanguageProviderConfig, SeedConfig
from seed.language_provider import activate_language_provider
from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    TSKV8Adapter,
)


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
    assert status.to_dict()["chat_enabled"] == "false"
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


def _passing_realization_gate_report() -> dict[str, object]:
    split = {
        "passed": True,
        "output_nonempty_rate": 1.0,
        "readable_rate": 1.0,
        "required_term_coverage": 1.0,
        "structured_leakage_free_rate": 1.0,
        "fallback_rate": 0.0,
    }
    return {
        "format": "taiji-language-realization-gate-v1",
        "corpus": {"round_trip": True, "split_disjoint": True},
        "train": split,
        "holdout": split,
        "rollback": {"checked": True, "outputs_match_reference": True},
        "checkpoint": {"checked": True, "outputs_match": True},
        "gate": {"passed": True},
    }


def test_product_chat_requires_a_passing_realization_and_safety_gate(monkeypatch) -> None:
    reports = {
        "training": {
            "training": {"training_applied": True},
            "expression_to_text_gate": _passing_realization_gate_report(),
        },
        "safety": {
            "adapted": {"safe_realization_rate": 1.0},
            "rollback": {"outputs_match_raw": True},
            "gate": {"passed": True},
        },
    }
    monkeypatch.setattr(
        "seed.language_provider._load_product_chat_report",
        lambda path, label: reports[label],
    )
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
    )

    class _Decoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del prompt, max_tokens, temperature
            return "已完成。"

    def fake_loader(adapter, artifact, **kwargs):
        del kwargs
        registry = LanguageBackendRegistry.default()
        registry.register(
            LanguageBackendSpec(
                backend_id=artifact.backend_id,
                family="external-causal-decoder-guarded",
                training_contract="expression-to-text-v1",
            )
        )
        adapter.attach_language_backend_registry(registry)
        adapter.attach_language_provider_artifact(artifact)
        adapter.attach_language_organ(
            ExternalTextDecoderLanguageOrgan(
                _Decoder(),
                prompt_builder=lambda expression: expression.content_id,
                backend_id=artifact.backend_id,
            )
        )
        return _Decoder()

    monkeypatch.setattr("seed.language_provider.load_qwen_language_provider", fake_loader)
    adapter = TSKV8Adapter()

    status, runtime = activate_language_provider(adapter, config)

    assert runtime is not None
    assert status.state == "active"
    assert status.chat_enabled is True
    assert adapter.language_organ is not None
    assert adapter.language_organ.backend_id == config.backend_id


def test_product_chat_rejects_legacy_provider_report(monkeypatch) -> None:
    monkeypatch.setattr(
        "seed.language_provider._load_product_chat_report",
        lambda path, label: (
            {"training": {"training_applied": True}}
            if label == "training"
            else {
                "adapted": {"safe_realization_rate": 1.0},
                "rollback": {"outputs_match_raw": True},
                "gate": {"passed": True},
            }
        ),
    )
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
    )

    status, runtime = activate_language_provider(TSKV8Adapter(), config)

    assert runtime is None
    assert status.state == "fallback"
    assert status.reason_code == "chat_gate_failed"
    assert status.chat_enabled is False
