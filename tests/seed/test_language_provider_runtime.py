from __future__ import annotations

from pathlib import Path

from seed import LanguageProviderConfig, SeedConfig
from seed.language_provider import (
    _verify_product_chat_artifact,
    activate_language_provider,
    auto_rollback_language_provider,
    build_provider_artifact,
    observe_language_provider,
    rotate_language_provider,
)
from taiji import (
    ExpressionPlan,
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageEmission,
    LanguageProviderArtifactRegistry,
    TSKV8Adapter,
    language_provider_content_digest,
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
    anchor = Path(__file__).resolve()
    anchor_digest = language_provider_content_digest(anchor)
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir=str(anchor),
        adapter_dir=str(anchor),
        training_corpus=str(anchor),
        training_report=str(anchor),
        safety_report=str(anchor),
        content_digests=(
            ("base_model", anchor_digest),
            ("adapter", anchor_digest),
            ("training_corpus", anchor_digest),
            ("training_report", anchor_digest),
            ("safety_report", anchor_digest),
        ),
        expires_at=4_000_000_000.0,
        chat_enabled=True,
    )

    class _Decoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return "数据库运行正常。"
            return "接口已经恢复。"

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
    anchor = Path(__file__).resolve()
    anchor_digest = language_provider_content_digest(anchor)
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir=str(anchor),
        adapter_dir=str(anchor),
        training_corpus=str(anchor),
        training_report=str(anchor),
        safety_report=str(anchor),
        content_digests=(
            ("base_model", anchor_digest),
            ("adapter", anchor_digest),
            ("training_corpus", anchor_digest),
            ("training_report", anchor_digest),
            ("safety_report", anchor_digest),
        ),
        expires_at=4_000_000_000.0,
        chat_enabled=True,
    )

    status, runtime = activate_language_provider(TSKV8Adapter(), config)

    assert runtime is None
    assert status.state == "fallback"
    assert status.reason_code == "chat_gate_failed"
    assert status.chat_enabled is False


def test_product_chat_artifact_rejects_content_drift_and_expiry() -> None:
    anchor = Path(__file__).resolve()
    digest = language_provider_content_digest(anchor)
    artifact = LanguageProviderConfig(
        mode="guarded",
        model_dir=str(anchor),
        adapter_dir=str(anchor),
        training_corpus=str(anchor),
        training_report=str(anchor),
        safety_report=str(anchor),
        content_digests=(
            ("base_model", digest),
            ("adapter", digest),
            ("training_corpus", digest),
            ("training_report", digest),
            ("safety_report", digest),
        ),
        expires_at=4_000_000_000.0,
        chat_enabled=True,
    )
    from seed.language_provider import build_provider_artifact

    manifest = build_provider_artifact(artifact)
    _verify_product_chat_artifact(manifest, artifact_root=None, now=0.0)

    drifted = manifest.__class__(
        **{
            **manifest.__dict__,
            "content_digests": tuple(
                ("base_model", "0" * 64) if role == "base_model" else (role, value)
                for role, value in manifest.content_digests
            ),
            "artifact_digest": "",
        }
    )
    try:
        _verify_product_chat_artifact(drifted, artifact_root=None, now=0.0)
    except ValueError as exc:
        assert "digest drifted" in str(exc)
    else:
        raise AssertionError("content drift must fail closed")

    expired = manifest.__class__(**{**manifest.__dict__, "expires_at": 0.0, "artifact_digest": ""})
    try:
        _verify_product_chat_artifact(expired, artifact_root=None, now=0.0)
    except ValueError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("expired artifact must fail closed")


def test_product_chat_first_chat_canary_failure_rolls_back(monkeypatch) -> None:
    monkeypatch.setattr(
        "seed.language_provider._verify_product_chat_artifact", lambda *args, **kwargs: {}
    )
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
    config = LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
    )

    adapter = TSKV8Adapter()
    status, runtime = activate_language_provider(adapter, config)

    assert runtime is None
    assert status.reason_code == "chat_canary_failed"
    assert status.chat_enabled is False
    assert adapter.language_organ is not None
    assert adapter.language_organ.backend_id == "native-readable"


def test_provider_rotation_stages_canary_and_preserves_live_version_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "seed.language_provider._verify_product_chat_artifact", lambda *args, **kwargs: {}
    )
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

    class _Decoder:
        def __init__(self, artifact_id: str) -> None:
            self.artifact_id = artifact_id

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if self.artifact_id == "provider-bad":
                return "已完成。"
            if "database-status" in prompt:
                return f"{self.artifact_id} 数据库运行正常。"
            return f"{self.artifact_id} 接口已经恢复。"

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
                _Decoder(artifact.artifact_id),
                prompt_builder=lambda expression: expression.content_id,
                backend_id=artifact.backend_id,
            )
        )
        return _Decoder(artifact.artifact_id)

    monkeypatch.setattr("seed.language_provider.load_qwen_language_provider", fake_loader)

    def config(artifact_id: str) -> LanguageProviderConfig:
        return LanguageProviderConfig(
            mode="guarded",
            model_dir="model",
            adapter_dir="adapter",
            artifact_id=artifact_id,
            training_corpus="corpus.json",
            training_report="training.json",
            safety_report="safety.json",
            chat_enabled=True,
        )

    first_config = config("provider-v1")
    second_config = config("provider-v2")
    bad_config = config("provider-bad")
    first = build_provider_artifact(first_config)
    second = build_provider_artifact(second_config)
    bad = build_provider_artifact(bad_config)
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(first, allow=True)
        .with_artifact(second, allow=True)
        .with_artifact(bad, allow=True)
        .activate(first.artifact_id)
    )
    adapter = TSKV8Adapter()
    first_status, first_runtime = activate_language_provider(adapter, first_config)
    assert first_status.state == "active"

    rotated = rotate_language_provider(
        adapter,
        registry,
        second_config,
        current_runtime=first_runtime,
    )
    assert rotated.committed is True
    assert rotated.registry.active_artifact_id == second.artifact_id
    assert rotated.registry.previous_artifact_id == first.artifact_id
    assert adapter.language_provider_artifact == second
    assert rotated.runtime is not first_runtime

    rejected = rotate_language_provider(
        adapter,
        rotated.registry,
        bad_config,
        current_runtime=rotated.runtime,
    )
    assert rejected.committed is False
    assert rejected.status.reason_code == "chat_canary_failed"
    assert rejected.runtime is rotated.runtime
    assert adapter.language_provider_artifact == second
    assert adapter.language_provider_artifact_registry.active_artifact_id == second.artifact_id
    assert adapter.language_provider_artifact_registry.previous_artifact_id == first.artifact_id
    assert (
        adapter.native_checkpoint()["components"]["language_provider_artifact_registry"][
            "active_artifact_id"
        ]
        == second.artifact_id
    )


def _install_guarded_provider_stubs(monkeypatch) -> None:
    """Make guarded activation depend only on the decoder, not on real artifacts."""

    monkeypatch.setattr(
        "seed.language_provider._verify_product_chat_artifact", lambda *args, **kwargs: {}
    )
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

    class _Decoder:
        def __init__(self, artifact_id: str) -> None:
            self.artifact_id = artifact_id

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return f"{self.artifact_id} 数据库运行正常。"
            return f"{self.artifact_id} 接口已经恢复。"

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
                _Decoder(artifact.artifact_id),
                prompt_builder=lambda expression: expression.content_id,
                backend_id=artifact.backend_id,
            )
        )
        return _Decoder(artifact.artifact_id)

    monkeypatch.setattr("seed.language_provider.load_qwen_language_provider", fake_loader)


def _health_config(artifact_id: str) -> LanguageProviderConfig:
    return LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        artifact_id=artifact_id,
        training_corpus="corpus.json",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
        health_failure_threshold=3,
        health_cooldown_seconds=100.0,
    )


def test_provider_health_watchdog_walks_the_full_degradation_ladder(monkeypatch) -> None:
    _install_guarded_provider_stubs(monkeypatch)
    old_config = _health_config("health-old")
    new_config = _health_config("health-new")
    old = build_provider_artifact(old_config)
    new = build_provider_artifact(new_config)
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(old, allow=True)
        .with_artifact(new, allow=True)
        .activate(old.artifact_id)
    )
    adapter = TSKV8Adapter()

    rotated = rotate_language_provider(adapter, registry, new_config)
    assert rotated.committed is True
    assert rotated.registry.previous_artifact_id == old.artifact_id
    policy = new_config.health_policy()

    nominal = auto_rollback_language_provider(adapter, rotated.registry, new_config, now=10.0)
    assert nominal.committed is False
    assert nominal.status.reason_code == "provider_health_nominal"
    assert adapter.language_provider_artifact == new

    for index in range(policy.failure_threshold):
        adapter.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=20.0 + index,
            policy=policy,
        )
    assert adapter.language_provider_health.degraded is True

    first = auto_rollback_language_provider(adapter, rotated.registry, new_config, now=30.0)
    assert first.committed is True
    assert first.status.state == "rollback"
    assert first.status.reason_code == "provider_health_rollback_previous"
    assert first.status.artifact_id == old.artifact_id
    assert first.registry.active_artifact_id == old.artifact_id
    assert new.artifact_id not in first.registry.allowed_artifact_ids
    assert first.status.health_rollback_count == 1

    for index in range(policy.failure_threshold):
        adapter.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=31.0 + index,
            policy=policy,
        )
    suppressed = auto_rollback_language_provider(adapter, first.registry, new_config, now=40.0)
    assert suppressed.committed is False
    assert suppressed.status.reason_code == "provider_health_cooldown_active"
    assert adapter.language_provider_artifact == old

    native = auto_rollback_language_provider(adapter, suppressed.registry, new_config, now=200.0)
    assert native.committed is True
    assert native.status.artifact_id == "native-readable"
    assert native.status.reason_code == "provider_health_rollback_native"
    assert native.status.chat_enabled is False
    assert native.registry.active_artifact_id is None
    assert native.registry.allowed_artifact_ids == ()
    assert adapter.language_organ is not None
    assert adapter.language_organ.backend_id == "native-readable"

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.language_provider_health == adapter.language_provider_health


def test_provider_health_probe_folds_live_emissions_and_never_upgrades(monkeypatch) -> None:
    _install_guarded_provider_stubs(monkeypatch)
    old_config = _health_config("probe-old")
    new_config = _health_config("probe-new")
    old = build_provider_artifact(old_config)
    new = build_provider_artifact(new_config)
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(old, allow=True)
        .with_artifact(new, allow=True)
        .activate(old.artifact_id)
    )
    adapter = TSKV8Adapter()
    rotated = rotate_language_provider(adapter, registry, new_config)
    assert rotated.committed is True

    expression = ExpressionPlan(
        expression_id="probe:live",
        content_id="content:live",
        modality="text",
        channel="chat",
        confidence=1.0,
    )
    healthy = LanguageEmission(
        expression=expression,
        text_bytes=b"\xe6\x8e\xa5\xe5\x8f\xa3\xe5\xb7\xb2\xe7\xbb\x8f\xe6\x81\xa2\xe5\xa4\x8d\xe3\x80\x82",  # "接口已经恢复。"
        backend=new_config.backend_id,
    )
    accepted = observe_language_provider(
        adapter, new_config, expression=expression, emission=healthy, now=1.0
    )
    assert accepted.committed is False
    assert accepted.status.reason_code == "provider_health_nominal"
    assert adapter.language_provider_health.consecutive_failures == 0
    assert adapter.language_provider_health.probe_count == 1

    leaking = LanguageEmission(
        expression=expression,
        text_bytes=b'{"semantic_slots": 1}',
        backend=new_config.backend_id,
    )
    result = accepted
    for index in range(new_config.health_failure_threshold):
        result = observe_language_provider(
            adapter, new_config, expression=expression, emission=leaking, now=2.0 + index
        )
    assert result.committed is True
    assert result.status.reason_code == "provider_health_rollback_previous"
    assert result.status.artifact_id == old.artifact_id
    assert adapter.language_provider_artifact == old

    settled = observe_language_provider(
        adapter, new_config, expression=expression, emission=healthy, now=50.0
    )
    assert settled.committed is False
    assert adapter.language_provider_artifact == old
    assert new.artifact_id not in adapter.language_provider_artifact_registry.allowed_artifact_ids


def test_watchdog_thresholds_round_trip_and_fail_closed() -> None:
    configured = SeedConfig(
        language_provider=LanguageProviderConfig(
            health_failure_threshold=5,
            health_cooldown_seconds=42.5,
        )
    )
    restored = SeedConfig.from_dict(configured.to_dict())
    assert restored == configured
    assert restored.language_provider.health_policy().failure_threshold == 5
    assert restored.language_provider.health_policy().cooldown_seconds == 42.5

    for invalid in ({"health_failure_threshold": 0}, {"health_cooldown_seconds": -1.0}):
        try:
            LanguageProviderConfig(**invalid)
        except ValueError:
            continue
        raise AssertionError(f"invalid watchdog threshold must fail closed: {invalid}")
