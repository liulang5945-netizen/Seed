"""SeedRuntime 层 provider 健康看门狗集成的回归测试。

覆盖 adapter 层测试无法触及的 api 边界修复：
  1. 原生模式 ``language_provider_status`` 必须与 guarded 模式同样输出完整
     14 键形状（键集一致性）。
  2. 看门狗回退到可读原生时，必须清空残留的 ``_provider_config``，防止
     降级版本的配置被下一次观察或状态上报误用。
  3. 名义探针（未触发回退）也必须把健康计数实时叠加进状态，让 status
     API 反映真实健康负载，而不是在回退之前恒定显示 0 探针。
"""

from __future__ import annotations

from collections.abc import Callable

from seed import LanguageProviderConfig, Seed
from seed.language_provider import build_provider_artifact
from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifactRegistry,
)

_PASSING_SPLIT = {
    "passed": True,
    "output_nonempty_rate": 1.0,
    "readable_rate": 1.0,
    "required_term_coverage": 1.0,
    "structured_leakage_free_rate": 1.0,
    "fallback_rate": 0.0,
}

_REPORTS = {
    "training": {
        "training": {"training_applied": True},
        "expression_to_text_gate": {
            "format": "taiji-language-realization-gate-v1",
            "corpus": {"round_trip": True, "split_disjoint": True},
            "train": _PASSING_SPLIT,
            "holdout": _PASSING_SPLIT,
            "rollback": {"checked": True, "outputs_match_reference": True},
            "checkpoint": {"checked": True, "outputs_match": True},
            "gate": {"passed": True},
        },
    },
    "safety": {
        "adapted": {"safe_realization_rate": 1.0},
        "rollback": {"outputs_match_raw": True},
        "gate": {"passed": True},
    },
}


def _wire_guarded(
    seed: Seed,
    monkeypatch,
    decoder_factory: Callable[[], object],
) -> tuple[LanguageProviderConfig, object, object]:
    """把 seed.substrate 装配成显式 guarded provider，并只允许该单一版本。

    registry 仅 allow 且 activate 当前版本，previous 为空，因此一旦该版本
    降级，看门狗必然回退到可读原生而不是另一个保留版本——这正是测试
    原生回退清理配置所需的最小前提。
    """

    import seed.language_provider as provider

    monkeypatch.setattr(provider, "_verify_product_chat_artifact", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        provider,
        "_load_product_chat_report",
        lambda path, label: _REPORTS[label],
    )
    decoder = decoder_factory()

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
                decoder,
                prompt_builder=lambda expression: expression.content_id,
                backend_id=artifact.backend_id,
            )
        )
        return decoder

    monkeypatch.setattr(provider, "load_qwen_language_provider", fake_loader)

    config = LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        artifact_id="runtime-guarded",
        training_corpus="corpus.json",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
        health_failure_threshold=3,
        health_cooldown_seconds=100.0,
    )
    status, runtime = provider.activate_language_provider(seed.substrate, config)
    assert status.state == "active"
    artifact = build_provider_artifact(config)
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(artifact, allow=True)
        .activate(artifact.artifact_id)
    )
    seed.substrate.attach_language_provider_artifact_registry(registry)
    return config, status, runtime


def test_seed_runtime_native_status_uses_full_key_set() -> None:
    from api.seed_runtime import SeedRuntime
    from seed.language_provider import LanguageProviderStatus

    runtime = SeedRuntime(Seed(episode_id="health-toggle-status"))
    keys = set(runtime.language_provider_status)
    expected = set(
        LanguageProviderStatus(
            mode="native",
            state="active",
            provider="native",
            backend_id="native-readable",
            artifact_id="native-readable",
        ).to_dict()
    )
    assert keys == expected
    assert any(key.startswith("health") for key in keys)


def _healthy_decoder() -> object:
    class _Decoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return "数据库运行正常。"
            return "接口已经恢复。"

    return _Decoder()


def test_nominal_probe_refreshes_health_counters_without_flipping_state(monkeypatch) -> None:
    from api.seed_runtime import SeedRuntime

    seed = Seed(episode_id="health-nominal")
    config, status, runtime = _wire_guarded(seed, monkeypatch, _healthy_decoder)
    runtime_host = SeedRuntime(
        seed,
        provider_status=status.to_dict(),
        provider_runtime=runtime,
        provider_config=config,
    )

    runtime_host.chat("检查数据库状态", learn=False)

    observed = runtime_host.language_provider_status
    assert observed["mode"] == "guarded"
    assert observed["state"] == "active"  # 名义探针不得翻转角色语义
    assert int(observed["health_probes"]) == 1
    assert observed["health_degraded"] == "false"


def _leaking_decoder() -> object:
    class _Decoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            # canary 用例必须通过；只有真实 chat 表达（content_id 不含标记）
            # 才产出带结构化泄漏的表面，从而稳定触发看门狗降级。
            if "database-status" in prompt:
                return "数据库运行正常。"
            if "interface-recovery" in prompt:
                return "接口已经恢复。"
            return '{"semantic_slots": 1}'

    return _Decoder()


def test_watchdog_native_rollback_clears_stale_provider_config(monkeypatch) -> None:
    from api.seed_runtime import SeedRuntime

    seed = Seed(episode_id="health-rollback")
    config, status, runtime = _wire_guarded(seed, monkeypatch, _leaking_decoder)
    runtime_host = SeedRuntime(
        seed,
        provider_status=status.to_dict(),
        provider_runtime=runtime,
        provider_config=config,
    )

    for _ in range(config.health_failure_threshold):
        runtime_host.chat("检查数据库状态", learn=False)

    payload = runtime_host.language_provider_status
    assert payload["mode"] == "native"
    assert payload["state"] == "rollback"
    assert payload["artifact_id"] == "native-readable"
    assert runtime_host._provider_config is None
    assert runtime_host.chat_language_backend == "native-readable"
    assert int(payload["health_rollback_count"]) == 1


def test_restart_keeps_watchdog_quarantine_instead_of_resurrecting(monkeypatch) -> None:
    """看门狗隔离过的 provider 不可在重启后被 config 偷偷复活。

    回退到原生后，隔离版本被移出持久 registry 的 allowlist。若重启路径用
    config 直接重建该版本，会绕过 ``require_allowed`` 复活劣化 provider，
    从而在每次重启后重新进入「复活→劣化→回退」的死循环，违背
    「degraded stays degraded across a restart」的持久化承诺。此测试证伪该
    回归：checkpoint 往返后再次用原 config 激活，必须保持原生。
    """
    import seed.language_provider as provider
    from api.seed_runtime import SeedRuntime

    seed = Seed(episode_id="health-restart-quarantine")
    config, status, runtime = _wire_guarded(seed, monkeypatch, _leaking_decoder)
    runtime_host = SeedRuntime(
        seed,
        provider_status=status.to_dict(),
        provider_runtime=runtime,
        provider_config=config,
    )
    for _ in range(config.health_failure_threshold):
        runtime_host.chat("检查数据库状态", learn=False)
    assert runtime_host.language_provider_status["mode"] == "native"

    # checkpoint 往返（还原被隔离的 registry + 健康记录）。
    restored = Seed.from_checkpoint(seed.checkpoint())
    restored_status, restored_runtime = provider.activate_language_provider(
        restored.substrate, config
    )
    assert restored_status.mode == "native"
    assert restored_status.state == "fallback"
    assert restored_status.reason_code == "provider_health_quarantined"
    assert restored_runtime is None
    # 原生回退不得再次挂载外部 provider 器官。
    assert restored.substrate.language_provider_artifact is None
