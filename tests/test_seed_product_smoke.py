"""阶段 4 产品接入冒烟测试：Seed 原生运行时的聊天/切换/健康检查分支。

不加载真实 800K 检查点——用默认小配置 Seed 注入运行时单例，验证
路由分支、SSE 事件协议与健康状态字段。
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def seed_client(monkeypatch):
    import api.seed_runtime as seed_runtime
    from api.app import create_app
    from api.seed_runtime import SeedRuntime
    from seed import Seed

    runtime = SeedRuntime(Seed(episode_id="smoke"))
    monkeypatch.setattr(seed_runtime, "_runtime", runtime)

    # 健康检查只在启动完成后返回完整负载，测试里直接标记就绪。
    from seed_platform.app_state import app_state

    monkeypatch.setattr(app_state, "startup_complete", True)
    monkeypatch.setattr(app_state, "startup_error", None)

    app = create_app(startup_tasks=False)
    client = TestClient(app)
    yield client
    monkeypatch.setattr(seed_runtime, "_runtime", None)


def test_health_reports_seed_active(seed_client):
    response = seed_client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["seed_active"] is True
    assert payload["model_loaded"] is True


def test_chat_stream_uses_seed_branch(seed_client):
    response = seed_client.post(
        "/api/chat/stream",
        json={"prompt": "你好", "history": []},
    )
    assert response.status_code == 200
    body = response.text
    # SSE 协议：final 事件 + 结束标记
    assert '"type": "final"' in body or '"type":"final"' in body
    assert '"readable": true' in body or '"readable":true' in body
    assert "native-readable" in body
    assert "[DONE]" in body


def test_chat_stream_with_history(seed_client):
    response = seed_client.post(
        "/api/chat/stream",
        json={"prompt": "再见", "history": [["你好", "你好，很高兴见到你。"]]},
    )
    assert response.status_code == 200
    assert "[DONE]" in response.text


def test_switch_model_rejects_unknown_type():
    from api.app import create_app

    client = TestClient(create_app(startup_tasks=False))
    response = client.post("/api/system/switch_model", json={"model_type": "unknown"})
    assert response.status_code == 410
    payload = response.json()
    assert payload["code"] == "legacy_endpoint_deprecated"
    assert payload["replacement"] == "/api/runtime/activate"


def test_runtime_activation_rejects_escape_path(seed_client):
    response = seed_client.post("/api/runtime/activate", json={"checkpoint_id": "../outside.pt"})
    assert response.status_code == 400


def test_runtime_watchdog_rolls_back_degraded_provider_after_publish(monkeypatch) -> None:
    """发布时 canary 通过、运行时劣化的外部 provider 在请求层面被自动回退。

    有状态解码器对发布 canary 提示词返回可读文本（通过发布 Gate），但对
    真实 chat 提示词输出结构化泄漏文本——复现 roadmap 「发布后运行时劣化没有
    探针」的缺口，并证明 api 层的请求级探针 + 自动回退闭环已接通。
    """
    from seed import LanguageProviderConfig, Seed
    from seed.language_provider import build_provider_artifact, rotate_language_provider
    from taiji import (
        ExternalTextDecoderLanguageOrgan,
        LanguageBackendRegistry,
        LanguageBackendSpec,
        LanguageProviderArtifactRegistry,
    )

    monkeypatch.setattr(
        "seed.language_provider._verify_product_chat_artifact", lambda *args, **kwargs: {}
    )
    reports = {
        "training": {
            "training": {"training_applied": True},
            "expression_to_text_gate": {
                "format": "taiji-language-realization-gate-v1",
                "corpus": {"round_trip": True, "split_disjoint": True},
                "train": {
                    "passed": True,
                    "output_nonempty_rate": 1.0,
                    "readable_rate": 1.0,
                    "required_term_coverage": 1.0,
                    "structured_leakage_free_rate": 1.0,
                    "fallback_rate": 0.0,
                },
                "holdout": {
                    "passed": True,
                    "output_nonempty_rate": 1.0,
                    "readable_rate": 1.0,
                    "required_term_coverage": 1.0,
                    "structured_leakage_free_rate": 1.0,
                    "fallback_rate": 0.0,
                },
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
    monkeypatch.setattr(
        "seed.language_provider._load_product_chat_report",
        lambda path, label: reports[label],
    )

    class _DegradingDecoder:
        """canary 提示词返回可读文本，chat 提示词输出结构化泄漏。"""

        def __init__(self, artifact_id: str) -> None:
            self.artifact_id = artifact_id

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return f"{self.artifact_id} 数据库运行正常。"
            if "interface-recovery" in prompt:
                return f"{self.artifact_id} 接口已经恢复。"
            return '{"semantic_slots": 1}'

    become_previous = _DegradingDecoder("watchdog-old")
    active = _DegradingDecoder("watchdog-new")

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
        decoder = active if artifact.artifact_id == "watchdog-new" else become_previous
        adapter.attach_language_organ(
            ExternalTextDecoderLanguageOrgan(
                decoder,
                prompt_builder=lambda expression: expression.content_id,
                backend_id=artifact.backend_id,
            )
        )
        return decoder

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
            health_failure_threshold=2,
            health_cooldown_seconds=100.0,
        )

    old_config = config("watchdog-old")
    new_config = config("watchdog-new")
    old = build_provider_artifact(old_config)
    new = build_provider_artifact(new_config)
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(old, allow=True)
        .with_artifact(new, allow=True)
        .activate(old.artifact_id)
    )

    seed = Seed(episode_id="watchdog-smoke")
    rotated = rotate_language_provider(seed.substrate, registry, new_config)
    assert rotated.committed is True
    assert rotated.status.artifact_id == "watchdog-new"

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime(
        seed,
        provider_status=rotated.status.to_dict(),
        provider_runtime=rotated.runtime,
        provider_config=new_config,
    )
    assert runtime._provider_status["chat_enabled"] == "true"

    # 连续健康失败达到阈值 → api 层自动回退到 previous 版本。
    for _ in range(new_config.health_failure_threshold):
        runtime.chat("检查库存", learn=False)

    assert runtime._provider_status["artifact_id"] == "watchdog-old"
    assert runtime._provider_status["reason_code"] == "provider_health_rollback_previous"
    assert runtime._provider_status["chat_enabled"] == "true"
    assert (
        new.artifact_id
        not in seed.substrate.language_provider_artifact_registry.allowed_artifact_ids
    )
