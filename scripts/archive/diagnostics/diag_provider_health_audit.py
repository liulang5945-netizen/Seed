"""Live audit of provider health watchdog edge cases and state consistency.

Targets the concrete failure candidates identified by code review:
  1. native-mode `_provider_status` key set vs guarded-mode 14-key set
  2. rollback-to-previous re-anchors health on the target artifact
  3. api-layer `_observe_provider_health` keeps `_provider_config` after rollback
  4. post-rollback `_chat_organ` vs adapter.language_organ sync
  5. `_native_status()` lock versus `SeedRuntime.rotate_language_provider` lock
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from seed import LanguageProviderConfig, Seed
from seed.language_provider import build_provider_artifact
from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifactRegistry,
)


def _install_stubs(provider):
    provider._verify_product_chat_artifact = lambda *args, **kwargs: {}

    split = {
        "passed": True,
        "output_nonempty_rate": 1.0,
        "readable_rate": 1.0,
        "required_term_coverage": 1.0,
        "structured_leakage_free_rate": 1.0,
        "fallback_rate": 0.0,
    }
    reports = {
        "training": {
            "training": {"training_applied": True},
            "expression_to_text_gate": {
                "format": "taiji-language-realization-gate-v1",
                "corpus": {"round_trip": True, "split_disjoint": True},
                "train": split,
                "holdout": split,
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
    provider._load_product_chat_report = lambda path, label: reports[label]

    class _DegradingDecoder:
        def __init__(self, artifact_id: str) -> None:
            self.artifact_id = artifact_id

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return f"{self.artifact_id} 数据库运行正常。"
            if "interface-recovery" in prompt:
                return f"{self.artifact_id} 接口已经恢复。"
            return '{"semantic_slots": 1}'

    old_decoder = _DegradingDecoder("audit-old")
    new_decoder = _DegradingDecoder("audit-new")

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
        decoder = new_decoder if artifact.artifact_id == "audit-new" else old_decoder
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

    provider.load_qwen_language_provider = fake_loader


def _config(artifact_id: str, threshold: int = 3) -> LanguageProviderConfig:
    return LanguageProviderConfig(
        mode="guarded",
        model_dir="model",
        adapter_dir="adapter",
        artifact_id=artifact_id,
        training_corpus="corpus.json",
        training_report="training.json",
        safety_report="safety.json",
        chat_enabled=True,
        health_failure_threshold=threshold,
        health_cooldown_seconds=100.0,
    )


def main() -> None:
    import seed.language_provider as provider

    _install_stubs(provider)
    report: dict[str, object] = {}

    # --- Probe 1: native-mode key set consistency --------------------------
    from api.seed_runtime import SeedRuntime

    rt = SeedRuntime(Seed(episode_id="audit"))
    native_keys = set(rt.language_provider_status)
    report["native_key_count"] = len(native_keys)
    report["native_has_health_keys"] = any(k.startswith("health") for k in native_keys)

    # --- Probe 2/3/4: guarded rollback via real adatper --------------------
    old = build_provider_artifact(_config("audit-old"))
    new = build_provider_artifact(_config("audit-new"))
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(old, allow=True)
        .with_artifact(new, allow=True)
        .activate(old.artifact_id)
    )
    seed = Seed(episode_id="audit")
    rotated = provider.rotate_language_provider(seed.substrate, registry, _config("audit-new"))
    report["rotation_committed"] = rotated.committed
    report["rotation_active"] = rotated.registry.active_artifact_id
    report["rotation_previous"] = rotated.registry.previous_artifact_id

    policy = _config("audit-new").health_policy()
    for index in range(policy.failure_threshold):
        seed.substrate.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=20.0 + index,
            policy=policy,
        )
    rolled = provider.auto_rollback_language_provider(
        seed.substrate, rotated.registry, _config("audit-new"), now=30.0
    )
    report["rollback_committed"] = rolled.committed
    report["rollback_artifact"] = rolled.status.artifact_id
    report["rollback_active_registry"] = rolled.registry.active_artifact_id
    report["rollback_adapter_artifact"] = (
        None if seed.substrate.language_provider_artifact is None else seed.substrate.language_provider_artifact.artifact_id
    )
    report["rollback_health_anchor"] = seed.substrate.language_provider_health.artifact_id
    report["rollback_organ_backend"] = seed.substrate.language_organ.backend_id

    # --- Probe 4: api-layer _chat_organ vs adapter organ after rollback ---
    from api.seed_runtime import SeedRuntime as SR2

    runtime = SR2(
        seed,
        provider_status=rotated.status.to_dict(),
        provider_runtime=rotated.runtime,
        provider_config=_config("audit-new"),
    )
    report["api_chat_organ_before"] = runtime.chat_language_backend
    for _ in range(policy.failure_threshold):
        runtime.chat("检查库存", learn=False)
    report["api_chat_organ_after"] = runtime.chat_language_backend
    report["api_provider_mode"] = runtime._provider_status.get("mode")
    report["api_provider_artifact"] = runtime._provider_status.get("artifact_id")
    report["api_provider_config_artifact"] = (
        None if runtime._provider_config is None else runtime._provider_config.artifact_id
    )
    report["api_adapter_organ"] = runtime.model.architecture.language_organ.backend_id

    report["PASSED"] = bool(
        native_keys
        and report["rollback_committed"]
        and report["rollback_adapter_artifact"] == report["rollback_artifact"]
        and report["rollback_health_anchor"] == report["rollback_artifact"]
    )
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
