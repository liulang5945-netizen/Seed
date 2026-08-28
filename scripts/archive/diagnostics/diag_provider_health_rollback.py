"""Live check of the provider health watchdog auto-rollback path."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import seed.language_provider as provider_module
from seed.config import LanguageProviderConfig
from seed.language_provider import auto_rollback_language_provider
from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifact,
    LanguageProviderArtifactRegistry,
    TSKV8Adapter,
    language_provider_content_digest,
)


class _Decoder:
    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del max_tokens, temperature
        if "database-status" in prompt:
            return "数据库运行正常。"
        return "接口已经恢复。"


def _fake_loader(adapter, artifact, **kwargs):
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=artifact.backend_id,
            family="external-causal-decoder-guarded",
            training_contract="expression-to-text-v1",
        )
    )
    organ = ExternalTextDecoderLanguageOrgan(
        _Decoder(artifact.artifact_id),
        prompt_builder=lambda plan: plan.expression_id,
        backend_id=artifact.backend_id,
    )
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_provider_artifact(artifact)
    adapter.attach_language_organ(organ)
    return _Decoder(artifact.artifact_id)


def _training_report(artifact_id: str) -> dict[str, object]:
    split = {
        "passed": True,
        "output_nonempty_rate": 1.0,
        "readable_rate": 1.0,
        "required_term_coverage": 1.0,
        "structured_leakage_free_rate": 1.0,
        "fallback_rate": 0.0,
    }
    return {
        "artifact_id": artifact_id,
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
    }


def _safety_report(artifact_id: str) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "adapted": {"safe_realization_rate": 1.0},
        "rollback": {"outputs_match_raw": True},
        "gate": {"passed": True},
    }


def _write_tree(root: Path, artifact_id: str) -> dict[str, Path]:
    paths = {
        "base_model": root / f"{artifact_id}-model",
        "adapter": root / f"{artifact_id}-adapter",
        "training_corpus": root / f"{artifact_id}-corpus.jsonl",
        "training_report": root / f"{artifact_id}-report.json",
        "safety_report": root / f"{artifact_id}-safety.json",
    }
    paths["base_model"].mkdir(parents=True, exist_ok=True)
    (paths["base_model"] / "config.json").write_text(f'{{"id": "{artifact_id}"}}', encoding="utf-8")
    paths["adapter"].mkdir(parents=True, exist_ok=True)
    (paths["adapter"] / "adapter.json").write_text(f'{{"id": "{artifact_id}"}}', encoding="utf-8")
    paths["training_corpus"].write_text(f'{{"id": "{artifact_id}"}}\n', encoding="utf-8")
    paths["training_report"].write_text(
        json.dumps(_training_report(artifact_id), ensure_ascii=False), encoding="utf-8"
    )
    paths["safety_report"].write_text(
        json.dumps(_safety_report(artifact_id), ensure_ascii=False), encoding="utf-8"
    )
    return paths


def _artifact(root: Path, artifact_id: str) -> tuple[LanguageProviderArtifact, dict[str, Path]]:
    paths = _write_tree(root, artifact_id)
    digests = tuple(
        sorted((role, language_provider_content_digest(path)) for role, path in paths.items())
    )
    draft = LanguageProviderArtifact(
        artifact_id=artifact_id,
        backend_id=f"{artifact_id}-backend",
        mode="guarded",
        base_model=str(paths["base_model"]),
        adapter_path=str(paths["adapter"]),
        training_corpus=str(paths["training_corpus"]),
        training_report=str(paths["training_report"]),
        safety_report=str(paths["safety_report"]),
        content_digests=digests,
        expires_at=4.0e9,
    )
    return draft, paths


def _config(artifact: LanguageProviderArtifact) -> LanguageProviderConfig:
    return LanguageProviderConfig(
        mode="guarded",
        provider="qwen",
        backend_id=artifact.backend_id,
        model_dir=artifact.base_model,
        adapter_dir=str(artifact.adapter_path),
        artifact_id=artifact.artifact_id,
        training_corpus=str(artifact.training_corpus),
        training_report=str(artifact.training_report),
        safety_report=str(artifact.safety_report),
        content_digests=artifact.content_digests,
        artifact_digest=artifact.artifact_digest,
        expires_at=artifact.expires_at,
        chat_enabled=True,
        health_failure_threshold=3,
        health_cooldown_seconds=100.0,
    )


def main() -> None:
    provider_module.load_qwen_language_provider = _fake_loader
    root = Path("artifacts/diagnostics/provider_health_rollback")
    root.mkdir(parents=True, exist_ok=True)
    old, _ = _artifact(root, "health-old")
    new, _ = _artifact(root, "health-new")

    adapter = TSKV8Adapter()
    registry = LanguageProviderArtifactRegistry()
    registry = registry.with_artifact(old, allow=True).with_artifact(new, allow=True)
    registry = registry.activate(old.artifact_id)

    rotated = provider_module.rotate_language_provider(adapter, registry, _config(new))
    report: dict[str, object] = {
        "rotation_committed": rotated.committed,
        "rotation_reason_code": rotated.status.reason_code,
        "rotation_reason": rotated.status.reason,
        "rotation_active": rotated.registry.active_artifact_id,
        "rotation_previous": rotated.registry.previous_artifact_id,
    }
    registry = rotated.registry
    config = _config(new)
    policy = config.health_policy()

    nominal = auto_rollback_language_provider(adapter, registry, config, now=10.0)
    report["nominal_committed"] = nominal.committed
    report["nominal_reason_code"] = nominal.status.reason_code

    for index in range(policy.failure_threshold):
        adapter.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=20.0 + index,
            policy=policy,
        )
    report["degraded"] = adapter.language_provider_health.degraded

    first = auto_rollback_language_provider(adapter, registry, config, now=30.0)
    report["first_committed"] = first.committed
    report["first_state"] = first.status.state
    report["first_artifact"] = first.status.artifact_id
    report["first_reason_code"] = first.status.reason_code
    report["first_active"] = first.registry.active_artifact_id
    report["first_previous"] = first.registry.previous_artifact_id
    report["first_allowlist"] = list(first.registry.allowed_artifact_ids)
    report["first_health_rollbacks"] = first.status.health_rollback_count
    registry = first.registry

    for index in range(policy.failure_threshold):
        adapter.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=31.0 + index,
            policy=policy,
        )
    second = auto_rollback_language_provider(adapter, registry, config, now=40.0)
    report["cooldown_committed"] = second.committed
    report["cooldown_reason_code"] = second.status.reason_code

    third = auto_rollback_language_provider(adapter, second.registry, config, now=200.0)
    report["final_committed"] = third.committed
    report["final_state"] = third.status.state
    report["final_artifact"] = third.status.artifact_id
    report["final_reason_code"] = third.status.reason_code
    report["final_active"] = third.registry.active_artifact_id
    report["final_allowlist"] = list(third.registry.allowed_artifact_ids)
    report["final_chat_enabled"] = third.status.chat_enabled

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    report["health_roundtrip"] = (
        restored.language_provider_health == adapter.language_provider_health
    )

    checks = {
        "rotation_retains_previous": report["rotation_previous"] == "health-old",
        "nominal_is_noop": not nominal.committed
        and report["nominal_reason_code"] == "provider_health_nominal",
        "threshold_trips": report["degraded"] is True,
        "first_rolls_to_previous": first.committed
        and report["first_artifact"] == "health-old"
        and report["first_active"] == "health-old",
        "degraded_quarantined": "health-new" not in report["first_allowlist"],
        "cooldown_suppresses": not second.committed
        and report["cooldown_reason_code"] == "provider_health_cooldown_active",
        "no_previous_falls_native": third.committed
        and report["final_artifact"] == "native-readable"
        and report["final_reason_code"] == "provider_health_rollback_native",
        "native_disables_chat": report["final_chat_enabled"] is False,
        "health_survives_checkpoint": report["health_roundtrip"] is True,
    }
    report["checks"] = checks
    report["PASSED"] = all(checks.values())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
