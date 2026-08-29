"""Run the deterministic W7-R1 provider watchdog S0 Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (
    ExpressionPlan,
    LanguageBackendSpec,
    LanguageEmission,
    LanguageProviderArtifact,
    LanguageProviderArtifactRegistry,
    LanguageProviderHealthGate,
    LanguageProviderHealthPolicy,
    NativeReadableTextLanguageOrgan,
    TSKV8Adapter,
)

MANIFEST = "plans/manifests/taiji_w7_r1_provider_watchdog_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_w7_r1_provider_watchdog_20260829.json"
DEFAULT_REPLAY_REPORT = PROJECT_ROOT / "reports" / "taiji_w7_r1_provider_watchdog_s1_20260829.json"


class _DegradedLanguageOrgan:
    """Deterministic unhealthy provider that leaks a structured surface."""

    backend_id = "s0-degraded-provider"

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        return LanguageEmission(
            expression=expression,
            text_bytes=b'{"semantic_slots": 1}',
            backend=self.backend_id,
        )

    def checkpoint(self) -> dict[str, str]:
        return {"backend": self.backend_id}


def build_report() -> dict[str, object]:
    policy = LanguageProviderHealthPolicy(
        failure_threshold=2,
        cooldown_seconds=30.0,
        minimum_accepted_rate=0.5,
        minimum_rate_probes=8,
    )
    report = LanguageProviderHealthGate(policy).evaluate(
        NativeReadableTextLanguageOrgan(),
        degraded_organ=_DegradedLanguageOrgan(),
        artifact_id="s0-active-artifact",
        artifact_digest="a" * 64,
        rollback_artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
        rollback_artifact_digest=None,
        now=100.0,
    )
    return {
        "manifest": MANIFEST,
        "format": report["format"],
        "gate_id": report["gate_id"],
        "policy": report["policy"],
        "metrics": report["metrics"],
        "health_state": report["health_state"],
        "gate": report["gate"],
        "controls": [
            "healthy provider stays active",
            "structured leakage trips the consecutive-failure threshold",
            "cooldown suppresses a second rollback",
            "health state round-trips through checkpoint",
            "content digest changes reset the health record",
        ],
    }


def build_replay_report() -> dict[str, object]:
    """Replay provider health through the real native adapter checkpoint."""

    artifact = LanguageProviderArtifact(
        artifact_id="s1-replay-artifact",
        backend_id="s1-replay-provider",
        mode="raw",
        base_model="models/s1-replay",
        content_digests=(("base_model", "b" * 64),),
    )
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(artifact, allow=True)
        .activate(artifact.artifact_id)
    )
    policy = LanguageProviderHealthPolicy(failure_threshold=3, cooldown_seconds=60.0)
    adapter = TSKV8Adapter()
    adapter.language_backend_registry.register(
        LanguageBackendSpec(
            backend_id=artifact.backend_id,
            family="external-causal-decoder-raw",
            training_contract="expression-to-text-v1",
        )
    )
    adapter.attach_language_provider_artifact(artifact)
    adapter.attach_language_provider_artifact_registry(registry)
    for index in range(2):
        adapter.observe_language_provider_health(
            accepted=False,
            reason_code="probe_unreadable",
            now=float(index),
            policy=policy,
        )

    parent_health = adapter.language_provider_health
    parent_checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(parent_checkpoint)
    restored_health = restored.language_provider_health
    restored.observe_language_provider_health(
        accepted=False,
        reason_code="probe_unreadable",
        now=2.0,
        policy=policy,
    )
    continued_health = restored.language_provider_health
    restored_again = TSKV8Adapter.from_native_checkpoint(restored.native_checkpoint())
    replay_passed = bool(
        restored.language_provider_artifact == artifact
        and restored.language_provider_artifact_registry == registry
        and restored_health == parent_health
        and restored_health.artifact_digest == artifact.artifact_digest
        and restored_health.probe_count == 2
        and continued_health.probe_count == 3
        and continued_health.consecutive_failures == 3
        and continued_health.degraded
        and continued_health.rollback_pending
        and restored_again.language_provider_health == continued_health
    )
    return {
        "manifest": MANIFEST,
        "stage": "S1",
        "checkpoint": {
            "format": parent_checkpoint["format"],
            "artifact_id": artifact.artifact_id,
            "artifact_digest": artifact.artifact_digest,
            "registry_revision": registry.revision,
            "active_artifact_id": registry.active_artifact_id,
        },
        "health": {
            "parent": parent_health.checkpoint(),
            "restored": restored_health.checkpoint(),
            "continued": continued_health.checkpoint(),
            "restored_again": restored_again.language_provider_health.checkpoint(),
        },
        "controls": [
            "artifact digest survives native adapter checkpoint",
            "registry selection survives native adapter checkpoint",
            "health counters survive restore without reset",
            "the next probe continues the same failure lineage",
            "threshold crossing remains observable after restore",
        ],
        "gate": {
            "passed": replay_passed,
            "criterion": "provider artifact identity, registry selection, health counters, digest binding, and threshold continuation survive a native adapter checkpoint round-trip",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("s0", "s1"), default="s0")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.stage == "s1":
        report = build_replay_report()
        report_path = args.report or DEFAULT_REPLAY_REPORT
    else:
        report = build_report()
        report_path = args.report or DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
