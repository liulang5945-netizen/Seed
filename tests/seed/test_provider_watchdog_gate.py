"""S0 evidence for the content-addressed provider watchdog contract."""

from __future__ import annotations

from scripts.training.eval_taiji_provider_watchdog import build_replay_report, build_report
from taiji import LanguageProviderHealthPolicy, LanguageProviderHealthState


def test_provider_watchdog_s0_gate_passes() -> None:
    report = build_report()
    assert report["gate"]["passed"] is True
    assert report["metrics"]["rollback_count"] == 1
    assert report["metrics"]["probes_before_rollback"] == 2
    assert report["metrics"]["rollback_suppressed_count"] >= 1
    assert report["metrics"]["checkpoint_roundtrip_matches"] is True
    assert report["health_state"]["artifact_id"] == "native-readable"
    assert report["health_state"]["artifact_digest"] is None


def test_provider_health_does_not_inherit_counters_across_digest_replacement() -> None:
    policy = LanguageProviderHealthPolicy(failure_threshold=3)
    state = LanguageProviderHealthState()
    for index in range(2):
        state = state.observe(
            artifact_id="same-id",
            artifact_digest="a" * 64,
            accepted=False,
            reason_code="probe_unreadable",
            now=float(index),
            policy=policy,
        )
    replaced = state.observe(
        artifact_id="same-id",
        artifact_digest="b" * 64,
        accepted=True,
        reason_code="probe_accepted",
        now=2.0,
        policy=policy,
    )
    assert replaced.artifact_id == "same-id"
    assert replaced.artifact_digest == "b" * 64
    assert replaced.probe_count == 1
    assert replaced.accepted_count == 1
    assert replaced.consecutive_failures == 0
    assert replaced.degraded is False


def test_legacy_health_checkpoint_without_digest_remains_loadable() -> None:
    restored = LanguageProviderHealthState.from_checkpoint(
        {
            "format": "taiji-language-provider-health-v1",
            "artifact_id": "legacy-artifact",
            "probe_count": 2,
            "accepted_count": 1,
            "consecutive_failures": 1,
            "rollback_count": 0,
            "degraded": False,
            "rollback_pending": False,
            "cooldown_until": 0.0,
            "last_reason_code": "probe_unreadable",
            "last_probe_at": 1.0,
        }
    )
    assert restored.artifact_id == "legacy-artifact"
    assert restored.artifact_digest is None
    assert restored.probe_count == 2


def test_provider_watchdog_s1_checkpoint_replay_continues_the_same_failure_lineage() -> None:
    report = build_replay_report()
    assert report["gate"]["passed"] is True
    assert len(report["checkpoint"]["artifact_digest"]) == 64
    assert report["checkpoint"]["artifact_digest"] == report["health"]["parent"]["artifact_digest"]
    assert report["health"]["parent"]["probe_count"] == 2
    assert report["health"]["restored"]["probe_count"] == 2
    assert report["health"]["continued"]["probe_count"] == 3
    assert report["health"]["continued"]["consecutive_failures"] == 3
    assert report["health"]["continued"]["degraded"] is True
    assert report["health"]["restored_again"] == report["health"]["continued"]
