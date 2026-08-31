from __future__ import annotations

import copy

import pytest

from taiji import (
    ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE,
    ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY,
    ArtifactConsumptionAudit,
    ArtifactConsumptionPolicy,
    TSKV8Adapter,
)


def test_policy_and_audit_are_content_addressed_and_roundtrip() -> None:
    policy = ArtifactConsumptionPolicy.verified_only(reason="new-growth-test")
    assert ArtifactConsumptionPolicy.from_payload(policy.to_payload()) == policy

    tampered_policy = policy.to_payload()
    tampered_policy["reason"] = "changed-after-signing"
    with pytest.raises(ValueError, match="digest mismatch"):
        ArtifactConsumptionPolicy.from_payload(tampered_policy)

    audit = ArtifactConsumptionAudit.create(
        "batch:policy-test",
        policy,
        {"candidate:b": "verified", "candidate:a": "legacy_unverified"},
        result="rejected",
        error_code="artifact_rejected",
    )
    assert ArtifactConsumptionAudit.from_payload(audit.to_payload()) == audit

    tampered_audit = audit.to_payload()
    tampered_audit["artifact_statuses"]["candidate:a"] = "verified"
    with pytest.raises(ValueError, match="digest mismatch"):
        ArtifactConsumptionAudit.from_payload(tampered_audit)


def test_new_runtime_defaults_verified_only_and_policy_checkpoint_is_reversible() -> None:
    model = TSKV8Adapter()
    initial_checkpoint = model.native_checkpoint()
    assert model.artifact_consumption_policy.mode == ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY

    legacy_policy = model.set_artifact_consumption_policy(
        ArtifactConsumptionPolicy.legacy_compatible(reason="historical-replay-test")
    )
    assert legacy_policy.mode == ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE
    legacy_checkpoint = model.native_checkpoint()

    restored_legacy = TSKV8Adapter()
    restored_legacy.restore_native(legacy_checkpoint)
    assert restored_legacy.artifact_consumption_policy == legacy_policy

    restored_initial = TSKV8Adapter()
    restored_initial.restore_native(initial_checkpoint)
    assert (
        restored_initial.artifact_consumption_policy.mode
        == ARTIFACT_CONSUMPTION_MODE_VERIFIED_ONLY
    )


def test_old_checkpoint_migrates_to_explicit_legacy_policy() -> None:
    model = TSKV8Adapter()
    old_checkpoint = copy.deepcopy(model.native_checkpoint())
    structural_runtime = old_checkpoint["components"]["structural_runtime"]
    structural_runtime.pop("artifact_consumption_policy")

    restored = TSKV8Adapter()
    restored.restore_native(old_checkpoint)
    policy = restored.artifact_consumption_policy
    assert policy.mode == ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE
    assert policy.reason == "historical-checkpoint-migration"


def test_policy_and_legacy_boolean_cannot_be_combined() -> None:
    model = TSKV8Adapter()
    policy = ArtifactConsumptionPolicy.verified_only(reason="explicit-test")
    with pytest.raises(ValueError, match="policy or legacy boolean"):
        model.resolve_artifact_consumption_policy(
            policy,
            require_verified_measurements=True,
        )

    legacy = model.resolve_artifact_consumption_policy(
        require_verified_measurements=False
    )
    assert legacy.mode == ARTIFACT_CONSUMPTION_MODE_LEGACY_COMPATIBLE
    assert legacy.reason == "legacy-boolean-compatibility"
