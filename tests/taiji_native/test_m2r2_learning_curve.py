from __future__ import annotations

from scripts.training.eval_taiji_m2r2_ablation import (
    _ngram_bpb,
    _owner_attribution_for_variant,
    shuffled_bytes,
)
from scripts.training.eval_taiji_m2r2_delay_probe import (
    _delay_target_index,
    build_delay_records,
)
from scripts.training.eval_taiji_m2r2_learning_curve import owner_attribution_check
from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def _owners() -> dict[str, str]:
    return {
        "fabric": "same-fabric",
        "motor": "same-motor",
        "memory": "same-memory",
        "predictive_context": "same-context",
        "protected_predictive_readout": "same-protected",
        "active_predictive_readout": "same-active",
    }


def test_active_registration_is_not_training_attribution() -> None:
    before = _owners()
    after = dict(before)
    after["active_predictive_readout"] = "trained-active"

    result = owner_attribution_check("active_readout", before, after)

    assert result["contract_passed"] is True
    assert result["changed_expected_owners"] == ["active_predictive_readout"]
    assert result["required_owner_changed"] is True


def test_frozen_arm_allows_no_owner_change() -> None:
    result = owner_attribution_check("frozen", _owners(), _owners())

    assert result["contract_passed"] is True
    assert result["changed_expected_owners"] == []
    assert result["required_owner_changed"] is False


def test_unexpected_owner_change_fails_contract() -> None:
    before = _owners()
    after = dict(before)
    after["fabric"] = "unexpected-fabric"

    result = owner_attribution_check("predictive_context", before, after)

    assert result["contract_passed"] is False
    assert result["unexpected_changed_owners"] == ["fabric"]


def test_joint_arm_requires_both_predictive_owners_only() -> None:
    before = _owners()
    after = dict(before)
    after["predictive_context"] = "trained-context"
    after["protected_predictive_readout"] = "trained-protected"

    result = owner_attribution_check("joint_predictive", before, after)

    assert result["contract_passed"] is True
    assert result["changed_expected_owners"] == [
        "predictive_context",
        "protected_predictive_readout",
    ]
    assert result["required_owner_changed"] is True


def test_shuffled_stream_is_deterministic_and_preserves_byte_counts() -> None:
    data = b"abracadabra" * 8

    first = shuffled_bytes(data, seed=7001)
    second = shuffled_bytes(data, seed=7001)

    assert first == second
    assert len(first) == len(data)
    assert sorted(first) == sorted(data)
    assert first != data


def test_additive_ngram_reference_is_finite_for_unseen_contexts() -> None:
    score = _ngram_bpb(b"abca" * 3, b"zzzz", order=3)

    assert score > 0.0
    assert score < float("inf")


def test_protected_readout_ablation_owner_contract_is_explicit() -> None:
    before = _owners()
    after = dict(before)
    after["protected_predictive_readout"] = "trained-protected"

    result = _owner_attribution_for_variant("readout_only", before, after)

    assert result["contract_passed"] is True
    assert result["changed_expected_owners"] == ["protected_predictive_readout"]
    assert result["unexpected_changed_owners"] == []


def test_delay_probe_has_disjoint_splits_and_known_target_index() -> None:
    records = build_delay_records(
        8,
        seed=101,
        train_count=8,
        dev_count=4,
        test_count=4,
    )

    assert not set(records["train"]) & set(records["dev"])
    assert not set(records["train"]) & set(records["test"])
    assert not set(records["dev"]) & set(records["test"])
    assert all(_delay_target_index(value, 8) == 10 for value in records["test"])
    assert all(value[2] == value[-1] for value in records["test"])


def _candidate_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        predictive_context_fan_in=2,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        identity_organ_enabled=False,
        seed=707,
    )


def test_gated_candidate_is_opt_in_and_legacy_checkpoint_is_byte_stable() -> None:
    model = Taiji(_candidate_config())
    model.learn_bytes(
        b"abcd" * 4,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
    )
    legacy_payload = model.checkpoint()
    legacy_digest = content_digest(legacy_payload)
    restored = Taiji.from_checkpoint(legacy_payload)

    assert restored.gated_temporal_candidate_enabled is False
    assert content_digest(restored.checkpoint()) == legacy_digest


def test_gated_candidate_preserves_output_before_learning_and_round_trips() -> None:
    source = Taiji(_candidate_config())
    source.learn_bytes(
        b"abcd" * 4,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
    )
    baseline = Taiji.from_checkpoint(source.checkpoint())
    candidate = Taiji.from_checkpoint(source.checkpoint())
    expected = baseline.generate(b"Taiji", 12)

    candidate.enable_gated_temporal_candidate()
    assert candidate.generate(b"Taiji", 12) == expected
    assert candidate.gated_temporal_candidate_enabled is True

    candidate_payload = candidate.checkpoint()
    restored = Taiji.from_checkpoint(candidate_payload)
    assert restored.gated_temporal_candidate_enabled is True
    assert content_digest(restored.checkpoint()) == content_digest(candidate_payload)
    assert restored.generate(b"Taiji", 12) == candidate.generate(b"Taiji", 12)


def test_gated_candidate_training_writes_candidate_without_legacy_context() -> None:
    model = Taiji(_candidate_config())
    model.enable_gated_temporal_candidate()
    legacy_before = content_digest(model.predictive_context.to_payload())
    candidate_before = content_digest(model.checkpoint()["gated_temporal_candidate"])

    model.learn_bytes(
        b"abcd" * 8,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=True,
        learn_predictive_readout=False,
    )

    assert content_digest(model.predictive_context.to_payload()) == legacy_before
    assert content_digest(model.checkpoint()["gated_temporal_candidate"]) != candidate_before
