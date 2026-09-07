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
