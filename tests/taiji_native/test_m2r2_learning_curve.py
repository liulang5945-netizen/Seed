from __future__ import annotations

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
