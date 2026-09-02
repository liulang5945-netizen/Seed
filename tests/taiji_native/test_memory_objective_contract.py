from __future__ import annotations

import pytest

from taiji import (
    MEMORY_OBJECTIVE_COMPONENTS,
    MEMORY_OBJECTIVE_CREDIT_AXES,
    EpisodicObjectiveContract,
)


def test_default_contract_roundtrips_by_content_digest() -> None:
    contract = EpisodicObjectiveContract()
    restored = EpisodicObjectiveContract.from_dict(contract.to_dict())

    assert contract.digest == restored.digest
    assert contract.source_partitions == (
        "phase_a_train",
        "phase_b_train",
        "replay_train",
    )
    assert contract.protected_partition == "phase_b_train"
    assert not contract.default_enabled


def test_contract_serializes_the_complete_objective_surface() -> None:
    payload = EpisodicObjectiveContract().to_dict()

    assert tuple(payload["components"]) == MEMORY_OBJECTIVE_COMPONENTS
    assert tuple(payload["credit_axes"]) == MEMORY_OBJECTIVE_CREDIT_AXES
    assert payload["positive_binding"] == "cue_identity_to_event"
    assert payload["negative_competition"] == "cross_cue_event_exclusion"
    assert set(payload["source_partitions"]).isdisjoint(
        payload["prohibited_partitions"]
    )


def test_contract_rejects_invalid_partition_and_credit_configuration() -> None:
    with pytest.raises(ValueError, match="protected partition"):
        EpisodicObjectiveContract(protected_partition="phase_a_holdout")
    with pytest.raises(ValueError, match="disjoint"):
        EpisodicObjectiveContract(
            source_partitions=("phase_a_train", "phase_b_train", "phase_a_holdout"),
            prohibited_partitions=("phase_a_holdout",),
        )
    with pytest.raises(ValueError, match="unsupported"):
        EpisodicObjectiveContract(credit_axes=("unknown",))


def test_contract_rejects_wrong_format_or_components() -> None:
    payload = EpisodicObjectiveContract().to_dict()
    payload["format"] = "legacy"
    with pytest.raises(ValueError, match="format"):
        EpisodicObjectiveContract.from_dict(payload)

    payload = EpisodicObjectiveContract().to_dict()
    payload["components"] = ["cue_identity"]
    with pytest.raises(ValueError, match="components"):
        EpisodicObjectiveContract.from_dict(payload)
