"""E2c 状态信用候选的判据册：通断两支都要能为假，恢复与冻结组各一支负对照。"""

from __future__ import annotations

import pytest
import torch

from taiji.internalization import content_digest
from taiji.joint_state_consumption import PersistentContentFeatureV1
from taiji.joint_state_consumption_course import E1_ATTRIBUTE, E1_COLORS
from taiji.joint_state_credit import (
    MODULE_GROUPS,
    STATE_CREDIT_FORMAT,
    CreditCandidateV1,
    CreditCaseV1,
    StateCreditCandidateV1,
    checkpoint_holds_no_graph,
)


def _case() -> CreditCaseV1:
    facts = (("钥匙", "黄色"), ("杯子", "红色"))
    candidates = tuple(
        CreditCandidateV1.from_value(
            feature=PersistentContentFeatureV1.create(
                entity=entity,
                attribute=E1_ATTRIBUTE,
                value=value,
                source_input_id=f"test:{index}",
                source_sha256=content_digest({"test": index, "value": value}),
                turn_index=0,
                revision=1,
                valid_from=0,
                confidence=1.0,
                continuous=[0.0] * 64,
            ),
            value=value,
        )
        for index, (entity, value) in enumerate(facts)
    )
    return CreditCaseV1(
        case_id="test:two-entities",
        entity=facts[0][0],
        attribute=E1_ATTRIBUTE,
        candidates=candidates,
        label=E1_COLORS.index(facts[0][1]),
    )


def _grads(mode: str) -> dict[str, float]:
    candidate = StateCreditCandidateV1(mode=mode, seed=0)
    out = candidate.forward(_case())
    return candidate.backward(out["loss"])


def test_credit_on_reaches_the_encoder() -> None:
    assert _grads("credit_on")["encoder"] > 0.0


def test_credit_stop_cuts_the_encoder() -> None:
    gradients = _grads("credit_stop")
    assert gradients["encoder"] == 0.0
    assert gradients["readout"] > 0.0


def test_the_selector_gets_no_answer_credit_in_either_arm() -> None:
    assert _grads("credit_on")["selector"] == 0.0
    assert _grads("credit_stop")["selector"] == 0.0


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="credit mode"):
        StateCreditCandidateV1(mode="credit_maybe")


def test_checkpoint_round_trip_restores_every_piece() -> None:
    candidate = StateCreditCandidateV1(mode="credit_on", seed=0)
    for _ in range(2):
        out = candidate.forward(_case())
        candidate.backward(out["loss"])
        candidate.step()
    torch.random.seed()
    before = {
        "parameters": candidate.parameters_digest(),
        "optimizer": content_digest(candidate.optimizer.state_dict()),
        "cursor": candidate.cursor,
    }
    payload = candidate.checkpoint()
    assert payload["format"] == STATE_CREDIT_FORMAT
    restored = StateCreditCandidateV1.from_checkpoint(payload)
    assert restored.parameters_digest() == before["parameters"]
    assert content_digest(restored.optimizer.state_dict()) == before["optimizer"]
    assert restored.cursor == before["cursor"] == 2
    assert torch.equal(torch.random.get_rng_state().clone(), payload["torch_rng_state"].clone())


def test_a_foreign_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="format"):
        StateCreditCandidateV1.from_checkpoint({"format": "something-else"})


def test_freezing_a_group_keeps_it_unchanged() -> None:
    candidate = StateCreditCandidateV1(mode="credit_on", seed=0, freeze=("encoder",))
    before = candidate.parameters_digest()
    out = candidate.forward(_case())
    candidate.backward(out["loss"])
    candidate.step()
    after = candidate.parameters_digest()
    assert before["encoder"] == after["encoder"]
    assert before["readout"] != after["readout"]
    assert len(MODULE_GROUPS) == 3


def test_checkpoint_graph_guard_can_be_false() -> None:
    assert checkpoint_holds_no_graph({"w": torch.zeros(2)}) is True
    assert checkpoint_holds_no_graph({"w": torch.zeros(2, requires_grad=True)}) is False
