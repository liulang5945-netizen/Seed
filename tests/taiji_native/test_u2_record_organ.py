"""Matched-filter RecordOrgan contract: neutral birth, differentiability, multi-record, recovery."""

from __future__ import annotations

from dataclasses import replace
from types import MethodType

import pytest
import torch

from scripts.training.first_a_support_22_5 import generate_examples, output, prepare
from taiji.u2_events import observation_intent
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_record_organ import RecordOrgan
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Packet, U2Record, U2Subject


def prepare_with_records(
    model: U2Subject, public: dict, identity: str, records: list[U2Record]
) -> dict:
    state = model.set_goal(model.initial(identity), public["goal"])
    state = observation_intent(model, state, "recorded-observation", 0.05)
    packet = U2Packet(
        "recorded-public-frame",
        "sensor",
        "world",
        public["t_start"],
        public["t_end"],
        0.0,
        public["observations"],
        public["coordinates"],
        public["outcome"],
        "recorded-observation",
    )
    state, _ = model.assimilate(state, packet)
    if records:
        state = model.set_records(state, records)
    state = model.set_request(state, "current-request", public["request"])
    return model.begin_utterance(state)


def make_parent(seed: int = 20261110, d: int = 8) -> U2Subject:
    config = replace(U2Config.small(), seed=seed, readout="pixel-salience", d=d)
    return U2Subject(config).metadata_candidate().request_candidate()


def make_record(raw: str, record_id: str, version: int = 0) -> U2Record:
    return U2Record(
        record_id=record_id,
        raw=raw.encode("utf-8"),
        version=version,
        source="sensor",
        scope="world",
        observed_at=float(version),
        writable=True,
    )


def finite_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    valid = torch.isfinite(left) & torch.isfinite(right)
    assert torch.equal(torch.isfinite(left), torch.isfinite(right))
    return float((left[valid] - right[valid]).detach().abs().max())


def test_record_organ_fork_is_deterministic_and_born_neutral():
    parent = make_parent()
    child_a = parent.record_organ_candidate()
    child_b = parent.record_organ_candidate()

    # Deterministic seeding inside fork_rng.
    for name, value in child_a.state_dict().items():
        assert torch.equal(value, child_b.state_dict()[name]), name

    # Every inherited parent tensor is bit-identical.
    for name, value in parent.state_dict().items():
        assert torch.equal(value, child_a.state_dict()[name]), name

    # Bit-identical output on both record-free and record-present states at birth.
    example = generate_examples("train", 4)[0]
    p_state = prepare(parent, example["public"], example["id"])
    c_state = prepare(child_a, example["public"], example["id"])
    assert finite_delta(output(parent, p_state)[0], output(child_a, c_state)[0]) == 0.0

    rec = make_record("通知：红色圆片改为左边。", "r0")
    p_rec_state = prepare_with_records(parent, example["public"], example["id"], [rec])
    c_rec_state = prepare_with_records(child_a, example["public"], example["id"], [rec])
    assert finite_delta(output(parent, p_rec_state)[0], output(child_a, c_rec_state)[0]) == 0.0


def test_birth_matched_filter_separates_without_hardcoded_suffix():
    torch.manual_seed(20261110)
    organ = RecordOrgan()
    rec_match = "通知：红色圆片改为左边。".encode()
    rec_near_miss = "通知：绿色圆片改为左边。".encode()
    for req in (
        "红色圆片现在在哪里？".encode(),
        "红色圆片上一次在哪里？".encode(),
        "请问红色圆片现在在哪里？".encode(),
    ):
        s_match = float(organ.match_statistic(req, rec_match).detach())
        s_miss = float(organ.match_statistic(req, rec_near_miss).detach())
        assert s_match == pytest.approx(1.0, abs=1e-5)
        assert s_miss < 0.7


def test_end_to_end_differentiable_credit_through_output_freezes_parent():
    parent = make_parent()
    model = parent.record_organ_candidate()
    parent_keys = {k for k in model.state_dict() if not k.startswith("record_organ.")}
    before = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("record_organ."))

    example = generate_examples("train", 4)[0]
    rec = make_record("通知：红色圆片改为右边。", "r-corr")
    optimizer = torch.optim.Adam(model.record_organ.parameters(), lr=0.05)

    def _byte_loss() -> torch.Tensor:
        state = prepare_with_records(model, example["public"], example["id"], [rec])
        terms = []
        for sym in [*"右边".encode(), 256]:
            logp, _, details = output(model, state)
            terms.append(-logp[sym])
            state = model.accept_byte(state, sym, details)
        return torch.stack(terms).mean()

    # Step 0: organ_mix == 0.0 still receives exact non-zero finite gradient.
    optimizer.zero_grad(set_to_none=True)
    loss0 = _byte_loss()
    loss0.backward()
    assert model.record_organ.organ_mix.grad is not None
    assert torch.isfinite(model.record_organ.organ_mix.grad)
    assert float(model.record_organ.organ_mix.grad.abs()) > 0.0
    optimizer.step()

    # Step 1: gradients flow cleanly into matched-filter route, salience, and span convs.
    optimizer.zero_grad(set_to_none=True)
    loss1 = _byte_loss()
    loss1.backward()
    for name in (
        "route_head.weight",
        "span_conv1.weight",
        "span_conv2.weight",
        "start_head.weight",
        "len_head.weight",
        "byte.weight",
        "salience.weight",
    ):
        param = dict(model.record_organ.named_parameters())[name]
        assert param.grad is not None, name
        assert bool(torch.isfinite(param.grad).all()), name
        assert float(param.grad.abs().sum()) > 0.0, name
    optimizer.step()

    after = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    assert before == after


def test_multi_record_selection_and_recency_superseding():
    torch.manual_seed(20261110)
    organ = RecordOrgan()
    with torch.no_grad():
        organ.organ_mix.fill_(1.0)
    req = "红色圆片现在在哪里？".encode()

    # Matching record placed at index 1 (NOT records[0]) among distractors.
    recs = [
        make_record("通知：绿色圆片改为左边。", "d0", version=0),
        make_record("通知：红色圆片改为右边。", "m1", version=0),
        make_record("通知：蓝色圆片改为中间。", "d2", version=0),
    ]
    plan = organ.plan(req, recs)
    assert float(plan["match_prob"].detach()) > 0.65
    assert int(plan["record_weights"].detach().argmax()) == 1

    # Distractors only: sentinel wins and match_prob stays low.
    plan_dist = organ.plan(req, [recs[0], recs[2]])
    assert float(plan_dist["match_prob"].detach()) < 0.35

    # Superseding correction: newer version for the same entity wins.
    superseded = [
        make_record("通知：红色圆片改为左边。", "r-old", version=1),
        make_record("通知：红色圆片改为右边。", "r-new", version=2),
    ]
    plan_sup = organ.plan(req, superseded)
    assert int(plan_sup["record_weights"].detach().argmax()) == 1


def test_frozen_e5_route_override_is_honoured_through_plan():
    """The E5.10r3 card patched ``organ.route`` per instance; the S0 organ keeps that frozen interface."""
    organ = RecordOrgan()
    request = "红色圆片现在在哪里？".encode()
    record = "通知：红色圆片改为左边。".encode()
    window = "左边".encode()
    organ.route = MethodType(lambda self, req, rec: window, organ)
    plan = organ.plan(request, [record])
    assert plan["legacy_hard"] is True
    assert float(plan["route_prob"]) == 1.0
    assert int(plan["record_weights"].argmax()) == 0
    for offset, symbol in enumerate([*window, 256]):
        _, probs = organ.emit(plan, offset)
        assert int(probs.argmax()) == symbol
    organ.route = MethodType(lambda self, req, rec: None, organ)
    plan = organ.plan(request, [record])
    assert plan["legacy_hard"] is True
    assert float(plan["route_prob"]) == 0.0
    assert plan["entries"] == ()
    del organ.route
    plan = organ.plan(request, [record])
    assert "legacy_hard" not in plan


def test_causal_cut_and_multi_turn_isolation_and_bundle_recovery(tmp_path):
    parent = make_parent()
    model = parent.record_organ_candidate()
    with torch.no_grad():
        model.record_organ.organ_mix.fill_(1.0)

    example = generate_examples("train", 4)[0]
    rec = make_record("通知：红色圆片改为左边。", "r0")
    state_with_rec = prepare_with_records(model, example["public"], example["id"], [rec])
    assert "organ_plan" in state_with_rec

    # Active organ diverges from parent on record-present state.
    p_with_rec = prepare_with_records(parent, example["public"], example["id"], [rec])
    parent_logp = output(parent, p_with_rec)[0]
    organ_logp = output(model, state_with_rec)[0]
    assert finite_delta(parent_logp, organ_logp) > 0.0

    # Verification cut restores exact parent output.
    model._verification_cuts = frozenset({"record_organ"})
    try:
        cut_logp = output(model, state_with_rec)[0]
        assert finite_delta(parent_logp, cut_logp) == 0.0
    finally:
        model._verification_cuts = frozenset()

    # Mid-utterance export_bundle / from_bundle preserves state & outputs bit-for-bit.
    bundle = model.export_bundle(state_with_rec)
    restored, restored_state = U2Subject.from_bundle(bundle)
    assert value_digest(restored_state) == value_digest(state_with_rec)
    assert (
        finite_delta(output(model, state_with_rec)[0], output(restored, restored_state)[0]) == 0.0
    )

    # Multi-turn isolation: finishing utterance clears organ_plan; next record-free turn is clean.
    _, _, details = output(model, state_with_rec)
    finished = model.accept_byte(state_with_rec, 256, details)
    assert "organ_plan" not in finished
    next_state = model.set_records(finished, [])
    next_state = model.set_request(next_state, "next-q", example["public"]["request"])
    next_turn = model.begin_utterance(next_state)
    assert "organ_plan" not in next_turn

    # Full U2Runtime persistence roundtrip with record_organ + pixel-salience.
    runtime = U2Runtime(model, subject_id="organ-recovery-test")
    runtime.set_goal("依据当前可核验观测，用自然中文回答请求。")
    runtime.set_request("q1", "红色圆片现在在哪里？")
    runtime.save(tmp_path / "organ_runtime.bin")
    loaded = U2Runtime.load(tmp_path / "organ_runtime.bin")
    assert_values_equal(runtime.state, loaded.state)
