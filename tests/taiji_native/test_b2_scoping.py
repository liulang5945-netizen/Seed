"""B.2 presence-gated correction adapter (v7): exact zero without records, open with them.

Zero-training contract tests. B.1 measured that the ungated adapter learns a
record-blind global shift; B.2's single structural variable is the presence
gate. These tests pin: the gate is exactly zero on record-free states (so the
record-free retention check is structural and must NOT be cited as earned),
the gate opens when a record is present (so the distractor-record retention
check is failable), knockout restores v4 even with records present, gradients
reach the adapter only through record-present episodes, and the v7 schema
round-trips without disturbing v6.
"""

import pytest
import torch

from taiji.u2_history import assert_values_equal
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Record, U2Subject

ADAPTER_TENSORS = {
    "correction_adapter.first.weight",
    "correction_adapter.first.bias",
    "correction_adapter.last.weight",
    "correction_adapter.last.bias",
}


def v4():
    torch.set_num_threads(1)
    return U2Subject(U2Config.small()).metadata_candidate().request_candidate()


def v7():
    return v4().correction_candidate("v2")


def record():
    return U2Record(
        record_id="b2-test",
        raw="红色圆片现在在左边。".encode(),
        version=0,
        source="sensor",
        scope="world",
        observed_at=0.0,
        writable=True,
    )


def runtime(model, subject_id):
    rt = U2Runtime(model, subject_id=subject_id)
    rt.set_goal("依据可核验观测完成任务。")
    rt.set_request("query", "红色圆片现在在哪里？")
    return rt


def test_v7_fork_inherits_and_births_four():
    base = v4()
    candidate = base.correction_candidate("v2")
    assert candidate.c.correction == "v2"
    assert len(list(candidate.parameters())) == 198
    assert sum(p.numel() for p in candidate.parameters()) == 23150
    assert set(candidate.state_dict()) - set(base.state_dict()) == ADAPTER_TENSORS
    for name, value in base.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    assert len(candidate.state_registry()) == 42
    assert candidate.parameter_revision == base.parameter_revision + 1


def test_invalid_fork_version_refused():
    with pytest.raises(ValueError, match="version"):
        v4().correction_candidate("v3")


def test_gate_is_exact_zero_without_records():
    base = runtime(v4(), "b2-base")
    gated = runtime(v7(), "b2-gated")
    assert len(gated.state["records"]) == 0
    with torch.no_grad():
        gated.model.correction_adapter.last.weight.add_(0.5)
    assert torch.equal(gated.output()[0], base.output()[0])


def test_gate_opens_with_a_record_and_knockout_restores_v4():
    base = runtime(v4(), "b2-base")
    gated = runtime(v7(), "b2-gated")
    base.state = base.model.set_records(base.state, [record()])
    gated.state = gated.model.set_records(gated.state, [record()])
    with torch.no_grad():
        gated.model.correction_adapter.last.weight.add_(0.5)
    base_out = base.output()[0].clone()
    assert not torch.equal(gated.output()[0], base_out)
    gated.model._verification_cuts = frozenset({"correction"})
    assert torch.equal(gated.output()[0], base_out)
    gated.model._verification_cuts = frozenset()
    assert not torch.equal(gated.output()[0], base_out)


def test_gradient_reaches_adapter_only_through_record_present_episodes():
    model = v7()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("correction_adapter.") or name.startswith("record_"):
            parameter.requires_grad_(True)
    rt = runtime(model, "b2-grad")

    def loss_on(state):
        state = model.begin_utterance(state)
        query = model.language_goal(state)
        parts = []
        for byte in [*"左边".encode(), 256]:
            probability, _, output = model.output(state, query, state["prediction_action"])
            parts.append(-probability[byte])
            state = model.accept_byte(state, byte, output)
        return torch.stack(parts).mean()

    model.zero_grad(set_to_none=True)
    loss_on(rt.state).backward()
    assert model.correction_adapter.last.weight.grad is None or bool(
        (model.correction_adapter.last.weight.grad == 0).all()
    )
    model.zero_grad(set_to_none=True)
    # A fresh runtime: the first backward freed the graph behind rt.state.
    rt_open = runtime(model, "b2-grad-open")
    rt_open.state = model.set_records(rt_open.state, [record()])
    loss_on(rt_open.state).backward()
    grad = model.correction_adapter.last.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.any())
    assert model.byte_head.weight.grad is None


def test_v7_bundle_roundtrip_and_schema_mismatches():
    rt = runtime(v7(), "b2-bundle")
    bundle = rt.model.export_bundle(rt.state)
    assert bundle["schema"] == "taiji-u2-subject-v7"
    assert bundle["config"]["correction"] == "v2"
    _, state = U2Subject.from_bundle(bundle)
    assert_values_equal(rt.state, state)
    with pytest.raises(ValueError, match="schema"):
        U2Subject.from_bundle({**bundle, "schema": "taiji-u2-subject-v6"})
    v6_rt = runtime(v4().correction_candidate("v1"), "b2-v6")
    v6_bundle = v6_rt.model.export_bundle(v6_rt.state)
    with pytest.raises(ValueError, match="schema"):
        U2Subject.from_bundle({**v6_bundle, "schema": "taiji-u2-subject-v7"})
