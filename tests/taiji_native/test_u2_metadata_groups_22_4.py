"""Zero-update metadata credit and actual group-local neuron growth checks."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Record, U2Subject


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def candidate(precision="float64"):
    parent = U2Subject(U2Config.small()).to(dtype=getattr(torch, precision))
    return parent, parent.metadata_candidate()


def prepared(precision="float64"):
    parent, model = candidate(precision)
    runtime = U2Runtime(model)
    runtime.set_goal("红盒现在在哪里？")
    runtime.set_records(
        (
            U2Record("a", "红盒在左边。".encode(), 0, observed_at=-1.0),
            U2Record("b", "红盒在右边。".encode(), 1),
        )
    )
    return parent, runtime


def credit(model, group=0):
    value = model.native_w.new_zeros(model.c.groups, model.c.d, 8 * model.c.d)
    value[group, 0, 0], value[group, 1, 1] = 3, 2
    return value


def native_value(runtime):
    model = runtime.model
    return model.native(
        runtime.state,
        model.goal(runtime.state["goal_raw"]),
        model.native_w.new_zeros(15),
        model.native_w.new_full((model.c.branches, model.c.d), 0.1),
    )


def answer_loss(runtime):
    model = runtime.model
    state = model.begin_utterance(runtime.state)
    goal = model.goal(state["goal_raw"])
    terms = []
    for symbol in [*"右边".encode(), 256]:
        probability, _, output = model.output(state, goal, state["prediction_action"])
        terms.append(-probability[symbol])
        state = model.accept_byte(state, symbol, output)
    return torch.stack(terms).mean()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_metadata_fork_inherits_every_parent_parameter(tmp_path, precision):
    parent, model = candidate(precision)
    for name, value in parent.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    assert len(list(parent.parameters())) == 191
    assert len(list(model.parameters())) == 194
    assert model.parameter_revision == parent.parameter_revision + 1


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_only_versions_change_record_routes_with_fixed_content_and_state(tmp_path, precision):
    _, runtime = prepared(precision)
    model = runtime.model
    goal = model.goal(runtime.state["goal_raw"])
    before = model.record_read(runtime.state, goal)
    records = runtime.state["records"]
    alternate = model.set_records(
        runtime.state, (replace(records[0], version=1), replace(records[1], version=0))
    )
    after = model.record_read(alternate, goal)
    assert torch.equal(runtime.state["H"], alternate["H"])
    assert torch.equal(runtime.state["F"], alternate["F"])
    assert torch.equal(before[3], after[3])
    assert not torch.equal(before[1], after[1])


def test_only_observation_time_changes_representation(tmp_path):
    _, runtime = prepared()
    model = runtime.model
    before = model.record_units(runtime.state)[0]
    records = runtime.state["records"]
    alternate = model.set_records(
        runtime.state, (replace(records[0], observed_at=-2.0), records[1])
    )
    after = model.record_units(alternate)[0]
    assert not torch.equal(before, after)
    assert [record.raw for record in records] == [record.raw for record in alternate["records"]]


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_answer_loss_reaches_all_new_metadata_parameters(tmp_path, precision):
    _, runtime = prepared(precision)
    parameters = [
        value
        for name, value in runtime.model.named_parameters()
        if name.startswith(("record_metadata.", "record_provenance."))
    ]
    gradients = torch.autograd.grad(answer_loss(runtime), parameters)
    assert len(gradients) == 3
    assert all(
        bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
        for gradient in gradients
    )


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_selected_group_gains_real_parameters_and_is_neutral(tmp_path, precision):
    _, runtime = prepared(precision)
    before = native_value(runtime).detach()
    old = runtime.model.state_dict()
    original_count = sum(value.numel() for value in runtime.model.parameters())
    runtime.birth_transaction(credit(runtime.model), groups=(0,))
    assert runtime.model.c.group_units == (13, 12)
    assert set(runtime.model.group_extensions) == {"0"}
    assert runtime.model.native_r.shape[1] == 12
    assert (
        sum(value.numel() for value in runtime.model.parameters()) - original_count
        == 9 * runtime.model.c.d
    )
    for name, value in old.items():
        assert torch.equal(value, runtime.model.state_dict()[name]), name
    tolerance = (
        {"atol": 1e-10, "rtol": 1e-8} if precision == "float64" else {"atol": 1e-5, "rtol": 1e-4}
    )
    assert torch.allclose(before, native_value(runtime), **tolerance)


def test_new_incoming_receives_task_credit_and_outgoing_starts_at_zero_credit(tmp_path):
    _, runtime = prepared()
    runtime.birth_transaction(credit(runtime.model), groups=(0,))
    extension = runtime.model.group_extensions["0"]
    incoming, outgoing = torch.autograd.grad(
        answer_loss(runtime), (extension.incoming, extension.outgoing)
    )
    assert float(incoming.norm()) > 0 and bool(torch.isfinite(incoming).all())
    assert torch.equal(outgoing, torch.zeros_like(outgoing))


def test_two_groups_and_repeated_birth_keep_actual_widths_and_lineage(tmp_path):
    _, runtime = prepared()
    runtime.birth_transaction(credit(runtime.model, 1), groups=(1,), birth_id="second-group-first")
    runtime.birth_transaction(credit(runtime.model, 0), groups=(0,), birth_id="first-group-next")
    runtime.birth_transaction(credit(runtime.model, 0), groups=(0,), birth_id="first-group-again")
    assert runtime.model.c.group_units == (14, 13)
    assert [len(row) for row in runtime.model.unit_ids] == [14, 13]
    assert list(runtime.model.group_extensions) == ["0", "1"]
    path = runtime.save(tmp_path / "ragged.bin")
    restored = U2Runtime.load(path)
    assert_values_equal(runtime.model.structure(), restored.model.structure())
    assert value_digest(runtime.model.state_dict()) == value_digest(restored.model.state_dict())
    assert_values_equal(runtime.state, restored.state)
    state, _ = rebuild_state(restored.model, restored.history, restored.state)
    assert_values_equal(restored.state, state)


def test_optimizer_owners_and_repeated_birth_moments_are_preserved(tmp_path):
    _, runtime = prepared()
    model = runtime.model
    rest = [
        value for name, value in model.named_parameters() if name not in {"native_r", "native_p"}
    ]
    runtime.optimizer = torch.optim.Adam(
        [
            {"params": [model.native_r], "lr": 0.001},
            {"params": [model.native_p], "lr": 0.002},
            {"params": rest},
        ],
        lr=0.003,
    )
    runtime.birth_transaction(credit(model), groups=(0,))
    extension = runtime.model.group_extensions["0"]
    runtime.optimizer.state[extension.incoming] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(extension.incoming, 0.2),
        "exp_avg_sq": torch.full_like(extension.incoming, 0.3),
    }
    runtime.optimizer.state[extension.outgoing] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(extension.outgoing, 0.4),
        "exp_avg_sq": torch.full_like(extension.outgoing, 0.5),
    }
    runtime.birth_transaction(credit(runtime.model), groups=(0,))
    extension = runtime.model.group_extensions["0"]
    incoming = runtime.optimizer.state[extension.incoming]
    outgoing = runtime.optimizer.state[extension.outgoing]
    assert torch.equal(incoming["exp_avg"][:1], torch.full_like(incoming["exp_avg"][:1], 0.2))
    assert not bool(incoming["exp_avg"][1:].any())
    assert torch.equal(outgoing["exp_avg"][:, :1], torch.full_like(outgoing["exp_avg"][:, :1], 0.4))
    assert not bool(outgoing["exp_avg"][:, 1:].any())
    assert float(incoming["step"]) == float(outgoing["step"]) == 3
    assert any(value is extension.incoming for value in runtime.optimizer.param_groups[0]["params"])
    assert any(value is extension.outgoing for value in runtime.optimizer.param_groups[1]["params"])
    restored = U2Runtime.from_envelope(runtime.to_envelope())
    assert value_digest(restored.optimizer.state_dict()) == value_digest(
        runtime.optimizer.state_dict()
    )


def test_missing_optimizer_owner_rejects_group_birth(tmp_path):
    _, runtime = prepared()
    runtime.optimizer = torch.optim.Adam(runtime.model.byte_head.parameters())
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError, match="optimizer owner"):
        runtime.birth_transaction(credit(runtime.model), groups=(0,))
    assert value_digest(runtime.to_envelope()) == before


@pytest.mark.parametrize("groups", [(), (0, 0), (-1,), (2,)])
def test_invalid_groups_reject_atomically(tmp_path, groups):
    _, runtime = prepared()
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError):
        runtime.birth_transaction(credit(runtime.model), groups=groups)
    assert value_digest(runtime.to_envelope()) == before


def test_only_selected_credit_is_required(tmp_path):
    _, runtime = prepared()
    runtime.birth_transaction(credit(runtime.model, 1), groups=(1,))
    assert runtime.model.c.group_units == (12, 13)
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError, match="insufficient"):
        runtime.birth_transaction(credit(runtime.model, 1), groups=(0,))
    assert value_digest(runtime.to_envelope()) == before


@pytest.mark.parametrize("mutation", ["ids", "branches", "counts", "schema"])
def test_malformed_ragged_bundle_is_rejected(tmp_path, mutation):
    _, runtime = prepared()
    runtime.birth_transaction(credit(runtime.model), groups=(0,))
    bundle = runtime.model.export_bundle(runtime.state)
    if mutation == "ids":
        bundle["structure"]["unit_ids"][0].pop()
    elif mutation == "branches":
        bundle["structure"]["unit_branches"][0][-1] = 3
    elif mutation == "counts":
        bundle["config"]["group_units"] = (14, 12)
    else:
        bundle["schema"] = "unsupported"
    with pytest.raises((ValueError, RuntimeError)):
        U2Subject.from_bundle(bundle)


def test_legacy_bundle_without_optional_fields_still_loads(tmp_path):
    parent = U2Subject(U2Config.small()).double()
    state = parent.initial()
    bundle = parent.export_bundle(state)
    for key in ("record_metadata", "base_units", "group_units"):
        bundle["config"].pop(key)
    restored, values = U2Subject.from_bundle(bundle)
    assert len(list(restored.parameters())) == 191
    assert value_digest(parent.state_dict()) == value_digest(restored.state_dict())
    assert_values_equal(state, values)
