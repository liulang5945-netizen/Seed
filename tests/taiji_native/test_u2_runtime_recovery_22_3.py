"""Finite-window U2 recovery regressions using synthetic operations and no updates."""

from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from taiji.u2_recovery import ReferenceExecutor, U2Runtime, atomic_load, atomic_save
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def serialized(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def settings(tmp_path, *, executor=None, max_operations=32):
    model = U2Subject(U2Config.small()).double()
    executor = executor or ReferenceExecutor(tmp_path / "executor.bin")
    runtime = U2Runtime(model, executor=executor, max_operations=max_operations)
    fixture = synthetic_fixture(model.c, torch.float64)
    runtime.set_goal(fixture["goal"])
    runtime.set_records(fixture["records"])
    return runtime, fixture


def issue_round(runtime, fixture, tmp_path, index):
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    decision = runtime.choose(legal)
    receipt_id = runtime.issue(decision, dt=0.05)
    runtime.execute_pending(checkpoint=tmp_path / f"intent-{index}.bin")
    runtime.set_process_time((index + 1) * 0.05)
    event = fixture["events"][index]
    return U2Packet(
        f"runtime-packet-{index}",
        "sensor",
        "world",
        index * 0.05,
        (index + 1) * 0.05,
        (index + 1) * 0.05,
        event["observations"],
        event["coordinates"],
        event["outcome"],
        receipt_id,
        feedback_record_ids=("red",),
    )


def pending_after_one_observation(tmp_path):
    runtime, fixture = settings(tmp_path)
    first = issue_round(runtime, fixture, tmp_path, 0)
    runtime.receive(first)
    second = issue_round(runtime, fixture, tmp_path, 1)
    return runtime, fixture, second


def selected_gradients(model, loss, names):
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(
        loss, [parameters[name] for name in names], allow_unused=True, retain_graph=True
    )
    return dict(zip(names, gradients, strict=True))


def assert_same_gradients(before, after):
    assert set(before) == set(after)
    for name, left in before.items():
        right = after[name]
        assert left is not None, name
        assert right is not None, name
        assert bool(torch.isfinite(right).all()), name
        assert torch.allclose(left, right, atol=1e-10, rtol=1e-8), name


def test_window_replay_restores_pending_motion_native_fast_and_critic_credit(tmp_path):
    runtime, _, packet = pending_after_one_observation(tmp_path)
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    live = runtime.receive(packet)
    resumed = restored.receive(packet)
    names = ("initial_h", "motion.first.weight", "native_w", "f_key.weight", "f_value.first.weight")
    first = selected_gradients(runtime.model, -live["predictive_logp"].sum(), names)
    second = selected_gradients(restored.model, -resumed["predictive_logp"].sum(), names)
    assert_same_gradients(first, second)
    assert float(first["motion.first.weight"].norm()) > 0
    assert float(first["f_key.weight"].norm()) > 0
    critic_names = tuple(
        name for name, _ in runtime.model.named_parameters() if name.startswith("critic.")
    )
    assert_same_gradients(
        selected_gradients(runtime.model, live["critic"], critic_names),
        selected_gradients(restored.model, resumed["critic"], critic_names),
    )


def test_values_only_restore_is_a_discriminating_theta_credit_negative_control(tmp_path):
    runtime, _, packet = pending_after_one_observation(tmp_path)
    envelope = serialized(runtime.to_envelope())
    values_model, values_state = U2Subject.from_bundle(envelope["subject_bundle"])
    replayed = U2Runtime.from_envelope(envelope, executor=runtime.executor)
    _, values_receipt = values_model.receive(values_state, packet)
    full_receipt = replayed.receive(packet)
    values_gradient = torch.autograd.grad(
        values_receipt["predictive_logp"].sum(), values_model.motion.first.weight, allow_unused=True
    )[0]
    full_gradient = torch.autograd.grad(
        full_receipt["predictive_logp"].sum(), replayed.model.motion.first.weight, allow_unused=True
    )[0]
    assert values_gradient is None
    assert full_gradient is not None and float(full_gradient.norm()) > 0


def test_pending_values_cannot_become_an_undeclared_truncated_anchor(tmp_path):
    runtime, _, _ = pending_after_one_observation(tmp_path)
    model, state = U2Subject.from_bundle(serialized(runtime.model.export_bundle(runtime.state)))
    with pytest.raises(ValueError, match="committed subject boundary"):
        U2Runtime(model, state=state, executor=runtime.executor)


def test_restored_rng_and_next_decision_reproduce_recorded_actor(tmp_path):
    runtime, _ = settings(tmp_path)
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    legal = torch.zeros(9, dtype=torch.bool)
    legal[[0, 2, 8]] = True
    before = runtime.choose(legal)
    after = restored.choose(legal)
    assert before["index"] == after["index"]
    assert before["operation"] == after["operation"]
    assert torch.equal(before["action"], after["action"])
    assert torch.equal(before["log_behavior"], after["log_behavior"])
    assert torch.equal(runtime.state["rng"], restored.state["rng"])


def test_goal_switch_unknown_outcome_and_duplicate_packet_survive_window_replay(tmp_path):
    runtime, fixture = settings(tmp_path)
    packet = issue_round(runtime, fixture, tmp_path, 0)
    old_goal = runtime.state["pending"]["goal_raw"]
    runtime.set_goal(b"replacement while previous action is pending")
    unknown = packet.outcome.clone()
    unknown[4:] = 0
    assert runtime.receive(replace(packet, outcome=unknown))["status"] == "awaiting_outcome"
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    complete = replace(packet, packet_id="runtime-complete")
    live = runtime.receive(complete)
    resumed = restored.receive(complete)
    assert torch.equal(live["critic"], resumed["critic"])
    assert restored.state["evidence"][-1]["old_goal"] == old_goal
    assert all(not bool(value.any()) for value in restored.state["eligibility"].values())
    assert restored.receive(complete) == {}
    assert restored.state["settled"].count(packet.receipt_id) == 1


def test_mid_character_and_actual_self_output_replay_without_world_fact_write(tmp_path):
    runtime, fixture = settings(tmp_path)
    runtime.receive(issue_round(runtime, fixture, tmp_path, 0))
    world = {key: runtime.state[key].detach().clone() for key in ("p", "v", "s", "F", "logw")}
    evidence_count = len(runtime.state["evidence"])
    runtime.begin_utterance()
    _, _, output = runtime.output()
    runtime.accept_byte(0xE7, output)
    assert runtime.state["utf8_remaining"] == 2
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    assert restored.state["utf8_remaining"] == 2
    assert torch.equal(runtime.output()[0], restored.output()[0])
    for symbol in (0xBA, 0xA2, 256):
        runtime.accept_byte(symbol)
        restored.accept_byte(symbol)
    assert not restored.state["utterance_active"]
    assert torch.equal(runtime.state["H"], restored.state["H"])
    assert len(restored.state["evidence"]) == evidence_count
    for key, value in world.items():
        assert torch.equal(restored.state[key], value), key
    assert any(operation for operation in restored.to_envelope()["window"]["operations"])


def test_internal_replay_and_wait_preserve_physical_clock_on_restore(tmp_path):
    runtime, fixture = settings(tmp_path)
    runtime.receive(issue_round(runtime, fixture, tmp_path, 0))
    before_time = runtime.state["world_time"]
    before_position = runtime.state["p"].detach().clone()
    for index in (4, 8):
        legal = torch.zeros(9, dtype=torch.bool)
        legal[index] = True
        runtime.internal(runtime.choose(legal))
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    assert runtime.state["world_time"] == restored.state["world_time"] == before_time
    assert torch.equal(restored.state["p"], before_position)
    assert torch.equal(runtime.state["F"], restored.state["F"])
    assert restored.state["wake_condition"] == runtime.state["wake_condition"]


@pytest.mark.parametrize("mutation", ["schema", "source", "window", "operations", "parameters"])
def test_bad_envelope_never_falls_back_to_a_values_only_success(tmp_path, mutation):
    runtime, _ = settings(tmp_path)
    envelope = serialized(runtime.to_envelope())
    if mutation == "schema":
        envelope["schema"] = "unsupported-envelope"
    elif mutation == "source":
        key = next(iter(envelope["source_pins"]))
        envelope["source_pins"][key] = "0" * 64
    elif mutation == "window":
        del envelope["window"]
    elif mutation == "operations":
        envelope["window"]["operations"] = []
    else:
        envelope["window"]["parameter_digest"] = "0" * 64
    with pytest.raises(ValueError):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


@pytest.mark.parametrize("mutation", ["truncate", "payload", "header"])
def test_atomic_file_rejects_corruption_before_deserializing(tmp_path, mutation):
    path = tmp_path / "checkpoint.bin"
    payload = {"value": torch.tensor([0.1, 0.2], dtype=torch.float64)}
    atomic_save(payload, path)
    assert torch.equal(atomic_load(path)["value"], payload["value"])
    raw = bytearray(path.read_bytes())
    if mutation == "truncate":
        raw = raw[:-17]
    elif mutation == "payload":
        raw[-17] ^= 1
    else:
        raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        atomic_load(path)


def test_failed_atomic_replace_keeps_previous_checkpoint_and_cleans_staging(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.bin"
    atomic_save({"generation": 0}, path)
    before = path.read_bytes()

    def fail_replace(self, target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_save({"generation": 1}, path)
    assert path.read_bytes() == before
    assert atomic_load(path) == {"generation": 0}
    assert list(tmp_path.iterdir()) == [path]


def test_execute_before_ack_recovery_does_not_repeat_external_effect(tmp_path):
    runtime, fixture = settings(tmp_path)
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    receipt_id = runtime.issue(runtime.choose(legal), dt=0.05)
    checkpoint = tmp_path / "durable-intent.bin"
    with pytest.raises(RuntimeError):
        runtime.execute_pending(checkpoint=checkpoint, fault="after_execute_before_ack")
    assert runtime.executor.effect_count == 1
    executed_ledger = (tmp_path / "executor.bin").read_bytes()
    restored = U2Runtime.load(checkpoint, executor=ReferenceExecutor(tmp_path / "executor.bin"))
    assert (tmp_path / "executor.bin").read_bytes() == executed_ledger
    restored.execute_pending(checkpoint=checkpoint)
    assert restored.executor.effect_count == 1
    restored.set_process_time(0.05)
    event = fixture["events"][0]
    packet = U2Packet(
        "after-crash",
        "sensor",
        "world",
        0.0,
        0.05,
        0.05,
        event["observations"],
        event["coordinates"],
        event["outcome"],
        receipt_id,
    )
    restored.receive(packet)
    assert restored.receive(packet) == {}
    assert restored.state["settled"].count(receipt_id) == 1


def test_uncertain_executor_status_waits_without_reissuing(tmp_path):
    class UnknownExecutor(ReferenceExecutor):
        def query(self, action_id):
            return {"status": "unknown", "action_id": action_id}

    runtime, _ = settings(tmp_path, executor=UnknownExecutor())
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    runtime.issue(runtime.choose(legal), dt=0.05)
    result = runtime.execute_pending(checkpoint=tmp_path / "unknown-intent.bin")
    assert result["status"] == "awaiting_execution_status"
    assert runtime.executor.effect_count == 0
    assert runtime.state["pending"] is not None


def test_window_overflow_is_rejected_before_mutating_rng(tmp_path):
    runtime, _ = settings(tmp_path, max_operations=2)
    before = runtime.state["rng"].clone()
    with pytest.raises(ValueError):
        runtime.choose(torch.ones(9, dtype=torch.bool))
    assert torch.equal(runtime.state["rng"], before)


def test_full_window_rejects_execution_before_any_external_effect(tmp_path):
    runtime, _ = settings(tmp_path, max_operations=4)
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    runtime.issue(runtime.choose(legal))
    ledger = (tmp_path / "executor.bin").read_bytes()
    with pytest.raises(ValueError, match="learning_window_budget_stop"):
        runtime.execute_pending(checkpoint=tmp_path / "over-budget.bin")
    assert runtime.executor.effect_count == 0
    assert (tmp_path / "executor.bin").read_bytes() == ledger
    assert not (tmp_path / "over-budget.bin").exists()


def test_repeated_acknowledgment_is_idempotent_at_window_limit(tmp_path):
    runtime, _ = settings(tmp_path, max_operations=5)
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    identity = runtime.issue(runtime.choose(legal))
    runtime.execute_pending(checkpoint=tmp_path / "acknowledged.bin")
    assert runtime.operation_count == 5
    assert runtime.execute_pending() == {"status": "acknowledged", "action_id": identity}
    assert runtime.operation_count == 5
    assert runtime.executor.effect_count == 1


@pytest.mark.parametrize("mutation", ["H", "F", "logw", "records", "eligibility", "resources"])
def test_invalid_value_anchor_cannot_certify_itself_by_roundtrip(tmp_path, mutation):
    runtime, _ = settings(tmp_path)
    runtime.begin_new_window()
    envelope = serialized(runtime.to_envelope())
    for state in (envelope["window"]["anchor"]["state"], envelope["subject_bundle"]["state"]):
        if mutation == "H":
            state["H"].fill_(2)
        elif mutation == "F":
            state["F"].fill_(10)
        elif mutation == "logw":
            state["logw"] = None
        elif mutation == "records":
            state["records"] = (None,)
        elif mutation == "eligibility":
            state["eligibility"][next(iter(state["eligibility"]))] = None
        else:
            del state["resources"]["events"]
    with pytest.raises(ValueError):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


@pytest.mark.parametrize("operation", ["window", "birth"])
def test_parameter_change_cannot_be_blessed_as_a_new_boundary(tmp_path, operation):
    runtime, _ = settings(tmp_path)
    parent = runtime.model
    with torch.no_grad():
        parent.native_w.add_(0.1)
    with pytest.raises(ValueError, match="Frozen window parameters"):
        if operation == "window":
            runtime.begin_new_window()
        else:
            runtime.birth_transaction(birth_credit(parent))
    assert runtime.model is parent
    assert runtime.model.c.units == U2Config.small().units


def test_source_change_cannot_be_relabelled_at_save_or_boundary(tmp_path, monkeypatch):
    runtime, _ = settings(tmp_path)
    monkeypatch.setattr("taiji.u2_recovery._source_pins", lambda: {"changed": "0" * 64})
    for operation in (runtime.to_envelope, runtime.begin_new_window):
        with pytest.raises(ValueError, match="source changed"):
            operation()


@pytest.mark.parametrize("location", ["runtime", "anchor_runtime", "anchor", "operation"])
def test_unregistered_runtime_nested_fields_are_rejected(tmp_path, location):
    runtime, _ = settings(tmp_path)
    envelope = serialized(runtime.to_envelope())
    if location == "runtime":
        envelope["runtime"]["unregistered"] = 1
    elif location == "operation":
        envelope["window"]["operations"][0]["unregistered"] = 1
    else:
        envelope["window"][location]["unregistered"] = 1
    with pytest.raises(ValueError):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


def birth_credit(model):
    credit = torch.zeros((model.c.groups, model.c.d, 8 * model.c.d), dtype=torch.float64)
    credit[:, 0, 0] = 3
    credit[:, 1, 1] = 2
    return credit


def test_failed_shadow_birth_keeps_parent_and_optimizer_untouched(tmp_path):
    runtime, _ = settings(tmp_path)
    optimizer = torch.optim.Adam(runtime.model.parameters(), lr=1e-3)
    runtime.optimizer = optimizer
    model = runtime.model
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    structure = copy.deepcopy(model.structure())
    with pytest.raises(RuntimeError):
        runtime.birth_transaction(birth_credit(model), birth_id="fault-birth", fault="after_birth")
    assert runtime.model is model
    assert runtime.optimizer is optimizer
    assert model.structure() == structure
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_two_shadow_births_preserve_moments_lineage_and_neutral_outputs(tmp_path):
    runtime, _ = settings(tmp_path)
    optimizer = torch.optim.Adam(runtime.model.parameters(), lr=1e-3)
    runtime.optimizer = optimizer
    for parameter in (runtime.model.native_r, runtime.model.native_p):
        optimizer.state[parameter] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(parameter, 0.2),
            "exp_avg_sq": torch.full_like(parameter, 0.3),
        }
    original_ids = copy.deepcopy(runtime.model.unit_ids)
    parent_units = runtime.model.c.units
    old_r_moment = optimizer.state[runtime.model.native_r]["exp_avg"].clone()
    old_p_moment = optimizer.state[runtime.model.native_p]["exp_avg"].clone()
    event = torch.full((runtime.model.c.branches, runtime.model.c.d), 0.1, dtype=torch.float64)
    goal = runtime.model.goal(runtime.state["goal_raw"])
    action = torch.zeros(15, dtype=torch.float64)
    before = runtime.model.native(runtime.state, goal, action, event).detach()
    for generation, branch in ((1, 0), (2, 1)):
        runtime.birth_transaction(
            birth_credit(runtime.model), branch=branch, birth_id=f"generation-{generation}"
        )
        assert runtime.model.c.units == parent_units + generation
        assert runtime.state["structure"] == runtime.model.structure()
        assert all(
            ids[:parent_units] == old
            for ids, old in zip(runtime.model.unit_ids, original_ids, strict=True)
        )
        after = runtime.model.native(
            runtime.state, runtime.model.goal(runtime.state["goal_raw"]), action, event
        )
        assert torch.allclose(before, after, atol=1e-10, rtol=0)
    r_moment = runtime.optimizer.state[runtime.model.native_r]["exp_avg"]
    p_moment = runtime.optimizer.state[runtime.model.native_p]["exp_avg"]
    assert torch.equal(r_moment[:, :parent_units], old_r_moment)
    assert torch.equal(p_moment[..., :parent_units], old_p_moment)
    assert not bool(r_moment[:, parent_units:].any())
    assert not bool(p_moment[..., parent_units:].any())
    restored = U2Runtime.from_envelope(serialized(runtime.to_envelope()), executor=runtime.executor)
    assert restored.model.structure() == runtime.model.structure()
