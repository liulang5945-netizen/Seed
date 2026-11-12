"""Normal-entry evidence credit, provenance and recovery regressions."""

import copy
from dataclasses import replace

import pytest
import torch

from taiji.u2_evidence_runtime import EvidenceRuntime
from taiji.u2_evidence_subject import EvidenceSubject, evidence_digest
from taiji.u2_history import assert_values_equal, detached_tree
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_subject import U2Config, U2Subject
from taiji.u2_task_executor import PhysicalExecutor
from taiji.u2_task_world import PhysicalWorld


def make_runtime(root, dtype=torch.float32):
    torch.set_num_threads(1)
    parent = U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate().request_candidate()
    model = EvidenceSubject.from_parent(parent)
    executor = PhysicalExecutor.create(
        PhysicalWorld.reset(34, {"body_count": 2}), root / "world.bin"
    )
    runtime = EvidenceRuntime(model, executor=executor, subject_id="evidence-reader")
    runtime.set_goal("依据公开观测回答请求。")
    return parent, runtime


def observe(runtime, root, identity="frame"):
    runtime.begin_observation(identity)
    runtime.execute_pending(root / "intent.bin")
    packet = runtime.executor.packet(runtime)
    runtime.receive(packet)
    return packet


def answer_loss(runtime):
    model, state = runtime.model, runtime.state
    state = model.begin_utterance(state)
    losses = []
    for symbol in [*"左边".encode(), 256]:
        probability, _, output = model.output(
            state, model.language_goal(state), state["prediction_action"]
        )
        losses.append(-probability[symbol])
        state = model.accept_byte(state, symbol, output)
    return torch.stack(losses).mean()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inheritance_normal_answer_credit_and_composition(tmp_path, dtype):
    parent, runtime = make_runtime(tmp_path, dtype)
    for name, parameter in parent.state_dict().items():
        assert torch.equal(parameter, runtime.model.state_dict()[name]), name
    assert len(runtime.model.state_registry()) == 44
    assert sum(p.numel() for p in runtime.model.parameters()) == 23326
    observe(runtime, tmp_path)
    assert runtime.model.evidence_tokens(runtime.state)[0].shape[0] == 24
    runtime.set_request("question", "红色圆片现在在哪里？")
    loss = answer_loss(runtime)
    names, owners = zip(*runtime.model.evidence_reader.named_parameters(), strict=True)
    gradients = torch.autograd.grad(
        loss, [*owners, runtime.model.encoder["rgb"].weight], retain_graph=True
    )
    assert all(bool(torch.isfinite(g).all()) and bool(g.any()) for g in gradients)
    assert len(names) == 7
    model, state = runtime.model, runtime.state
    encoded, metadata = model.evidence_tokens(state)
    request = model.language_goal(state)
    h = model.read_h(state, request)
    original = U2Subject.branch_context(model, state, request, state["prediction_action"])
    _, attention, _ = model.evidence_reader(
        request, h, original, encoded, metadata, torch.ones(encoded.shape[0], dtype=torch.bool)
    )
    derivative = torch.autograd.grad(
        attention[1].square().sum(), model.evidence_reader.value.weight
    )[0]
    assert bool(derivative.any())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_restore_partial_expression_and_reencoding_twice(tmp_path, dtype):
    _, runtime = make_runtime(tmp_path, dtype)
    observe(runtime, tmp_path)
    runtime.set_request("question", "红色圆片现在在哪里？")
    runtime.begin_utterance()
    runtime.accept_byte(0xE5)
    runtime.save(tmp_path / "subject.bin")
    restored = EvidenceRuntime.load(tmp_path / "subject.bin")
    assert isinstance(restored.model, EvidenceSubject)
    assert_values_equal(runtime.state, restored.state)
    assert_values_equal(runtime.output()[0], restored.output()[0])
    for symbol in (0xB7, 0xA6, 256):
        runtime.accept_byte(symbol)
        restored.accept_byte(symbol)
    assert_values_equal(runtime.state, restored.state)
    for _ in range(2):
        with torch.no_grad():
            runtime.model.encoder["rgb"].weight.add_(0.001)
        runtime.model.mark_parameter_update()
        state, _ = rebuild_state(runtime.model, runtime.history, runtime.state)
        before, _ = runtime.model.evidence_tokens(runtime.state)
        after, _ = runtime.model.evidence_tokens(state)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        assert_values_equal(state["evidence_directory"], runtime.state["evidence_directory"])
        runtime.state = state


def test_duplicate_missing_tamper_and_revocation(tmp_path):
    _, runtime = make_runtime(tmp_path)
    packet = observe(runtime, tmp_path)
    before = copy.deepcopy(runtime.state["evidence_directory"])
    runtime.receive(packet)
    assert runtime.state["evidence_directory"] == before
    missing = detached_tree(runtime.state)
    missing["evidence"].clear()
    with pytest.raises(ValueError, match="missing_canonical"):
        runtime.model.evidence_tokens(missing)
    tampered = detached_tree(runtime.state)
    tampered["evidence"][0]["observations"]["rgb"].zero_()
    with pytest.raises(ValueError, match="digest_mismatch"):
        runtime.model.evidence_tokens(tampered)
    runtime.revoke_evidence(packet.packet_id)
    assert runtime.model.evidence_tokens(runtime.state)[0].shape == (0, 8)
    runtime.save(tmp_path / "revoked.bin")
    restored = EvidenceRuntime.load(tmp_path / "revoked.bin")
    assert_values_equal(runtime.state, restored.state)
    rebuilt, _ = rebuild_state(restored.model, restored.history, restored.state)
    assert_values_equal(restored.state["revoked_evidence"], rebuilt["revoked_evidence"])
    envelope = restored.to_envelope()
    row = envelope["subject_bundle"]["state"]["evidence"][0]
    row["observations"]["rgb"].zero_()
    envelope["subject_bundle"]["state"]["evidence_directory"][row["packet_id"]] = evidence_digest(
        row
    )
    with pytest.raises(ValueError, match="Recovery value mismatch"):
        EvidenceRuntime.from_envelope(envelope)


def test_request_and_sensor_interventions_change_normal_output(tmp_path):
    _, runtime = make_runtime(tmp_path)
    observe(runtime, tmp_path)
    runtime.set_request("question", "红色圆片现在在哪里？")
    model, state = runtime.model, runtime.state
    goal = model.language_goal(state)
    actual = model.branch_context(state, goal, state["prediction_action"])
    changed = model.branch_context(
        state, model.goal("蓝色圆片现在在哪里？".encode()), state["prediction_action"]
    )
    assert not torch.equal(actual, changed)
    model._verification_cuts = ("query_evidence",)
    cut = model.branch_context(state, goal, state["prediction_action"])
    assert not torch.equal(actual, cut)
    assert_values_equal(
        cut, U2Subject.branch_context(model, state, goal, state["prediction_action"])
    )


def test_fifth_active_event_stops_without_silent_eviction(tmp_path):
    _, runtime = make_runtime(tmp_path)
    for i in range(4):
        observe(runtime, tmp_path, f"frame-{i}")
    before = dict(runtime.state["evidence_directory"])
    with pytest.raises(ValueError, match="evidence_event_budget_stop"):
        observe(runtime, tmp_path, "frame-4")
    assert runtime.state["evidence_directory"] == before


@pytest.mark.parametrize("field,value", [("source", "host"), ("scope", "private")])
def test_source_scope_rejected_even_with_matching_digest(tmp_path, field, value):
    _, runtime = make_runtime(tmp_path)
    observe(runtime, tmp_path)
    state = detached_tree(runtime.state)
    row = state["evidence"][0]
    row[field] = value
    state["evidence_directory"][row["packet_id"]] = evidence_digest(row)
    with pytest.raises(ValueError, match="source_not_admitted"):
        runtime.model.evidence_tokens(state)


def test_incomplete_observation_is_not_admitted_until_settled(tmp_path):
    _, runtime = make_runtime(tmp_path)
    runtime.begin_observation("incomplete")
    runtime.execute_pending(tmp_path / "intent.bin")
    packet = runtime.executor.packet(runtime)
    partial = replace(packet, outcome=torch.zeros_like(packet.outcome))
    runtime.receive(partial)
    assert runtime.state["pending"] is not None
    assert not runtime.state["evidence_directory"]
    assert runtime.model.evidence_tokens(runtime.state)[0].shape[0] == 0


def test_evidence_clock_and_catalog_completeness(tmp_path):
    _, runtime = make_runtime(tmp_path)
    observe(runtime, tmp_path, "first")
    observe(runtime, tmp_path, "second")
    encoded, coordinates = runtime.model.evidence_tokens(runtime.state)
    assert encoded.shape[0] == 48
    assert bool((coordinates[:24, 3] < coordinates[24:, 3]).all())
    state = detached_tree(runtime.state)
    state["evidence_directory"].clear()
    with pytest.raises(ValueError, match="incomplete_evidence_directory"):
        runtime.model.export_bundle(state)


def test_modality_and_token_budget_rejected(tmp_path):
    _, runtime = make_runtime(tmp_path)
    observe(runtime, tmp_path)
    state = detached_tree(runtime.state)
    row = state["evidence"][0]
    row["observations"] = {"rgb": torch.zeros(129, 48)}
    row["coordinates"] = {"rgb": torch.zeros(129, 8)}
    state["evidence_directory"][row["packet_id"]] = evidence_digest(row)
    with pytest.raises(ValueError, match="token_budget_stop"):
        runtime.model.evidence_tokens(state)
    row["observations"] = {"private": torch.zeros(1, 48)}
    row["coordinates"] = {"private": torch.zeros(1, 8)}
    state["evidence_directory"][row["packet_id"]] = evidence_digest(row)
    with pytest.raises(ValueError, match="modality_not_admitted"):
        runtime.model.evidence_tokens(state)
