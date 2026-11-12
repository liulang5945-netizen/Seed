"""Persistent goals, native request assimilation and recoverable answer ownership."""

import copy

import pytest
import torch

from taiji.u2_history import assert_values_equal
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject


def subject(dtype=torch.float32):
    torch.set_num_threads(1)
    model = U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate().request_candidate()
    runtime = U2Runtime(model, subject_id="native-requests")
    runtime.set_goal("依据可核验观测完成任务。")
    return runtime


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_request_fork_inherits_all_parameters(dtype):
    base = U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate()
    candidate = base.request_candidate()
    assert len(list(candidate.parameters())) == 194
    for name, value in base.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    assert len(candidate.state_registry()) == 42
    assert len(base.state_registry()) == 40


def test_request_identity_is_protocol_only_and_content_changes_reading():
    first = subject()
    second = U2Runtime.from_envelope(first.to_envelope())
    third = U2Runtime.from_envelope(first.to_envelope())
    before = copy.copy(first.state)
    first.set_request("one", "红色圆片现在在哪里？")
    second.set_request("two", "蓝色圆片现在在哪里？")
    third.set_request("three", "红色圆片现在在哪里？")
    assert not torch.equal(first.state["H"], second.state["H"])
    assert not torch.equal(first.output()[0], second.output()[0])
    assert torch.equal(first.state["H"], third.state["H"])
    assert torch.equal(first.output()[0], third.output()[0])
    for name in ("goal_raw", "goal_version", "world_time"):
        assert first.state[name] == before[name]
    for name in ("h", "p", "v", "s", "l", "F"):
        assert torch.equal(first.state[name], before[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_answer_credit_reaches_request_and_native_parameters(dtype):
    runtime = subject(dtype)
    runtime.set_request("query", "请输出左边。")
    state = runtime.model.begin_utterance(runtime.state)
    query = runtime.model.language_goal(state)
    parts = []
    for byte in [*"左边".encode(), 256]:
        probability, _, output = runtime.model.output(state, query, state["prediction_action"])
        parts.append(-probability[byte])
        state = runtime.model.accept_byte(state, byte, output)
    loss = torch.stack(parts).mean()
    gradients = torch.autograd.grad(
        loss, [query, runtime.model.text_input.weight, runtime.model.native_w]
    )
    assert all(bool(torch.isfinite(g).all()) and bool(g.any()) for g in gradients)


def test_partial_utterance_request_survives_restore_and_finishes(tmp_path):
    runtime = subject()
    goal = runtime.state["goal_raw"]
    version = runtime.state["goal_version"]
    runtime.set_request("query", "请输出左边。")
    runtime.begin_utterance()
    runtime.accept_byte(0xE5)
    runtime.save(tmp_path / "partial.bin")
    restored = U2Runtime.load(tmp_path / "partial.bin")
    assert_values_equal(runtime.state, restored.state)
    assert restored.state["utterance_request"] == ("query", 1)
    for byte in (0xB7, 0xA6, 256):
        runtime.accept_byte(byte)
        restored.accept_byte(byte)
    assert_values_equal(runtime.state, restored.state)
    assert restored.state["answer_request"]["status"] == "answered"
    assert restored.state["goal_raw"] == goal and restored.state["goal_version"] == version
    rebuilt, _ = rebuild_state(
        restored.model, restored.history, restored.state, reset_eligibility=False
    )
    assert_values_equal(restored.state, rebuilt)


def test_unresolved_request_cannot_be_silently_replaced():
    runtime = subject()
    runtime.set_request("first", "第一个问题。")
    with pytest.raises(ValueError, match="boundary"):
        runtime.set_request("second", "第二个问题。")
    runtime.cancel_request("first", 1)
    runtime.set_request("second", "第二个问题。")
    assert runtime.state["answer_request"]["id"] == "second"


def test_goal_version_and_request_utterance_boundaries():
    runtime = subject()
    runtime.set_request("first", "第一个问题。")
    runtime.begin_utterance()
    with pytest.raises(ValueError, match="boundary"):
        runtime.set_goal("替换任务。")
    with pytest.raises(ValueError, match="boundary"):
        runtime.cancel_request("first", 1)


def test_request_bundle_declares_distinct_schema():
    runtime = subject()
    bundle = runtime.model.export_bundle(runtime.state)
    assert bundle["schema"] == "taiji-u2-subject-v4"
    _, state = U2Subject.from_bundle(bundle)
    assert_values_equal(runtime.state, state)
    bundle["schema"] = "taiji-u2-subject-v3"
    with pytest.raises(ValueError, match="schema"):
        U2Subject.from_bundle(bundle)


def test_request_schema_without_metadata_branch():
    model = U2Subject(U2Config.small()).request_candidate()
    bundle = model.export_bundle(model.initial())
    assert bundle["schema"] == "taiji-u2-subject-v4"
    U2Subject.from_bundle(bundle)


def test_derived_pending_measurement_must_match_request():
    runtime = subject()
    runtime.set_request("one", "问题。")
    runtime.state["nu"][4] = 0
    with pytest.raises(ValueError, match="measurement_mismatch"):
        runtime.model.export_bundle(runtime.state)
