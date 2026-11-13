"""Normal-entry fusion, causal evidence timing, typed state and optimizer inheritance."""

import copy
from unittest.mock import patch

import pytest
import torch

from scripts.training.evidence_a_learner_22_5 import fresh_optimizer
from scripts.training.first_a_support_22_5 import make_model, output
from scripts.training.joint_token_learner_22_5 import (
    fork_with_optimizer,
    load_learner,
    save_learner,
)
from scripts.training.paired_episode_22_5 import start_episode
from scripts.training.paired_world_22_5 import build_partition
from scripts.training.task_history_loss_22_5 import score_answer
from taiji.u2_events import observation_intent
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_joint_subject import (
    JointGoalRuntime,
    JointGoalSubject,
    JointQueryRuntime,
    JointQuerySubject,
)
from taiji.u2_subject import U2Packet
from taiji.u2_task_goal import ActiveQuerySubject


@pytest.fixture(scope="module")
def partition():
    torch.set_num_threads(1)
    return build_partition("train")


def parent():
    return ActiveQuerySubject.from_parent(EvidenceSubject.from_parent(make_model("native")))


@pytest.mark.parametrize("cls", [JointQuerySubject, JointGoalSubject])
def test_normal_task_has_shared_expression_and_native_credit(partition, cls):
    model = cls.from_parent(parent())
    assert sum(p.numel() for p in model.parameters()) == 25086
    assert len(list(model.parameters())) == 214
    assert len(model.state_registry()) == 44
    assert all(not key.startswith("joint_fusion.") for key, _ in model.policy_named_parameters())
    for cut in ((), ("joint_native",), ("joint_expression",)):
        model._verification_cuts = cut
        state = start_episode(model, partition, partition["episodes"][0])
        assert len(state["evidence_directory"]) == 1
        loss, _ = score_answer(model, state, partition["episodes"][0]["targets"][0])
        values = torch.autograd.grad(loss, list(model.joint_fusion.parameters()))
        assert all(bool(torch.isfinite(value).all()) and bool(value.norm() > 0) for value in values)


def test_closed_fusion_matches_parent_and_retains_old_gradient_ownership(partition):
    original = parent()
    model = JointQuerySubject.from_parent(original)
    model._verification_cuts = ("joint_tokens",)
    episode = partition["episodes"][0]
    before = start_episode(original, partition, episode)
    after = start_episode(model, partition, episode)
    assert_values_equal(before["H"], after["H"])
    assert_values_equal(output(original, before)[0], output(model, after)[0])
    old_loss, _ = score_answer(original, before, episode["targets"][0])
    new_loss, _ = score_answer(model, after, episode["targets"][0])
    old_grad = torch.autograd.grad(old_loss, list(original.parameters()), allow_unused=True)
    new_grad = torch.autograd.grad(new_loss, list(model.parameters()), allow_unused=True)
    assert_values_equal(old_grad, new_grad[: len(old_grad)])
    assert all(value is None for value in new_grad[len(old_grad) :])


def test_current_packet_enters_fusion_after_the_original_correction_transitions(partition):
    original_model = parent()
    model = JointQuerySubject.from_parent(original_model)
    episode = partition["episodes"][0]
    public = partition["worlds"][episode["world_id"]]["public"]
    state = model.set_goal(model.initial(), public["goal"])
    state = observation_intent(model, state, "packet-action", 0.05)
    packet = U2Packet(
        "frame",
        "sensor",
        "world",
        public["t_start"],
        public["t_end"],
        0.0,
        public["observations"],
        public["coordinates"],
        public["outcome"],
        "packet-action",
    )
    prefixes = []
    original = model.joint_feature

    def observe(current):
        prefixes.append(tuple(current["evidence_directory"]))
        return original(current)

    with (
        patch.object(model, "joint_feature", side_effect=observe),
        patch.object(model, "native", wraps=model.native) as native,
    ):
        result, _ = model.assimilate(state, packet)
        fused_calls = native.call_count
    original_state = original_model.set_goal(original_model.initial(), public["goal"])
    original_state = observation_intent(original_model, original_state, "packet-action", 0.05)
    with patch.object(original_model, "native", wraps=original_model.native) as native:
        original_model.assimilate(original_state, packet)
        assert fused_calls == native.call_count == 3
    assert prefixes and all(not prefix for prefix in prefixes)
    assert set(result["evidence_directory"]) == {"frame"}
    assert model.evidence_tokens(result)[0].shape[0] > 0


@pytest.mark.parametrize(
    "cls,runtime_cls,architecture",
    [
        (JointQuerySubject, JointQueryRuntime, "native-joint-query"),
        (JointGoalSubject, JointGoalRuntime, "native-joint-control"),
    ],
)
def test_typed_bundle_and_real_moment_learner_roundtrip(tmp_path, cls, runtime_cls, architecture):
    base = parent()
    optimizer = fresh_optimizer(base)
    base.byte_head.weight.square().sum().backward()
    optimizer.step()
    model, inherited = fork_with_optimizer(base, optimizer, cls)
    assert len(inherited.state) == len(optimizer.state)
    assert all(parameter not in inherited.state for parameter in model.joint_fusion.parameters())
    path = tmp_path / "learner.bin"
    save_learner(
        path,
        model,
        inherited,
        architecture,
        {"parent": "fixture"},
        update=0,
        cursor=0,
        order=[0],
        data_digest="fixture",
    )
    restored, other, _ = load_learner(path)
    assert_values_equal(restored.state_dict(), model.state_dict())
    assert_values_equal(other.state_dict(), inherited.state_dict())
    runtime = runtime_cls(model)
    runtime.set_goal("保持目标。")
    runtime.set_request("one", "红色圆片在哪里？")
    runtime.save(tmp_path / "runtime.bin")
    loaded = runtime_cls.load(tmp_path / "runtime.bin")
    assert value_digest(loaded.state) == value_digest(runtime.state)
    wrong = copy.deepcopy(model.export_bundle(model.initial()))
    wrong["joint_tokens"]["owner"] = "Psi"
    with pytest.raises(ValueError, match="registry"):
        cls.from_bundle(wrong)
