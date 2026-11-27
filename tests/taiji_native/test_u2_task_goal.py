"""Neutral task conditioning, Theta ownership and actual history regression guards."""

import pytest
import torch

from scripts.training.evidence_a_learner_22_5 import fresh_optimizer
from scripts.training.first_a_support_22_5 import accept, make_model, output
from scripts.training.paired_episode_22_5 import episode_loss
from scripts.training.paired_world_22_5 import build_partition
from scripts.training.task_goal_learner_22_5 import fork_with_optimizer, load_learner, save_learner
from scripts.training.task_history_loss_22_5 import history_loss
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_task_goal import (
    ActiveQueryRuntime,
    ActiveQuerySubject,
    PersistentGoalRuntime,
    PersistentGoalSubject,
)


@pytest.fixture(scope="module")
def data():
    torch.set_num_threads(1)
    return build_partition("train")


def base(dtype=torch.float32):
    return EvidenceSubject.from_parent(make_model("native")).to(dtype=dtype)


@pytest.mark.parametrize("cls", [ActiveQuerySubject, PersistentGoalSubject])
def test_projection_registers_only_theta_and_preserves_old_order(cls):
    parent = base()
    model = cls.from_parent(parent)
    assert sum(p.numel() for p in model.parameters()) == 23390
    assert len(list(model.parameters())) == 202
    assert list(dict(model.named_parameters()))[:-1] == list(dict(parent.named_parameters()))
    assert len(model.state_registry()) == 44
    assert list(dict(model.policy_named_parameters())) == list(
        dict(parent.policy_named_parameters())
    )
    assert "task" in model.parameter_owners()["native_task_projection.weight"]
    assert "actor" not in model.parameter_owners()["native_task_projection.weight"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_projection_is_neutral_and_inactive_after_answer(data, dtype):
    parent = base(dtype)
    models = [ActiveQuerySubject.from_parent(parent), PersistentGoalSubject.from_parent(parent)]
    state = parent.initial("same")
    state = parent.set_goal(state, "测试持续目标。")
    for model in models:
        child = model.initial("same")
        child = model.set_goal(child, "测试持续目标。")
        original = parent.set_request(state, "one", "红色圆片现在在哪里？")
        current = model.set_request(child, "one", "红色圆片现在在哪里？")
        assert_values_equal(original["H"], current["H"])
        old_policy, new_policy = parent.policy(original), model.policy(current)
        for key in old_policy.keys() - {"psi_version"}:
            assert_values_equal(old_policy[key], new_policy[key], f"neutral.policy.{key}")
        model_state = model.begin_utterance(current)
        parent_state = parent.begin_utterance(original)
        for symbol in [*"未知".encode(), 256]:
            left, _, details_left = output(parent, parent_state)
            right, _, details_right = output(model, model_state)
            assert_values_equal(left, right)
            parent_state = accept(parent, parent_state, symbol, details_left)
            model_state = accept(model, model_state, symbol, details_right)
        assert_values_equal(parent_state["H"], model_state["H"])
        # Pending/answering masks have finished, so a nonzero projection is inactive.
        with torch.no_grad():
            model.native_task_projection.weight.fill_(0.2)
        goal, action = model.goal(model_state["goal_raw"]), model.native_w.new_zeros(15)
        a = model.native(model_state, goal, action, goal[None].expand(model.c.branches, -1))
        b = EvidenceSubject.native(
            model, model_state, goal, action, goal[None].expand(model.c.branches, -1)
        )
        assert_values_equal(a, b)


@pytest.mark.parametrize(
    "runtime_cls,model_cls",
    [(ActiveQueryRuntime, ActiveQuerySubject), (PersistentGoalRuntime, PersistentGoalSubject)],
)
def test_typed_partial_restore_and_inherited_optimizer(tmp_path, data, runtime_cls, model_cls):
    parent = base()
    optimizer = fresh_optimizer(parent)
    for parameter in parent.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(parameter, 0.1),
            "exp_avg_sq": torch.full_like(parameter, 0.2),
        }
    model, child_optimizer = fork_with_optimizer(parent, optimizer, model_cls)
    assert model.native_task_projection.weight not in child_optimizer.state
    checkpoint = tmp_path / "parent.bin"
    architecture = "native-goal-query" if model_cls is ActiveQuerySubject else "native-goal-control"
    save_learner(
        checkpoint,
        model,
        child_optimizer,
        architecture,
        {"scope": "synthetic moment fixture"},
        update=0,
        cursor=0,
        order=[],
        data_digest=value_digest(data),
    )
    other, restored_optimizer, _ = load_learner(checkpoint)
    assert_values_equal(model.state_dict(), other.state_dict())
    assert_values_equal(child_optimizer.state_dict(), restored_optimizer.state_dict())
    runtime = runtime_cls(model)
    runtime.set_goal("持续目标。")
    runtime.set_request("query", "红色圆片现在在哪里？")
    runtime.begin_utterance()
    runtime.accept_byte(0xE6)
    runtime.save(tmp_path / "partial.bin")
    restored = runtime_cls.load(tmp_path / "partial.bin")
    assert_values_equal(runtime.state, restored.state)
    assert_values_equal(runtime.output()[0], restored.output()[0])
    wrong_runtime = (
        PersistentGoalRuntime if runtime_cls is ActiveQueryRuntime else ActiveQueryRuntime
    )
    with pytest.raises(ValueError, match="task_goal_schema_mismatch"):
        wrong_runtime.load(tmp_path / "partial.bin")


def test_teacher_loss_is_original_and_actual_history_is_distinct(data):
    model = base()
    episode = data["episodes"][0]
    original = episode_loss(model, data, episode)
    result = history_loss(model, data, episode, "teacher")
    torch.testing.assert_close(original, result["loss"])
    assert result["first_history_bytes"] == episode["targets"][0]
    actual = history_loss(model, data, episode, "actual-self")
    assert actual["history_source"] == "actual-self"
    assert actual["selection_credit"].startswith("conditional_state_only")
    with pytest.raises(ValueError, match="unsupported_task_history_source"):
        history_loss(model, data, episode, "unregistered")


def test_projection_activation_cancel_and_goal_version():
    model = ActiveQuerySubject.from_parent(base())
    state = model.set_goal(model.initial(), "原始目标。")
    state = model.set_request(state, "one", "红色圆片现在在哪里？")
    changed = model.set_goal(state, "新目标。")
    assert changed["answer_request"]["goal_version"] < changed["goal_version"]
    canceled = model.cancel_request(changed, "one", 1)
    with torch.no_grad():
        model.native_task_projection.weight.copy_(torch.eye(8))
    goal, action = model.goal(changed["goal_raw"]), model.native_w.new_zeros(15)
    event = goal[None].expand(model.c.branches, -1)
    assert not torch.equal(
        model.native(changed, goal, action, event),
        EvidenceSubject.native(model, changed, goal, action, event),
    )
    assert_values_equal(
        model.native(canceled, goal, action, event),
        EvidenceSubject.native(model, canceled, goal, action, event),
    )


def test_actual_history_never_uses_the_teacher_state(monkeypatch):
    from scripts.training import task_history_loss_22_5 as trainer

    parameter = torch.tensor(2.0, requires_grad=True)
    seen = []
    episode = {"targets": (b"gold-first", b"gold-second")}
    monkeypatch.setattr(trainer, "start_episode", lambda *_: {"turn": 0, "origin": "initial"})

    def score(_model, state, _target):
        if state["turn"] == 0:
            return parameter.square(), {"turn": 0, "origin": "teacher"}
        seen.append(state["origin"])
        return (3 * parameter).square(), state

    monkeypatch.setattr(trainer, "score_answer", score)
    monkeypatch.setattr(
        trainer,
        "generate_first",
        lambda *_: ({"turn": 0, "origin": "actual"}, b"wrong-self-output", True),
    )
    monkeypatch.setattr(
        trainer, "next_request", lambda _m, state, _e, turn: {**state, "turn": turn}
    )
    result = trainer.history_loss(object(), {}, episode, "actual-self")
    assert seen == ["actual"] and result["first_history_bytes"] == b"wrong-self-output"
    torch.testing.assert_close(result["loss"], torch.tensor(20.0))


def test_unfinished_actual_history_keeps_blocked_cost_and_first_credit(monkeypatch):
    from scripts.training import task_history_loss_22_5 as trainer

    parameter = torch.tensor(2.0, requires_grad=True)
    monkeypatch.setattr(trainer, "start_episode", lambda *_: {})
    monkeypatch.setattr(
        trainer, "score_answer", lambda *_: (parameter.square(), {"origin": "teacher-ended"})
    )
    monkeypatch.setattr(
        trainer, "generate_first", lambda *_: ({"origin": "actual-open"}, b"unfinished", False)
    )

    def forbidden(*_args):
        raise AssertionError("An unfinished answer must not be replaced with teacher history")

    monkeypatch.setattr(trainer, "next_request", forbidden)
    result = trainer.history_loss(object(), {}, {"targets": (b"one", b"two")}, "actual-self")
    torch.testing.assert_close(result["loss"], torch.tensor(2.5))
    torch.testing.assert_close(torch.autograd.grad(result["loss"], parameter)[0], torch.tensor(2.0))
    assert result["blocked_outcome"] == 1 and result["second_ce"] == 0
