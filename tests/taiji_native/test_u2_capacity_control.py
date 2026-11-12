"""Capacity-control ownership and typed episodic learner checkpoint regressions."""

import copy

import pytest
import torch

from scripts.training.evidence_a_learner_22_5 import fresh_optimizer, load_learner, save_learner
from scripts.training.first_a_support_22_5 import generate_examples, task_loss
from taiji.u2_capacity_control import CapacityResidual, CapacityRuntime, CapacitySubject
from taiji.u2_evidence_runtime import EvidenceRuntime
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save, value_digest
from taiji.u2_subject import U2Config, U2Subject


@pytest.fixture(scope="module")
def examples():
    torch.set_num_threads(1)
    return generate_examples("price", 8)


def parent(dtype):
    return U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate().request_candidate()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_control_has_584_active_parameters_and_inherits_parent(dtype, examples):
    base = parent(dtype)
    model = CapacitySubject.from_parent(base)
    assert sum(p.numel() for p in model.parameters()) == 23326
    assert sum(p.numel() for p in model.evidence_reader.parameters()) == 584
    assert len(list(model.evidence_reader.parameters())) == 5
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    batch = copy.deepcopy(examples[:2])
    for item in batch:
        for group in ("observations", "coordinates"):
            item["public"][group] = {
                key: value.to(dtype=dtype) for key, value in item["public"][group].items()
            }
        item["public"]["outcome"] = item["public"]["outcome"].to(dtype=dtype)
    gradients = torch.autograd.grad(
        task_loss(model, batch), list(model.evidence_reader.parameters())
    )
    assert all(bool(torch.isfinite(g).all()) and bool(g.any()) for g in gradients)


def test_control_only_uses_request_h_and_context():
    torch.manual_seed(19)
    control = CapacityResidual(8)
    request, h, context = torch.randn(8), torch.randn(2, 8), torch.randn(2, 8)
    encoded, metadata = torch.randn(24, 8, requires_grad=True), torch.randn(24, 8)
    valid = torch.ones(24, dtype=torch.bool)
    original = control(request, h, context, encoded, metadata, valid)[0]
    changed = control(request, h, context, encoded * 100, metadata + 3, valid)[0]
    assert torch.equal(original, changed)
    assert torch.autograd.grad(original.sum(), encoded, allow_unused=True)[0] is None
    assert not torch.equal(
        original, control(request.roll(1), h, context, encoded, metadata, valid)[0]
    )


def test_capacity_runtime_has_distinct_bundle_identity(tmp_path):
    model = CapacitySubject.from_parent(parent(torch.float32))
    runtime = CapacityRuntime(model)
    runtime.set_goal("依据观测回答请求。")
    runtime.set_request("query", "红色圆片现在在哪里？")
    runtime.begin_utterance()
    runtime.accept_byte(0xE5)
    runtime.save(tmp_path / "capacity.bin")
    restored = CapacityRuntime.load(tmp_path / "capacity.bin")
    assert_values_equal(runtime.state, restored.state)
    assert_values_equal(runtime.output()[0], restored.output()[0])
    with pytest.raises(ValueError, match="schema_mismatch"):
        EvidenceRuntime.load(tmp_path / "capacity.bin")


@pytest.mark.parametrize("architecture", ["native-old", "native-query", "native-capacity"])
def test_learner_checkpoint_and_next_batch_gradient(tmp_path, examples, architecture):
    base = parent(torch.float32)
    factories = {"native-query": EvidenceSubject, "native-capacity": CapacitySubject}
    model = base if architecture == "native-old" else factories[architecture].from_parent(base)
    optimizer = fresh_optimizer(model)
    path = tmp_path / "learner.bin"
    save_learner(
        path,
        model,
        optimizer,
        architecture,
        {"parent": "fixture"},
        update=0,
        cursor=0,
        order=list(range(8)),
        data_digest=value_digest(examples),
    )
    other, opt, payload = load_learner(path)
    assert not optimizer.state and not opt.state and payload["update"] == 0
    first = task_loss(model, examples[:2])
    second = task_loss(other, examples[:2])
    assert torch.equal(first, second)
    left = torch.autograd.grad(first, list(model.parameters()), allow_unused=True)
    right = torch.autograd.grad(second, list(other.parameters()), allow_unused=True)
    assert_values_equal(left, right)
    with pytest.raises(FileExistsError):
        save_learner(
            path,
            model,
            optimizer,
            architecture,
            {},
            update=0,
            cursor=0,
            order=[],
            data_digest="fixture",
        )
    broken = atomic_load(path)
    broken["source_pins"]["taiji/u2_subject.py"] = "bad"
    atomic_save(broken, tmp_path / "bad.bin")
    with pytest.raises(ValueError, match="source_identity"):
        load_learner(tmp_path / "bad.bin")
