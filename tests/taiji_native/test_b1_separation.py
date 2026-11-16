"""B.1 correction adapter: separated plasticity target, neutral birth, live retention path.

These are zero-training contract tests. They pin the graph accounting, the
bit-exact inheritance, the birth neutrality, the knockout equivalence, the
gradient path (adapter and record channel learn, emission stays frozen) and the
v6 bundle schema. The behavioral claim that retention can fail is pinned here
without any project data: on a record-free state, perturbing the adapter moves
the answer distribution, and the ``correction`` cut restores it exactly.
"""

import pytest
import torch

from taiji.u2_history import assert_values_equal
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject

ADAPTER_TENSORS = {
    "correction_adapter.first.weight",
    "correction_adapter.first.bias",
    "correction_adapter.last.weight",
    "correction_adapter.last.bias",
}


def v4(dtype=torch.float32):
    torch.set_num_threads(1)
    return U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate().request_candidate()


def v6(dtype=torch.float32):
    return v4(dtype).correction_candidate()


def test_correction_fork_inherits_all_and_births_exactly_four():
    base = v4()
    candidate = base.correction_candidate()
    assert len(list(base.parameters())) == 194
    assert sum(p.numel() for p in base.parameters()) == 22742
    assert len(list(candidate.parameters())) == 198
    assert sum(p.numel() for p in candidate.parameters()) == 23150
    born = set(candidate.state_dict()) - set(base.state_dict())
    assert born == ADAPTER_TENSORS
    for name, value in base.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    assert len(candidate.state_registry()) == len(base.state_registry()) == 42
    assert candidate.parameter_revision == base.parameter_revision + 1
    last = candidate.correction_adapter.last
    assert bool((last.weight == 0).all() and (last.bias == 0).all())


def test_config_rejects_invalid_correction_profiles():
    with pytest.raises(ValueError, match="correction"):
        U2Config(correction="v3")
    with pytest.raises(ValueError, match="binding v2"):
        U2Config(binding="v2", correction="v1")
    with pytest.raises(ValueError, match="binding v2"):
        U2Config(binding="v2", correction="v2")


def test_double_fork_is_refused():
    with pytest.raises(ValueError, match="already enabled"):
        v6().correction_candidate()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_birth_is_bit_neutral_on_the_answer_path(dtype):
    base = U2Runtime(v4(dtype), subject_id="b1-base")
    candidate = U2Runtime(v6(dtype), subject_id="b1-candidate")
    for runtime in (base, candidate):
        runtime.set_goal("依据可核验观测完成任务。")
        runtime.set_request("query", "红色圆片现在在哪里？")
    assert torch.equal(base.output()[0], candidate.output()[0])


def test_retention_path_is_live_without_records_and_knockout_is_exact():
    base = U2Runtime(v4(), subject_id="b1-base")
    candidate = U2Runtime(v6(), subject_id="b1-candidate")
    for runtime in (base, candidate):
        runtime.set_goal("依据可核验观测完成任务。")
        runtime.set_request("query", "红色圆片现在在哪里？")
    assert len(candidate.state["records"]) == 0
    before = candidate.output()[0].clone()
    assert torch.equal(before, base.output()[0])
    with torch.no_grad():
        candidate.model.correction_adapter.last.weight.add_(0.5)
    perturbed = candidate.output()[0]
    assert not torch.equal(perturbed, before)
    candidate.model._verification_cuts = frozenset({"correction"})
    assert torch.equal(candidate.output()[0], before)
    candidate.model._verification_cuts = frozenset()
    assert torch.equal(candidate.output()[0], perturbed)


def test_gradient_reaches_adapter_and_record_but_not_emission():
    model = v6()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith("correction_adapter.") or name.startswith("record_"):
            parameter.requires_grad_(True)
    runtime = U2Runtime(model, subject_id="b1-grad")
    runtime.set_goal("依据可核验观测完成任务。")
    runtime.set_request("query", "请输出左边。")
    state = model.begin_utterance(runtime.state)
    query = model.language_goal(state)
    parts = []
    for byte in [*"左边".encode(), 256]:
        probability, _, output = model.output(state, query, state["prediction_action"])
        parts.append(-probability[byte])
        state = model.accept_byte(state, byte, output)
    loss = torch.stack(parts).mean()
    loss.backward()
    adapter_grad = model.correction_adapter.last.weight.grad
    record_grad = model.record_output.weight.grad
    assert adapter_grad is not None
    assert bool(torch.isfinite(adapter_grad).all()) and bool(adapter_grad.any())
    assert record_grad is not None
    assert bool(torch.isfinite(record_grad).all()) and bool(record_grad.any())
    assert model.byte_head.weight.grad is None
    assert model.joint_readout.first.weight.grad is None


def test_v6_bundle_roundtrip_and_schema_mismatches():
    runtime = U2Runtime(v6(), subject_id="b1-bundle")
    runtime.set_goal("依据可核验观测完成任务。")
    bundle = runtime.model.export_bundle(runtime.state)
    assert bundle["schema"] == "taiji-u2-subject-v6"
    assert bundle["config"]["correction"] == "v1"
    _, state = U2Subject.from_bundle(bundle)
    assert_values_equal(runtime.state, state)
    wrong = {**bundle, "schema": "taiji-u2-subject-v4"}
    with pytest.raises(ValueError, match="schema"):
        U2Subject.from_bundle(wrong)
    base_runtime = U2Runtime(v4(), subject_id="b1-base-bundle")
    base_runtime.set_goal("依据可核验观测完成任务。")
    base_bundle = base_runtime.model.export_bundle(base_runtime.state)
    forged = {**base_bundle, "schema": "taiji-u2-subject-v6"}
    with pytest.raises(ValueError, match="schema"):
        U2Subject.from_bundle(forged)


def test_old_bundles_still_load_with_correction_off():
    runtime = U2Runtime(v4(), subject_id="b1-legacy")
    runtime.set_goal("依据可核验观测完成任务。")
    bundle = runtime.model.export_bundle(runtime.state)
    assert "correction" in bundle["config"] and bundle["config"]["correction"] == "off"
    legacy = {**bundle, "config": {k: v for k, v in bundle["config"].items() if k != "correction"}}
    model, _ = U2Subject.from_bundle(legacy)
    assert model.c.correction == "off"
