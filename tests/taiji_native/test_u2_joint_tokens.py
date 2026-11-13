"""Local joint-token credit, masking and branch isolation before subject integration."""

import pytest
import torch

from taiji.u2_joint_tokens import JointTokenFusion, rms


@pytest.fixture(params=[torch.float32, torch.float64])
def case(request):
    torch.set_num_threads(1)
    dtype = request.param
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261113)
        model = JointTokenFusion(8).to(dtype=dtype)
    generator = torch.Generator().manual_seed(37)
    values = [
        torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
        for shape in ((4, 8), (5, 8), (5, 8), (2, 3, 8))
    ]
    return model, values, torch.tensor([True, False, True, True, True])


def test_all_declared_parameters_and_inputs_receive_local_credit(case):
    model, values, valid = case
    query, evidence, metadata, h = [value.requires_grad_() for value in values]
    result = model(query, evidence, metadata, valid, h)
    assert result.shape == (2, 8) and result.square().mean(-1).max() < 1
    target = torch.arange(1, 17, dtype=result.dtype).reshape(2, 8) / 16
    gradients = torch.autograd.grad(
        (result * target).sum(), [*model.parameters(), query, evidence, metadata, h]
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 1696
    assert all(bool(torch.isfinite(value).all()) and bool(value.norm() > 0) for value in gradients)
    assert torch.equal(gradients[-3][1], torch.zeros(8, dtype=result.dtype))
    assert torch.equal(gradients[-2][1], torch.zeros(8, dtype=result.dtype))


def test_invalid_nan_tokens_are_masked_and_all_absent_matches_empty_evidence(case):
    model, (query, evidence, metadata, h), valid = case
    expected = model(query, evidence, metadata, valid, h)
    evidence[1], metadata[1] = torch.nan, torch.nan
    actual = model(query, evidence, metadata, valid, h)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    missing = model(query, evidence, metadata, torch.zeros_like(valid), h)
    empty = model(query, evidence[:0], metadata[:0], valid[:0], h)
    torch.testing.assert_close(missing, empty)


def test_branches_stay_isolated_and_group_evidence_permutations_preserve_readout(case):
    model, (query, evidence, metadata, h), valid = case
    original = model(query, evidence, metadata, valid, h)
    changed = h.clone()
    changed[0] += 0.5
    isolated = model(query, evidence, metadata, valid, changed)
    torch.testing.assert_close(isolated[1], original[1], atol=0, rtol=0)
    assert not torch.equal(isolated[0], original[0])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        query, evidence[permutation], metadata[permutation], valid[permutation], h.flip(1)
    )
    torch.testing.assert_close(permuted, original)


def test_empty_query_is_neutral_and_invalid_shapes_values_and_budgets_are_rejected(case):
    model, (query, evidence, metadata, h), valid = case
    assert torch.equal(model(query[:0], evidence, metadata, valid, h), torch.zeros_like(h[:, 0]))
    with pytest.raises(ValueError, match="input_shape"):
        model(query, evidence, metadata[:, :7], valid, h)
    with pytest.raises(ValueError, match="nonfinite_input"):
        model(query * torch.nan, evidence, metadata, valid, h)
    with pytest.raises(ValueError, match="sequence_budget_stop"):
        model(query[:1].expand(513, -1), evidence, metadata, valid, h)
    with pytest.raises(ValueError, match="native_budget_stop"):
        model(query, evidence, metadata, valid, h[:, :1].expand(-1, 17, -1))
    with pytest.raises(ValueError, match="precision_mismatch"):
        other = torch.float64 if query.dtype == torch.float32 else torch.float32
        model(query.to(dtype=other), evidence, metadata, valid, h)


def test_attention_allocation_is_bounded_before_building_the_matrix():
    model = JointTokenFusion(8, heads=8)
    with pytest.raises(ValueError, match="attention_budget_stop"):
        model(
            torch.zeros(512, 8),
            torch.zeros(128, 8),
            torch.zeros(128, 8),
            torch.ones(128, dtype=torch.bool),
            torch.zeros(8, 16, 8),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rms_zero_has_the_declared_finite_jacobian(dtype):
    value = torch.zeros(4, dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(rms, value)
    torch.testing.assert_close(jacobian, 100 * torch.eye(4, dtype=dtype))
