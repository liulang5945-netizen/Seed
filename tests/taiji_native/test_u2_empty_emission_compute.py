"""Empty-copy fast path preserves support, output fields and higher-order credit."""

import json
from pathlib import Path

import pytest
import torch

from scripts.training.verify_paired_compute_22_5 import load_reference
from taiji.u2_operators import emission, initial_cursor

ROOT = Path(__file__).resolve().parents[2]


def fixture(dtype, records, positions):
    torch.manual_seed(32)
    state = initial_cursor(3, records, positions, dtype)
    state["log_m"].requires_grad_()
    state["log_zeta"].requires_grad_()
    state["log_eta"].requires_grad_()
    valid = torch.ones(records, positions, dtype=torch.bool)
    alpha_raw = torch.randn(3, records + 1, dtype=dtype, requires_grad=True)
    if not valid.numel():
        route = torch.zeros_like(alpha_raw, dtype=torch.bool)
        route[:, -1] = True
        alpha = alpha_raw.masked_fill(~route, -torch.inf).log_softmax(-1)
    else:
        alpha = alpha_raw.log_softmax(-1)
    beta_raw = torch.randn(3, records, positions, dtype=dtype, requires_grad=True)
    beta = beta_raw.log_softmax(-1) if positions else beta_raw
    lam = torch.tensor([0.0, 0.4, 1.0], dtype=dtype, requires_grad=True)
    gamma = torch.tensor([1.0, 0.6, 0.0], dtype=dtype, requires_grad=True)
    generator_raw = torch.randn(3, 257, dtype=dtype, requires_grad=True)
    generator_mask = torch.ones_like(generator_raw, dtype=torch.bool)
    generator_mask[:, 13] = False
    generator = generator_raw.masked_fill(~generator_mask, -torch.inf).log_softmax(-1)
    ids = torch.full((records, positions), 65, dtype=torch.long)
    mask = torch.ones(257, dtype=torch.bool)
    mask[14] = False
    inputs = [
        state["log_m"],
        state["log_zeta"],
        state["log_eta"],
        alpha_raw,
        beta_raw,
        lam,
        gamma,
        generator_raw,
    ]
    return (state, alpha, beta, lam, gamma, generator, ids, valid, mask), inputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (2, 0), (1, 3)])
def test_emission_fields_gradients_hvp_and_impossible_support(dtype, shape):
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_5_3_compute_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    reference = load_reference(contract["reference_operator"])
    arguments, inputs = fixture(dtype, *shape)
    expected = reference.emission(*arguments)
    actual = emission(*arguments)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)
    vectors = []
    for result in (expected, actual):
        weights = torch.linspace(0.1, 1.0, 257, dtype=dtype)
        loss = (result["log_output"].exp() * weights).sum()
        gradients = torch.autograd.grad(
            loss, inputs, create_graph=True, retain_graph=True, allow_unused=True
        )
        product = torch.stack(
            [g.sum() for g in gradients if g is not None and g.requires_grad]
        ).sum()
        second = torch.autograd.grad(product, inputs, retain_graph=True, allow_unused=True)
        vectors.append((gradients, second))
    for left, right in zip(vectors[0], vectors[1], strict=True):
        for a, b in zip(left, right, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(
                    a, b, **contract["equivalence"][str(dtype).split(".")[-1]]
                )
    impossible = actual["log_branch"][:, 13].sum()
    gradient = torch.autograd.grad(impossible, inputs[-1], allow_unused=True)[0]
    assert gradient is not None and bool((gradient == 0).all())


def test_invalid_mask_is_rejected_before_fast_path():
    arguments, _ = fixture(torch.float64, 0, 0)
    changed = [*arguments[:-1], torch.ones(256, dtype=torch.bool)]
    with pytest.raises(ValueError, match="Invalid emission dimensions"):
        emission(*changed)
