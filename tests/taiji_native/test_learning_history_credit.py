"""Local finite-support checks for the U2 history-risk gradient specification."""

import torch


def fixture():
    theta = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
    probability = theta.softmax(0)
    states = torch.stack(
        [torch.sin(theta[0] + action * theta[1]) + theta[2].square() for action in range(3)]
    )
    downstream = (states - torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64)).square()
    first_loss = theta.square().sum()
    return theta, probability, downstream, first_loss


def test_full_policy_gradient_requires_state_and_distribution_credit():
    theta, probability, downstream, first = fixture()
    objective = first + (probability * downstream).sum()
    exact = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    conditional = torch.autograd.grad(
        first + (probability.detach() * downstream).sum(), theta, retain_graph=True
    )[0]
    baseline = (probability * downstream).sum().detach()
    distribution = sum(
        (probability[action] * (downstream[action] - baseline)).detach()
        * torch.autograd.grad(probability[action].log(), theta, retain_graph=True)[0]
        for action in range(3)
    )
    torch.testing.assert_close(exact, conditional + distribution, atol=1e-12, rtol=1e-12)
    assert float((exact - conditional).abs().max()) > 1e-4


def test_candidate_normalization_and_complete_expectation_hvp():
    theta, probability, downstream, _ = fixture()
    support = torch.tensor([0, 2])
    normalized = probability[support] / probability[support].sum()
    from_logits = theta[support].softmax(0)
    torch.testing.assert_close(normalized, from_logits, atol=1e-12, rtol=1e-12)
    losses = [(normalized * downstream[support]).sum(), (from_logits * downstream[support]).sum()]
    values = []
    direction = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
    for loss in losses:
        gradient = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True)[0]
        hvp = torch.autograd.grad((gradient * direction).sum(), theta, retain_graph=True)[0]
        values.append((gradient, hvp))
    for expected, actual in zip(values[0], values[1], strict=True):
        torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-12)


def test_mean_token_nll_preserves_equal_length_ranks_but_changes_policy():
    log_sequence = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.float64)
    true_conditional = log_sequence.softmax(0)
    from_mean_nll = (log_sequence / 7).softmax(0)
    assert true_conditional.argmax() == from_mean_nll.argmax()
    assert not torch.allclose(true_conditional, from_mean_nll)
