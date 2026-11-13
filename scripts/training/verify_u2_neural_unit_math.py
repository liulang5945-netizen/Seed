"""Local witnesses for the stateful-unit proposal; no U2 runtime or training."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_u2_neural_unit_math_20261113.json"


def rollout(x, gates, r, p, w, c, eta, b, z, a):
    """Synchronous scalar units; x is an externally fixed legal input sequence."""
    outputs, states, adaptations = [], [], []
    for inputs, gate in zip(x, gates, strict=True):
        drive = r @ inputs + w @ (gate * z) - b * a
        z = c * z + (1 - c) * drive.tanh()
        a = (1 - eta) * a + eta * z
        outputs.append(p @ (gate * z))
        states.append(z)
        adaptations.append(a)
    return torch.stack(outputs), torch.stack(states), torch.stack(adaptations)


def verify(dtype):
    gen = torch.Generator().manual_seed(2026111307)

    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, generator=gen)

    checks = []
    atol = 2e-6 if dtype == torch.float32 else 1e-12

    def equal(name, actual, expected):
        error = float((actual - expected).detach().abs().max())
        checks.append({"name": name, "max_abs_error": error, "tolerance": atol})
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)

    def witness(name, condition, **values):
        checks.append({"name": name, **values})
        if not condition:
            raise AssertionError(name)

    x = rand(12, 3)
    gates = rand(12, 3).sigmoid()
    r, p = rand(3, 3) * 0.3, rand(2, 3) * 0.3
    zeros, eta = torch.zeros(3, dtype=dtype), torch.full((3,), 0.1, dtype=dtype)
    w = torch.zeros(3, 3, dtype=dtype)
    y, _, _ = rollout(x, gates, r, p, w, zeros, eta, zeros, zeros, zeros)
    equal("old_RP_compatibility", y, ((x @ r.T).tanh() * gates) @ p.T)

    c = torch.tensor([0.1, 0.5, 0.9], dtype=dtype)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=dtype)
    w = rand(3, 3) * 0.5
    initial_z, initial_a = rand(3).tanh(), rand(3).tanh()
    y, z, a = rollout(x, gates, r, p, w, c, eta, b, initial_z, initial_a)
    witness("bounded_states", bool(z.abs().max() <= 1 and a.abs().max() <= 1))

    # Zero input/edges/adaptation and nonzero output: exact neutral birth.
    r_old, p_old, w_old = [v.clone().requires_grad_() for v in (r, p, w)]
    reference, _, _ = rollout(x, gates, r_old, p_old, w_old, c, eta, b, initial_z, initial_a)
    r_new = torch.cat([r_old, torch.zeros(1, 3, dtype=dtype)])
    p_new = torch.cat([p_old, rand(2, 1)], 1)
    w_new = torch.nn.functional.pad(w_old, (0, 1, 0, 1))
    new_gates = torch.cat([gates, torch.full((12, 1), 0.7, dtype=dtype)], 1)
    child, child_z, _ = rollout(
        x,
        new_gates,
        r_new,
        p_new,
        w_new,
        torch.cat([c, c.new_tensor([0.5])]),
        torch.cat([eta, eta.new_tensor([0.1])]),
        torch.cat([b, zeros[:1]]),
        torch.cat([initial_z, zeros[:1]]),
        torch.cat([initial_a, zeros[:1]]),
    )
    equal("neutral_birth_outputs", child, reference)
    equal("neutral_birth_states", child_z[:, :3], z)
    equal("untrained_new_unit_stays_zero", child_z[:, 3], x.new_zeros(12))
    parent_grad = torch.autograd.grad(reference.square().sum(), (r_old, p_old, w_old))
    child_grad = torch.autograd.grad(child.square().sum(), (r_old, p_old, w_old))
    for name, actual, expected in zip(("r", "p", "w"), child_grad, parent_grad, strict=True):
        equal(f"neutral_old_gradient_{name}", actual, expected)

    # Analytic derivative at birth, including temporal input credit.
    direction, delta = rand(2), rand(12, 2)
    gate = new_gates[:, -1]
    for carry in (0.0, 0.5, 0.9):
        incoming = torch.zeros(3, dtype=dtype, requires_grad=True)

        def objective(row, carry=carry):
            state = x.new_zeros(())
            loss = x.new_zeros(())
            for step in range(len(x)):
                state = carry * state + (1 - carry) * (row @ x[step]).tanh()
                loss = loss + gate[step] * state * (direction @ delta[step])
            return loss

        measured = torch.autograd.grad(objective(incoming), incoming)[0]
        eligibility, expected = x.new_zeros(3), x.new_zeros(3)
        for step in range(len(x)):
            eligibility = carry * eligibility + (1 - carry) * x[step]
            expected = expected + gate[step] * (direction @ delta[step]) * eligibility
        equal(f"birth_credit_c_{carry}", measured, expected)
        if dtype == torch.float64:
            h = 1e-4
            basis = torch.eye(3, dtype=dtype) * h
            numerical = torch.stack([(objective(v) - objective(-v)) / (2 * h) for v in basis])
            cubic, absolute = x.new_zeros(3), x.new_zeros(3)
            bound, magnitude = x.new_zeros(3), x.new_zeros(3)
            for step in range(len(x)):
                cubic = carry * cubic + (1 - carry) * x[step].abs().pow(3)
                absolute = carry * absolute + (1 - carry) * x[step].abs()
                coefficient = (gate[step] * (direction @ delta[step])).abs()
                bound = bound + coefficient * cubic * h**2 / 3
                magnitude = magnitude + coefficient * absolute
            # |tanh(u)-u| <= |u|^3/3, propagated through the linear carry.
            # The rounding guard is an engineering tolerance, not a libm proof.
            bound = bound + 64 * torch.finfo(dtype).eps * magnitude
            error = (numerical - expected).abs()
            witness(
                f"finite_difference_c_{carry}",
                bool((error <= bound).all()),
                max_abs_error=float(error.max()),
                max_derived_bound=float(bound.max()),
            )
        if carry:
            old_formula = ((gate * (delta @ direction))[:, None] * x).sum(0)
            difference = float((measured - old_formula).norm())
            witness(f"instantaneous_credit_wrong_c_{carry}", difference > 0.01, gap=difference)

    # Complete recurrent widening: clone unit 1, divide every source column.
    mapping = torch.tensor([0, 1, 2, 1])
    multiplicity = torch.bincount(mapping).to(dtype)
    denom = multiplicity[mapping]
    split, split_z, split_a = rollout(
        x,
        gates[:, mapping],
        r[mapping],
        p[:, mapping] / denom,
        w[mapping][:, mapping] / denom,
        c[mapping],
        eta[mapping],
        b[mapping],
        initial_z[mapping],
        initial_a[mapping],
    )
    equal("recurrent_split_outputs", split, y)
    equal("recurrent_split_states", split_z, z[:, mapping])
    equal("recurrent_split_adaptation", split_a, a[:, mapping])

    # Rank bounds independent directions, not the count of nonzero gradients.
    credit = torch.tensor([[1.0, 0], [0, 0]], dtype=dtype)
    output_columns = torch.tensor([[1.0, 1], [0, 0]], dtype=dtype)
    gradients = credit.T @ output_columns
    live = int((gradients.norm(dim=0) > 0).sum())
    rank = int(torch.linalg.matrix_rank(credit))
    witness("rank_one_two_live_units", rank == 1 and live == 2, rank=rank, live=live)

    # A zero direct readout is insufficient for safe retirement.
    inputs = torch.ones(3, 1, dtype=dtype)
    readout = torch.tensor([[1.0, 0]], dtype=dtype)
    incoming = torch.tensor([[0.0], [1.0]], dtype=dtype)
    edges = torch.tensor([[0.0, 1], [0, 0]], dtype=dtype)
    rest = torch.zeros(2, dtype=dtype)
    args = (inputs, torch.ones(3, 2, dtype=dtype), incoming, readout)
    active = rollout(*args, edges, rest, rest, rest, rest, rest)[0]
    cut = rollout(*args, edges * 0, rest, rest, rest, rest, rest)[0]
    witness("zero_readout_not_safe_to_delete", float((active - cut).abs().max()) > 0.1)

    # Zero-valued new attention tokens still change an old normalizer.
    original = torch.tensor([0.0], dtype=dtype).softmax(0) @ torch.tensor([1.0], dtype=dtype)
    extended = torch.zeros(2, dtype=dtype).softmax(0) @ torch.tensor([1.0, 0], dtype=dtype)
    witness("attention_support_breaks_neutrality", float((original - extended).abs()) == 0.5)
    return {"dtype": str(dtype), "checks": checks, "passed": len(checks)}


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    results = [verify(dtype) for dtype in (torch.float32, torch.float64)]
    payload = {
        "scope": "local_stateful_unit_algebra_and_counterexamples",
        "runtime_implemented": False,
        "capability_evidence": False,
        "optimizer_calls": 0,
        "project_data_reads": 0,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "results": results,
        "passed": sum(result["passed"] for result in results),
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "report": str(REPORT)}))


if __name__ == "__main__":
    main()
