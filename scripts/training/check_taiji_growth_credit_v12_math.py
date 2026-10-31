"""Review local growth-credit claims; no model, optimizer, corpus or checkpoint.

Counterexamples reject overbroad claims. Random matrix checks supplement the
algebra in the design, and do not establish growth or replay efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

SEED = 20261031
RANK_TOL = 1e-10


def rank(matrix: np.ndarray, tolerance: float = RANK_TOL) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def gradients_close(test: np.ndarray, numeric: np.ndarray, atol: float, rtol: float) -> bool:
    return bool(
        np.all(np.abs(test - numeric) <= atol + rtol * np.maximum(np.abs(test), np.abs(numeric)))
    )


def allocate_coverage(spectra: list[np.ndarray], budget: int) -> tuple[int, ...]:
    """Exact enumeration for tiny fixtures, not a scalable allocation solver."""
    best_key = None
    best = None
    for counts in itertools.product(*(range(min(len(s), budget) + 1) for s in spectra)):
        if sum(counts) > budget:
            continue
        coverage = [
            float((s[:r] ** 2).sum() / (s**2).sum()) for s, r in zip(spectra, counts, strict=True)
        ]
        key = (min(coverage), sum(coverage), -sum(counts), tuple(-r for r in counts))
        if best_key is None or key > best_key:
            best_key, best = key, counts
    if best is None:
        raise ValueError("No feasible credit allocation")
    return best


def validate() -> dict:
    rng = np.random.default_rng(SEED)
    checks: list[dict] = []

    def record(name: str, passed: bool, category: str, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), "category": category, **evidence})

    for _ in range(50):
        d, p, n, r = (int(v) for v in rng.integers(1, 8, size=4))
        delta, x, b = rng.normal(size=(d, n)), rng.normal(size=(p, n)), rng.normal(size=(d, r))
        c = delta @ x.T
        assert rank(b.T @ c) <= min(rank(c), rank(b)) <= min(d, p, n)
    record("independent_gradient_rank_bound", True, "local_identity", random_cases=50)

    c = np.diag([1.0, 0.0, 0.0, 0.0])
    b = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    gradient = b.T @ c
    live = int((np.linalg.norm(gradient, axis=1) > RANK_TOL).sum())
    record(
        "nonzero_unit_count_is_not_bounded_by_credit_rank",
        live == 4 and rank(c) == rank(gradient) == 1 and np.allclose(b.T @ b, np.eye(4)),
        "counterexample",
        credit_rank=rank(c),
        nonzero_rows=live,
        gradient_rank=rank(gradient),
        per_unit_norms=np.linalg.norm(gradient, axis=1).tolist(),
        scope="Even orthonormal outgoing columns can give redundant nonzero gradients.",
    )

    c = np.diag([0.29, 0.0, 0.0, 0.0])
    u, _, _ = np.linalg.svd(c)
    norms = np.linalg.norm((0.1 * u).T @ c, axis=1)
    record(
        "top_svd_zero_credit_fraction_has_limited_scope",
        np.allclose(norms, [0.029, 0, 0, 0]),
        "local_identity",
        request=4,
        rank=rank(c),
        zero_credit_fraction=float((norms <= RANK_TOL).mean()),
        scope="Exact-rank statement for this top-SVD initializer, not every neutral birth.",
    )

    for _ in range(50):
        d, p, n, groups = 4, 5, 9, 3
        delta, x = rng.normal(size=(d, n)), rng.normal(size=(p, n))
        ids = rng.integers(groups, size=n)
        matrices = [(delta * (ids == group)) @ x.T for group in range(groups)]
        c = delta @ x.T
        total = sum(rank(matrix) for matrix in matrices)
        assert np.allclose(sum(matrices), c) and rank(c) <= total <= n
    record("hard_partition_exact_rank_bounds", True, "local_identity", random_cases=50)

    x = np.array([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]], dtype=float)
    delta = -(x[:, 0] * x[:, 1]) / 4
    hard = (x[:, 0] > 0).astype(float)
    matrices = [((g * delta) @ x)[None, :] for g in (hard, 1 - hard)]
    record(
        "conditional_block_rank_can_exceed_global_rank",
        rank(sum(matrices)) == 0 and sum(rank(matrix) for matrix in matrices) == 2,
        "witness",
        global_rank=rank(sum(matrices)),
        sum_branch_ranks=sum(rank(matrix) for matrix in matrices),
        scope="Blockwise credit ranks, not a theorem about total network capacity.",
    )

    soft_matrices = [np.array([[0.5]]), np.array([[0.5]])]
    lifted = np.array([[0.5, 0.5]])
    record(
        "soft_partition_rank_sum_can_exceed_sample_count",
        sum(rank(matrix) for matrix in soft_matrices) == 2 and rank(lifted) == 1,
        "counterexample",
        sample_slots=1,
        sum_branch_ranks=2,
        lifted_feature_rank=rank(lifted),
        scope="Duplicate soft gates count the same sample twice; no hard-partition upper bound.",
    )
    tiny = [np.array([[0.06]]), np.array([[0.06]])]
    record(
        "numerical_rank_sum_does_not_inherit_exact_rank_inequality",
        sum(rank(matrix, 0.1) for matrix in tiny) == 0 and rank(sum(tiny), 0.1) == 1,
        "counterexample",
        absolute_rank_threshold=0.1,
    )

    one_x, one_delta = np.array([[1.0, 2.0]]), np.array([[3.0, -1.0]])
    base = one_delta.T @ one_x
    repeated = np.repeat(one_delta, 10, axis=0).T @ np.repeat(one_x, 10, axis=0) / 10
    record(
        "duplicate_replay_does_not_create_rank",
        np.allclose(base, repeated) and rank(base) == rank(repeated) == 1,
        "counterexample",
    )
    direction = np.ones(2) / np.sqrt(2)
    before, after = np.eye(2), np.eye(2) - np.outer(direction, direction)
    record(
        "adding_a_residual_sample_can_reduce_aggregate_rank",
        rank(before) == 2 and rank(after) == 1,
        "counterexample",
        before_rank=rank(before),
        after_rank=rank(after),
        scope="More distinct sample directions do not imply larger aggregate credit rank.",
    )
    h, other_inputs = np.zeros((2, 2)), np.eye(2)
    full_x = np.concatenate([h, other_inputs], axis=1)
    record(
        "low_hidden_rank_does_not_bound_full_input_credit_rank",
        rank(h) == 0 and rank(np.eye(2) @ full_x) == 2,
        "counterexample",
        hidden_rank=rank(h),
        full_input_credit_rank=2,
    )

    gate_rng = np.random.default_rng(SEED)
    gates = gate_rng.uniform(size=(2000, 4))
    c_plus = np.einsum("bn,n,np->bp", gates, delta, x)
    c_minus = np.einsum("bn,n,np->bp", 1 - gates, delta, x)
    norms = 0.1 * np.sqrt((c_plus**2).sum(axis=1) + (c_minus**2).sum(axis=1))
    live_ratio = float(
        np.concatenate([np.linalg.norm(c_plus, axis=1), np.linalg.norm(c_minus, axis=1)]).min()
        > RANK_TOL
    )
    record(
        "random_gates_can_have_all_units_receive_nonzero_credit",
        live_ratio == 1.0 and abs(np.median(norms) - 0.034747082173609056) < 1e-12,
        "counterexample",
        random_gate_nonzero_unit_fraction=live_ratio,
        median_gradient_norm=float(np.median(norms)),
        scope="Raw live fraction cannot by itself distinguish true from random gates.",
    )

    cluster = np.diag([1.0, 0.0])
    q = np.array([[1.0], [0.0]])
    record(
        "shared_branch_cluster_coverage_can_still_cancel",
        np.linalg.norm(q.T @ cluster) > 0
        and np.linalg.norm(q.T @ -cluster) > 0
        and np.array_equal(q.T @ (cluster - cluster), np.zeros((1, 2))),
        "counterexample",
        scope="A per-cluster max-min objective also needs aggregate-credit feasibility.",
    )

    spectra = [np.array([10.0, 9.0]), np.array([1.0])]
    allocation = allocate_coverage(spectra, budget=2)
    coverage = [(s[:r] ** 2).sum() / (s**2).sum() for s, r in zip(spectra, allocation, strict=True)]
    record(
        "coverage_allocation_differs_from_total_energy_allocation",
        allocation == (1, 1) and np.isclose(min(coverage), 100 / 181),
        "operator_fixture",
        allocation=list(allocation),
        per_branch_coverage=[float(value) for value in coverage],
        total_energy_allocation=[2, 0],
        total_energy_minimum_coverage=0,
        scope="Finite integer optimum only; no task or growth advantage is inferred.",
    )

    # Check the actual conditioned forward, not just B^T C against itself.
    x_small = rng.normal(size=(6, 3))
    delta_small = rng.normal(size=(6, 2)) / 6
    g = 1 / (1 + np.exp(-x_small[:, 0]))
    gates_small = [g, 1 - g]
    matrices = [delta_small.T @ (gate[:, None] * x_small) for gate in gates_small]
    outgoing = [0.1 * np.linalg.svd(matrix, full_matrices=False)[0][:, :1] for matrix in matrices]
    incoming = [torch.zeros((1, 3), dtype=torch.float64, requires_grad=True) for _ in matrices]
    outputs = []
    for gate, b, r in zip(gates_small, outgoing, incoming, strict=True):
        outputs.append(
            torch.tensor(gate[:, None])
            * (torch.tanh(torch.tensor(x_small) @ r.T) @ torch.tensor(b).T)
        )
    addition = sum(outputs)
    loss = (addition * torch.tensor(delta_small)).sum()
    automatic = torch.autograd.grad(loss, incoming)
    analytic = [b.T @ matrix for b, matrix in zip(outgoing, matrices, strict=True)]
    analytic_flat = np.concatenate([value.ravel() for value in analytic])
    max_error = max(
        float(np.max(np.abs(auto.numpy() - a))) for auto, a in zip(automatic, analytic, strict=True)
    )

    def objective(flat: np.ndarray) -> float:
        params = flat.reshape(2, 1, 3)
        prediction = sum(
            gate[:, None] * (np.tanh(x_small @ r.T) @ b.T)
            for gate, b, r in zip(gates_small, outgoing, params, strict=True)
        )
        return float((prediction * delta_small).sum())

    errors = {}
    correct, zero_rejected, sign_rejected = True, True, True
    atol, rtol = 1e-11, 1e-6
    for eps in (1e-4, 1e-5, 1e-6):
        numeric = np.zeros(6)
        for index in range(6):
            step = np.zeros(6)
            step[index] = eps
            numeric[index] = (objective(step) - objective(-step)) / (2 * eps)

        correct = correct and gradients_close(analytic_flat, numeric, atol, rtol)
        zero_rejected = zero_rejected and not gradients_close(np.zeros(6), numeric, atol, rtol)
        sign_rejected = sign_rejected and not gradients_close(-analytic_flat, numeric, atol, rtol)
        errors[str(eps)] = float(np.max(np.abs(analytic_flat - numeric)))
    record(
        "conditioned_birth_actual_forward_credit_and_negative_controls",
        float(addition.detach().abs().max()) == 0
        and max_error < 1e-12
        and correct
        and zero_rejected
        and sign_rejected,
        "local_derivative",
        autograd_absolute_error=max_error,
        finite_difference_errors=errors,
        atol=atol,
        rtol=rtol,
        zero_gradient_rejected=zero_rejected,
        reversed_gradient_rejected=sign_rejected,
    )

    return {
        "schema": "taiji-growth-credit-v12-local-math-review",
        "date": "2026-10-31",
        "baseline_commit": "d545b3ccc",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "seed": SEED,
        "dtype": "float64",
        "rank_tolerance_for_fixtures": RANK_TOL,
        "checks": checks,
        "status": "passed_local_review_only" if all(c["passed"] for c in checks) else "failed",
        "optimizer_steps": 0,
        "data_splits_read": [],
        "full_subject_created": False,
        "limits": "Counterexamples and local identities only. No rank-monotone replay, "
        "neuron-count impossibility, novel theory, actual growth benefit or task ability is established.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "failed": [c["name"] for c in result["checks"] if not c["passed"]],
            }
        )
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
