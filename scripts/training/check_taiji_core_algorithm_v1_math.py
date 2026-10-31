"""Synthetic algebra checks for the proposed Taiji core algorithm, not a model trial.

No checkpoint, corpus or evaluation split is opened. No optimizer is constructed.
The finite-difference checks validate specified local derivatives only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def run_checks() -> dict:
    rng = np.random.default_rng(20261031)
    checks: list[dict] = []

    def record(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    # Convex leaky update preserves the coordinate bound, not useful cognition.
    state = rng.uniform(-1, 1, size=(4, 8))
    tau = np.array([1.0, 4.0, 16.0, 64.0])[:, None]
    leak = 1 - np.exp(-1 / tau)
    for _ in range(100):
        candidate = np.tanh(rng.normal(size=state.shape) * 30)
        state = (1 - leak) * state + leak * candidate
    record("bounded_neural_state", np.max(np.abs(state)) <= 1, max_abs=float(np.max(np.abs(state))))

    # GDN2-oriented state: keys x values. Read F_t, write only afterward.
    key = rng.normal(size=5)
    key /= np.linalg.norm(key)
    value = rng.normal(size=3)
    initial = np.zeros((5, 3))
    prior_read = initial.T @ key
    after = initial + np.outer(key, value)
    record(
        "read_before_write_and_next_read",
        np.array_equal(prior_read, np.zeros(3)) and np.allclose(after.T @ key, value),
        next_read_error=float(np.max(np.abs(after.T @ key - value))),
    )

    old_memory = rng.normal(size=(5, 3))
    # The permission-zero branch is a literal no-op, including no projection.
    permission = False
    permitted_result = after if permission else old_memory.copy()
    record("permission_zero_no_write", np.array_equal(permitted_result, old_memory))

    # The added projection bounds F; this says nothing about recall fidelity.
    limit = 2.0
    large = after * 100
    projected = large * min(1.0, limit / np.linalg.norm(large))
    record(
        "fast_memory_norm_bound",
        np.linalg.norm(projected) <= limit + 1e-12,
        norm=float(np.linalg.norm(projected)),
    )

    # Parent residual plus zero incoming / nonzero outgoing newborn neurons.
    inputs = rng.normal(size=(7, 6))
    errors = rng.normal(size=(7, 4))
    parent_in = rng.normal(size=(3, 6))
    parent_out = rng.normal(size=(4, 3))
    parent_output = np.tanh(inputs @ parent_in.T) @ parent_out.T
    covariance = errors.T @ inputs / len(inputs)
    left, singular, _ = np.linalg.svd(covariance, full_matrices=False)
    newborn_out = 0.1 * left[:, :2]
    newborn_in = np.zeros((2, 6))
    child_output = parent_output + np.tanh(inputs @ newborn_in.T) @ newborn_out.T
    record(
        "birth_function_neutral",
        np.array_equal(parent_output, child_output),
        max_delta=float(np.max(np.abs(child_output - parent_output))),
    )

    # Test local loss derivative against finite differences, not a stored formula alone.
    def local_loss(incoming: np.ndarray) -> float:
        addition = np.tanh(inputs @ incoming.T) @ newborn_out.T
        return float(np.sum(errors * addition) / len(inputs))

    analytic = newborn_out.T @ covariance
    numerical = np.zeros_like(newborn_in)
    step = 1e-6
    for index in np.ndindex(newborn_in.shape):
        plus, minus = newborn_in.copy(), newborn_in.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (local_loss(plus) - local_loss(minus)) / (2 * step)
    derivative_error = float(np.max(np.abs(analytic - numerical)))
    record(
        "newborn_incoming_credit",
        np.linalg.norm(analytic) > 1e-8 and derivative_error < 1e-8,
        gradient_norm=float(np.linalg.norm(analytic)),
        finite_difference_error=derivative_error,
        singular_values=singular[:2].tolist(),
    )
    dead_gradient = np.zeros_like(newborn_out).T @ covariance
    record("all_zero_birth_negative_control", np.linalg.norm(dead_gradient) == 0)

    # An inner write gate has a real derivative to a subsequent read loss.
    gate_logit = -0.7

    def gate_loss(logit: float) -> float:
        gate = 1 / (1 + np.exp(-logit))
        memory = np.outer(key, gate * value)
        residual = memory.T @ key - value
        return float(residual @ residual / 2)

    gate = 1 / (1 + np.exp(-gate_logit))
    gate_analytic = (gate - 1) * (value @ value) * gate * (1 - gate)
    gate_numerical = (gate_loss(gate_logit + step) - gate_loss(gate_logit - step)) / (2 * step)
    record(
        "inner_write_gate_to_later_task_loss",
        abs(gate_analytic) > 1e-8 and abs(gate_analytic - gate_numerical) < 1e-8,
        derivative=float(gate_analytic),
        finite_difference_error=float(abs(gate_analytic - gate_numerical)),
    )

    # Position-aware scores distinguish two occurrences of the same byte.
    raw_bytes = np.array([65, 65, 66, 67])
    legal = np.array([True, True, True, False])
    scores = np.array([1.0, -1.0, 0.2, 1000.0])
    attention = np.zeros(4)
    attention[legal] = softmax(scores[legal])
    copied = np.zeros(257)
    np.add.at(copied, raw_bytes, attention)
    generated = softmax(rng.normal(size=257))
    mixture = 0.4 * generated + 0.6 * copied
    record(
        "copy_normalization_position_and_source_mask",
        abs(mixture.sum() - 1) < 1e-12 and attention[3] == 0 and attention[0] != attention[1],
        probability_sum=float(mixture.sum()),
        duplicate_byte_position_weights=attention[:2].tolist(),
    )
    record("no_source_uses_generation_only", np.array_equal(1.0 * generated, generated))

    # One protected-gradient halfspace: local linear condition, not retention proof.
    protected = rng.normal(size=12)
    proposed = protected.copy()  # deliberate first-order violation
    violation = float(protected @ proposed)
    constrained = proposed - max(0.0, violation) * protected / (protected @ protected)
    record(
        "one_anchor_first_order_projection",
        protected @ constrained <= 1e-10,
        dot=float(protected @ constrained),
    )

    return {
        "schema": "taiji-core-algorithm-v1-synthetic-math-checks",
        "date": "2026-10-31",
        "seed": 20261031,
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "array_dtype": "float64",
        "status": "passed_static_properties_only" if all(c["passed"] for c in checks) else "failed",
        "checks": checks,
        "optimizer_updates": 0,
        "full_model_created": False,
        "data_splits_read": [],
        "limits": "No language, lifelong-learning, autonomous-activity or growth-efficacy claim follows from these array checks.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        if args.output.exists():
            raise FileExistsError("math-check output is write-once")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
