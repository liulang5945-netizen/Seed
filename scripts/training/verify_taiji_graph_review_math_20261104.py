"""Exact algebraic counterexamples for the E1--E11 graph review.

This script evaluates rational identities only. It constructs no Taiji model,
loads no weights or data, and invokes no optimizer or autograd engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from fractions import Fraction as F
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def determinant(rows: list[list[F]]) -> F:
    work = [row[:] for row in rows]
    result = F(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return F(0)
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            result = -result
        value = work[column][column]
        result *= value
        for row in range(column + 1, len(work)):
            ratio = work[row][column] / value
            for entry in range(column + 1, len(work)):
                work[row][entry] -= ratio * work[column][entry]
            work[row][column] = F(0)
    return result


def tanh_log2(multiple: int) -> F:
    """tanh(multiple * log(2)), computed exactly without floating point."""
    power = F(4) ** multiple
    return (power - 1) / (power + 1)


def additive_rank() -> dict[str, Any]:
    # tanh((log(a)+log(b))/2) = (a*b-1)/(a*b+1).
    scores = [[F(a * b - 1, a * b + 1) for b in (1, 2, 4)] for a in (1, 2, 3)]
    value = determinant(scores)
    assert value == -F(94, 61425)
    mixed_second_derivative = -2 * F(3, 5) * (1 - F(3, 5) ** 2)
    assert mixed_second_derivative == -F(96, 125)
    return {
        "score_matrix": scores,
        "determinant": value,
        "mixed_second_derivative_at_log2": mixed_second_derivative,
        "scope": "Refutes global rank<=2 and universal third-order-only interaction.",
    }


def query_reordering() -> dict[str, Any]:
    def score(query: int, key: int) -> F:
        return tanh_log2(query + key) - 2 * tanh_log2(query + key - 2)

    rows = [[score(query, key) for key in (0, 1)] for query in (0, 2)]
    assert rows == [[F(30, 17), F(9, 5)], [F(15, 17), -F(3, 13)]]
    winners = [max(range(2), key=row.__getitem__) for row in rows]
    assert winners == [1, 0]
    goal_key_rows = [[F(goal * key) for key in (-1, 1)] for goal in (-1, 1)]
    assert [max(range(2), key=row.__getitem__) for row in goal_key_rows] == [0, 1]
    return {
        "additive_tanh_scores": rows,
        "winners": winners,
        "existing_goal_query_bilinear_scores": goal_key_rows,
        "scope": "Finite legal scorer subfamilies; no claim that training finds them.",
    }


def independent_controls() -> dict[str, Any]:
    chi, eta, squared_direction, kv = F(1, 2), F(3), F(1, 4), F(1)
    jacobian = [[chi / (1 + eta * squared_direction), F(0)], [F(0), chi * kv]]
    assert determinant(jacobian) == F(1, 7)
    # F=1, eta=0, alpha=w=1/2: erase=-1/2 and write=+1/2 cancel.
    assert (F(1, 2) - 1) + F(1, 2) == 0
    forget_partial, write_partial = -chi, chi
    assert forget_partial < 0 < write_partial
    return {
        "retention_write_control_jacobian": jacobian,
        "jacobian_determinant": determinant(jacobian),
        "chi_partial_in_cancellation_case": F(0),
        "independent_forget_write_partials": [forget_partial, write_partial],
        "scope": "Unprojected interior controls; shared chi can cancel but does not force all gradients.",
    }


def erase_sensitivity() -> dict[str, Any]:
    eta, squared_direction, alpha = F(15), F(1, 4), F(1, 1000)
    keep = 1 / (1 + eta * squared_direction)
    assert keep == F(4, 19) and keep > F(1, 17)
    assert alpha * keep == F(4, 19000) < F(1, 17)
    profiles = []
    for scale in (2, 20, 200):
        eta_i, direction = F(scale**2), F(1, scale)
        value = 1 / (1 + eta_i * direction**2)
        derivative = -2 * eta_i * direction / (1 + eta_i * direction**2) ** 2
        assert value == F(1, 2) and abs(derivative) == F(scale, 2)
        profiles.append(
            {"eta": eta_i, "direction": direction, "abs_direction_partial": abs(derivative)}
        )
    return {
        "Ae_keep": keep,
        "complete_decay_erase_keep": alpha * keep,
        "bounded_forward_unbounded_direction_sensitivity_family": profiles,
        "scope": "Fixed-direction state norm is distinct from input/parameter sensitivity.",
    }


def time_and_units() -> dict[str, Any]:
    expected_discount = (F(1, 2) + F(1, 8)) / 2
    discount_of_mean_time = F(1, 4)
    assert expected_discount - discount_of_mean_time == F(1, 16)
    return {
        "E_exp_minus_beta_duration": expected_discount,
        "exp_minus_beta_mean_duration": discount_of_mean_time,
        "head_count": 2,
        "head_shape": [32, 32],
        "scalar_entries": 2 * 32 * 32,
        "block_diagonal_matrix_rank_upper_bound": 2 * 32,
        "scope": "Matrix rank is not parameter count; no storage-capacity theorem inferred.",
    }


def kkt_example() -> dict[str, Any]:
    solution, bound, dual = F(1), F(3, 2), F(0)
    soft_gradient = solution
    stationarity = (solution - 2) + soft_gradient + dual
    slack = solution - bound
    assert stationarity == 0 and slack < 0 and dual * slack == 0
    return {
        "problem": "min .5*(x-2)^2+.5*x^2 subject to x<=1.5",
        "solution": solution,
        "soft_gradient": soft_gradient,
        "hard_constraint_dual": dual,
        "stationarity": stationarity,
        "slack": slack,
        "scope": "A soft objective coefficient need not equal a hard-constraint dual variable.",
    }


def stability_notions() -> dict[str, Any]:
    # f(F)=(1-chi+chi*sigmoid(4F-4))*F at F=1.
    chi, gate, gate_partial, state = F(1, 2), F(1, 2), F(1), F(1)
    energy_factor = 1 - chi + chi * gate
    full_derivative = energy_factor + chi * state * gate_partial
    assert energy_factor == F(3, 4) and full_derivative == F(5, 4)
    a = [[F(1, 2), F(2)], [F(0), F(1, 2)]]
    b = [[F(1, 2), F(0)], [F(2), F(1, 2)]]
    product = [[sum(b[i][k] * a[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    trace = product[0][0] + product[1][1]
    characteristic_at_one = 1 - trace + determinant(product)
    assert characteristic_at_one == -F(55, 16)
    assert 1 / (1 - F(2)) == -1  # invertible I-J; Neumann powers diverge
    return {
        "pointwise_energy_factor": energy_factor,
        "full_derivative": full_derivative,
        "individual_triangular_spectral_radius": F(1, 2),
        "switch_product": product,
        "product_characteristic_at_one": characteristic_at_one,
        "implicit_scalar_J": F(2),
        "implicit_inverse_I_minus_J": F(-1),
        "scope": "Energy, contraction, switching stability and implicit differentiability differ.",
    }


def birth_tangent() -> dict[str, Any]:
    output_weights, delta_r, sample = [F(1), F(2)], [F(3), F(-1)], [F(2), F(1)]
    delta_w = [[coefficient * entry for entry in delta_r] for coefficient in output_weights]
    direct = [
        coefficient * sum(r * x for r, x in zip(delta_r, sample, strict=True))
        for coefficient in output_weights
    ]
    inherited = [sum(w * x for w, x in zip(row, sample, strict=True)) for row in delta_w]
    assert direct == inherited == [F(5), F(10)]
    # b*tanh(r), b=1,r=0,y=1. Ordinary SGD with eta=log(2)
    # gives r=log(2), then tanh(r)=3/5 and the outgoing gradient=-6/25.
    initial_incoming, initial_outgoing = F(-1), F(0)
    outgoing_after_symbolic_step = (F(3, 5) - 1) * F(3, 5)
    assert outgoing_after_symbolic_step == -F(6, 25)
    return {
        "delta_W_equals_B_delta_R": delta_w,
        "birth_and_free_parent_tangent": direct,
        "initial_incoming_gradient": initial_incoming,
        "initial_outgoing_gradient": initial_outgoing,
        "outgoing_gradient_after_analytic_SGD_step": outgoing_after_symbolic_step,
        "scope": "Free allowed parent W; constrained/frozen parent groups require a new tangent definition.",
    }


def op2_emission() -> dict[str, Any]:
    a, null, lam = [F(1, 10), F(2, 5)], F(1, 2), F(4, 5)
    cursor = [[F(0), F(1, 2)], [F(0), F(0)]]
    fresh = [[F(99, 100), F(1, 100)], [F(1, 2), F(1, 2)]]
    raw = [[0, 1], [1, 0]]
    q = []
    for record, m, p in zip(a, cursor, fresh, strict=True):
        mass = sum(m)
        eta = [lam * x + (1 - lam * mass) * y for x, y in zip(m, p, strict=True)]
        assert sum(eta) == 1
        q.append([record * value for value in eta])
    assert a[0] < lam * sum(cursor[0])  # violates preservation, not OP2 legality
    assert sum(map(sum, q)) + null == 1
    gamma, gen = F(3, 10), [F(1, 5), F(3, 10), F(1, 2)]
    copy = [
        sum(q[i][j] for i in range(2) for j in range(2) if raw[i][j] == byte) for byte in range(3)
    ]
    mix = gamma + (1 - gamma) * null
    normalized_copy = [value / (1 - null) for value in copy]
    op2 = [mix * g + (1 - gamma) * c for g, c in zip(gen, copy, strict=True)]
    older_form = [mix * g + (1 - mix) * c for g, c in zip(gen, normalized_copy, strict=True)]
    assert op2 == older_form and sum(op2) == 1
    return {
        "OP2_valid_below_preservation_requirement": True,
        "position_Q": q,
        "generator_mix": mix,
        "identical_emission": op2,
        "scope": "Emission algebra only; head parameterizations and state kernels remain versioned.",
    }


def duplicate_copy() -> dict[str, Any]:
    raw = [0, 1, 0, 0, 1, 0]
    cursor, outputs = [F(0)] * len(raw), []
    for expected_position, target in enumerate(raw):
        fresh = [F(index == 0) for index in range(len(raw))]
        eta = [old + (1 - sum(cursor)) * new for old, new in zip(cursor, fresh, strict=True)]
        assert eta == [F(index == expected_position) for index in range(len(raw))]
        probability = sum(weight for weight, byte in zip(eta, raw, strict=True) if byte == target)
        assert probability == 1
        posterior = [
            weight * F(byte == target) / probability for weight, byte in zip(eta, raw, strict=True)
        ]
        cursor = [F(0), *posterior[:-1]]
        outputs.append(target)
    assert outputs == raw
    return {
        "raw": raw,
        "copied": outputs,
        "scope": "Hard kernel with fixed record/start/length controls, not learned routing or EOS validation.",
    }


def projected_proximal_kkt() -> dict[str, Any]:
    # Fbar=(0,1/4), k=(1/2,0), beta=1, v=5/8.
    # Hessian=diag(5/4,1), b=(5/16,1/4), unconstrained optimum=(1/4,1/4).
    unconstrained, radial = [F(1, 4), F(1, 4)], [F(1, 5), F(1, 5)]
    radius_squared = F(2, 25)
    assert sum(value**2 for value in radial) == radius_squared
    assert F(1, 4) ** 2 < radius_squared < sum(value**2 for value in unconstrained)
    gradient = [F(5, 4) * radial[0] - F(5, 16), radial[1] - F(1, 4)]
    implied_duals = [-g / x for g, x in zip(gradient, radial, strict=True)]
    assert implied_duals == [F(5, 16), F(1, 4)]
    assert implied_duals[0] != implied_duals[1]
    return {
        "unconstrained_optimum": unconstrained,
        "radial_projection": radial,
        "radius_squared": radius_squared,
        "incompatible_KKT_multipliers": implied_duals,
        "scope": "Euclidean clipping of a non-isotropic quadratic optimum need not solve its constrained problem.",
    }


def serializable(value: Any) -> Any:
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = None
    if args.report is not None:
        report_path = args.report.resolve()
        if report_path.parent != (ROOT / "reports").resolve() or report_path.exists():
            parser.error("Report must be a new file directly inside reports/.")
    checks = {}
    for function in (
        additive_rank,
        query_reordering,
        independent_controls,
        erase_sensitivity,
        time_and_units,
        kkt_example,
        stability_notions,
        birth_tangent,
        op2_emission,
        duplicate_copy,
        projected_proximal_kkt,
    ):
        checks[function.__name__] = {"passed": True, "details": function()}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = serializable(
        {
            "schema": "taiji-graph-review-exact-witnesses-v1",
            "protocol_date": "2026-11-04",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "basis_commit": revision,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "passed": True,
            "checks": checks,
            "scope": "Exact algebraic witnesses and counterexamples, not a full-model capability assessment.",
            "accounting": {
                "model_constructions": 0,
                "weights_loaded": False,
                "autograd_calls": 0,
                "optimizer_invocations": 0,
                "analytic_SGD_step_witnesses": 1,
                "project_data_reads": 0,
            },
            "proof_boundary": "Finite counterexamples refute general claims; general theorems require the written derivations.",
        }
    )
    if report_path is not None:
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "exact_witness_groups": len(checks),
                "report": str(report_path) if report_path else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
