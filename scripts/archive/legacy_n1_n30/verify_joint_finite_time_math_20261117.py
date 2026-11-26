"""M06/M43：联合有限时域稳定的精确代数与数值核验（不运行任务或训练）。"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def main() -> None:
    # 两个谱半径均为1/2的矩阵，交替乘积的谱半径却为9/4+sqrt(5)。
    a = np.array([[0.5, 2.0], [0.0, 0.5]])
    b = np.array([[0.5, 0.0], [2.0, 0.5]])
    ba = b @ a
    ba_exact = (
        (Fraction(1, 4), Fraction(1)),
        (Fraction(1), Fraction(17, 4)),
    )
    assert np.array_equal(ba, np.array([[0.25, 1.0], [1.0, 4.25]]))
    radius = 9.0 / 4.0 + float(np.sqrt(5.0))
    eigenvalues = sorted(np.linalg.eigvals(ba).real.tolist())
    assert np.allclose(eigenvalues, [9.0 / 4.0 - np.sqrt(5.0), radius], atol=1e-12, rtol=0.0)
    assert float(np.linalg.norm(a, 2)) > 0.5  # 谱半径不是诱导范数。

    direction = np.linalg.eig(ba)[1][:, 1]
    growth = []
    for k in range(1, 5):
        moved = np.linalg.matrix_power(ba, k) @ direction
        measured = float(np.linalg.norm(moved) / np.linalg.norm(direction))
        expected = radius**k
        assert np.isclose(measured, expected, atol=1e-10, rtol=0.0)
        growth.append({"k": k, "measured": measured, "expected": expected})

    # z/a/H 分块速率的一个精确例子。
    c, eta, ell = 0.5, 0.5, 0.5
    w_norm, b_norm, p_norm, gate_norm = 0.1, 0.05, 0.1, 1.0
    rho_z = c + (1.0 - c) * (w_norm * gate_norm + b_norm)
    rho_a = 1.0 - eta + eta * rho_z
    rho_h = 1.0 - ell + ell * p_norm * gate_norm * rho_z
    joint_l = max(rho_z, rho_a, rho_h)
    assert np.isclose(rho_z, 0.575, atol=0.0, rtol=0.0)
    assert np.isclose(rho_a, 0.7875, atol=0.0, rtol=0.0)
    assert np.isclose(rho_h, 0.52875, atol=0.0, rtol=0.0)
    assert np.isclose(joint_l, 0.7875, atol=0.0, rtol=0.0)
    assert joint_l < 1.0

    # 核验有限时域递推与闭式上界在常数界下一致。
    horizon = 8
    initial_y = 0.01
    input_bound = 0.001
    initial_f = 0.02
    m, n = 0.3, 0.2
    read_y = initial_y
    for _ in range(horizon):
        read_y = joint_l * read_y + m * input_bound + n * initial_f
    closed_form = joint_l**horizon * initial_y + (
        (m * input_bound + n * initial_f) * (1.0 - joint_l**horizon) / (1.0 - joint_l)
    )
    assert np.isclose(read_y, closed_form, atol=1e-14, rtol=0.0)

    # 写入时持久记忆只要求增量有界，不要求记忆本身收缩。
    write_increment = 0.002
    write_y = initial_y
    f = initial_f
    for _ in range(horizon):
        write_y = joint_l * write_y + m * input_bound + n * f
        f += write_increment
    write_closed_form = joint_l**horizon * initial_y + sum(
        joint_l ** (horizon - 1 - k) * (m * input_bound + n * (initial_f + k * write_increment))
        for k in range(horizon)
    )
    assert np.isclose(write_y, write_closed_form, atol=1e-14, rtol=0.0)
    assert np.isclose(f, initial_f + horizon * write_increment, atol=1e-15, rtol=0.0)

    report = {
        "schema": "taiji-joint-finite-time-math-v1",
        "status": "scalar_example_and_closed_form_checks_passed",
        "behavior_validation": "not_run",
        "training_steps": 0,
        "counterexample": {
            "individual_spectral_radii": [0.5, 0.5],
            "product": ba.tolist(),
            "product_eigenvalues": eigenvalues,
            "product_spectral_radius": radius,
            "exact_product": [[str(value) for value in row] for row in ba_exact],
            "alternating_growth": growth,
            "conclusion": "bounded_steps_do_not_imply_joint_contraction",
        },
        "rate_example": {
            "rho_z": rho_z,
            "rho_a": rho_a,
            "rho_h": rho_h,
            "joint_L": joint_l,
        },
        "read_only_bound": {
            "horizon": horizon,
            "recursive": read_y,
            "closed_form": closed_form,
            "read_only_f": initial_f,
        },
        "write_bound": {
            "horizon": horizon,
            "recursive_transient": write_y,
            "closed_form_transient": write_closed_form,
            "persistent_memory_error": f,
            "increment_per_write": write_increment,
        },
    }
    destination = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "taiji_joint_finite_time_math_20261117.json"
    )
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
