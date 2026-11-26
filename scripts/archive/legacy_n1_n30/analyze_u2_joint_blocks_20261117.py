"""当前 U2 核心的局部雅可比分析；不读取任务数据、不训练、不声称全轨迹稳定。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.u2_operators import fp1
from taiji.u2_subject import U2Config, U2Subject


def spectral_norm(value: torch.Tensor, input_numel: int) -> float:
    matrix = value.detach().reshape(-1, input_numel)
    if matrix.numel() == 0:
        return 0.0
    return float(torch.linalg.svdvals(matrix).max())


def main() -> None:
    torch.set_num_threads(1)
    subject = U2Subject(U2Config.small()).double()
    config = subject.c
    state = subject.initial()
    state = subject.begin_utterance(state)
    branches, groups, d, key = config.branches, config.groups, config.d, config.key
    h0 = state["H"].detach().clone().requires_grad_(True)
    f0 = state["F"].detach().clone().requires_grad_(True)
    event0 = torch.zeros(branches, d, dtype=torch.float64, requires_grad=True)
    goal = torch.zeros(d, dtype=torch.float64)
    action = torch.zeros(15, dtype=torch.float64)

    def native(h: torch.Tensor, f: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        current = dict(state)
        current["H"] = h
        current["F"] = f
        return subject.native(current, goal, action, event)

    jac_h, jac_f, jac_event = torch.autograd.functional.jacobian(
        native, (h0, f0, event0), create_graph=False, vectorize=False
    )
    direction = torch.linspace(-0.1, 0.1, h0.numel(), dtype=torch.float64).reshape(h0.shape)
    _, jvp = torch.autograd.functional.jvp(
        native,
        (h0, f0, event0),
        (direction, torch.zeros_like(f0), torch.zeros_like(event0)),
    )
    epsilon = 1e-5
    finite_difference = (
        native(h0 + epsilon * direction, f0, event0) - native(h0 - epsilon * direction, f0, event0)
    ) / (2 * epsilon)
    native_jvp_max_abs = float((jvp.detach() - finite_difference.detach()).abs().max())

    # FP1在零记忆、约束未激活点的导数同时用autograd和闭式解核对。
    memory = torch.zeros(key, key, dtype=torch.float64, requires_grad=True)
    keys = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.5, 0.3]], dtype=torch.float64).T
    values = torch.tensor([[0.4, -0.2, 0.1], [-0.3, 0.2, 0.5]], dtype=torch.float64).T
    alpha = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    gains = torch.tensor([0.3, 0.7], dtype=torch.float64)
    chi = torch.tensor(0.5, dtype=torch.float64)
    radius = torch.tensor(4.0, dtype=torch.float64)

    def write(value: torch.Tensor) -> torch.Tensor:
        return fp1(value, keys, values, alpha, gains, chi, radius)

    write_jac = torch.autograd.functional.jacobian(write, memory, create_graph=False)
    beta = chi * gains
    hessian = torch.eye(key, dtype=torch.float64) + (keys * beta) @ keys.T
    coefficients = 1 - chi + chi * alpha
    analytic = torch.linalg.solve(hessian, torch.diag(coefficients))
    analytic_jac = torch.kron(analytic.contiguous(), torch.eye(key, dtype=torch.float64))
    write_jac_flat = write_jac.reshape(key * key, key * key)
    max_write_error = float((write_jac_flat - analytic_jac).abs().max())
    assert native_jvp_max_abs < 1e-6, native_jvp_max_abs
    assert max_write_error < 1e-12, max_write_error

    # 当前核心没有z/a；Y的共享连接F/W也不是该U2Subject的字段。
    z_a_present = all(name in state for name in ("z", "a"))
    y_shared_present = all(name in state for name in ("W",))
    full_contract_instantiated = bool(z_a_present and y_shared_present)
    native_l = spectral_norm(jac_h, h0.numel())
    native_n = spectral_norm(jac_f, f0.numel())
    native_m_event = spectral_norm(jac_event, event0.numel())
    fp1_l = spectral_norm(write_jac_flat, key * key)
    report = {
        "schema": "taiji-u2-local-jacobian-v1",
        "status": "partial_local_jacobian_only",
        "joint_status": "full_z_a_H_F_contract_not_instantiated",
        "behavior_validation": "not_run",
        "task_data_read": False,
        "training_steps": 0,
        "candidate": {
            "class": "U2Subject",
            "config": "U2Config.small",
            "seed": config.seed,
            "branches": branches,
            "groups": groups,
            "d": d,
            "key": key,
            "source_sha256": {
                "u2_subject.py": hashlib.sha256(
                    (ROOT / "taiji" / "u2_subject.py").read_bytes()
                ).hexdigest(),
                "u2_operators.py": hashlib.sha256(
                    (ROOT / "taiji" / "u2_operators.py").read_bytes()
                ).hexdigest(),
            },
        },
        "contract": {
            "H_present": True,
            "FP1_F_present": True,
            "z_a_present": z_a_present,
            "Y_shared_F_W_present": y_shared_present,
            "full_z_a_H_F_contract_instantiated": full_contract_instantiated,
            "joint_L": None,
            "joint_M": None,
            "joint_N": None,
        },
        "local_native": {
            "point": "initial_state_zero_H_zero_F_zero_event",
            "L_native_dH_dH": native_l,
            "N_native_dH_dF": native_n,
            "M_native_dH_devent": native_m_event,
            "directional_jvp_max_abs_error": native_jvp_max_abs,
            "interpretation": "pointwise spectral norms only; not trajectory Lipschitz certificates",
        },
        "local_fp1_write": {
            "point": "zero_memory_inactive_ball",
            "L_fp1_dF_dF": fp1_l,
            "analytic_L_fp1_dF_dF": spectral_norm(analytic_jac, key * key),
            "max_abs_autograd_vs_analytic": max_write_error,
            "alpha": alpha.tolist(),
            "gains": gains.tolist(),
            "chi": float(chi),
            "radius": float(radius),
        },
        "boundary": [
            "z/a blocks are absent, so the proof's joint z/a/H/F rate cannot be instantiated",
            "Y shared F/W is not a field of the current U2Subject",
            "the pending query-binding readout has no frozen implementation identity",
            "local Jacobians do not prove a trajectory-wide Lipschitz bound or behavioral stability",
        ],
    }
    destination = ROOT / "reports" / "taiji_u2_local_jacobian_20261117.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "joint_status": report["joint_status"],
                "native_L": native_l,
                "native_N": native_n,
                "native_M_event": native_m_event,
                "fp1_L": fp1_l,
                "write_error": max_write_error,
                "report": str(destination),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
