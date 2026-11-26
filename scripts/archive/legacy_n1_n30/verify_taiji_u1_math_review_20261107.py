"""Independent numerical review of U1 unified algebra propositions (2026-11-07).

Scope: algebra only. No production Taiji import, no checkpoint, no dataset,
no sealed face, no training. Synthetic float64 arrays. Writes one NEW report
file and refuses to overwrite an existing one (write-once discipline).

Four red/green checks, each with a positive (green) and negative (red) arm:

R1  Theorem 5.1 claim "beta -> infinity makes the read X^T k converge exactly
    to v" under the *constrained* FP1 definition of U1 section 5.
    Green arm: unconstrained solve / constraint inactive -> error -> 0.
    Red arm:   constraint active (||V K^+||_F > C_F) -> error has a strictly
               positive limit ||v|| - C_F*||k||; beta cannot remove it.

R2  Proposition 4.1 normalization premise "sum_p w_p = 1 - q_null" versus
    U1 section 4 construction w = lambda*m + (1-lambda*m_mass)*anchor, with
    anchor mass = 1 - alpha_null from section 3.
    Green arm: q_null bound to alpha_null*(1-lambda*m_mass) -> sum P = 1.
    Red arm:   q_null independent -> sum P != 1 (premise broken).

R3  Proposition 2.2 startup credit: Read_R output layer zero-initialized.
    Green arm: W_rR random -> dL/dW_out nonzero.
    Red arm:   W_rR also zero -> both gradients exactly zero (dead saddle).

R4  Proposition 7.1 causal gate inherits only t_end, while MGW1 section 2
    requires t_end AND t_arrival on two different clocks.
    Logical record of the disagreement (no numeric model needed).

R5  Independent-review finding: the constrained FP1 solution
    (A+mu*I)^-1 B is NOT the same map as solving unconstrained and then
    applying the Euclidean ball projection Pi(A^-1 B), except when A is
    proportional to I or the solution lies in one eigenspace.
    Green arm: F0=0, single pair -> the two coincide.
    Red arm:   F0!=0, several pairs -> they differ.

R6  Independent-review finding: when (k,v) pairs conflict, the difference
    between sequential per-domain updates and the joint solve is not purely
    a "domain-order artifact" (the joint objective has an irreducible
    residual). Green arm: orthogonal keys -> sequential == joint.
    Red arm:   same key with conflicting values -> sequential diverges and
    the joint read lands on the beta-weighted average with residual > 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "taiji_u1_math_review_20261107.json"


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def solve_unconstrained(f0: np.ndarray, k: np.ndarray, v: np.ndarray, beta: float):
    dim = k.size
    a = np.eye(dim) + beta * np.outer(k, k)
    b = f0 + beta * np.outer(k, v)
    return np.linalg.solve(a, b)


def solve_constrained(f0: np.ndarray, k: np.ndarray, v: np.ndarray, beta: float, cap: float):
    dim = k.size
    a = np.eye(dim) + beta * np.outer(k, k)
    b = f0 + beta * np.outer(k, v)
    x0 = np.linalg.solve(a, b)
    if np.linalg.norm(x0, "fro") <= cap:
        return x0, 0.0, False

    def at(mu: float) -> np.ndarray:
        return np.linalg.solve(a + mu * np.eye(dim), b)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if np.linalg.norm(at(hi), "fro") <= cap:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(at(mid), "fro") > cap:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    return at(mu), mu, True


def check_r1() -> dict:
    dim, vdim = 6, 4
    k = np.zeros(dim)
    k[0] = 1.0
    f0 = np.zeros((dim, vdim))
    cap = 4.0
    betas = [1.0, 10.0, 100.0, 1e3, 1e6]
    arms = {}
    for label, vnorm in (("green_constraint_inactive", 2.0), ("red_constraint_active", 5.0)):
        v = np.zeros(vdim)
        v[0] = vnorm
        rows = []
        for beta in betas:
            xu = solve_unconstrained(f0, k, v, beta)
            xc, mu, active = solve_constrained(f0, k, v, beta, cap)
            rows.append(
                {
                    "beta": beta,
                    "unconstrained_read_error": float(np.linalg.norm(xu.T @ k - v)),
                    "constrained_read_error": float(np.linalg.norm(xc.T @ k - v)),
                    "mu": float(mu),
                    "constraint_active": bool(active),
                    "norm_unconstrained": float(np.linalg.norm(xu, "fro")),
                    "norm_constrained": float(np.linalg.norm(xc, "fro")),
                }
            )
        predicted_floor = max(0.0, vnorm - cap * float(np.linalg.norm(k)))
        arms[label] = {
            "v_norm": vnorm,
            "cap": cap,
            "min_norm_solution_norm": vnorm / float(np.linalg.norm(k)),
            "predicted_error_floor": predicted_floor,
            "asymptotic_constrained_read_error": rows[-1]["constrained_read_error"],
            "rows": rows,
        }
    red = arms["red_constraint_active"]
    green = arms["green_constraint_inactive"]
    # Explicit, readable verdict.
    red_refutes_claim = (
        red["asymptotic_constrained_read_error"] > 0.5
        and abs(red["asymptotic_constrained_read_error"] - red["predicted_error_floor"]) < 1e-6
        and red["rows"][-1]["unconstrained_read_error"] < 1e-5
        and red["rows"][-1]["constraint_active"]
    )
    green_control = (
        green["rows"][-1]["unconstrained_read_error"] < 1e-5
        and green["asymptotic_constrained_read_error"] < 1e-5
        and not green["rows"][-1]["constraint_active"]
    )
    return {
        "verdict": "theorem_5_1_beta_limit_refuted_under_active_constraint",
        "red_refutes_claim": bool(red_refutes_claim),
        "green_control_passes": bool(green_control),
        "arms": arms,
        "note": (
            "With F0=0, ||k||=1, v=vnorm*e0, C_F=cap, the constrained solution reads "
            "X^T k = (||k||*C_F/||v||)*v, so the error tends to ||v||-C_F*||k|| > 0 "
            "whenever ||v||/||k|| > C_F. The unconstrained beta-limit does converge to v."
        ),
    }


def check_r2() -> dict:
    n_rec, n_pos = 2, 3
    rng = np.random.default_rng(20261107)
    alpha = softmax(np.array([1.2, 0.4, -0.3]))  # last = null
    a_null = float(alpha[-1])
    betas = np.stack([softmax(rng.normal(size=n_pos)) for _ in range(n_rec)])
    anchor = np.concatenate([alpha[i] * betas[i] for i in range(n_rec)])
    m = np.zeros(n_rec * n_pos)
    m[1], m[4] = 0.20, 0.10
    m_mass = float(m.sum())
    lam = 0.5
    w = lam * m + (1.0 - lam * m_mass) * anchor
    sum_w = float(w.sum())
    gamma = 0.3
    q_null_independent = 0.15
    q_null_bound = a_null * (1.0 - lam * m_mass)

    def emission_sum(q_null: float) -> float:
        # sum_y C_b(y) = sum_p w_p (each position carries exactly one byte)
        return 1.0 + (1.0 - gamma) * (sum_w - (1.0 - q_null))

    green = {
        "q_null": q_null_bound,
        "required_q_null": q_null_bound,
        "sum_P": emission_sum(q_null_bound),
    }
    red = {
        "q_null": q_null_independent,
        "required_q_null": q_null_bound,
        "sum_P": emission_sum(q_null_independent),
        "deviation": emission_sum(q_null_independent) - 1.0,
    }
    return {
        "verdict": "proposition_4_1_premise_not_closed_by_the_w_definition",
        "alpha_null": a_null,
        "lambda": lam,
        "m_mass": m_mass,
        "sum_w_from_definition": sum_w,
        "sum_w_1_minus_alpha_null_scaled": 1.0 - a_null * (1.0 - lam * m_mass),
        "required_binding": "q_null == alpha_null * (1 - lambda * m_mass)",
        "green_control": green,
        "red_counterexample": red,
        "red_detects_break": bool(abs(red["deviation"]) > 1e-9),
        "green_control_passes": bool(abs(green["sum_P"] - 1.0) < 1e-12),
        "note": (
            "U1 section 4 asserts sum_p w_p = 1 - q_null, but its own construction "
            "w = lambda*m + (1-lambda*m_mass)*anchor with anchor mass 1-alpha_null gives "
            "sum_p w_p = 1 - alpha_null*(1-lambda*m_mass). Without the binding, "
            "proposition 4.1's normalization premise fails. The upstream U1 precheck "
            "hard-codes w to satisfy the assertion, so it cannot detect this."
        ),
    }


def check_r3() -> dict:
    d = 4
    rng = np.random.default_rng(20261107)
    phi = rng.normal(size=d)
    c = rng.normal(size=d)
    blocks = [rng.normal(size=d) for _ in range(7)]
    ell = 0.37
    w_out = np.zeros((d, d))
    arms = {}
    for label, w_rr in (
        ("green_W_rR_random", rng.normal(size=(d, d))),
        ("red_W_rR_also_zero", np.zeros((d, d))),
    ):
        w8 = np.concatenate(
            [rng.normal(size=(d, 5 * d)), w_rr, rng.normal(size=(d, 2 * d))], axis=1
        )
        x = np.concatenate(blocks[:5] + [np.zeros(d)] + blocks[5:])
        pre = w8 @ x
        g_r = ell * (1.0 - np.tanh(pre) ** 2) * c  # dL/dh' folded through tanh
        r_r = w_out @ phi  # zero-initialized output layer -> zero vector
        grad_w_out = np.outer(w_rr.T @ g_r, phi)
        grad_w_rr = np.outer(g_r, r_r)
        arms[label] = {
            "grad_W_out_norm": float(np.linalg.norm(grad_w_out)),
            "grad_W_rR_norm": float(np.linalg.norm(grad_w_rr)),
            "rR_norm": float(np.linalg.norm(r_r)),
        }
    return {
        "verdict": "proposition_2_2_startup_credit_requires_nonzero_W_rR",
        "green_control": arms["green_W_rR_random"],
        "red_counterexample": arms["red_W_rR_also_zero"],
        "red_detects_dead_saddle": bool(
            arms["red_W_rR_also_zero"]["grad_W_out_norm"] == 0.0
            and arms["red_W_rR_also_zero"]["grad_W_rR_norm"] == 0.0
        ),
        "green_control_passes": bool(arms["green_W_rR_random"]["grad_W_out_norm"] > 1e-3),
        "note": (
            "With Read_R output layer zero, dL/dW_out = (W_rR^T g)*phi^T. If W_rR is "
            "also zero-initialized, both dL/dW_out and dL/dW_rR are exactly zero and "
            "the record path can never start. U1 proposition 2.2 states the gradient "
            "is 'generally nonzero' but does not register W_rR != 0 as a premise; "
            "upstream precheck U3 uses a random W8 (hence random W_rR) only."
        ),
    }


def check_r4() -> dict:
    now_physical = 10.0
    now_wall = 12.0
    packet = {"t_end": 9.0, "t_arrival": 12.5, "source_legal": True}
    u1_single_gate = packet["t_end"] <= now_physical
    mgw1_double_gate = (
        packet["t_end"] <= now_physical
        and packet["t_arrival"] <= now_wall
        and packet["source_legal"]
    )
    return {
        "verdict": "proposition_7_1_inherits_only_half_of_the_MGW1_causal_gate",
        "packet": packet,
        "now_physical_clock": now_physical,
        "now_wall_clock": now_wall,
        "u1_single_t_end_gate_readable": bool(u1_single_gate),
        "mgw1_t_end_and_t_arrival_gate_readable": bool(mgw1_double_gate),
        "disagreement": bool(u1_single_gate and not mgw1_double_gate),
        "note": (
            "MGW1 section 2 gates on t_end (environment physical clock) AND t_arrival "
            "(monotone wall clock) plus source legality, and explicitly forbids comparing "
            "the two clocks numerically. U1 section 7 states only 't_end <= now' and "
            "collapses both clocks into one 'now'. A packet with t_end <= now but "
            "t_arrival > now is future information under MGW1 yet readable under U1, "
            "so proposition 7.1's 'no future leakage' conclusion is not proven for "
            "this case. Upstream precheck U7 varies t_end only."
        ),
    }


def _batch_solve(f0: np.ndarray, pairs, dim: int):
    a = np.eye(dim) + sum(b * np.outer(k, k) for k, _, b in pairs)
    b = f0 + sum(b * np.outer(k, v) for k, v, b in pairs)
    return np.linalg.solve(a, b), a, b


def _batch_solve_constrained(f0: np.ndarray, pairs, dim: int, cap: float):
    x0, a, b = _batch_solve(f0, pairs, dim)
    if np.linalg.norm(x0, "fro") <= cap:
        return x0, False

    def at(mu: float) -> np.ndarray:
        return np.linalg.solve(a + mu * np.eye(dim), b)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if np.linalg.norm(at(hi), "fro") <= cap:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(at(mid), "fro") > cap:
            lo = mid
        else:
            hi = mid
    return at(0.5 * (lo + hi)), True


def check_r5() -> dict:
    dim, vdim, cap = 6, 4, 1.0
    rng = np.random.default_rng(20261107)
    arms = {}

    # Green: F0 = 0, single pair -> X_uc lies in a single eigenspace of A.
    k = np.zeros(dim)
    k[0] = 1.0
    v = np.zeros(vdim)
    v[0] = 4.0
    green_pairs = [(k, v, 100.0)]
    f0_green = np.zeros((dim, vdim))
    x_uc, _, _ = _batch_solve(f0_green, green_pairs, dim)
    x_proj = x_uc * min(1.0, cap / np.linalg.norm(x_uc, "fro"))
    x_con, active = _batch_solve_constrained(f0_green, green_pairs, dim, cap)
    arms["green_single_eigenspace"] = {
        "constraint_active": bool(active),
        "gap_euclidean_projection_vs_constrained_proximal": float(
            np.linalg.norm(x_proj - x_con, "fro")
        ),
    }

    # Red: F0 != 0, several pairs -> A has several distinct eigenvalues.
    f0_red = 0.5 * rng.normal(size=(dim, vdim))
    red_pairs = [
        (rng.normal(size=dim) * 0.4, rng.normal(size=vdim), 50.0),
        (rng.normal(size=dim) * 0.4, rng.normal(size=vdim), 80.0),
        (rng.normal(size=dim) * 0.4, rng.normal(size=vdim), 120.0),
    ]
    x_uc_r, _, _ = _batch_solve(f0_red, red_pairs, dim)
    x_proj_r = x_uc_r * min(1.0, cap / np.linalg.norm(x_uc_r, "fro"))
    x_con_r, active_r = _batch_solve_constrained(f0_red, red_pairs, dim, cap)
    gap_r = float(np.linalg.norm(x_proj_r - x_con_r, "fro"))
    arms["red_multiple_eigenspaces"] = {
        "constraint_active": bool(active_r),
        "gap_euclidean_projection_vs_constrained_proximal": gap_r,
        "norm_unconstrained": float(np.linalg.norm(x_uc_r, "fro")),
    }
    return {
        "verdict": "constrained_proximal_solution_is_not_post_hoc_euclidean_projection",
        "arms": arms,
        "red_detects_mismatch": bool(active_r and gap_r > 1e-3),
        "green_control_passes": bool(
            arms["green_single_eigenspace"]["gap_euclidean_projection_vs_constrained_proximal"]
            < 1e-10
        ),
        "note": (
            "KKT gives X_mu=(A+mu I)^-1 B: each eigendirection of A is shrunk by "
            "alpha_i/(alpha_i+mu), which is non-uniform. The edit_fast Pi operator "
            "shrinks all entries by one scalar. The two coincide only when A is "
            "proportional to I or X_uc lies in one eigenspace (F0=0, single pair). "
            "So U1 theorem 5.1 silently compares three different maps."
        ),
    }


def check_r6() -> dict:
    dim, vdim = 4, 3
    beta = 1e6
    f0 = np.zeros((dim, vdim))
    k = np.zeros(dim)
    k[0] = 1.0
    v1 = np.array([3.0, 0.0, 0.0])
    v2 = np.array([0.0, 3.0, 0.0])

    # Red: one shared key, two conflicting values.
    joint, _, _ = _batch_solve(f0, [(k, v1, beta), (k, v2, beta)], dim)
    s_joint = joint.T @ k
    x1, _, _ = _batch_solve(f0, [(k, v1, beta)], dim)
    x2, _, _ = _batch_solve(x1, [(k, v2, beta)], dim)
    x1b, _, _ = _batch_solve(f0, [(k, v2, beta)], dim)
    x2b, _, _ = _batch_solve(x1b, [(k, v1, beta)], dim)
    s_fwd, s_rev = x2.T @ k, x2b.T @ k
    joint_read_residual = float(np.linalg.norm(s_joint - v1) + np.linalg.norm(s_joint - v2))
    weighted_average = 0.5 * (v1 + v2)

    # Green: orthogonal keys -> sequential re-anchoring commutes with the joint solve.
    k2 = np.zeros(dim)
    k2[1] = 1.0
    joint_g, _, _ = _batch_solve(f0, [(k, v1, beta), (k2, v2, beta)], dim)
    g1, _, _ = _batch_solve(f0, [(k, v1, beta)], dim)
    g2, _, _ = _batch_solve(g1, [(k2, v2, beta)], dim)
    gap_green = float(np.linalg.norm(joint_g - g2, "fro"))

    return {
        "verdict": "domain_order_artifact_over_attributed_when_pairs_conflict",
        "red_arm": {
            "joint_read": s_joint.tolist(),
            "beta_weighted_average": weighted_average.tolist(),
            "sequential_v1_then_v2_read": s_fwd.tolist(),
            "sequential_v2_then_v1_read": s_rev.tolist(),
            "sequential_gap": float(np.linalg.norm(s_fwd - s_rev)),
            "joint_read_residual": joint_read_residual,
        },
        "green_arm": {"orthogonal_key_gap": gap_green},
        "red_detects_conflict": bool(
            np.linalg.norm(s_fwd - s_rev) > 1.0
            and joint_read_residual > 1.0
            and np.linalg.norm(s_joint - weighted_average) < 1e-4
        ),
        "green_control_passes": bool(gap_green < 1e-8),
        "note": (
            "With beta1=beta2 and one key, the joint read tends to "
            "(v1+v2)/2 with irreducible residual, while sequential updates tend "
            "to the last-written value. That gap is conflict allocation, not a "
            "domain-order artifact; the joint solve removes only the order "
            "dependence. With orthogonal keys the two orders commute and both "
            "equal the joint solve (green control)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out: Path = args.output
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {out}")
    if not out.parent.is_dir():
        raise SystemExit(f"output directory does not exist: {out.parent}")

    try:
        basis = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        basis = "unavailable"

    payload = {
        "schema": "taiji-u1-math-review-20261107-v1",
        "date": "2026-11-07",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reviewed_spec": "plans/reference/SPEC-TAIJI-UNIFIED_CORE_U1_20261107.md",
        "basis_commit": basis,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Independent algebra/numerical review of four U1 propositions. Synthetic "
            "float64 data only; 0 production graph, 0 checkpoint, 0 dataset, 0 training, "
            "0 sealed-face access. Not a capability claim and not a production gate."
        ),
        "checks": {
            "R1_beta_limit_under_sphere_constraint": check_r1(),
            "R2_emission_normalization_premise": check_r2(),
            "R3_zero_init_startup_saddle": check_r3(),
            "R4_causal_gate_inheritance": check_r4(),
            "R5_constrained_proximal_vs_euclidean_projection": check_r5(),
            "R6_conflict_pairs_vs_domain_order": check_r6(),
        },
    }
    payload["all_red_arms_detect_defect"] = all(
        [
            payload["checks"]["R1_beta_limit_under_sphere_constraint"]["red_refutes_claim"],
            payload["checks"]["R2_emission_normalization_premise"]["red_detects_break"],
            payload["checks"]["R3_zero_init_startup_saddle"]["red_detects_dead_saddle"],
            payload["checks"]["R4_causal_gate_inheritance"]["disagreement"],
            payload["checks"]["R5_constrained_proximal_vs_euclidean_projection"][
                "red_detects_mismatch"
            ],
            payload["checks"]["R6_conflict_pairs_vs_domain_order"]["red_detects_conflict"],
        ]
    )
    payload["all_green_controls_pass"] = all(
        [
            payload["checks"]["R1_beta_limit_under_sphere_constraint"]["green_control_passes"],
            payload["checks"]["R2_emission_normalization_premise"]["green_control_passes"],
            payload["checks"]["R3_zero_init_startup_saddle"]["green_control_passes"],
            payload["checks"]["R5_constrained_proximal_vs_euclidean_projection"][
                "green_control_passes"
            ],
            payload["checks"]["R6_conflict_pairs_vs_domain_order"]["green_control_passes"],
        ]
    )
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, check in payload["checks"].items():
        print(f"  {name}: {check['verdict']}")
    print(
        f"red arms detect defect: {payload['all_red_arms_detect_defect']}; "
        f"green controls pass: {payload['all_green_controls_pass']}"
    )


if __name__ == "__main__":
    main()
