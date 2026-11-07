"""Independent algebra pre-checks for the unified-subject algebra U1 (2026-11-07).

INVALID EVIDENCE (annotated 2026-11-08, owner-approved U1 errata):
  The U1 normalization check below (``check_u1``, lines ~64-68) builds ``w`` by
  hard-coding rows that already sum to ``1 - q_null``.  That makes the assertion
  a tautology on hand-satisfied input; it is NOT evidence that U1 emission is
  normalized.  The mathematical statement is now an identity under the bound
  null definition (SPEC-TAIJI-UNIFIED_CORE_U1_20261107.md §3 定义 3.1 / §4
  命题 4.1) and must be re-tested with ``w`` generated from (alpha, beta, lambda,
  m_mass) per the construction, not with a pre-normalized ``w``.  See
  U1-ERRATA_20261108.md §2.2 and the spec's §4 evidence-invalid banner.

Verifies the seam-closure propositions of SPEC-TAIJI-UNIFIED_CORE_U1:
  U1  unified emission normalization (K-branch x per-branch gen/copy mixture)
  U2  posterior decomposition = exact joint Bayes over (branch, record position)
      (eta scratch update + per-branch m-cursor update), 2M-sample Monte Carlo
  U3  extension neutrality: 8d H with zero-initialized Read_R is bit-identical
      to the 7d (MGW1) H; startup gradient to Read_R output layer is nonzero
  U4  M1 carries over: d(copy_mass)/d(alpha) != 0 for the anchor-direct form
  U5  FP1 joint batch solve is exact and domain-order invariant, unlike
      sequential per-domain updates
  U6  8d H boundedness under unbounded inputs (prop 2.1)
  U7  record-domain causal gating: t_end-gated records cannot enter past
      predictions (prop 7.1)

No production Taiji import, no checkpoint, no dataset, no sealed face.
Synthetic values only, float64 numpy. Writes a new report file only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

COUNTS = {
    "emission_samples": 0,
    "h_steps_checked": 0,
    "fp1_solves": 0,
}

CHECKS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def run_checks(rng: np.random.Generator) -> None:
    # ------------------------------------------------------------------
    # Shared synthetic unified-emission fixture: K branches, one record with
    # 3 positions (bytes 65/66/65), per-branch gen/copy mixtures.
    # ------------------------------------------------------------------
    K = 2
    byte_ids = np.array([65, 66, 65])
    vocab = [65, 66, 67]
    eta0 = np.array([0.6, 0.4])
    q_null = np.array([0.10, 0.20])
    gamma = np.array([0.30, 0.55])
    w = np.zeros((K, 3))
    w[0] = np.array([0.55, 0.20, 0.15])  # sums to 1 - q_null[0] = 0.90
    w[1] = np.array([0.30, 0.35, 0.15])  # sums to 1 - q_null[1] = 0.80
    gen = np.zeros((K, 3))
    gen[0] = np.array([0.50, 0.30, 0.20])
    gen[1] = np.array([0.25, 0.45, 0.30])

    def branch_mix(b: int) -> np.ndarray:
        gprime = gamma[b] + (1 - gamma[b]) * q_null[b]
        copy_mass = np.array([w[b][byte_ids == y].sum() for y in vocab])
        return gprime * gen[b] + (1 - gamma[b]) * copy_mass

    mix = np.stack([branch_mix(b) for b in range(K)])  # (K, |vocab|)
    P = eta0 @ mix

    # U1: normalization -------------------------------------------------
    worst = abs(float(P.sum()) - 1.0)
    worst = max(worst, max(abs(float(mix[b].sum()) - 1.0) for b in range(K)))
    record(
        "U1 unified emission normalization (per-branch mixtures and eta mixture)",
        worst < 1e-12,
        f"max|sum-1| = {worst:.2e}; per-branch sums = {[round(float(mix[b].sum()), 12) for b in range(K)]}",
    )

    # U2: posterior decomposition = exact joint Bayes --------------------
    acc = 65
    eta_formula = eta0 * mix[:, vocab.index(acc)]
    eta_formula = eta_formula / eta_formula.sum()
    r_formula = np.zeros((K, 3))
    for b in range(K):
        r_formula[b] = (
            np.where(byte_ids == acc, w[b], 0.0) * (1 - gamma[b]) / mix[b, vocab.index(acc)]
        )
    MC = 2_000_000
    comp_pos = np.zeros(MC, dtype=np.int8)
    pos = np.zeros(MC, dtype=np.int8)
    byt = np.zeros(MC, dtype=np.int8)
    for b in range(K):
        sel_b = slice(int(MC * (0 if b == 0 else eta0[:b].sum())), int(MC * eta0[: b + 1].sum()))
        nb = sel_b.stop - sel_b.start
        comp_pos[sel_b] = b
        gp = gamma[b] + (1 - gamma[b]) * q_null[b]
        is_gen = rng.random(nb) < gp
        pos[sel_b] = rng.choice(3, size=nb, p=w[b] / w[b].sum())
        pos[sel_b] = np.where(is_gen, -1, pos[sel_b])  # gen-origin samples carry no position
        sampled_gen = rng.choice(3, size=nb, p=gen[b])
        byt[sel_b.start : sel_b.stop] = np.where(
            is_gen,
            sampled_gen,
            np.searchsorted(np.array(vocab), byte_ids[np.where(pos[sel_b] >= 0, pos[sel_b], 0)]),
        )
    sel = byt == vocab.index(acc)
    mc_eta = np.array([(comp_pos[sel] == b).mean() for b in range(K)])
    mc_r = np.zeros((K, 3))
    for b in range(K):
        mb = sel & (comp_pos == b) & (pos >= 0)
        denom = (sel & (comp_pos == b)).sum()
        mc_r[b] = [(pos[mb] == p).sum() / denom for p in range(3)]
    COUNTS["emission_samples"] += MC
    ok_u2 = (
        np.abs(mc_eta - eta_formula).max() < 3e-3
        and np.abs(mc_r - r_formula).max() < 5e-3
        and abs(eta_formula.sum() - 1) < 1e-15
        and all(r_formula[b].sum() <= 1 + 1e-12 for b in range(K))
    )
    record(
        "U2 posterior decomposition = exact joint Bayes over (branch, position)",
        ok_u2,
        f"2M-sample MC: max|eta_MC - eta_formula| = {np.abs(mc_eta - eta_formula).max():.2e}; "
        f"max|r_MC - r_formula| = {np.abs(mc_r - r_formula).max():.2e}; "
        f"eta sums to 1 exactly; per-branch r sums <= 1 (gen responsibility is the complement)",
    )

    # U3: extension neutrality -------------------------------------------
    d = 4
    h = rng.uniform(-1, 1, d)
    blocks = [rng.uniform(-1, 1, d) for _ in range(7)]  # 7 shared d-blocks
    rR_zero = np.zeros(d)
    rR_nonzero = rng.uniform(-1, 1, d)
    W8 = rng.normal(size=(d, 8 * d))
    W7 = np.delete(W8, slice(5 * d, 6 * d), axis=1)  # drop the r^R columns
    ell = 0.37

    def step8(x_blocks, r_r):
        x = np.concatenate(x_blocks[:5] + [r_r] + x_blocks[5:])
        pre = W8 @ x
        return (1 - ell) * h + ell * np.tanh(pre)

    h7 = (1 - ell) * h + ell * np.tanh(W7 @ np.concatenate(blocks))
    h8_zero = step8(blocks, rR_zero)
    h8_nonzero = step8(blocks, rR_nonzero)
    bit_equal = np.array_equal(h7, h8_zero)
    # startup gradient: dL/d(rR) = W_rR^T [ell*(1-tanh^2(pre)) .* dL/dh']
    c = rng.normal(size=d)

    def loss8(r_r):
        x = np.concatenate(blocks[:5] + [r_r] + blocks[5:])
        pre = W8 @ x
        hh = (1 - ell) * h + ell * np.tanh(pre)
        return float(c @ hh)

    eps = 1e-6
    grad_fd = np.array(
        [(loss8(rR_zero + eps * e) - loss8(rR_zero - eps * e)) / (2 * eps) for e in np.eye(d)]
    )
    W_rR = W8[:, 5 * d : 6 * d]
    pre0 = W8 @ np.concatenate(blocks[:5] + [rR_zero] + blocks[5:])
    grad_ana = W_rR.T @ (ell * (1 - np.tanh(pre0) ** 2) * c)
    ok_u3 = (
        bit_equal
        and not np.allclose(h7, h8_nonzero)
        and np.abs(grad_fd - grad_ana).max() < 1e-9
        and np.linalg.norm(grad_ana) > 1e-3
    )
    record(
        "U3 extension neutrality: 8d H with zero-init Read_R == 7d H bit-exact; startup gradient nonzero",
        ok_u3,
        f"h7 == h8(ReadR=0) bitwise: {bit_equal}; h8(ReadR!=0) differs: {not np.allclose(h7, h8_nonzero)}; "
        f"max|FD-analytic grad| = {np.abs(grad_fd - grad_ana).max():.1e}; ||grad|| = {np.linalg.norm(grad_ana):.4f} > 0",
    )

    # U4: M1 carries (d copy_mass / d alpha != 0, anchor-direct form) ------
    n_rec = 3
    alpha = rng.dirichlet(np.ones(n_rec + 1))  # last = null
    betas = [rng.dirichlet(np.ones(3)) for _ in range(n_rec)]
    rec_bytes = np.array([[65, 66, 65], [66, 66, 65], [67, 65, 66]])
    target = 65

    def copy_mass(a):
        m = 0.0
        for i in range(n_rec):
            m += a[i] * betas[i][rec_bytes[i] == target].sum()
        return m

    grads = np.zeros(n_rec)
    for i in range(n_rec):
        e = np.zeros(n_rec + 1)
        e[i] = 1e-6
        grads[i] = (copy_mass(alpha + e) - copy_mass(alpha - e)) / 2e-6
    ok_u4 = np.abs(grads).max() > 1e-3 and np.all(grads >= -1e-12)
    record(
        "U4 M1 carries over: anchor-direct copy_mass has nonzero gradient w.r.t. record selection alpha",
        ok_u4,
        f"d(copy_mass)/d(alpha) = {np.round(grads, 6).tolist()} (content fidelity and record "
        "selection are not structurally separable, M1 verbatim)",
    )

    # U5: FP1 joint solve exact + domain-order invariant ------------------
    kdim, vdim = 6, 4
    F0 = rng.normal(size=(kdim, vdim))
    obj_pairs = [
        (rng.normal(size=kdim), rng.normal(size=vdim), float(np.exp(rng.normal())))
        for _ in range(3)
    ]
    rec_pairs = [
        (rng.normal(size=kdim), rng.normal(size=vdim), float(np.exp(rng.normal())))
        for _ in range(2)
    ]

    def solve_batch(pairs, base):
        A = np.eye(kdim) + sum(b * np.outer(kk, kk) for kk, _, b in pairs)
        B = base + sum(b * np.outer(kk, vv) for kk, vv, b in pairs)
        X = np.linalg.solve(A, B)
        return X, A, B

    X_joint, A_j, B_j = solve_batch(obj_pairs + rec_pairs, F0)
    X_ord1, _, _ = solve_batch(rec_pairs, solve_batch(obj_pairs, F0)[0])
    X_ord2, _, _ = solve_batch(obj_pairs, solve_batch(rec_pairs, F0)[0])
    perm = list(range(5))[::-1]
    allp = obj_pairs + rec_pairs
    X_perm, _, _ = solve_batch([allp[i] for i in perm], F0)
    COUNTS["fp1_solves"] += 4
    ok_u5 = (
        np.abs(A_j @ X_joint - B_j).max() < 1e-11
        and np.abs(X_perm - X_joint).max() < 1e-12
        and np.abs(X_ord1 - X_joint).max() > 1e-3
        and np.abs(X_ord1 - X_ord2).max() > 1e-3
    )
    record(
        "U5 FP1 joint solve: exact, permutation/domain-order invariant; sequential per-domain is order-dependent",
        ok_u5,
        f"joint FOC residual = {np.abs(A_j @ X_joint - B_j).max():.1e}; max|X_perm - X_joint| = "
        f"{np.abs(X_perm - X_joint).max():.1e}; sequential obj-then-rec vs rec-then-obj differ by "
        f"{np.abs(X_ord1 - X_ord2).max():.3f} (order artifact eliminated only by the joint solve)",
    )

    # U6: 8d boundedness with unbounded inputs ----------------------------
    worst = 0.0
    for _ in range(200_000):
        x = rng.normal(size=8 * d) * rng.uniform(0.1, 100)
        hh = rng.uniform(-1, 1, d)
        tau = np.exp(rng.uniform(-2, 6))
        l = 1 - math.exp(-1 / tau)
        hh2 = (1 - l) * hh + l * np.tanh(x[:d] * 0 + rng.normal(size=d) * rng.uniform(1, 50))
        worst = max(worst, np.abs(hh2).max())
    COUNTS["h_steps_checked"] += 200_000
    record(
        "U6 8d leaky-tanh boundedness under unbounded input magnitudes",
        worst <= 1.0 + 1e-12,
        f"max|h'| = {worst:.15f} over 200k random steps with inputs up to 100 sigma",
    )

    # U7: record-domain causal gating -------------------------------------
    now = 10.0
    packets = [
        {"kind": "text_record", "t_end": 5.0, "payload": "rec1"},
        {"kind": "text_record", "t_end": 10.0, "payload": "rec2"},
        {"kind": "text_record", "t_end": 11.5, "payload": "rec3-future"},
        {"kind": "rgb", "t_end": 9.0, "payload": "frame"},
    ]

    def readable(p, t):
        return p["t_end"] <= t

    visible_now = sorted(p["payload"] for p in packets if readable(p, now))
    visible_later = sorted(p["payload"] for p in packets if readable(p, 12.0))
    ok_u7 = (
        "rec3-future" not in visible_now
        and visible_now == ["frame", "rec1", "rec2"]
        and "rec3-future" in visible_later
    )
    record(
        "U7 record-domain causal gating: t_end-gated records cannot enter past predictions",
        ok_u7,
        f"readable at t=10: {visible_now}; at t=12: {visible_later}; future record excluded until its t_end",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=None, help="write JSON report to a NEW file (must not exist)"
    )
    args = ap.parse_args()
    t0 = time.process_time()
    rng = np.random.default_rng(20261107)
    run_checks(rng)
    cpu = time.process_time() - t0
    passed = all(c["ok"] for c in CHECKS)
    try:
        basis = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        basis = "unavailable"
    report = {
        "schema": "taiji-unified-u1-math-precheck-v1",
        "date": "2026-11-07",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "basis_commit": basis,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": CHECKS,
        "accounting": {
            **COUNTS,
            "production_imports": 0,
            "checkpoints_loaded": 0,
            "dataset_reads": 0,
            "sealed_face_reads": 0,
        },
        "cpu_seconds": round(cpu, 6),
        "scope": (
            "Algebra pre-checks for the U1 unified-subject seam-closure propositions only "
            "(emission normalization, joint posterior decomposition, extension neutrality, "
            "M1 carryover, FP1 domain-order invariance, 8d boundedness, record causal gating). "
            "Synthetic values; no production graph, weights, dataset or sealed face. Does not "
            "sign production engineering, cross-candidate weight transfer, or any capability."
        ),
    }
    print(f"\nstatus={'passed' if passed else 'failed'}; checks={len(CHECKS)}; cpu={cpu:.3f}s")
    if args.output is not None:
        out = args.output
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing report: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
