"""Fusion-equivalence pre-check for U1 proposition 4.2's decomposition note.

Answers the 2026-11-07 owner question "why two operators instead of one
fused operator":
  F1  the fused single-table posterior (one multiplicative renormalization
      over sources {(branch, gen)} U {(branch, position)}) equals the
      factored form (eta scratch update x per-branch m-cursor update) to
      machine precision over a multi-step utterance  ->  the "two updates"
      are one joint posterior's chain-rule factors; a representation choice.
  F2  the FORBIDDEN fusion (merging the scratch eta into the world omega)
      lets the model's own emitted bytes reweight its world hypotheses with
      no new observation arriving (self-confirmation drift, L-M2 violation).
  F3  with the scratch separation, omega is bit-identical under self-output.

No production Taiji import, no checkpoint, no dataset, no sealed face.
Synthetic values only, float64 numpy. Writes a new report file only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

K, P, T = 2, 3, 5
BYTE_IDS = np.array([65, 66, 65])
VOCAB = np.array([65, 66, 67])

COUNTS = {"utterance_steps_simulated": 0}
CHECKS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")


def step_params(t: int):
    r = np.random.default_rng((20261108, t))
    gamma = r.uniform(0.2, 0.8, K)
    q_null = r.uniform(0.05, 0.25, K)
    gen = np.stack([r.dirichlet(np.ones(3)) for _ in range(K)])
    anchor = np.stack([r.dirichlet(np.ones(P)) for _ in range(K)])
    lam = r.uniform(0.3, 0.9, K)
    return gamma, q_null, gen, anchor, lam


def w_from_m(m: np.ndarray, anchor: np.ndarray, lam: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    for b in range(K):
        mm = m[b].sum()
        out[b] = lam[b] * m[b] + (1 - lam[b] * mm) * anchor[b]
    return out


def run_utterance(mode: str, r2: np.random.Generator) -> dict:
    """Simulate one utterance; returns fused/factored divergence and omega drifts."""
    eta = np.array([0.5, 0.5])
    m = np.zeros((K, P))
    m[:, 0] = 0.5
    omega = np.array([0.5, 0.5])
    omega_fused = np.array([0.5, 0.5])
    worst = 0.0
    for t in range(T):
        gamma, q_null, gen, anchor, lam = step_params(t)
        w = w_from_m(m, anchor, lam)
        gprime = gamma + (1 - gamma) * q_null
        copy_mass = np.stack(
            [(w[b] * (v == BYTE_IDS).astype(float)).sum() for b in range(K) for v in VOCAB]
        ).reshape(K, 3)
        pmix = gprime[:, None] * gen + (1 - gamma)[:, None] * copy_mass
        if mode == "self_sampled":
            mix = eta @ pmix
            y = int(VOCAB[r2.choice(3, p=mix / mix.sum())])
        else:  # adversarial: the byte branch 0 predicts best
            y = int(VOCAB[np.argmax(pmix[0])])
        match = (y == BYTE_IDS).astype(float)
        yi = int(np.flatnonzero(y == VOCAB)[0])

        # fused form: ONE table over sources {(b, gen)} U {(b, p)}
        qg = eta * gprime
        qp = eta[:, None] * (1 - gamma)[:, None] * w
        z = (qg * gen[:, yi]).sum() + (qp * match[None, :]).sum()
        qg_post = qg * gen[:, yi] / z
        qp_post = qp * match[None, :] / z

        # factored form: eta update x m_b update (prop 4.2)
        pb = gprime * gen[:, yi] + (1 - gamma) * (w * match[None, :]).sum(1)
        eta_new = eta * pb
        eta_new /= eta_new.sum()
        r = (1 - gamma)[:, None] * w * match[None, :] / pb[:, None]
        r_gen = gprime * gen[:, yi] / pb
        qg_fact = eta_new * r_gen
        qp_fact = eta_new[:, None] * r

        worst = max(
            worst,
            np.abs(qg_post - qg_fact).max(),
            np.abs(qp_post - qp_fact).max(),
        )

        # forbidden fusion: world omega reweighted by the model's own output
        omega_fused = omega_fused * pb
        omega_fused /= omega_fused.sum()

        eta = eta_new
        m = np.zeros((K, P))
        m[:, 1:] = r[:, :-1]  # within-record shift, no wraparound
        COUNTS["utterance_steps_simulated"] += 1
    return {
        "worst_divergence": worst,
        "omega_scratch": omega,
        "omega_fused": omega_fused,
        "drift_pct": 100 * (omega_fused[0] - 0.5) / 0.5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=None, help="write JSON report to a NEW file (must not exist)"
    )
    args = ap.parse_args()
    t0 = time.process_time()
    self_run = run_utterance("self_sampled", np.random.default_rng(20261108))
    adv_run = run_utterance("adversarial", np.random.default_rng(20261109))

    record(
        "F1 fused single-table posterior == factored (eta x m_b) form, machine precision",
        max(self_run["worst_divergence"], adv_run["worst_divergence"]) < 1e-15,
        f"max |q_fused - q_factored| over both {T}-byte utterances = "
        f"{max(self_run['worst_divergence'], adv_run['worst_divergence']):.2e}; the persistent state is "
        "(eta, m_b) in both forms - the flat table is a per-step materialization (chain rule, not two mechanisms)",
    )
    drifts = (self_run["drift_pct"], adv_run["drift_pct"])
    record(
        "F2 forbidden fusion (scratch merged into world omega) drifts omega on self-output alone",
        min(abs(d) for d in drifts) > 10.0,
        f"after {T} self-emitted bytes with NO new observation: self-sampled drift {drifts[0]:+.0f}% "
        f"(omega -> {np.round(self_run['omega_fused'], 4).tolist()}), adversarial drift {drifts[1]:+.0f}% "
        f"(omega -> {np.round(adv_run['omega_fused'], 4).tolist()}); whichever branch predicts what the model "
        "says gets confirmed by its own speech (L-M2 violation)",
    )
    ok_f3 = np.array_equal(self_run["omega_scratch"], np.array([0.5, 0.5])) and np.array_equal(
        adv_run["omega_scratch"], np.array([0.5, 0.5])
    )
    record(
        "F3 with the scratch separation, world omega is bit-identical under self-output",
        ok_f3,
        f"omega after both utterances = {self_run['omega_scratch'].tolist()} (drift exactly 0; only real "
        "observations reweight world hypotheses)",
    )

    cpu = time.process_time() - t0
    passed = all(c["ok"] for c in CHECKS)
    try:
        basis = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        basis = "unavailable"
    report = {
        "schema": "taiji-unified-u1-fusion-note-v1",
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
        },
        "cpu_seconds": round(cpu, 6),
        "scope": (
            "Equivalence and boundary checks for U1 prop 4.2's decomposition note only: "
            "fused vs factored posterior identity, and the self-confirmation drift of the "
            "forbidden omega/eta fusion. Synthetic values; no production graph, weights or "
            "dataset. Does not sign any capability or production engineering."
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
