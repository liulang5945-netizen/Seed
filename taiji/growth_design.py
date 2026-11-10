"""Growth certificates and the growth decision rule (design layer, numpy only).

Implements plans/reference/SPEC-TAIJI-GROWTH_DESIGN_20261108.md:

* ``key_spectrum`` / ``read_margin_stats`` -- is the read-key space collapsed,
  and is the collapse a shared offset (repairable by re-keying) or an intrinsic
  low-rank budget (candidate for dimension growth)?
* ``capacity_certificate`` -- is a FP1 domain short of capacity (``delta_cap``)
  or structurally inseparable (``delta_comp``)?
* ``store_decision`` / ``operator_decision`` -- the two mechanisms decide
  separately and never speak for each other.
* ``allocate`` -- integer max-min allocation, never an even spread.

This module is deliberately numpy-only and is NOT imported by any training
runner (see plans/reference/FROZEN-READSET_20261108.md). Nothing here trains,
loads weights or touches a checkpoint.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np

R_MAX = 8
ENUMERATION_CAP = 6561

# Relatedness band for a new growth direction (spec §4A).  Starting values from
# the constructive pre-review; recalibrated per domain after the first report.
BAND = {"o_min": 0.15, "o_max": 0.85, "s_min": 0.30}

# Nominal, uncalibrated thresholds.  The capacity branch is algebraic and does
# not use them; the key-space branch stays ``no_growth`` until calibrated.
THRESHOLDS = {
    "top_cos": 0.95,
    "margin": 0.02,
    "entropy": 0.35,
    "top_share_raw": 0.90,
    "top_share_centered": 0.50,
    "effective_rank_fraction": 0.90,
    "struct_epsilon": 1e-6,
}


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return cast(np.ndarray, values / np.clip(norms, 1e-12, None))


def _spectrum(values: np.ndarray) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"top_share": float("nan"), "effective_rank": float(values.shape[0])}
    gram = values @ values.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    denom = float((eigenvalues**2).sum())
    return {
        "top_share": float(eigenvalues[-1] / total) if total > 0 else float("nan"),
        "effective_rank": float(total**2 / denom) if denom > 0 else 0.0,
    }


def key_spectrum(keys: np.ndarray) -> dict[str, float]:
    """Raw and centered spectrum of unit-normalised read keys ``(n, d)``."""
    normalized = unit_rows(np.asarray(keys, dtype=np.float64))
    off = normalized @ normalized.T
    np.fill_diagonal(off, 0.0)
    raw = _spectrum(normalized)
    centered = _spectrum(normalized - normalized.mean(axis=0, keepdims=True))
    count = normalized.shape[0]
    return {
        "occupied": float(count),
        "width": float(normalized.shape[1]),
        "mean_abs_cos": float(np.abs(off).sum() / max(count * (count - 1), 1)),
        "top_share": raw["top_share"],
        "effective_rank": raw["effective_rank"],
        "top_share_centered": centered["top_share"],
        "effective_rank_centered": centered["effective_rank"],
        "effective_rank_fraction": raw["effective_rank"] / max(count, 1),
    }


def read_margin_stats(keys: np.ndarray, temperature: float = 0.1) -> dict[str, float]:
    """Top-1/top-2 cosine margins and normalised softmax entropy per query."""
    normalized = unit_rows(np.asarray(keys, dtype=np.float64))
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    ordered = np.sort(gram, axis=1)[:, ::-1]
    top, second = ordered[:, 0], ordered[:, 1]
    margins = top - second
    weights = np.exp((gram - gram.max(axis=1, keepdims=True)) / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    entropy = -(weights * np.log(np.clip(weights, 1e-300, None))).sum(axis=1)
    entropy /= np.log(max(normalized.shape[0] - 1, 2))
    return {
        "p95_top_cos": float(np.percentile(top, 95)),
        "median_top_cos": float(np.median(top)),
        "median_margin": float(np.median(margins)),
        "p95_margin": float(np.percentile(margins, 95)),
        "median_entropy": float(np.median(entropy)),
    }


def direction_band(
    existing: np.ndarray,
    candidate: np.ndarray,
    *,
    band: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Relatedness band for a new unit direction (spec §4A).

    ``existing`` is ``(n, d)`` unit directions already in the subject,
    ``candidate`` is ``(d,)``.  Returns ``o/r/s/t`` plus one of
    ``growth_new_direction`` / ``growth_is_copy`` / ``growth_isolated``.
    Only ``growth_new_direction`` may be counted as a capacity gain.
    """
    tau = {**BAND, **(band or {})}
    old = unit_rows(np.asarray(existing, dtype=np.float64))
    new = unit_rows(np.asarray(candidate, dtype=np.float64).reshape(1, -1))[0]
    if old.shape[0] == 0:
        overlap, nearest = 0.0, 0.0
    else:
        _, _, basis = np.linalg.svd(old, full_matrices=False)
        rank = int(np.linalg.matrix_rank(old))
        basis = basis[: max(rank, 1)]
        overlap = float(np.linalg.norm(basis @ new))
        nearest = float(np.abs(old @ new).max())
    overlap = min(overlap, 1.0)
    novelty = float(np.sqrt(max(1.0 - overlap**2, 0.0)))
    separation = 1.0 - nearest
    if overlap > tau["o_max"] or separation < tau["s_min"]:
        verdict = "growth_is_copy"
    elif overlap < tau["o_min"]:
        verdict = "growth_isolated"
    else:
        verdict = "growth_new_direction"
    return {
        "overlap": overlap,
        "novelty": novelty,
        "separation": separation,
        "nearest_cos": nearest,
        "o_min": tau["o_min"],
        "o_max": tau["o_max"],
        "s_min": tau["s_min"],
        "r_min": float(np.sqrt(max(1.0 - tau["o_max"] ** 2, 0.0))),
        "verdict": verdict,
        "counts_as_capacity": verdict == "growth_new_direction",
    }


def capacity_certificate(K: np.ndarray, V: np.ndarray, c_f: float) -> dict[str, Any]:
    """Capacity vs structure for one FP1 domain.

    ``K`` is ``(d, m)``, ``V`` is ``(r, m)``: the association target is
    ``X^T K = V`` under ``||X||_F <= c_f``.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    pinv = np.linalg.pinv(K)
    rho = float(np.linalg.norm(V @ pinv))
    residual = V - V @ pinv @ K
    delta_comp = float(np.linalg.norm(residual))
    singular = np.linalg.svd(K, compute_uv=False)
    sigma_plus = float(singular[singular > 1e-12].min()) if np.any(singular > 1e-12) else 0.0
    delta_cap = max(0.0, rho - c_f)
    return {
        "rho": rho,
        "c_f": float(c_f),
        "delta_cap": delta_cap,
        "delta_comp": delta_comp,
        "sigma_plus": sigma_plus,
        "readout_lower_bound": sigma_plus * delta_cap,
        "rank_K": int(np.linalg.matrix_rank(K)),
        "rank_consistent": bool(delta_comp <= 1e-9 * max(1.0, float(np.linalg.norm(V)))),
    }


def store_decision(
    spectrum: dict[str, float] | None,
    margins: dict[str, float] | None,
    *,
    calibrated: bool = False,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Key-space decision.  Diagnosis is always reported; the action is gated."""
    tau = {**THRESHOLDS, **(thresholds or {})}
    diagnosis = "insufficient_occupancy"
    collapsed = False
    evidence = {"spectrum": spectrum, "margins": margins}
    if margins:
        collapsed = (
            margins["p95_top_cos"] >= tau["top_cos"] and margins["median_margin"] <= tau["margin"]
        )
        projection_bound = bool(
            spectrum
            and spectrum["effective_rank"] >= tau["effective_rank_fraction"] * spectrum["width"]
        )
        shared_offset = bool(
            spectrum
            and spectrum["top_share"] >= tau["top_share_raw"]
            and spectrum["top_share_centered"] <= tau["top_share_centered"]
        )
        if collapsed and shared_offset:
            diagnosis = "collapsed_shared_offset"
        elif collapsed and projection_bound:
            diagnosis = "projection_bottleneck"
        elif collapsed:
            diagnosis = "collapsed_unattributed"
        elif projection_bound:
            diagnosis = "projection_bottleneck"
        else:
            diagnosis = "separable_no_growth"
    result = {
        "decision": "no_growth",
        "diagnosis": diagnosis,
        "collapse_detected": collapsed,
        "evidence": evidence,
    }
    if not calibrated:
        return result
    if diagnosis == "collapsed_shared_offset":
        result["decision"] = "representation_repair"
    elif diagnosis == "projection_bottleneck":
        result["decision"] = "key_dimension_growth"
    return result


def operator_decision(
    certificate: dict[str, Any], *, thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    """FP1 decision: proof-based, so it does not need calibration."""
    tau = {**THRESHOLDS, **(thresholds or {})}
    epsilon = tau["struct_epsilon"] * max(1.0, certificate["rho"])
    if certificate["delta_comp"] > epsilon:
        decision, diagnosis = "structural_repair", "delta_comp_positive"
    elif certificate["delta_cap"] > 0.0:
        decision, diagnosis = "capacity_growth", "delta_cap_positive"
    else:
        decision, diagnosis = "no_growth", "operational_sufficient"
    return {"decision": decision, "diagnosis": diagnosis, "evidence": certificate}


def allocate(
    rho_tables: dict[str, list[float]] | None,
    caps: dict[str, int],
    r_max: int = R_MAX,
) -> dict[str, int]:
    """Integer max-min allocation: maximise the worst marginal gain first.

    ``rho_tables[name][u]`` is the certificate gain (e.g. residual reduction) of
    giving ``name`` exactly ``u`` units.  Ties: larger sum, then fewer units,
    then lexicographic -- deterministic and free of RNG.
    """
    if not rho_tables:
        return {}
    names = sorted(rho_tables)
    combinations = itertools.product(*(range(min(caps.get(name, 0), r_max) + 1) for name in names))
    best_key, best = None, None
    for index, combination in enumerate(combinations):
        if index >= ENUMERATION_CAP or sum(combination) > r_max:
            continue
        gains = [rho_tables[name][combination[i]] for i, name in enumerate(names)]
        key = (min(gains), sum(gains), -sum(combination), tuple(-value for value in combination))
        if best_key is None or key > best_key:
            best_key, best = key, dict(zip(names, combination, strict=True))
    return {name: units for name, units in (best or {}).items() if units > 0}


def slot_growth_bytes(
    added_slots: int,
    *,
    width: int = 128,
    groups: int = 4,
    fast_heads: int = 2,
    key_width: int = 32,
    value_width: int = 32,
    itemsize: int = 4,
) -> int:
    """State bytes added by extra addressed-store slots (keys + age + count)."""
    per_slot = width + groups * fast_heads * key_width * value_width + 2
    return int(added_slots) * per_slot * itemsize
