"""Independent algebra pre-checks for the MGW1 lifecycle supplement (2026-11-06).

Covers the 2026-11-06 supplement of SPEC-TAIJI-MGW1_MULTIMODAL_WORLD:
  - section 3.1 slot lifecycle operators (birth-from-null / retire / merge),
  - section 5.1 branch collapse diagnostics (ESS),
  - the section 8/11 emission loop (64-byte strict UTF-8 FSM, branch-level
    mask-renormalize then eta mixture, scratch-eta update by the emitted
    byte's branch probability, world omega invariance, truncated semantics).

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

CFG = {
    "theta_birth": 0.15,
    "kappa_birth": 0.6,
    "theta_idle": 0.02,
    "t_retire": 8,
    "delta_merge": 0.05,
    "delta_v": 0.5,
    "kappa_merge": 0.8,
    "e_storm": 5,
    "l_retire": -8.0,
    "bins": 4,
    "max_bytes": 64,
    "ess_warn": 1.2,
    "ess_streak": 10,
}

COUNTS = {
    "birth_calls": 0,
    "births_applied": 0,
    "retires_applied": 0,
    "merges_applied": 0,
    "emission_runs": 0,
    "bytes_emitted": 0,
}

CHECKS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------- lifecycle


def fresh_slots(m: int, rng: np.random.Generator) -> dict[int, dict]:
    return {
        i: {
            "h": rng.normal(size=4) * 0.1,
            "p": rng.uniform(-1, 1, 2),
            "v": np.zeros(2),
            "s": 0.1,
            "l": 0.0,
            "g": 0.5,
        }
        for i in range(m)
    }


def birth_from_null(
    slots: dict[int, dict],
    alpha_null: np.ndarray,
    xy: np.ndarray,
    journal: list[dict],
    event_id: int,
    storm: dict,
    cfg: dict = CFG,
) -> tuple[str, str]:
    """Spec 3.1 birth operator. Mutates slots/journal only on status 'birth'."""
    COUNTS["birth_calls"] += 1
    if any(e["event_id"] == event_id and e["op"] == "birth" for e in journal):
        return "already_applied", "same event_id already has a birth entry"
    a = np.asarray(alpha_null, float)
    total = float(a.sum())
    if total < cfg["theta_birth"]:
        storm["streak"] = 0
        return "no_birth", f"null mass {total:.3f} < theta_birth"
    nb = cfg["bins"]
    idx = np.clip(((np.asarray(xy, float) + 1.0) / 2.0 * nb).astype(int), 0, nb - 1)
    bin_id = idx[:, 0] * nb + idx[:, 1]
    q = np.zeros(nb * nb)
    np.add.at(q, bin_id, a)
    b = int(np.argmax(q))
    if q[b] < cfg["kappa_birth"] * total:
        storm["streak"] = 0
        return "no_birth", f"dominant bin mass {q[b]:.3f} < kappa_birth*A"
    idle = [(sigmoid(s["l"]), sid) for sid, s in slots.items() if s["g"] < cfg["theta_idle"]]
    if not idle:
        storm["streak"] = 0
        return "no_birth", "no idle slot below theta_idle"
    if storm["streak"] >= cfg["e_storm"]:
        # birth is demanded but the consecutive-birth gate is closed; the
        # streak stands (the instrument should have halted at the report).
        return "churn_stop", f"birth streak {storm['streak']} reached e_storm"
    _, sid = min(idle)  # lowest rho, tie -> smallest stable slot id
    m = bin_id == b
    w = a[m]
    c = np.asarray(xy, float)[m]
    centroid = (w[:, None] * c).sum(0) / w.sum()
    old = slots[sid]
    slots[sid] = {
        "h": np.zeros_like(old["h"]),  # graph reseeds h from cluster tokens; marker here
        "p": centroid,
        "v": np.zeros(2),
        "s": 0.1,
        "l": 0.0,
        "g": old["g"],
    }
    journal.append(
        {
            "event_id": event_id,
            "op": "birth",
            "slot_id": sid,
            "A": total,
            "bin": b,
            "Q_bin": float(q[b]),
        }
    )
    storm["streak"] += 1
    COUNTS["births_applied"] += 1
    return "birth", f"slot {sid} reseeded, centroid={np.round(centroid, 4).tolist()}"


def retire_slot(slot: dict, journal: list[dict], event_id: int, sid: int, reason: str) -> None:
    slot["l"] = CFG["l_retire"]
    slot["retired"] = True
    journal.append({"event_id": event_id, "op": "retire", "slot_id": sid, "reason": reason})
    COUNTS["retires_applied"] += 1


def maybe_retire(
    slots: dict[int, dict],
    g_history: list[dict],
    null_mass_history: list[float],
    journal: list[dict],
    event_id: int,
    scene_closed: bool = False,
    cfg: dict = CFG,
) -> list[int]:
    """Spec 3.1 retire. g_history/null_mass_history are per prior fusion events."""
    if scene_closed:
        retired = []
        for sid in sorted(slots):
            if slots[sid].get("retired"):
                continue
            retire_slot(slots[sid], journal, event_id, sid, "scene_closed")
            retired.append(sid)
        return retired
    if len(g_history) < cfg["t_retire"]:
        return []
    recent_g = g_history[-cfg["t_retire"] :]
    recent_a = null_mass_history[-cfg["t_retire"] :]
    retired = []
    for sid in sorted(slots):
        if slots[sid].get("retired"):
            continue
        if all(h.get(sid, 1.0) < cfg["theta_idle"] for h in recent_g) and all(
            a < cfg["theta_birth"] for a in recent_a
        ):
            retire_slot(slots[sid], journal, event_id, sid, "t_retire_idle")
            retired.append(sid)
    return retired


def merge_slots(
    slots: dict[int, dict],
    mass_vectors: dict[int, np.ndarray],
    journal: list[dict],
    event_id: int,
    cfg: dict = CFG,
) -> list[tuple[int, int]]:
    """Spec 3.1 merge: at most one merge per event; keeps slot i, retires slot j."""
    if any(e["event_id"] == event_id and e["op"] == "merge" for e in journal):
        return []  # rate limit: one merge per event, idempotent refusal
    live = [sid for sid in sorted(slots) if not slots[sid].get("retired")]
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            i, j = live[a], live[b]
            si, sj = slots[i], slots[j]
            if not (
                np.linalg.norm(si["p"] - sj["p"]) < cfg["delta_merge"]
                and np.linalg.norm(si["v"] - sj["v"]) < cfg["delta_v"]
            ):
                continue
            mi, mj = mass_vectors.get(i), mass_vectors.get(j)
            if mi is None or mj is None:
                continue
            denom = float(np.linalg.norm(mi) * np.linalg.norm(mj))
            if denom <= 0:
                continue
            cos = float(mi @ mj / denom)
            if cos > cfg["kappa_merge"]:
                gi, gj = si["g"], sj["g"]
                w = gi + gj + 1e-12
                si["h"] = np.tanh((gi * si["h"] + gj * sj["h"]) / w)
                si["p"] = (gi * si["p"] + gj * sj["p"]) / w
                si["v"] = (gi * si["v"] + gj * sj["v"]) / w
                si["s"] = (gi * si["s"] + gj * sj["s"]) / w
                si["l"] = logit((sigmoid(si["l"]) + sigmoid(sj["l"])) / 2.0)
                retire_slot(slots[j], journal, event_id, j, "merged")
                slots[j]["merged_into"] = i
                journal.append(
                    {"event_id": event_id, "op": "merge", "keep": i, "retire": j, "cos": cos}
                )
                COUNTS["merges_applied"] += 1
                return [(i, j)]
    return []


# ------------------------------------------------------------------- ESS


def ess(omega) -> float:
    w = np.asarray(omega, float)
    return float(1.0 / np.sum(w * w))


def ess_warning_stream(values: list[float], cfg: dict = CFG) -> list[int]:
    """Fire at each event whose low-ESS streak has reached T_ess; high ESS resets."""
    fires, streak = [], 0
    for t, v in enumerate(values):
        streak = streak + 1 if v < cfg["ess_warn"] else 0
        if streak >= cfg["ess_streak"]:
            fires.append(t)
    return fires


# ------------------------------------------------------- UTF-8 FSM emission


def allowed_bytes(state, budget: int) -> dict[int, tuple | None]:
    """Strict UTF-8 FSM. state None = character boundary; (k, lo, hi) = k
    continuation bytes remain, the next one constrained to [lo, hi]."""
    out: dict[int, tuple | None] = {}
    if state is None:
        if budget >= 1:
            for b in range(0x00, 0x80):
                out[b] = None
        if budget >= 2:
            for b in range(0xC2, 0xE0):
                out[b] = (1, 0x80, 0xBF)
        if budget >= 3:
            out[0xE0] = (2, 0xA0, 0xBF)
            for b in range(0xE1, 0xED):
                out[b] = (2, 0x80, 0xBF)
            out[0xED] = (2, 0x80, 0x9F)
            for b in range(0xEE, 0xF0):
                out[b] = (2, 0x80, 0xBF)
        if budget >= 4:
            out[0xF0] = (3, 0x90, 0xBF)
            for b in range(0xF1, 0xF4):
                out[b] = (3, 0x80, 0xBF)
            out[0xF4] = (3, 0x80, 0x8F)
    else:
        k, lo, hi = state
        for b in range(lo, hi + 1):
            out[b] = (k - 1, 0x80, 0xBF) if k - 1 > 0 else None
    return out


def emit_sequence(
    logits_fn,
    k_branches: int,
    eta0: np.ndarray,
    omega_world: np.ndarray,
    rng: np.random.Generator,
    cfg: dict = CFG,
) -> dict:
    """Spec 8/11 emission loop (supplement-fixed convention): branch-level FSM
    mask then renormalize, eta mixture, sample; scratch eta updated by the
    emitted byte's branch probability; world omega untouched. At the byte cap,
    one final raw (unmasked) prediction decides terminated vs truncated."""
    eta = np.array(eta0, float)
    omega_snapshot = np.array(omega_world, float)
    state = None
    out = bytearray()
    terminated = truncated = False
    steps_log = []
    p_sums = []
    step = 0
    while len(out) < cfg["max_bytes"]:
        budget = cfg["max_bytes"] - len(out)
        logits = np.asarray(logits_fn(step, out[-1] if out else 0), float)
        allow = allowed_bytes(state, budget)
        eos_ok = state is None
        pb = np.zeros((k_branches, 257))
        for kb in range(k_branches):
            p = softmax(logits[kb])
            mask = np.zeros(257, bool)
            for b in allow:
                mask[b] = True
            if eos_ok:
                mask[256] = True
            pk = p * mask
            s = pk.sum()
            if s <= 0:
                pk = mask.astype(float)
                s = pk.sum()
            pb[kb] = pk / s
        mix = eta @ pb
        p_sums.append(abs(float(mix.sum()) - 1.0))
        r = rng.random() * float(mix.sum())
        bstar = int(np.searchsorted(np.cumsum(mix), r))
        bstar = min(bstar, 256)
        steps_log.append(
            {"pb": pb.copy(), "bstar": bstar, "eta_after": eta.copy() if bstar == 256 else None}
        )
        step += 1
        if bstar == 256:
            terminated = True
            break
        out.append(bstar)
        COUNTS["bytes_emitted"] += 1
        eta = eta * pb[:, bstar]
        eta = eta / eta.sum()
        steps_log[-1]["eta_after"] = eta.copy()
        state = allow[bstar]
    if not terminated:
        # one final EOS prediction on the raw (unmasked) mixed choice
        logits = np.asarray(logits_fn(step, out[-1] if out else 0), float)
        raw = np.stack([softmax(logits[kb]) for kb in range(k_branches)])
        mix_raw = eta @ raw
        r = rng.random() * float(mix_raw.sum())
        bstar = int(np.searchsorted(np.cumsum(mix_raw), r))
        terminated = bstar == 256
        truncated = not terminated
    COUNTS["emission_runs"] += 1
    return {
        "bytes": bytes(out),
        "terminated": terminated,
        "truncated": truncated,
        "steps_log": steps_log,
        "p_sums": p_sums,
        "omega_unchanged": bool(np.array_equal(omega_snapshot, np.asarray(omega_world, float))),
        "eta_final": eta,
    }


# ------------------------------------------------------------------- checks


def run_checks(rng: np.random.Generator) -> None:
    # A. birth -----------------------------------------------------------
    slots = fresh_slots(4, rng)
    slots[0]["g"], slots[1]["g"], slots[2]["g"], slots[3]["g"] = 0.001, 0.9, 0.8, 0.7
    before = {
        sid: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        for sid, s in slots.items()
    }
    journal: list[dict] = []
    storm = {"streak": 0}
    alpha = np.array([0.05, 0.05, 0.05, 0.05])
    xy = np.array([[0.10, 0.10], [0.15, 0.10], [0.10, 0.15], [-0.90, -0.90]])
    status, detail = birth_from_null(slots, alpha, xy, journal, 1, storm)
    untouched = all(
        all(
            (
                np.array_equal(
                    np.asarray(slots[sid][k], dtype=float), np.asarray(before[sid][k], dtype=float)
                )
                if isinstance(slots[sid][k], np.ndarray)
                else slots[sid][k] == before[sid][k]
            )
            for k in slots[sid]
        )
        for sid in (1, 2, 3)
    )
    centroid_expected = np.array([(0.10 + 0.15 + 0.10) / 3, (0.10 + 0.10 + 0.15) / 3])
    centroid_ok = np.allclose(slots[0]["p"], centroid_expected, atol=1e-12)
    record(
        "A1 birth neutrality + centroid correctness",
        status == "birth" and untouched and centroid_ok and len(journal) == 1,
        f"status={status}; slots 1-3 byte-identical={untouched}; centroid={np.round(slots[0]['p'], 6).tolist()} "
        f"expected={np.round(centroid_expected, 6).tolist()}; journal={len(journal)} entry",
    )

    # A2 zero-alpha / ordering invariance (operator reads only null mass + coords)
    slots2 = fresh_slots(4, rng)
    for sid in range(4):
        slots2[sid]["g"] = 0.001
    alpha2 = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
    xy2 = np.array([[0.10, 0.10], [0.15, 0.10], [0.10, 0.15], [0.9, 0.9], [-0.8, 0.7], [0.3, -0.9]])
    j2: list[dict] = []
    s2 = {"streak": 0}
    birth_from_null(slots2, alpha2, xy2, j2, 1, s2)
    c_first = slots2[0]["p"].copy()
    perm = [3, 4, 5, 0, 2, 1]  # permute zero-alpha tokens and reorder the rest
    birth_from_null(slots2, alpha2[perm], xy2[perm], j2, 2, s2)
    record(
        "A2 birth depends only on null mass and coordinates",
        np.allclose(c_first, slots2[0]["p"], atol=0) and np.array_equal(c_first, slots2[0]["p"]),
        f"centroid invariant under zero-alpha token perturbation and reordering: {np.array_equal(c_first, slots2[0]['p'])}",
    )

    # A3 idempotence / audit
    n_journal = len(journal)
    status, detail = birth_from_null(slots, alpha, xy, journal, 1, storm)
    record(
        "A3 birth idempotent refusal on same event_id",
        status == "already_applied" and len(journal) == n_journal,
        f"repeat call status={status}; journal unchanged at {n_journal}",
    )

    # A4 storm gate + reset
    slots3 = fresh_slots(4, rng)
    for sid in range(4):
        slots3[sid]["g"] = 0.001
    j3: list[dict] = []
    s3 = {"streak": 0}
    statuses = []
    for ev in range(1, 7):
        statuses.append(birth_from_null(slots3, alpha, xy, j3, ev, s3)[0])
    birth_from_null(
        slots3, np.array([0.01, 0.01, 0.01, 0.01]), xy, j3, 7, s3
    )  # below theta -> reset
    status_after_reset = birth_from_null(slots3, alpha, xy, j3, 8, s3)[0]
    record(
        "A4 birth storm gate (e_storm=5) with reset on quiet event",
        statuses == ["birth"] * 5 + ["churn_stop"] and status_after_reset == "birth",
        f"6 consecutive triggers -> {statuses}; after quiet event, next trigger -> {status_after_reset}",
    )

    # A5 retired slot exits binding competition (log(eps+rho) term)
    eps = 1e-12
    rho_ret = sigmoid(CFG["l_retire"])
    scores = np.array([math.log(eps + 0.5), math.log(eps + rho_ret), 0.0])
    masses = np.exp(scores - scores.max())
    masses /= masses.sum()
    record(
        "A5 retired slot binding mass < 1e-3 vs active rho=0.5",
        masses[1] < 1e-3,
        f"rho_retired={rho_ret:.3e}; softmax mass={masses.tolist()}; log-rho gap={math.log(eps+0.5)-math.log(eps+rho_ret):.3f} nats",
    )

    # B. retire ----------------------------------------------------------
    slots4 = fresh_slots(3, rng)
    for sid in range(3):
        slots4[sid]["g"] = 0.5
    before4 = {
        sid: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        for sid, s in slots4.items()
    }
    evidence = [{"raw": "evidence-bytes-keep"}]
    j4: list[dict] = []
    g_hist = [{0: 0.001, 1: 0.6, 2: 0.7}] * CFG["t_retire"]
    a_hist = [0.01] * CFG["t_retire"]
    retired = maybe_retire(slots4, g_hist, a_hist, j4, 100)
    slot0 = slots4[0]
    preserved = (
        np.array_equal(slot0["h"], before4[0]["h"])
        and np.array_equal(slot0["p"], before4[0]["p"])
        and np.array_equal(slot0["v"], before4[0]["v"])
        and slot0["s"] == before4[0]["s"]
    )
    record(
        "B1 retire keeps archived p/h/v/s, evidence untouched, journal appended",
        retired == [0]
        and preserved
        and slot0["l"] == CFG["l_retire"]
        and evidence == [{"raw": "evidence-bytes-keep"}]
        and len(j4) == 1,
        f"retired={retired}; archived fields preserved={preserved}; l={slot0['l']}; evidence unchanged",
    )

    slots5 = fresh_slots(3, rng)
    j5: list[dict] = []
    retired_scene = maybe_retire(slots5, [], [], j5, 200, scene_closed=True)
    record(
        "B2 scene close retires every slot with per-slot journal entries",
        retired_scene == [0, 1, 2]
        and len(j5) == 3
        and all(e["reason"] == "scene_closed" for e in j5),
        f"retired={retired_scene}; journal entries={len(j5)}",
    )

    # C. merge -----------------------------------------------------------
    slots6 = fresh_slots(3, rng)
    slots6[0].update(
        {
            "h": np.full(4, 0.5),
            "p": np.array([0.0, 0.0]),
            "v": np.zeros(2),
            "s": 0.1,
            "l": 0.0,
            "g": 0.5,
        }
    )
    slots6[1].update(
        {
            "h": np.full(4, -0.5),
            "p": np.array([0.02, 0.0]),
            "v": np.zeros(2),
            "s": 0.2,
            "l": 0.0,
            "g": 0.3,
        }
    )
    slots6[2].update({"p": np.array([0.9, 0.9]), "g": 0.6})
    before6 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in slots6[2].items()}
    j6: list[dict] = []
    mv = {0: np.array([1.0, 1.0, 0.0]), 1: np.array([1.0, 1.0, 0.0]), 2: np.array([0.0, 0.0, 1.0])}
    merged = merge_slots(slots6, mv, j6, 300)
    h_expected = np.tanh((0.5 * 0.5 + 0.3 * -0.5) / (0.8 + 1e-12))
    p_expected = (0.5 * np.array([0.0, 0.0]) + 0.3 * np.array([0.02, 0.0])) / 0.8
    s_expected = (0.5 * 0.1 + 0.3 * 0.2) / 0.8
    l_expected = logit(0.5)
    merge_ok = (
        merged == [(0, 1)]
        and np.allclose(slots6[0]["h"], h_expected, atol=1e-12)
        and np.allclose(slots6[0]["p"], p_expected, atol=1e-12)
        and abs(slots6[0]["s"] - s_expected) < 1e-12
        and abs(slots6[0]["l"] - l_expected) < 1e-12
        and slots6[1].get("retired")
        and slots6[1].get("merged_into") == 0
    )
    third_ok = all(
        (
            np.array_equal(
                np.asarray(slots6[2][k], dtype=float), np.asarray(before6[k], dtype=float)
            )
            if isinstance(slots6[2][k], np.ndarray)
            else slots6[2][k] == before6[k]
        )
        for k in before6
    )
    record(
        "C1 merge g-weighted fields + third-slot neutrality",
        merge_ok and third_ok,
        f"merged={merged}; h[0]={np.round(slots6[0]['h'], 6).tolist()} (expected tanh(0.125)={h_expected:.6f}); "
        f"p[0]={np.round(slots6[0]['p'], 6).tolist()}; s[0]={slots6[0]['s']:.6f}; slot2 untouched={third_ok}",
    )

    slots7 = fresh_slots(4, rng)
    slots7[0].update({"p": np.array([0.0, 0.0]), "v": np.zeros(2), "g": 0.5})
    slots7[1].update({"p": np.array([0.01, 0.0]), "v": np.zeros(2), "g": 0.4})
    slots7[2].update({"p": np.array([0.5, 0.0]), "v": np.zeros(2), "g": 0.5})
    slots7[3].update({"p": np.array([0.51, 0.0]), "v": np.zeros(2), "g": 0.4})
    j7: list[dict] = []
    mv7 = {i: np.array([1.0, 1.0]) for i in range(4)}
    merges = merge_slots(slots7, mv7, j7, 400)
    merges2 = merge_slots(slots7, mv7, j7, 400)
    record(
        "C2 merge rate limit: at most one per event",
        len(merges) == 1 and merges2 == [],
        f"first call merges={merges}; second call same event merges={merges2}",
    )

    # D. ESS -------------------------------------------------------------
    e_uniform, e_onehot, e_mix = (
        ess([1 / 3, 1 / 3, 1 / 3]),
        ess([1.0, 0.0, 0.0]),
        ess([0.5, 0.25, 0.25]),
    )
    record(
        "D1 ESS known values",
        abs(e_uniform - 3.0) < 1e-12 and abs(e_onehot - 1.0) < 1e-12 and abs(e_mix - 8 / 3) < 1e-12,
        f"uniform->3: {e_uniform:.12f}; one-hot->1: {e_onehot:.12f}; (0.5,0.25,0.25)->8/3: {e_mix:.12f}",
    )
    stream = [3.0] * 5 + [1.0] * 10 + [3.0] + [1.0] * 12
    fires = ess_warning_stream(stream)
    record(
        "D2 collapse warning fires at T_ess streak and resets on recovery",
        fires[0] == 14 and fires == [14, 25, 26, 27],
        f"stream of 28 events -> fires at t={fires} (first burst: 10th consecutive low at t=14; high at t=15 resets; second burst reaches streak 10 at t=25 and reports while it holds)",
    )

    # E. emission loop ---------------------------------------------------
    # E1/E2/E3: 200 seeded random runs
    k_br = 3
    eta0 = np.array([0.5, 0.3, 0.2])
    omega_world = np.array([0.4, 0.35, 0.25])
    all_decode, all_omega, worst_psum, worst_eta = True, True, 0.0, 0.0
    n_term = n_trunc = 0
    for run in range(200):
        seed = 20261106 * 1000 + run

        def logits_fn(t, prev, _seed=seed):
            r = np.random.default_rng((_seed, t, int(prev)))
            return r.normal(size=(k_br, 257))

        res = emit_sequence(logits_fn, k_br, eta0, omega_world, np.random.default_rng(seed))
        try:
            res["bytes"].decode("utf-8")
            dec = True
        except UnicodeDecodeError:
            dec = False
        all_decode &= dec
        all_omega &= res["omega_unchanged"]
        worst_psum = max(worst_psum, max(res["p_sums"]))
        # manual eta recomputation from the recorded branch probs and emitted bytes
        eta_manual = eta0.copy()
        for st in res["steps_log"]:
            if st["bstar"] == 256:
                break
            eta_manual = eta_manual * st["pb"][:, st["bstar"]]
            eta_manual = eta_manual / eta_manual.sum()
            worst_eta = max(worst_eta, float(np.abs(eta_manual - st["eta_after"]).max()))
        if res["terminated"]:
            n_term += 1
        else:
            n_trunc += 1
            all_decode &= len(res["bytes"]) == CFG["max_bytes"]
    record(
        "E1 200 random emissions: strict UTF-8 legality incl. truncated at char boundary",
        all_decode,
        f"terminated={n_term}, truncated={n_trunc}; every output decodes; truncated outputs are exactly 64 bytes at character boundaries",
    )
    record(
        "E2 world omega bit-identical and scratch eta matches manual update",
        all_omega and worst_eta == 0.0,
        f"omega unchanged in 200/200 runs; max|eta_manual - eta_recorded| = {worst_eta:.1e}",
    )
    record(
        "E3 masked-renormalized eta mixture sums to 1 at every step",
        worst_psum < 1e-12,
        f"max|sum(P)-1| over all steps of 200 runs = {worst_psum:.2e}",
    )

    # E4 budget mask excludes over-long starters
    a1 = allowed_bytes(None, 1)
    a2 = allowed_bytes(None, 2)
    a3 = allowed_bytes(None, 3)
    ok4 = (
        0x41 in a1
        and all(b not in a1 for b in range(0xC2, 0x100))
        and 0xC3 in a2
        and all(b not in a2 for b in range(0xE0, 0x100))
        and 0xE6 in a3
        and 0xF0 not in a3
        and 0xF0 in allowed_bytes(None, 4)
        and all(b not in a3 for b in list(range(0x80, 0xC2)) + [0xF5, 0xFF])
    )
    # a 3-byte starter at remaining=3 must complete: transitions stay legal at budget 2,1
    st1 = a3[0xE6]
    st2 = allowed_bytes(st1, 2)[0x80]
    st3 = allowed_bytes(st2, 1)[0x80]
    ok4 &= st3 is None
    record(
        "E4 FSM budget mask blocks starters that cannot complete; 3-byte char at remaining=3 completes",
        ok4,
        f"remaining=1: ASCII only; =2: +2-byte starters; =3: +3/4-byte starters; continuation chain returns to boundary: {st1}->{st2}->{st3}",
    )

    # E5 truncated vs terminated semantics
    def never_eos(t, prev):
        r = np.random.default_rng((1, t))
        lg = r.normal(size=(k_br, 257))
        lg[:, 256] = -100.0
        return lg

    res_trunc = emit_sequence(never_eos, k_br, eta0, omega_world, np.random.default_rng(7))
    ok_trunc = (
        res_trunc["truncated"] and not res_trunc["terminated"] and len(res_trunc["bytes"]) == 64
    )

    def eos_at_4(t, prev):
        r = np.random.default_rng((2, t))
        lg = r.normal(size=(k_br, 257)) * 0.01
        if t < 4:
            lg[:, 0x41] = 50.0  # force ASCII 'A' so step 4 sits at a char boundary
            lg[:, 256] = -100.0
        else:
            lg[:, 256] = 100.0
        return lg

    res_term = emit_sequence(eos_at_4, k_br, eta0, omega_world, np.random.default_rng(8))
    ok_term = res_term["terminated"] and not res_term["truncated"] and len(res_term["bytes"]) == 4
    record(
        "E5 final prediction does not fake voluntary stop",
        ok_trunc and ok_term,
        f"never-EOS logits -> truncated={res_trunc['truncated']}, 64 bytes, decodes; EOS-preferred at step 4 -> terminated={res_term['terminated']}, {len(res_term['bytes'])} bytes",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=None, help="write JSON report to a NEW file (must not exist)"
    )
    args = ap.parse_args()
    t0 = time.process_time()
    rng = np.random.default_rng(20261106)
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
        "schema": "taiji-mgw1-lifecycle-supplement-precheck-v1",
        "date": "2026-11-06",
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
            "Algebra pre-checks for the MGW1 2026-11-06 supplement only (slot lifecycle "
            "birth/retire/merge, ESS diagnostics, 64-byte UTF-8 FSM emission loop with "
            "scratch-mixture convention). Synthetic values; no production graph, weights, "
            "dataset or sealed face. Does not sign production engineering, object discovery, "
            "physical capability or task performance."
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
