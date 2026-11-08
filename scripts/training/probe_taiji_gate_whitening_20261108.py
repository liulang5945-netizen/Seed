"""Shrinkage whitening as the only variable: the last cut at the key-geometry layer.

The centering retest showed that subtracting the mean repairs the key collapse
(effective rank 2.557 -> 12.148) but destroys the carrier that lets a synonym
wording land on its own written key (floor 0.9648 -> 0.4494).  This run asks one
question: is there a *linear* read-time transform that repairs the collapse while
keeping the carrier?  Whitening keeps the mean (it is not subtracted) and only
re-weights directions by their variance, so in principle it could fix the
dominance of the shared offset without deleting it.

Three arms, one variable (the read-time transform applied to *both* the stored
keys and the query key; the store itself always keeps raw keys):

* ``A_raw``      -- today's read (identity).
* ``B_whiten``   -- ridge whitening with ``lambda = tr(Sigma)/d`` (the mean
  eigenvalue, a unit-information ridge fixed before any reading; no value is
  searched).  ``Sigma`` is the covariance of the occupied store keys about their
  mean, ``Sigma_s = Sigma + lambda I``, ``W = Sigma_s^{-1/2}``, and the transform
  is ``x -> W x`` on both sides.  The mean is *kept* in the keys.
* ``C_center``   -- centering, measured in the previous run; its numbers are
  copied from ``reports/taiji_gate_centering_20261108.json`` rather than
  recomputed (declared in the manifest).

Nothing is trained or optimised.  The only model operation is the frozen forward
used to build the store and the read path.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.living_core import (  # noqa: E402
    ADDRESSED_ALPHA,
    ADDRESSED_TEMPERATURE,
    LivingCoreConfig,
    LivingCoreReference,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

BASE_RUNNER = ROOT / "scripts/training/probe_taiji_gate_read_gate_20261108.py"
CENTERING_REPORT = ROOT / "reports/taiji_gate_centering_20261108.json"
LIVING_CORE = ROOT / "taiji/living_core.py"
REPORT = ROOT / "reports/taiji_gate_whitening_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_whitening_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_whitening_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_gate_whitening_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_whitening_20261108_watchdog.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
CLASSES = ("written", "unwritten", "new_token", "new_pairing", "no_value")
COLLAPSE_RANK_FACTOR = 2.0
COLLAPSE_TOP_SHARE = 0.75
CARRIER_FLOOR = 0.90
RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

CRITERIA = {
    "C1_control_identity": (
        "with the raw transform the runner's own read reproduces the real addressed_read on all "
        "154 queries: identical fast sha, top slot and max|delta| = 0"
    ),
    "C2_store_immutable": (
        "whitening is read-time only: store_keys/store_values/store_age hashes are identical "
        "before and after every whitening pass"
    ),
    "C3_transform_scope": (
        "the same W is applied to the occupied stored keys and to the query key; the store keeps "
        "the raw keys and the encoder is untouched"
    ),
    "C4_shrinkage_rule": (
        "the ridge is lambda = trace(Sigma)/d with Sigma the covariance of the occupied store keys "
        "about their mean; the value is fixed by this rule before any reading and is not searched"
    ),
    "C5_threshold_source": (
        "the gate threshold is the minimum top cosine of the written group in that same arm; no "
        "negative label enters the value"
    ),
    "C6_collapse_fixed": (
        "collapse_fixed = whitened effective rank >= 2x the raw effective rank, or whitened "
        "top_share <= 0.75 (thresholds fixed before the reading)"
    ),
    "C7_carrier_kept": "carrier_kept = written-group minimum top cosine >= 0.90",
    "V1_written_bytes": (
        "if whitening changes the returned fast sha of any of the 22 written faces against arm A, "
        "the verdict is whitening_changes_written_bytes (hard failure, dominant rule)"
    ),
    "V2_enables": (
        "otherwise, if at least two gaps are > 0 and the unwritten correct count does not drop, "
        "the verdict is whitening_enables_gate with the threshold interval (lower, upper]"
    ),
    "V3_ceiling": (
        "otherwise, if whitening either fixes the collapse or keeps the carrier but not both, the "
        "verdict is encoding_is_the_ceiling: the averaged-pooling byte encoding ties the two "
        "together, so neither the gate nor the offset nor another linear form can separate them"
    ),
    "unchanged": "no fixture, face, position, class, ridge or anchor is retuned after first reading",
}


# ---------------------------------------------------------------------------
# Discipline helpers.
# ---------------------------------------------------------------------------
def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with open(tmp, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    record = {**record, "at": round(time.time() - START_WALL, 3)}
    with open(PROGRESS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    LAST_READING[0] = time.time()


def touch() -> None:
    LAST_READING[0] = time.time()


def acquire_single_instance() -> str:
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            raise SystemExit(f"another instance holds {LOCK} ({age:.1f}s old)") from None
        LOCK.unlink()
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:  # pragma: no cover - surfaced as a hard stop
        if exc.errno != errno.EEXIST:
            raise
        raise SystemExit(f"cannot acquire {LOCK}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started={time.time():.3f}\n")
    return LOCK.name


def release_single_instance() -> None:
    if LOCK.exists():
        LOCK.unlink()


def start_watchdog(stop: threading.Event) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(2.0):
            idle = time.time() - LAST_READING[0]
            if idle > WATCHDOG_SECONDS:
                WATCHDOG_FIRED[0] = True
                write_json(
                    WATCHDOG_MARKER,
                    {"fired": True, "idle_seconds": round(idle, 3), "at": time.time()},
                )
                os._exit(3)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def load_base():
    spec = importlib.util.spec_from_file_location("read_gate_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Read-time transforms.  The store always keeps raw keys; only the comparison
# space changes.
# ---------------------------------------------------------------------------
def occupied(state) -> torch.Tensor:
    return torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()


def store_center(state) -> torch.Tensor:
    return state.store_keys[occupied(state)].mean(dim=0)


def fit_transform(state, mode: str) -> dict[str, Any]:
    valid = occupied(state)
    keys = state.store_keys[valid].detach().double()
    if mode == "raw":
        return {"mode": "raw"}
    center = keys.mean(dim=0)
    if mode == "center":
        return {"mode": "center", "center": center}
    if mode != "whiten":
        raise ValueError(f"unknown transform {mode}")
    centered = keys - center
    count = max(int(centered.shape[0]), 1)
    covariance = (centered.T @ centered) / count
    ridge = float(torch.diagonal(covariance).sum() / covariance.shape[0])
    shrunk = covariance + ridge * torch.eye(covariance.shape[0], dtype=covariance.dtype)
    values, vectors = torch.linalg.eigh(shrunk)
    values = values.clamp(min=1e-18)
    matrix = (vectors * values.rsqrt()) @ vectors.T
    spectrum = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
    return {
        "mode": "whiten",
        "center": center,
        "matrix": matrix,
        "ridge": ridge,
        "shrinkage": {
            "sample_count": count,
            "dimension": int(covariance.shape[0]),
            "trace_over_d": round(ridge, 12),
            "eigen_min": round(float(spectrum[0]), 12),
            "eigen_median": round(float(spectrum.median()), 12),
            "eigen_max": round(float(spectrum[-1]), 12),
            "eigen_sum": round(float(spectrum.sum()), 12),
            "scale_ratio_max_over_min": round(
                float(((spectrum[-1] + ridge) / max(spectrum[0] + ridge, 1e-30)) ** 0.5), 6
            ),
        },
    }


def apply_transform(keys: torch.Tensor, query: torch.Tensor, transform: dict[str, Any]):
    mode = transform["mode"]
    if mode == "raw":
        return keys, query
    if mode == "center":
        shift = transform["center"]
        return keys - shift, query - shift
    matrix = transform["matrix"]
    moved_keys = (matrix @ keys.detach().double().T).T
    moved_query = matrix @ query.detach().double()
    return moved_keys, moved_query


def cosine_scores(keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    return (keys @ query) / torch.clamp(keys.norm(dim=-1) * query.norm(), min=1e-12)


def scores(state, key: torch.Tensor, transform: dict[str, Any]) -> dict[str, Any]:
    valid = occupied(state)
    moved_keys, moved_query = apply_transform(state.store_keys[valid], key, transform)
    cosine = cosine_scores(moved_keys, moved_query)
    ordered = torch.sort(cosine.detach(), descending=True).values
    top = int(torch.argmax(cosine.detach()))
    return {
        "top_cosine": float(cosine[top].detach()),
        "second_cosine": float(ordered[1].detach()) if ordered.numel() > 1 else None,
        "margin": float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None,
        "top_slot": int(valid[top]),
        "weights": torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0),
    }


def read_fast(state, key: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    valid = occupied(state)
    moved_keys, moved_query = apply_transform(state.store_keys[valid], key, transform)
    cosine = cosine_scores(moved_keys, moved_query)
    weights = torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0).to(state.store_values.dtype)
    retrieved = torch.einsum("i,i...->...", weights, state.store_values[valid])
    return (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def store_signature(state) -> dict[str, Any]:
    return {
        "keys": tensor_sha(state.store_keys),
        "values": tensor_sha(state.store_values),
        "age": tensor_sha(state.store_age.float()),
        "count": int(state.store_count),
    }


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------
def describe(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 9),
        "p25": round(float(np.percentile(arr, 25)), 9),
        "median": round(float(np.median(arr)), 9),
        "p75": round(float(np.percentile(arr, 75)), 9),
        "max": round(float(arr.max()), 9),
    }


def item_key(subject, state, item: dict[str, Any]) -> torch.Tensor:
    key = item.get("key")
    if key is None:
        return subject.store_query_key(state, item.get("raw", b""))
    if isinstance(key, np.ndarray):
        return torch.tensor(key, dtype=subject.W.dtype, device=subject.W.device)
    return key.to(dtype=subject.W.dtype, device=subject.W.device)


def collect(subject, state, items, transform) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        record = scores(state, item_key(subject, state, item), transform)
        record.pop("weights", None)
        record.update({"class": item["class"], "tag": item["tag"]})
        if "face_tag" in item:
            record["face_tag"] = item["face_tag"]
        rows.append(record)
        touch()
    return rows


def group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    floor = min(row["top_cosine"] for row in records if row["class"] == "written")
    groups: dict[str, Any] = {}
    for name in CLASSES:
        tops = [row["top_cosine"] for row in records if row["class"] == name]
        margins = [row["margin"] for row in records if row["class"] == name]
        groups[name] = {
            "count": len(tops),
            "top_cosine": describe(tops),
            "margin": describe(margins),
        }
    gaps = {name: floor - groups[name]["top_cosine"]["max"] for name in CLASSES[2:]}
    return {
        "written_floor": float(floor),
        "written_floor_reported": round(float(floor), 9),
        "groups": groups,
        "gaps": {name: round(float(value), 9) for name, value in gaps.items()},
        "separable": [name for name, value in gaps.items() if value > 0],
    }


def written_baseline(subject, state, items) -> dict[str, Any]:
    out = {}
    for item in items:
        if item["class"] != "written":
            continue
        key = item_key(subject, state, item)
        after, info = subject.addressed_read(state, item.get("raw", b""), key=key)
        out[item["tag"]] = {"fast_sha": tensor_sha(after.fast), "top_slot": int(info["top_slot"])}
        touch()
    return out


def control_check(subject, state, items, raw_rows) -> dict[str, Any]:
    same_fast = 0
    same_slot = 0
    max_delta = 0.0
    for item, own in zip(items, raw_rows, strict=True):
        key = item_key(subject, state, item)
        after, _ = subject.addressed_read(state, item.get("raw", b""), key=key)
        fast = read_fast(state, key, {"mode": "raw"})
        max_delta = max(max_delta, float((after.fast - fast).abs().max()))
        if tensor_sha(after.fast) == tensor_sha(fast):
            same_fast += 1
        if int(own["top_slot"]) == int(scores(state, key, {"mode": "raw"})["top_slot"]):
            same_slot += 1
    return {
        "queries": len(items),
        "fast_sha_identical": same_fast,
        "top_slot_identical": same_slot,
        "max_abs_delta": max_delta,
    }


def unwritten_correct(records: list[dict[str, Any]]) -> dict[str, Any]:
    face_slot = {row["tag"]: row["top_slot"] for row in records if row["class"] == "written"}
    judged = [row for row in records if row["class"] == "unwritten"]
    hits = [row for row in judged if row["top_slot"] == face_slot.get(row.get("face_tag"))]
    return {
        "count": len(judged),
        "correct": len(hits),
        "correct_rate": round(len(hits) / max(len(judged), 1), 6),
        "misses": [row["tag"] for row in judged if row not in hits],
    }


def gate_pass(subject, state, items, transform, threshold: float, baseline) -> dict[str, Any]:
    matched = dict.fromkeys(CLASSES, 0)
    totals = dict.fromkeys(CLASSES, 0)
    written_bytes_same = 0
    written_slot_same = 0
    changed = []
    per_item = []
    for item in items:
        key = item_key(subject, state, item)
        record = scores(state, key, transform)
        record.pop("weights", None)
        hit = bool(record["top_cosine"] >= threshold)
        totals[item["class"]] += 1
        if hit:
            matched[item["class"]] += 1
        if item["class"] == "written":
            fast = read_fast(state, key, transform)
            if tensor_sha(fast) == baseline[item["tag"]]["fast_sha"]:
                written_bytes_same += 1
            else:
                changed.append(item["tag"])
            if record["top_slot"] == baseline[item["tag"]]["top_slot"]:
                written_slot_same += 1
        per_item.append({"class": item["class"], "tag": item["tag"], **record})
        touch()
    return {
        "threshold": round(float(threshold), 9),
        "groups": {
            name: {
                "matched": matched[name],
                "total": totals[name],
                "refused": totals[name] - matched[name],
                "refusal_rate": round((totals[name] - matched[name]) / max(totals[name], 1), 6),
            }
            for name in CLASSES
        },
        "written_bytes_identical_to_arm_a": written_bytes_same,
        "written_slot_identical_to_arm_a": written_slot_same,
        "written_faces_changed": changed,
        "unwritten_false_refusals": totals["unwritten"] - matched["unwritten"],
        "per_item": per_item,
    }


def key_spectrum(keys: torch.Tensor) -> dict[str, Any]:
    matrix = keys.detach().double()
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum())
    return {
        "count": int(matrix.shape[0]),
        "rank": int(torch.linalg.matrix_rank(matrix)),
        "effective_rank": round(float(singular.sum() ** 2 / max(total, 1e-30)), 6),
        "top_share": round(float(energy[0] / max(total, 1e-30)), 6),
    }


def transform_spectrum(state, transform) -> dict[str, Any]:
    valid = occupied(state)
    keys = state.store_keys[valid]
    if transform["mode"] == "raw":
        return key_spectrum(keys)
    moved, _ = apply_transform(keys, keys[0], transform)
    return key_spectrum(moved)


def verdict_block(stats, gate, unwritten, diagnostics) -> dict[str, Any]:
    gaps = stats["gaps"]
    positive = [name for name, value in gaps.items() if value > 0]
    bytes_same = gate["written_bytes_identical_to_arm_a"]
    total = gate["groups"]["written"]["total"]
    collapse_fixed = bool(
        diagnostics["whitened"]["effective_rank"]
        >= COLLAPSE_RANK_FACTOR * diagnostics["raw"]["effective_rank"]
        or diagnostics["whitened"]["top_share"] <= COLLAPSE_TOP_SHARE
    )
    carrier_kept = bool(stats["written_floor"] >= CARRIER_FLOOR)
    trade_off = collapse_fixed != carrier_kept
    substantive = "whitening_inconclusive"
    if (
        bytes_same >= total
        and len(positive) >= 2
        and unwritten["correct"] >= unwritten["raw_correct"]
    ):
        substantive = "whitening_enables_gate"
    elif trade_off:
        substantive = "encoding_is_the_ceiling"
    block = {
        "written_bytes_identical": bytes_same,
        "written_total": total,
        "written_slots_identical": gate["written_slot_identical_to_arm_a"],
        "positive_gaps": positive,
        "collapse_fixed": collapse_fixed,
        "carrier_kept": carrier_kept,
        "trade_off": trade_off,
        "substantive_verdict": substantive,
        "unwritten_correct": unwritten["correct"],
        "unwritten_raw_correct": unwritten["raw_correct"],
    }
    if bytes_same < total:
        block["verdict"] = "whitening_changes_written_bytes"
        block["reason"] = "whitening altered the returned fast bytes of written faces (rule V1)"
        return block
    if substantive == "whitening_enables_gate":
        lower = max(stats["groups"][name]["top_cosine"]["max"] for name in positive)
        unwritten_tops = sorted(
            row["top_cosine"] for row in gate["per_item"] if row["class"] == "unwritten"
        )
        upper = min(stats["written_floor"], unwritten_tops[1] if len(unwritten_tops) > 1 else 0.0)
        block["verdict"] = "whitening_enables_gate"
        block["reason"] = "written bytes unchanged, at least two gaps positive, unwritten not worse"
        block["threshold_interval"] = {
            "lower_open": round(float(lower), 9),
            "upper_inclusive": round(float(upper), 9),
            "width": round(float(upper - lower), 9),
        }
        return block
    if substantive == "encoding_is_the_ceiling":
        block["verdict"] = "encoding_is_the_ceiling"
        block["reason"] = (
            "whitening repairs the collapse or keeps the carrier, not both: the averaged-pooling "
            "byte encoding ties the two together"
        )
        return block
    block["verdict"] = "whitening_inconclusive"
    block["reason"] = "fewer than two positive gaps and no collapse/carrier trade-off detected"
    return block


def arm_block(subject, state, arm: str, items) -> dict[str, Any]:
    before = store_signature(state)
    raw = collect(subject, state, items, {"mode": "raw"})
    control = control_check(subject, state, items, raw)
    baseline = written_baseline(subject, state, items)
    whiten = fit_transform(state, "whiten")
    moved = collect(subject, state, items, whiten)
    stats_raw = group_stats(raw)
    stats_moved = group_stats(moved)
    gate_raw = gate_pass(
        subject, state, items, {"mode": "raw"}, stats_raw["written_floor"], baseline
    )
    gate_moved = gate_pass(subject, state, items, whiten, stats_moved["written_floor"], baseline)
    after = store_signature(state)
    diagnostics = {
        "raw": transform_spectrum(state, {"mode": "raw"}),
        "whitened": transform_spectrum(state, whiten),
        "shrinkage": whiten["shrinkage"],
    }
    unwritten_moved = unwritten_correct(moved)
    unwritten_moved["raw_correct"] = unwritten_correct(raw)["correct"]
    verdict = verdict_block(stats_moved, gate_moved, unwritten_moved, diagnostics)
    append_progress(
        {
            "phase": f"arm_{arm}",
            "control_fast_identical": control["fast_sha_identical"],
            "written_bytes_same": gate_moved["written_bytes_identical_to_arm_a"],
            "verdict": verdict["verdict"],
        }
    )
    return {
        "arm": arm,
        "transform": {
            "mode": "ridge_whitening",
            "ridge_rule": "lambda = trace(Sigma)/d",
            "ridge": round(float(whiten["ridge"]), 12),
            "applied_to": "occupied stored keys and query key, at read time",
            "store_keys": "raw (untransformed)",
        },
        "control": control,
        "store_immutable": before == after,
        "store_before": before,
        "store_after": after,
        "diagnostics": diagnostics,
        "arm_a": {
            "floor": stats_raw["written_floor_reported"],
            "gaps": stats_raw["gaps"],
            "separable": stats_raw["separable"],
            "m_spec_written": describe([row["margin"] for row in raw if row["class"] == "written"]),
            "m_spec_all": describe([row["margin"] for row in raw]),
            "group_top": {name: stats_raw["groups"][name]["top_cosine"] for name in CLASSES},
            "unwritten": unwritten_correct(raw),
            "gate": {key: gate_raw[key] for key in gate_raw if key != "per_item"},
        },
        "arm_b": {
            "floor": stats_moved["written_floor_reported"],
            "gaps": stats_moved["gaps"],
            "separable": stats_moved["separable"],
            "m_spec_written": describe(
                [row["margin"] for row in moved if row["class"] == "written"]
            ),
            "m_spec_all": describe([row["margin"] for row in moved]),
            "group_top": {name: stats_moved["groups"][name]["top_cosine"] for name in CLASSES},
            "unwritten": unwritten_moved,
            "gate": {key: gate_moved[key] for key in gate_moved if key != "per_item"},
        },
        "per_item_gate_b": [
            {
                "class": row["class"],
                "tag": row["tag"],
                "top_cosine": row["top_cosine"],
                "margin": row["margin"],
                "top_slot": row["top_slot"],
            }
            for row in gate_moved["per_item"]
        ],
        "verdict": verdict,
    }


def reused_centering() -> dict[str, Any]:
    if not CENTERING_REPORT.exists():
        return {"reused": False, "reason": "centering report missing"}
    source = json.loads(CENTERING_REPORT.read_text(encoding="utf-8"))
    out = {"reused": True, "source": str(CENTERING_REPORT.relative_to(ROOT))}
    for arm in ("real_forward", "fixture_continuity"):
        block = source["arms"][arm]
        out[arm] = {
            "floor_arm_a": block["arm_a"]["floor"],
            "floor_arm_b_centered": block["arm_b"]["floor"],
            "gaps_arm_a": block["arm_a"]["gaps"],
            "gaps_arm_b_centered": block["arm_b"]["gaps"],
            "m_spec_written_arm_a": block["arm_a"]["written_margin"]["median"],
            "m_spec_written_arm_b": block["arm_b"]["written_margin"]["median"],
            "written_bytes_same_arm_b": block["arm_b"]["gate"]["written_bytes_identical_to_arm_a"],
            "verdict": block["verdict"]["verdict"],
        }
    return out


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("whitening probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    PROGRESS.write_text("", encoding="utf-8")
    if WATCHDOG_MARKER.exists():
        WATCHDOG_MARKER.unlink()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261108)
    base = load_base()
    base.build_fixture()

    manifest = {
        "format": "taiji-whitening-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "does shrinkage whitening repair the key collapse while keeping the carrier that "
            "lets a synonym wording reach its own written key?"
        ),
        "written_before_any_reading": True,
        "sequence": [
            "build the real-forward store and the continuity store",
            "arm A: identity read, gate anchored on the written floor",
            "arm B: ridge whitening lambda = trace(Sigma)/d applied to keys and query at read time",
            "arm C: centering, reused from the previous report (not recomputed)",
            "verdict per the pre-written V1..V3 ordering (V1 dominant)",
        ],
        "criteria": CRITERIA,
        "constants": {
            "faces": base.FACE_COUNT,
            "base_occupied": base.BASE_OCCUPIED,
            "slots": base.SLOTS,
            "negative_per_class": base.NEG_PER_CLASS,
            "unwritten_per_rule": base.UNWRITTEN_PER_RULE,
            "temperature": ADDRESSED_TEMPERATURE,
            "alpha": ADDRESSED_ALPHA,
            "collapse_rank_factor": COLLAPSE_RANK_FACTOR,
            "collapse_top_share": COLLAPSE_TOP_SHARE,
            "carrier_floor": CARRIER_FLOOR,
            "query_count": base.FACE_COUNT + 3 * base.FACE_COUNT + 3 * base.NEG_PER_CLASS,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "black_invoked": False,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
        },
    }
    write_json(MANIFEST, manifest)
    append_progress({"phase": "manifest_written", "sha256": file_sha(MANIFEST)})

    real_subject = LivingCoreReference(LivingCoreConfig())
    real_state, real_rows = base.build_real_store(real_subject)
    real_unwritten = base.real_unwritten(real_rows)
    real_negatives = base.real_negatives(real_subject, real_state, real_rows)
    append_progress(
        {
            "phase": "real_store_built",
            "occupied": int(occupied(real_state).numel()),
            "faces": len(real_rows),
        }
    )

    continuity_subject = LivingCoreReference(LivingCoreConfig()).double()
    fixture_rows = base.fixture_faces()
    fixture_unwritten = base.fixture_unwritten(fixture_rows)
    fixture_negatives = base.fixture_negatives(fixture_rows)
    continuity_state = base.continuity_state(continuity_subject, fixture_rows)
    append_progress({"phase": "continuity_store_built", "faces": len(fixture_rows)})

    arms = {}
    for arm, subject, state, rows, unwritten, negatives in (
        ("real_forward", real_subject, real_state, real_rows, real_unwritten, real_negatives),
        (
            "fixture_continuity",
            continuity_subject,
            continuity_state,
            fixture_rows,
            fixture_unwritten,
            fixture_negatives,
        ),
    ):
        items = base.group_items(arm, rows, unwritten, negatives)
        block = arm_block(subject, state, arm, items)
        block["query_count"] = len(items)
        arms[arm] = block
        append_progress({"phase": f"arm_done_{arm}", "verdict": block["verdict"]["verdict"]})

    reused = reused_centering()
    overall = arms["real_forward"]["verdict"]["verdict"]
    report = {
        "probe": "taiji_gate_whitening",
        "date": "2026-11-08",
        "question": manifest["question"],
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "sha256": file_sha(MANIFEST),
            "written_before_any_reading": True,
        },
        "criteria": CRITERIA,
        "arms": arms,
        "arm_c_reused_centering": reused,
        "growth_implication": {
            "question": "must growth replace the encoding rather than apply another linear map?",
            "answer": (
                "yes if whitening cannot both repair the collapse and keep the carrier; the "
                "growth design's localized+centered encoding is then the candidate, not another "
                "linear transform on the averaged-pooling byte key"
            ),
            "evidence": "see arms.real_forward.verdict.{collapse_fixed,carrier_kept,trade_off}",
        },
        "verdict": {
            "real_forward": arms["real_forward"]["verdict"],
            "fixture_continuity": arms["fixture_continuity"]["verdict"],
            "overall": overall,
            "overall_source": "real_forward (the production read path)",
        },
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "living_core_sha256": file_sha(LIVING_CORE),
            "process_guard": process_guard,
            "single_instance_lock": lock_name,
            "watchdog": {
                "seconds": WATCHDOG_SECONDS,
                "fired": bool(WATCHDOG_FIRED[0]),
                "marker_exists": WATCHDOG_MARKER.exists(),
            },
            "write_then_rename": True,
        },
        "discipline": manifest["discipline"],
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap_seconds": CPU_CAP,
        },
    }
    write_json(REPORT, report)
    append_progress({"phase": "report_written", "verdict": overall})
    stop.set()

    real = arms["real_forward"]
    print(
        json.dumps(
            {
                "overall": overall,
                "substantive": real["verdict"]["substantive_verdict"],
                "real_written_bytes_same": real["verdict"]["written_bytes_identical"],
                "real_arm_a_gaps": real["arm_a"]["gaps"],
                "real_arm_b_gaps": real["arm_b"]["gaps"],
                "real_floor_a": real["arm_a"]["floor"],
                "real_floor_b": real["arm_b"]["floor"],
                "real_unwritten": [
                    real["arm_a"]["unwritten"]["correct"],
                    real["arm_b"]["unwritten"]["correct"],
                ],
                "collapse_fixed": real["verdict"]["collapse_fixed"],
                "carrier_kept": real["verdict"]["carrier_kept"],
                "cpu_seconds": report["cost"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
