"""Centering as the only variable: zero-training remeasure of the 154-query read set.

Arm A keeps the real ``store_query_key`` untouched.  Arm B subtracts the mean of the occupied
store keys from both the stored keys and the query key at read time; the encoder, the write
protocol, the gate form and its anchoring are unchanged.  Nothing is trained or optimised.
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
LIVING_CORE = ROOT / "taiji/living_core.py"
REPORT = ROOT / "reports/taiji_gate_centering_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_centering_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_centering_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_centering_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_centering_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_centering_20261108_watchdog.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
CLASSES = ("written", "unwritten", "new_token", "new_pairing", "no_value")
RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

CRITERIA = {
    "C1_control_identity": (
        "with center=None the runner's own read reproduces the real addressed_read on all 154 "
        "queries of both arms: identical fast sha, top slot, top/second cosine and margin"
    ),
    "C2_store_immutable": (
        "centering is read-time only: store_keys/store_values/store_age hashes are identical "
        "before and after every arm-B pass"
    ),
    "C3_threshold_source": (
        "the gate threshold is the minimum top cosine of the written group in that same arm and "
        "centering; no negative label enters the value"
    ),
    "C4_gap_sign": (
        "gap = written_floor - class_max_top_cosine per negative class, sign reported as measured; "
        "a non-positive gap is called 'not separable by this read gate'"
    ),
    "V1_written_bytes": (
        "if centering changes the returned fast sha of any of the 22 written faces against arm A, "
        "the verdict is centering_changes_written_bytes and no gate is proposed (dominant rule)"
    ),
    "V2_insufficient": (
        "otherwise, if all three gaps are <= 0 the verdict is centering_insufficient: the "
        "separability problem is not in the offset layer"
    ),
    "V3_enables": (
        "otherwise, if at least two gaps are > 0 and unwritten false refusals are <= 1/66, the "
        "verdict is centering_enables_gate and a threshold interval (lower, upper] is reported"
    ),
    "V4_interval_empty": (
        "if the V3 conditions hold but lower >= upper the verdict is centering_no_usable_interval"
    ),
    "V5_partial": "anything else is centering_partial with per-class signs reported",
    "unchanged": "no fixture, face, position, class or anchor is retuned after the first reading",
}


# ---------------------------------------------------------------------------
# Discipline helpers (same shape as the read-gate runner).
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


def reset_run_logs() -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    if WATCHDOG_MARKER.exists():
        WATCHDOG_MARKER.unlink()


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


# ---------------------------------------------------------------------------
# Base runner (fixtures, stores and the 154-query roster) reused verbatim.
# ---------------------------------------------------------------------------
def load_base():
    spec = importlib.util.spec_from_file_location("read_gate_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Read with an optional centering offset (read-time only; the store is untouched).
# ---------------------------------------------------------------------------
def occupied(state) -> torch.Tensor:
    return torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()


def store_center(state) -> torch.Tensor:
    return state.store_keys[occupied(state)].mean(dim=0)


def scores(subject, state, key: torch.Tensor, center: torch.Tensor | None) -> dict[str, Any]:
    valid = occupied(state)
    keys = state.store_keys[valid]
    shift = center if center is not None else torch.zeros_like(key)
    centered = keys - shift
    query = key - shift
    cosine = (centered @ query) / torch.clamp(centered.norm(dim=-1) * query.norm(), min=1e-12)
    ordered = torch.sort(cosine.detach(), descending=True).values
    top = int(torch.argmax(cosine.detach()))
    return {
        "top_cosine": float(cosine[top].detach()),
        "second_cosine": float(ordered[1].detach()) if ordered.numel() > 1 else None,
        "margin": float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None,
        "top_slot": int(valid[top]),
        "weights": [
            float(value.detach()) for value in torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0)
        ],
        "top_index": top,
    }


def read_fast(
    subject, state, key: torch.Tensor, center: torch.Tensor | None
) -> tuple[torch.Tensor, dict]:
    valid = occupied(state)
    keys = state.store_keys[valid]
    shift = center if center is not None else torch.zeros_like(key)
    centered = keys - shift
    query = key - shift
    cosine = (centered @ query) / torch.clamp(centered.norm(dim=-1) * query.norm(), min=1e-12)
    weights = torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0)
    retrieved = torch.einsum("i,i...->...", weights, state.store_values[valid])
    fast = (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved
    return fast, scores(subject, state, key, center)


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def store_signature(state) -> dict[str, str]:
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


def collect(subject, state, items, center) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        record = scores(subject, state, item_key(subject, state, item), center)
        record.update({"class": item["class"], "tag": item["tag"]})
        rows.append(record)
        touch()
    return rows


def item_key(subject, state, item: dict[str, Any]):
    key = item.get("key")
    if key is None:
        return subject.store_query_key(state, item.get("raw", b""))
    if isinstance(key, np.ndarray):
        return torch.tensor(key, dtype=subject.W.dtype, device=subject.W.device)
    return key.to(dtype=subject.W.dtype, device=subject.W.device)


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
            "top_cosines": tops,
        }
    gaps = {name: floor - groups[name]["top_cosine"]["max"] for name in CLASSES[2:]}
    return {
        "written_floor": floor,
        "groups": groups,
        "gaps": {name: round(value, 9) for name, value in gaps.items()},
        "separable": [name for name, value in gaps.items() if value > 0],
    }


def gate_pass(
    subject, state, items, center, threshold: float, baseline: dict[str, str]
) -> dict[str, Any]:
    matched = dict.fromkeys(CLASSES, 0)
    totals = dict.fromkeys(CLASSES, 0)
    written_bytes_same = 0
    written_slot_same = 0
    changed = []
    for item in items:
        key = item_key(subject, state, item)
        record = scores(subject, state, key, center)
        hit = bool(record["top_cosine"] >= threshold)
        totals[item["class"]] += 1
        if hit:
            matched[item["class"]] += 1
        if item["class"] == "written":
            fast, _ = read_fast(subject, state, key, center)
            if tensor_sha(fast) == baseline[item["tag"]]["fast_sha"]:
                written_bytes_same += 1
            else:
                changed.append(item["tag"])
            if record["top_slot"] == baseline[item["tag"]]["top_slot"]:
                written_slot_same += 1
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
    }


def baseline_written(subject, state, items) -> dict[str, Any]:
    out = {}
    for item in items:
        if item["class"] != "written":
            continue
        key = item_key(subject, state, item)
        after, info = subject.addressed_read(state, item.get("raw", b""), key=key)
        out[item["tag"]] = {
            "fast_sha": tensor_sha(after.fast),
            "top_slot": int(info["top_slot"]),
            "top_cosine": float(info["top_cosine"]),
        }
        touch()
    return out


def control_check(subject, state, items, center_none: list[dict[str, Any]]) -> dict[str, Any]:
    same_fast = 0
    same_slot = 0
    max_delta = 0.0
    for item, own in zip(items, center_none, strict=True):
        key = item_key(subject, state, item)
        after, info = subject.addressed_read(state, item.get("raw", b""), key=key)
        fast, _ = read_fast(subject, state, key, None)
        max_delta = max(max_delta, float((after.fast - fast).abs().max()))
        if tensor_sha(after.fast) == tensor_sha(fast):
            same_fast += 1
        if int(info["top_slot"]) == own["top_slot"]:
            same_slot += 1
    return {
        "queries": len(items),
        "fast_sha_identical": same_fast,
        "top_slot_identical": same_slot,
        "max_abs_delta": max_delta,
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
        "sigma1": round(float(singular[0]), 9),
        "sigma_last": round(float(singular[-1]), 9),
    }


def as_double(value) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.tensor(value, dtype=torch.float64)
    return value.detach().double()


def centroid_diagnostics(state, rows, negatives) -> dict[str, Any]:
    valid = occupied(state)
    keys = state.store_keys[valid].detach().double()
    center = keys.mean(dim=0)
    direction = center / torch.clamp(center.norm(), min=1e-12)
    key_cos = (keys @ direction) / torch.clamp(keys.norm(dim=-1), min=1e-12)
    mid_cos = []
    for index in range(len(negatives["no_value"])):
        first = as_double(rows[index]["key"])
        other = as_double(rows[(index + len(rows) // 2) % len(rows)]["key"])
        midpoint = first + other
        midpoint = midpoint / torch.clamp(midpoint.norm(), min=1e-12)
        mid_cos.append(float((midpoint @ direction) / torch.clamp(midpoint.norm(), min=1e-12)))
    return {
        "centroid_norm": round(float(center.norm()), 9),
        "stored_cosine_to_centroid": describe([float(value) for value in key_cos]),
        "no_value_midpoint_cosine_to_centroid": describe(mid_cos),
        "midpoint_above_stored_median": bool(np.median(mid_cos) > float(key_cos.median())),
    }


def verdict_block(stats: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    gaps = stats["gaps"]
    positive = [name for name, value in gaps.items() if value > 0]
    if gate["written_bytes_identical_to_arm_a"] < gate["groups"]["written"]["total"]:
        return {
            "verdict": "centering_changes_written_bytes",
            "reason": "centering altered the returned fast bytes of written faces; rule V1 wins",
            "positive_gaps": positive,
        }
    if not positive:
        return {
            "verdict": "centering_insufficient",
            "reason": "all three gaps stay non-positive; the problem is not the offset layer",
            "positive_gaps": [],
        }
    if len(positive) >= 2 and gate["unwritten_false_refusals"] <= 1:
        lower = max(stats["groups"][name]["top_cosine"]["max"] for name in positive)
        unwritten = sorted(
            row["top_cosine"] for row in gate["per_item"] if row["class"] == "unwritten"
        )
        upper = min(stats["written_floor"], unwritten[1] if len(unwritten) > 1 else unwritten[0])
        if lower < upper:
            return {
                "verdict": "centering_enables_gate",
                "reason": "at least two gaps are positive and at most one unwritten wording is refused",
                "positive_gaps": positive,
                "threshold_interval": {
                    "lower_open": round(float(lower), 9),
                    "upper_inclusive": round(float(upper), 9),
                    "width": round(float(upper - lower), 9),
                },
            }
        return {
            "verdict": "centering_no_usable_interval",
            "reason": "the rejecting lower bound is not below the accepting upper bound",
            "positive_gaps": positive,
            "threshold_interval": {
                "lower_open": round(float(lower), 9),
                "upper_inclusive": round(float(upper), 9),
                "width": round(float(upper - lower), 9),
            },
        }
    return {
        "verdict": "centering_partial",
        "reason": "fewer than two positive gaps, or more than one unwritten refusal",
        "positive_gaps": positive,
    }


def arm_block(subject, state, arm: str, items, rows, negatives) -> dict[str, Any]:
    center = store_center(state)
    before = store_signature(state)
    off = collect(subject, state, items, None)
    control = control_check(subject, state, items, off)
    baseline = baseline_written(subject, state, items)
    on = collect(subject, state, items, center)
    stats_off = group_stats(off)
    stats_on = group_stats(on)
    gate_off = gate_pass(subject, state, items, None, stats_off["written_floor"], baseline)
    gate_on = gate_pass(subject, state, items, center, stats_on["written_floor"], baseline)
    gate_on["per_item"] = on
    after = store_signature(state)
    valid = occupied(state)
    diagnostics = {
        "raw_keys": key_spectrum(state.store_keys[valid]),
        "centered_keys": key_spectrum(state.store_keys[valid] - center),
        "centroid": centroid_diagnostics(state, rows, negatives),
    }
    append_progress(
        {
            "phase": f"arm_{arm}",
            "control_fast_identical": control["fast_sha_identical"],
            "written_bytes_same": gate_on["written_bytes_identical_to_arm_a"],
        }
    )
    result = {
        "arm": arm,
        "center": "occupied_store_key_mean",
        "center_norm": round(float(center.norm()), 9),
        "control": control,
        "store_before": before,
        "store_after": after,
        "store_immutable": before == after,
        "arm_a": {
            "floor": round(stats_off["written_floor"], 9),
            "gaps": stats_off["gaps"],
            "separable": stats_off["separable"],
            "written_margin": describe([row["margin"] for row in off if row["class"] == "written"]),
            "all_margin": describe([row["margin"] for row in off]),
            "group_top": {name: stats_off["groups"][name]["top_cosine"] for name in CLASSES},
            "gate": {key: gate_off[key] for key in gate_off if key != "per_item"},
        },
        "arm_b": {
            "floor": round(stats_on["written_floor"], 9),
            "gaps": stats_on["gaps"],
            "separable": stats_on["separable"],
            "written_margin": describe([row["margin"] for row in on if row["class"] == "written"]),
            "all_margin": describe([row["margin"] for row in on]),
            "group_top": {name: stats_on["groups"][name]["top_cosine"] for name in CLASSES},
            "gate": {key: gate_on[key] for key in gate_on if key != "per_item"},
        },
        "diagnostics": diagnostics,
        "per_item_gate_on": [
            {
                "class": row["class"],
                "tag": row["tag"],
                "top_cosine": row["top_cosine"],
                "margin": row["margin"],
                "top_slot": row["top_slot"],
            }
            for row in on
        ],
    }
    result["verdict"] = verdict_block(stats_on, gate_on)
    return result


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("centering probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    reset_run_logs()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261108)
    base = load_base()
    base.build_fixture()

    manifest = {
        "format": "taiji-centering-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "with centering as the only variable, do the three negative classes separate from "
            "the written faces on the real read keys?"
        ),
        "written_before_any_reading": True,
        "sequence": [
            "build the real-forward store and the continuity store",
            "arm A: uncentered scores, gate anchored on the written floor",
            "arm B: center = mean of occupied store keys subtracted from keys and query",
            "control: center=None must reproduce the real addressed_read on all 154 queries",
            "verdict per the pre-written V1..V5 ordering (V1 dominant)",
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
        block = arm_block(subject, state, arm, items, rows, negatives)
        block["query_count"] = len(items)
        arms[arm] = block
        append_progress({"phase": f"arm_done_{arm}", "verdict": block["verdict"]["verdict"]})

    overall = arms["real_forward"]["verdict"]["verdict"]
    report = {
        "probe": "taiji_gate_centering",
        "date": "2026-11-08",
        "question": manifest["question"],
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "sha256": file_sha(MANIFEST),
            "written_before_any_reading": True,
        },
        "criteria": CRITERIA,
        "arms": arms,
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

    print(
        json.dumps(
            {
                "overall": overall,
                "real": overall,
                "continuity": arms["fixture_continuity"]["verdict"]["verdict"],
                "real_arm_a_gaps": arms["real_forward"]["arm_a"]["gaps"],
                "real_arm_b_gaps": arms["real_forward"]["arm_b"]["gaps"],
                "real_written_bytes_same": arms["real_forward"]["arm_b"]["gate"][
                    "written_bytes_identical_to_arm_a"
                ],
                "cpu_seconds": report["cost"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
